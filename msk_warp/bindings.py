import numpy as np
import opensim as osim
import torch

import msk_warp._src.forward as forward
import msk_warp._src.math as math
import msk_warp.utils.body_helper as body_helper
import msk_warp.utils.function_helper as function_helper
import msk_warp.utils.joint_helper as joint_helper
import msk_warp.utils.spatial_transform_helper as spatial_transform_helper
import msk_warp.utils.visual_helper as visual_helper
import msk_warp.utils.muscle_helper as muscle_helper
import msk_warp.utils.coordinate_force_helper as coordinate_force_helper
import msk_warp.utils.site_helper as site_helper
from msk_warp import Model, Data, MeshLoadResult, GeomType, IntegratorType, Option, ContactType, LimitType, \
    ActivationType, MobilizerType, MetabolicOptions, MuscleMetadata, ActuatorMetadata, IntegratorStateScratch, \
    IntegratorDotScratch, \
    MuscleLengthInfo, FiberVelocityInfo, MuscleDynamicsInfo, Contact, SpatialInertia, ArticulatedInertia, TileBlockDim, \
    vec5
from msk_warp.render.renderer import Renderer, RendererType
from msk_warp.utils.converted_objects import *
from msk_warp.utils.kinematic_tree import KinematicTree
from msk_warp.utils.python_util import string_list_to_ordering, apply_map_to_list, gather, \
    exclusive_scan, create_nested_list
from msk_warp.utils.warp_util import to_warp_array, make_full, make_zero


@dataclass
class ModelLoadResult:
    model: Model
    data: Data
    body_id_lookup: dict[str, int]
    dof_id_lookup: dict[str, int]
    qpos_id_lookup: dict[str, int]
    limit_id_lookup: dict[str, int]
    muscle_id_lookup: dict[str, int]
    actuator_id_lookup: dict[str, int]
    collider_id_lookup: dict[str, int]
    visuals: list[MeshLoadResult]


def prepare_contacts(
        geom_data: GeomData,
        body_parent_ids: list[int],
        ngeom: int,
):
    # precalculated geom pairs
    geom1, geom2 = np.triu_indices(ngeom, k=1)
    nxn_geom_pair = np.stack((geom1, geom2), axis=1)

    # Contact pair id: -1 if not pre-defined, -2 if skipped, id otherwise
    nxn_pairid_contact = -1 * np.ones(len(geom1), dtype=int)

    # filter out parent-child collisions and self-collisions
    geom_bodyid = np.array(geom_data.body_id)
    geom_pc_filter = np.array(geom_data.pc_filter)
    body_parentid = np.array(body_parent_ids)
    bodyid1, bodyid2 = geom_bodyid[geom1], geom_bodyid[geom2]
    parentid1, parentid2 = body_parentid[bodyid1], body_parentid[bodyid2]
    pc_filter1, pc_filter2 = geom_pc_filter[geom1], geom_pc_filter[geom2]

    self_collision = (bodyid1 == bodyid2)
    # Mask for whether collision is between parent-child
    parent_child_collision = (
            ((bodyid1 == parentid2) & (bodyid1 != 0))
            | ((bodyid2 == parentid1) & (bodyid2 != 0))
    )
    # For any parent-child collision, if both geoms have pc_filter=False, ensure the collision happens
    parent_child_collision &= ~(~pc_filter1 & ~pc_filter2)

    nxn_pairid_contact[parent_child_collision | self_collision] = -2
    nxn_pairid_collision = -1 * np.ones(len(geom1), dtype=int)
    include = (nxn_pairid_contact > -2) | (nxn_pairid_collision >= 0)
    nxn_pairid = np.hstack([nxn_pairid_contact.reshape((-1, 1)), nxn_pairid_collision.reshape((-1, 1))])
    nxn_pairid_filtered = nxn_pairid[include]
    nxn_geom_pair_filtered = nxn_geom_pair[include]

    # count contact pair types
    geom_types = geom_data.geom_type
    geom_type_pair_count = np.bincount([
        math.upper_trid_index(len(GeomType),
                              int(geom_types[geom1[i]]),
                              int(geom_types[geom2[i]]))
        for i in np.arange(len(geom1))
        if nxn_pairid_contact[i] > -2 or nxn_pairid_collision[i] > -1
    ], minlength=len(GeomType) * (len(GeomType) + 1) // 2, )
    return geom_types, geom_type_pair_count, nxn_geom_pair_filtered, nxn_pairid_filtered


def get_num_scratch_states(integrator: IntegratorType) -> tuple[int, int]:
    """ Returns number of additional copies of state and state_dot for integration """
    if integrator == IntegratorType.RK4_ADAPTIVE:
        return 2, 5
    elif integrator == IntegratorType.EULER_ADAPTIVE:
        return 2, 1
    else:
        return 0, 0


def load_model(
        model_path: str,
        n_worlds: int,
        integrator: IntegratorType,
        polynomial_data_path: str = None,
        render_kinematic_tree: bool = True,
) -> ModelLoadResult:
    # All the mesh files for visuals should be located here
    osim.ModelVisualizer.addDirToGeometrySearchPaths("data/geometry")
    model = osim.Model(model_path)
    model.initSystem()

    # Check every body has a mobilizer
    if model.getNumBodies() != model.getNumJoints():
        raise ValueError(f"Num bodies ({model.getNumBodies()}) does not match num Joints ({model.getNumJoints()})")

    # Parse bodies, joints, collision geometry, visuals
    converted_bodies = [GROUND_BODY] + [body_helper.convert_body(body) for body in model.getBodyList()]
    converted_joints = [GROUND_JOINT] + [joint_helper.convert_joint(joint) for joint in model.getJointList()]
    # converted_geoms = geom_helper.convert_geoms(model)
    converted_geoms = []  # TODO
    converted_visuals = visual_helper.convert_visuals(model)
    converted_spatial_transforms = spatial_transform_helper.convert_spatial_transforms(model)
    converted_dampers = coordinate_force_helper.convert_coordinate_linear_damper(model)
    converted_springs = coordinate_force_helper.convert_coordinate_linear_spring(model)
    converted_stops = coordinate_force_helper.convert_coordinate_linear_stop(model)
    converted_muscles = muscle_helper.convert_muscles(model)
    # any sites that aren't part of muscle paths + muscle path points. Note: muscle path points must come first
    converted_sites = muscle_helper.flatten_sites(converted_muscles) + site_helper.convert_sites(model)

    # Create a lookup from body name -> body data. Needed for fast joint->body lookup
    body_name_to_body = {body.name: body for body in converted_bodies}

    # Build the kinematic tree, storing the joint that connects each node to its parent
    tree = KinematicTree(root_body=body_name_to_body[GROUND], root_joint=converted_joints[0])
    for joint in converted_joints:
        if joint.parent_body_name != GROUND_PARENT:
            parent_body = body_name_to_body[joint.parent_body_name]
            child_body = body_name_to_body[joint.child_body_name]
            tree.add_edge(parent_body, child_body, joint)

    tree.verify()
    if render_kinematic_tree:
        tree.render()  # graphviz is such a great tool

    # Using the kinematic tree, compute a forward ordering, create an ordered list of bodies and joints
    tree_ordering = tree.forward_ordering()
    ordered_bodies = [node.body for node in tree_ordering]
    ordered_joints = [node.joint for node in tree_ordering]
    # Also re-order the spatial transforms
    joint_ordering = joint_helper.compute_joint_name_ordering(ordered_joints)
    ordered_spatial_transforms = spatial_transform_helper.order_spatial_transforms(
        converted_spatial_transforms, joint_ordering)

    # Body id -> parent id
    ordered_bodies_names = [body.name for body in ordered_bodies]
    body_parent_id = [joint_helper.get_joint_parent_id(joint, ordered_bodies_names) for joint in ordered_joints]

    # Body name -> body id
    body_ordering = string_list_to_ordering(ordered_bodies_names)

    # Create the "body-level" array (contains list of all bodies at level i)
    body_tree = tree.create_body_tree()
    body_tree_indices = [apply_map_to_list(level, body_ordering) for level in body_tree]
    body_tree_warp = tuple([to_warp_array(level, dtype=int) for level in body_tree_indices])

    # Get all the children of each body
    body_children = [node.get_children_no_roots() for node in tree_ordering]
    body_children_names = [[node.body.name for node in children] for children in body_children]
    body_children_indices = [apply_map_to_list(children, body_ordering) for children in body_children_names]
    # Flatten list, compute number of children, get address
    body_children_flattened = [child_idx for children_indices in body_children_indices for child_idx in
                               children_indices]
    body_children_num = [len(children) for children in body_children_indices]
    body_children_adr = exclusive_scan(body_children_num)

    # Starting address of joint's coordinates/speeds
    mob_qpos_adr, mob_dof_adr = joint_helper.compute_qpos_dof_adr(ordered_joints)

    # Create the *global* ordering lookup for each coordinate in qpos, dof.
    qpos_ordering = joint_helper.get_global_qpos_ordering_lookup(ordered_joints)
    dof_ordering = joint_helper.get_global_dof_ordering_lookup(ordered_joints)
    # Ordering lookup for qpos, dof relative to each joint. only really need one of these
    relative_qpos_ordering = joint_helper.get_relative_qpos_ordering_lookup(ordered_joints)
    relative_dof_ordering = joint_helper.get_relative_dof_ordering_lookup(ordered_joints)

    # Index of mobilizer -> index of custom joint (-1 if not custom)
    mob_to_cst_idx, cst_to_mob_idx = joint_helper.compute_mobilizer_custom_joint_index(ordered_joints)
    n_custom_jnts = joint_helper.compute_num_custom_joints(ordered_joints)

    # Spatial transforms: flatten all the axes
    ordered_transform_axes = spatial_transform_helper.get_flattened_transform_axes(ordered_spatial_transforms)

    nq = sum([joint.num_coordinates for joint in ordered_joints])
    nv = sum([joint.num_speeds for joint in ordered_joints])
    nb = len(ordered_bodies)
    nmuscle = len(converted_muscles)
    ngeom = len(converted_geoms)
    nvis = len(converted_visuals)
    nsite = len(converted_sites)
    nlinearstop = len(converted_stops)

    nz = nmuscle

    use_fn_path = False
    # needs shapes
    opt = Option(
        gravity=-9.80665,
        explicit_gravity=True,
        contact_type=ContactType.HUNT_CROSSLEY,
        limit_type=LimitType.EXPONENTIAL,
        activation_type=ActivationType.MILLARD,
        integrator=integrator,

        enable_drag=True,

        use_fn_path=use_fn_path,
        metabolic_options=MetabolicOptions(
            activation_maintenance_rate_on=True,
            shortening_rate_on=True,
            mechanical_work_rate_on=True,
            enforce_minimum_heat_rate=True,

            aerobic_factor=1.0,
            muscle_effort_scaling_factor=1.0,
            use_bhargava_recruitment=True,
            include_negative_mechanical_work=True,
            forbid_negative_total_power=True,
        ),

        safety=0.9,
        min_shrink=0.1,
        max_grow=5.0,
        hysteresis_low=0.9,
        hysteresis_high=1.2,
        accuracy=0.01,
        use_inf_norm=False,

        qvel_weights=wp.full(nv, 1.0, dtype=float),
        z_weights=wp.full(nz, 1.0, dtype=float),

        visuals=True
    )

    actuator_data = []  # fixme
    am = wp.array(actuator_data, dtype=ActuatorMetadata)

    nactuators = 0  # fixme
    geom_type_pair_count = []  # fixme
    nxn_geom_pair_filtered = []
    nxn_pairid_filtered = []
    naconmax = max(512, n_worlds * 32)

    # fixme
    qpos0 = [0.0] * nq
    root_free = MobilizerType.FREE in [joint.mob_type for joint in ordered_joints]
    if root_free:
        qpos0[0:4] = [0.0, 0.0, 0.0, 1.0]  # Root orientation (quaternion)
    qvel0 = [0.0] * nv

    dt = 1.0 / 500.0

    # Flatten out entries of dataclasses into lists
    body_data = dataclass_list_transpose(ordered_bodies, cls=BodyData)
    joint_data = dataclass_list_transpose(ordered_joints, cls=JointData)
    txfm_data = dataclass_list_transpose(ordered_transform_axes, cls=TransformAxisData)
    geom_data = dataclass_list_transpose(converted_geoms, cls=GeomData)
    vis_data = dataclass_list_transpose(converted_visuals, cls=VisualData)
    site_data = dataclass_list_transpose(converted_sites, cls=SiteData)
    stop_data = dataclass_list_transpose(converted_stops, cls=CoordinateLinearStopData)

    # Gather functions
    linear_fns, linear_fns_idx = function_helper.get_functions_of_type(ordered_transform_axes, cls=LinearFunctionData)
    const_fns, const_fns_idx = function_helper.get_functions_of_type(ordered_transform_axes, cls=ConstantFunctionData)
    poly_fns, poly_fns_idx = function_helper.get_functions_of_type(ordered_transform_axes, cls=PolynomialFunctionData)
    nlinearfn, nconstfn, npolyfn = len(linear_fns), len(const_fns), len(poly_fns)
    nfunctions = nlinearfn + nconstfn + npolyfn

    # Get all "relative" coordinate indices for each transform axis
    txfm_dofs = spatial_transform_helper.get_txfm_coordinate_names(ordered_transform_axes)
    txfm_qpos_relative_idx = apply_map_to_list(txfm_dofs, relative_dof_ordering)
    txfm_qpos_global_idx = apply_map_to_list(txfm_dofs, qpos_ordering)
    # Now, use gather to find the relative coordinate indices used for each function
    linear_fns_qpos_global_idx = gather(txfm_qpos_global_idx, linear_fns_idx)
    poly_fns_qpos_global_idx = gather(txfm_qpos_global_idx, poly_fns_idx)

    # We need to reshape the transform data to be (num_custom_joints, 6)
    cst_txfm_axes = create_nested_list(txfm_data["axis"], num_per_sublist=6)
    cst_txfm_dof = create_nested_list(txfm_qpos_relative_idx, num_per_sublist=6)

    linear_fn_mb = function_helper.get_linear_fn_mb(linear_fns)
    const_fn_vals = function_helper.get_const_fn_vals(const_fns)
    poly_coeffs = function_helper.get_flattened_poly_coeffs(poly_fns)
    poly_coeffs_num, poly_coeffs_adr = function_helper.get_poly_coeffs_num_adr(poly_fns)

    geom_body_id = apply_map_to_list(geom_data["body_name"], body_ordering)
    vis_body_id = apply_map_to_list(vis_data["body_name"], body_ordering)
    site_body_id = apply_map_to_list(site_data["body_name"], body_ordering)

    # Joint limits
    dof_damping = coordinate_force_helper.get_dof_damping(converted_dampers, dof_ordering)
    dof_stiffness = coordinate_force_helper.get_dof_stiffness(converted_springs, dof_ordering)
    qpos_spring_rest = coordinate_force_helper.get_qpos_spring_rest(converted_springs, qpos_ordering)
    stop_qpos_adr = apply_map_to_list(stop_data["coordinate"], qpos_ordering)
    stop_dof_adr = apply_map_to_list(stop_data["coordinate"], dof_ordering)

    # Muscles/sites
    muscle_pts_num = muscle_helper.get_muscle_pts_num(converted_muscles)
    muscle_pts_adr = exclusive_scan(muscle_pts_num)
    muscle_data = muscle_helper.create_muscle_metadata(converted_muscles)
    mm = wp.array(muscle_data, dtype=MuscleMetadata)

    m = Model(
        nbody=nb,
        nq=nq,
        nv=nv,
        nmuscle=nmuscle,
        nactuator=nactuators,
        nz=nz,
        njnts_cst=n_custom_jnts,
        ngeom=ngeom,
        nvis=nvis,
        nsite=nsite,
        nlinearstop=nlinearstop,
        nfunctions=nfunctions,
        nlinearfn=nlinearfn,
        nconstfn=nconstfn,
        npolyfn=npolyfn,

        opt=opt,
        muscle_metadata=mm,
        muscle_data=muscle_data,

        actuator_metadata=am,

        # warp arrays
        body_mass=to_warp_array(body_data["mass"], dtype=float),
        body_mass_center=to_warp_array(body_data["mass_center"], dtype=wp.vec3),
        body_unit_inertia_OB_B=to_warp_array(body_data["unit_inertia_OB_B"], dtype=wp.mat33),

        body_parentid=to_warp_array(body_parent_id, dtype=int),
        mob_type=to_warp_array(joint_data["mob_type"], dtype=int),
        mob_qposadr=to_warp_array(mob_qpos_adr, dtype=int),
        mob_dofadr=to_warp_array(mob_dof_adr, dtype=int),
        mob_dofnum=to_warp_array(joint_data["num_speeds"], dtype=int),
        mob_X_PF=to_warp_array(joint_data["transform_PF"], dtype=wp.transform),
        mob_X_MB=to_warp_array(joint_data["transform_MB"], dtype=wp.transform),
        mob_extra_info=to_warp_array(joint_data["extra_info"], dtype=wp.vec3),

        mob_to_cst_id=to_warp_array(mob_to_cst_idx, dtype=int),
        cst_to_mob_id=to_warp_array(cst_to_mob_idx, dtype=int),
        cst_txfm_axes=to_warp_array(cst_txfm_axes, dtype=wp.vec3),
        cst_txfm_dof=to_warp_array(cst_txfm_dof, dtype=int),

        linear_fn_mb=to_warp_array(linear_fn_mb, dtype=wp.vec2),
        const_fn_c=to_warp_array(const_fn_vals, dtype=float),
        poly_fn_coeff=to_warp_array(poly_coeffs, dtype=float),
        poly_fn_coeff_adr=to_warp_array(poly_coeffs_adr, dtype=int),
        poly_fn_coeff_num=to_warp_array(poly_coeffs_num, dtype=int),
        linear_fn_adr=to_warp_array(linear_fns_idx, dtype=int),
        const_fn_adr=to_warp_array(const_fns_idx, dtype=int),
        poly_fn_adr=to_warp_array(poly_fns_idx, dtype=int),
        linear_fn_qpos_adr=to_warp_array(linear_fns_qpos_global_idx, dtype=int),
        poly_fn_qpos_adr=to_warp_array(poly_fns_qpos_global_idx, dtype=int),

        dof_damping=to_warp_array(dof_damping, dtype=float),
        dof_stiffness=to_warp_array(dof_stiffness, dtype=float),
        qpos_spring_rest=to_warp_array(qpos_spring_rest, dtype=float),

        stop_qpos_range=to_warp_array(stop_data["range"], dtype=wp.vec2),
        stop_qpos_adr=to_warp_array(stop_qpos_adr, dtype=int),
        stop_dof_adr=to_warp_array(stop_dof_adr, dtype=int),
        stop_dof_stiffness_damping=to_warp_array(stop_data["stiffness_damping"], dtype=wp.vec2),

        geom_type=to_warp_array(geom_data["geom_type"], dtype=int),
        geom_bodyid=to_warp_array(geom_body_id, dtype=int),
        geom_X_loc=to_warp_array(geom_data["transform"], dtype=wp.transform),
        geom_size=to_warp_array(geom_data["size"], dtype=wp.vec3),
        geom_friction=to_warp_array(geom_data["friction"], dtype=wp.vec3),
        geom_stiffness=to_warp_array(geom_data["stiffness"], dtype=float),
        geom_dissipation=to_warp_array(geom_data["dissipation"], dtype=float),
        geom_transition_velocity=to_warp_array(geom_data["transition_velocity"], dtype=float),
        geom_priority=to_warp_array(geom_data["priority"], dtype=int),
        geom_aabb=to_warp_array(geom_data["aabb"], dtype=wp.vec3),
        geom_rbound=to_warp_array(geom_data["rbound"], dtype=float),

        geom_pair_type_count=tuple(geom_type_pair_count),
        nxn_geom_pair_filtered=wp.array(nxn_geom_pair_filtered, dtype=wp.vec2i),
        nxn_pairid_filtered=wp.array(nxn_pairid_filtered, dtype=wp.vec2i),

        vis_bodyid=to_warp_array(vis_body_id, dtype=int),
        vis_X_loc=to_warp_array(vis_data["transform"], dtype=wp.transform),

        site_bodyid=to_warp_array(site_body_id, dtype=int),
        site_pos=to_warp_array(site_data["offset"], dtype=wp.vec3),

        muscle_pts_num=to_warp_array(muscle_pts_num, dtype=int),
        muscle_pts_adr=to_warp_array(muscle_pts_adr, dtype=int),
        muscle_poly_coeffs=to_warp_array([], dtype=float),  # TODO
        muscle_poly_adr=to_warp_array([], dtype=int),
        muscle_poly_order=to_warp_array([], dtype=int),
        muscle_poly_qpos_adr=to_warp_array([], dtype=int),
        muscle_poly_dof_adr=to_warp_array([], dtype=int),
        muscle_dep_dof_num=to_warp_array([], dtype=int),
        muscle_dep_dof_adr=to_warp_array([], dtype=int),

        body_tree=body_tree_warp,
        body_children=to_warp_array(body_children_flattened, dtype=int),
        body_children_num=to_warp_array(body_children_num, dtype=int),
        body_children_adr=to_warp_array(body_children_adr, dtype=int),
        block_dim=TileBlockDim(),
    )

    n_int_states, n_int_dot_states = get_num_scratch_states(integrator)
    integrator_scratch = [
        IntegratorStateScratch(
            time=make_zero(n_worlds, dtype=float),
            qpos=make_zero((n_worlds, nq), dtype=float),
            qvel=make_zero((n_worlds, nv), dtype=float),
            m_state=make_zero((n_worlds, nmuscle), dtype=float),
            m_act=make_zero((n_worlds, nmuscle), dtype=float),
            a_act=make_zero((n_worlds, nactuators), dtype=float),
        ) for _ in range(n_int_states)
    ]

    integrator_dot_scratch = [
        IntegratorDotScratch(
            qvel=make_zero((n_worlds, nv), dtype=float),
            qacc=make_zero((n_worlds, nv), dtype=float),
            m_state_dot=make_zero((n_worlds, nmuscle), dtype=float),
            m_act_dot=make_zero((n_worlds, nmuscle), dtype=float),
            a_act_dot=make_zero((n_worlds, nactuators), dtype=float),
        ) for _ in range(n_int_dot_states)
    ]

    # Custom joints may need up to 6 additional vectors: [f(q), f'(q), f''(q)] for each 6 functions
    num_scratch = 3 if n_custom_jnts == 0 else 6

    d = Data(
        world_reset=make_full(True, n_worlds, dtype=bool),
        time=make_zero(n_worlds, dtype=float),

        # for adaptive integrators
        integrator_scratch=integrator_scratch,
        integrator_dot_scratch=integrator_dot_scratch,
        qvel_buffer=make_zero((n_worlds, nv), dtype=float),

        time1=make_zero(n_worlds, dtype=float),
        next_time=make_zero(n_worlds, dtype=float),
        step_size=make_full(dt, (n_worlds,), dtype=float),
        actual_step_size=make_full(dt, (n_worlds,), dtype=float),
        artificially_limited=make_zero(n_worlds, dtype=bool),
        step_accepted=make_zero(n_worlds, dtype=bool),
        integration_done=make_zero(n_worlds, dtype=bool),
        nintegrating=make_zero(1, dtype=int),

        qvel_scales=make_full(1.0, (n_worlds, nv), dtype=float),
        z_scales=make_full(1.0, (n_worlds, nz), dtype=float),
        qpos_diff=make_zero((n_worlds, nq), dtype=float),
        ninv_dq_tmp=make_zero((n_worlds, nv), dtype=float),
        qpos_diff_scaled=make_zero((n_worlds, nq), dtype=float),
        qvel_diff=make_zero((n_worlds, nv), dtype=float),
        z_diff=make_zero((n_worlds, nz), dtype=float),
        qpos_err=make_zero((n_worlds,), dtype=float),
        qvel_err=make_zero((n_worlds,), dtype=float),
        z_err=make_zero((n_worlds,), dtype=float),
        error=make_zero(n_worlds, dtype=float),
        steps_attempted=make_zero(n_worlds, dtype=int),

        qpos=wp.array(np.tile(qpos0, (n_worlds, 1)), dtype=float),
        qvel=wp.array(np.tile(qvel0, (n_worlds, 1)), dtype=float),
        m_act=make_zero((n_worlds, nmuscle), dtype=float),
        a_act=make_full(0.5, (n_worlds, nactuators), dtype=float),
        m_state=make_zero((n_worlds, nmuscle), dtype=float),

        grf=make_zero((n_worlds,), dtype=wp.vec3),
        joint_moments=make_zero((n_worlds, nv), dtype=float),

        qacc=make_zero((n_worlds, nv), dtype=float),
        m_act_dot=make_zero((n_worlds, nmuscle), dtype=float),
        a_act_dot=make_zero((n_worlds, nactuators), dtype=float),
        m_excitations=make_full(0.5, (n_worlds, nmuscle), dtype=float),
        a_excitations=make_zero((n_worlds, nactuators), dtype=float),
        m_state_dot=make_zero((n_worlds, nmuscle), dtype=float),

        cst_fn_output=make_zero((n_worlds, nfunctions), dtype=wp.vec3),

        mob_X_GB=make_zero((n_worlds, nb), dtype=wp.transform),
        mob_X_FM=make_zero((n_worlds, nb), dtype=wp.transform),
        mob_X_PB=make_zero((n_worlds, nb), dtype=wp.transform),
        mob_scratch=make_zero((n_worlds, nb, num_scratch), dtype=wp.vec3),
        mob_phi=make_zero((n_worlds, nb), dtype=wp.vec3),
        mob_coriolis_acc=make_zero((n_worlds, nb), dtype=wp.spatial_vector),

        body_COM_G=make_zero((n_worlds, nb), dtype=wp.vec3),
        body_Mk_G=make_zero((n_worlds, nb), dtype=SpatialInertia),
        body_P=make_zero((n_worlds, nb), dtype=ArticulatedInertia),
        body_PPlus=make_zero((n_worlds, nb), dtype=ArticulatedInertia),
        body_V_FM=make_zero((n_worlds, nb), dtype=wp.spatial_vector),
        body_V_PB_G=make_zero((n_worlds, nb), dtype=wp.spatial_vector),
        body_V_GB=make_zero((n_worlds, nb), dtype=wp.spatial_vector),
        body_VD_PB_G=make_zero((n_worlds, nb), dtype=wp.spatial_vector),
        body_A_GB=make_zero((n_worlds, nb), dtype=wp.spatial_vector),
        body_eps=make_zero((n_worlds, nb), dtype=wp.spatial_vector),

        body_gyro_force=make_zero((n_worlds, nb), dtype=wp.spatial_vector),
        body_total_coriolis_acc=make_zero((n_worlds, nb), dtype=wp.spatial_vector),
        body_total_centrifugal_force=make_zero((n_worlds, nb), dtype=wp.spatial_vector),
        body_articulated_centrifugal_force=make_zero((n_worlds, nb), dtype=wp.spatial_vector),
        body_zPlus=make_zero((n_worlds, nb), dtype=wp.spatial_vector),

        geom_X=make_zero((n_worlds, ngeom), dtype=wp.transform),
        geom_cforce=make_zero((n_worlds, ngeom), dtype=float),

        vis_X=make_zero((n_worlds, nvis), dtype=wp.transform),

        site_rpos=make_zero((n_worlds, nsite), dtype=wp.vec3),
        site_xpos=make_zero((n_worlds, nsite), dtype=wp.vec3),
        site_xvel=make_zero((n_worlds, nsite), dtype=wp.vec3),
        site_active=make_zero((n_worlds, nsite), dtype=bool),

        muscle_active_sites=make_zero((n_worlds, nsite), dtype=int),
        muscle_num_active=make_zero((n_worlds, nmuscle), dtype=int),
        muscle_moment_arm=make_zero((n_worlds, nmuscle, nv), dtype=float),

        site_diff_vec=make_zero((n_worlds, max(0, nsite - 1)), dtype=wp.vec3),
        site_diff_len=make_zero((n_worlds, max(0, nsite - 1)), dtype=float),
        site_diff_vel=make_zero((n_worlds, max(0, nsite - 1)), dtype=float),

        mob_H_FM=make_zero((n_worlds, nv), dtype=wp.spatial_vector),
        mob_H=make_zero((n_worlds, nv), dtype=wp.spatial_vector),
        mob_HDot_FM=make_zero((n_worlds, nv), dtype=wp.spatial_vector),
        mob_HDot=make_zero((n_worlds, nv), dtype=wp.spatial_vector),

        mob_G=make_zero((n_worlds, nv), dtype=wp.spatial_vector),
        mob_DI=make_zero((n_worlds, nv), dtype=wp.spatial_vector),

        muscle_length=make_zero((n_worlds, nmuscle), dtype=float),
        muscle_velocity=make_zero((n_worlds, nmuscle), dtype=float),
        muscle_actuation=make_zero((n_worlds, nmuscle), dtype=float),
        muscle_metabolic=make_zero((n_worlds, nmuscle), dtype=float),

        muscle_length_info=make_zero((n_worlds, nmuscle), dtype=MuscleLengthInfo),
        muscle_velocity_info=make_zero((n_worlds, nmuscle), dtype=FiberVelocityInfo),
        muscle_dynamics_info=make_zero((n_worlds, nmuscle), dtype=MuscleDynamicsInfo),

        body_F=make_zero((n_worlds, nb), dtype=wp.spatial_vector),
        body_F_gravity=make_zero((n_worlds, nb), dtype=wp.spatial_vector),
        xfrc_applied=make_zero((n_worlds, nb), dtype=wp.spatial_vector),
        body_F_contact=make_zero((n_worlds, nb), dtype=wp.spatial_vector),
        body_F_drag=make_zero((n_worlds, nb), dtype=wp.spatial_vector),
        body_F_muscle=make_zero((n_worlds, nb), dtype=wp.spatial_vector),

        qfrc_applied=make_zero((n_worlds, nv), dtype=float),
        qfrc_spring=make_zero((n_worlds, nv), dtype=float),
        qfrc_damper=make_zero((n_worlds, nv), dtype=float),
        qfrc_muscle=make_zero((n_worlds, nv), dtype=float),
        qfrc_actuator=make_zero((n_worlds, nv), dtype=float),
        qfrc_limit=make_zero((n_worlds, nv), dtype=float),

        qfrc_total=make_zero((n_worlds, nv), dtype=float),

        contact=Contact(
            dist=make_zero(naconmax, dtype=float),
            pos=make_zero(naconmax, dtype=wp.vec3),
            frame=make_zero(naconmax, dtype=wp.mat33),
            friction=make_zero(naconmax, dtype=vec5),
            dim=make_zero(naconmax, dtype=int),
            curvature=make_zero(naconmax, dtype=float),
            stiffness=make_zero(naconmax, dtype=float),
            dissipation=make_zero(naconmax, dtype=float),
            transition_velocity=make_zero(naconmax, dtype=float),
            geom=make_zero(naconmax, dtype=wp.vec2i),
            worldid=make_zero(naconmax, dtype=int),
        ),

        nworld=n_worlds,
        naconmax=naconmax,
        nacon=make_zero(n_worlds, dtype=int),

        # collision driver
        collision_pair=wp.zeros((naconmax,), dtype=wp.vec2i),
        collision_pairid=wp.zeros((naconmax,), dtype=wp.vec2i),
        collision_worldid=wp.zeros((naconmax,), dtype=int),
        ncollision=wp.zeros((1,), dtype=int),
    )

    # forward.reset(m, d)

    mesh_load_results = []
    for visual in converted_visuals:
        mesh_load_results.append(
            MeshLoadResult(
                file=visual.mesh_file,
                scale=visual.scale_factors
            )
        )

    return ModelLoadResult(
        model=m,
        data=d,
        body_id_lookup=body_ordering,
        qpos_id_lookup=qpos_ordering,
        dof_id_lookup=dof_ordering,
        limit_id_lookup={},
        muscle_id_lookup={},
        actuator_id_lookup={},
        collider_id_lookup={},
        visuals=mesh_load_results
    )


def reinitialize_model(
        m: Model,
        d: Data,
):
    """ Re-initialize the model (ie any parameters have changed). """
    # Ensure the muscle metadata is up to date
    mm = wp.array(m.muscle_data, dtype=MuscleMetadata)
    m.muscle_metadata = mm

    d.world_reset.fill_(True)
    forward.reset(m, d)


def create_renderer(
        load_result: ModelLoadResult,
        renderer_type: RendererType,
        draw_colliders: bool,
        draw_visuals: bool,
        draw_muscles: bool,
        draw_body_mass: bool,
        draw_beams: bool,
):
    viewer = Renderer(
        m=load_result.model,
        renderer_type=renderer_type,
        draw_colliders=draw_colliders,
        draw_visuals=draw_visuals,
        draw_muscles=draw_muscles,
        draw_body_mass=draw_body_mass,
        draw_beams=draw_beams
    )
    viewer.load_meshes(load_result.visuals)
    return viewer


def set_reset(d: Data, reset_worlds: torch.Tensor):
    d_reset_torch = wp.to_torch(d.world_reset)
    d_reset_torch[:] = reset_worlds.ravel()


# --- Model Fields ---
def damping(m: Model) -> torch.Tensor:
    return wp.to_torch(m.dof_damping)


def stiffness(m: Model) -> torch.Tensor:
    return wp.to_torch(m.jnt_stiffness)


def body_mass(m: Model) -> torch.Tensor:
    return wp.to_torch(m.body_mass)


def get_num_qpos(m: Model) -> int:
    return m.nq


def get_num_dofs(m: Model) -> int:
    return m.nv


def get_num_bodies(m: Model) -> int:
    return m.nbody


def get_num_visuals(m: Model) -> int:
    return m.nvis


def get_num_colliders(m: Model) -> int:
    return m.ngeom


def get_num_muscles(m: Model) -> int:
    return m.nmuscle


def get_num_actuators(m: Model) -> int:
    return m.nactuator


def get_num_limits(m: Model) -> int:
    return m.ndoflimit


def get_qpos_adr(m: Model, body_id: int) -> torch.Tensor:
    mob_qpos_adr = wp.to_torch(m.mob_qposadr)
    return mob_qpos_adr[body_id]


def get_dof_adr(m: Model, body_id: int) -> torch.Tensor:
    jnt_dof_adr = wp.to_torch(m.mob_dofadr)
    return jnt_dof_adr[body_id]


def get_qpos_num(m: Model, body_id: int) -> torch.Tensor:
    mob_qpos_num = wp.to_torch(m.mob_dofnum)
    return mob_qpos_num[body_id]


def get_dof_num(m: Model, body_id: int) -> torch.Tensor:
    jnt_dof_num = wp.to_torch(m.mob_dofnum)
    return jnt_dof_num[body_id]


def muscle_metadata(m: Model) -> list[MuscleMetadata]:
    return m.muscle_data


def gravity(m: Model) -> float:
    return m.opt.gravity


def set_drag_enabled(m: Model, enabled: bool):
    m.opt.enable_drag = enabled


def set_contact_type(m: Model, contact_type: ContactType):
    m.opt.contact_type = contact_type


def set_limit_type(m: Model, limit_type: LimitType):
    m.opt.limit_type = limit_type


def set_activation_type(m: Model, activation_type: ActivationType):
    m.opt.activation_type = activation_type


def steps_attempted(d: Data) -> torch.Tensor:
    return wp.to_torch(d.steps_attempted)


def set_integrator_accuracy(m: Model, accuracy: float):
    m.opt.accuracy = accuracy


def set_integrator_use_inf_norm(m: Model, use_inf_norm: bool):
    m.opt.use_inf_norm = use_inf_norm


def is_adaptive(integrator_type: IntegratorType) -> bool:
    return integrator_type in [
        IntegratorType.EULER_ADAPTIVE,
        IntegratorType.RK4_ADAPTIVE,
    ]


def joint_limit_ranges(m: Model) -> torch.Tensor:
    return wp.to_torch(m.limit_dof_range)


def joint_limit_qadr(m: Model) -> torch.Tensor:
    return wp.to_torch(m.limit_dof_qadr)


def exp_limit_forces(m: Model) -> torch.Tensor:
    return wp.to_torch(m.limit_dof_forces)


def exp_limit_shapes(m: Model) -> torch.Tensor:
    return wp.to_torch(m.limit_dof_shapes)


# --- Data Fields ---
def time(d: Data) -> torch.tensor:
    return wp.to_torch(d.time)


def body_positions(d: Data) -> torch.Tensor:
    return wp.to_torch(d.mob_X_GB)


def body_com_positions(d: Data) -> torch.Tensor:
    return wp.to_torch(d.body_COM_G)


def body_rotations(d: Data) -> torch.Tensor:
    return wp.to_torch(d.xquat)


def body_velocities(d: Data) -> torch.Tensor:
    return wp.to_torch(d.body_acc)


def body_com_velocities(d: Data) -> torch.Tensor:
    return wp.to_torch(d.xivel)


def body_user_forces(d: Data) -> torch.Tensor:
    return wp.to_torch(d.xfrc_applied)


def joint_positions(d: Data) -> torch.Tensor:
    return wp.to_torch(d.qpos)


def joint_velocities(d: Data) -> torch.Tensor:
    return wp.to_torch(d.qvel)


def joint_accelerations(d: Data) -> torch.Tensor:
    return wp.to_torch(d.qacc)


def qfrc_spring(d: Data) -> torch.Tensor:
    return wp.to_torch(d.qfrc_spring)


def qfrc_damper(d: Data) -> torch.Tensor:
    return wp.to_torch(d.qfrc_damper)


def qfrc_muscle(d: Data) -> torch.Tensor:
    return wp.to_torch(d.qfrc_muscle)


def qfrc_actuator(d: Data) -> torch.Tensor:
    return wp.to_torch(d.qfrc_actuator)


def qfrc_limit(d: Data) -> torch.Tensor:
    return wp.to_torch(d.qfrc_limit)


def subtree_com_positions(d: Data) -> torch.Tensor:
    return wp.to_torch(d.subtree_com)


# -- Muscles ---
def muscle_activations(d: Data) -> torch.Tensor:
    return wp.to_torch(d.m_act)


def muscle_activations_dot(d: Data) -> torch.Tensor:
    return wp.to_torch(d.m_act_dot)


def muscle_excitations(d: Data) -> torch.Tensor:
    return wp.to_torch(d.m_excitations)


def muscle_actuations(d: Data) -> torch.Tensor:
    return wp.to_torch(d.muscle_actuation)


def muscle_path_lengths(d: Data) -> torch.Tensor:
    return wp.to_torch(d.muscle_length)


def muscle_path_velocities(d: Data) -> torch.Tensor:
    return wp.to_torch(d.muscle_velocity)


def muscle_fiber_lengths(d: Data) -> torch.Tensor:
    return wp.to_torch(d.m_state)


def muscle_fiber_velocities(d: Data) -> torch.Tensor:
    return wp.to_torch(d.m_state_dot)


def muscle_powers(d: Data) -> torch.Tensor:
    return wp.to_torch(d.muscle_metabolic)


def muscle_moment_arms(d: Data) -> torch.Tensor:
    return wp.to_torch(d.muscle_moment_arm)


def muscle_metadata_np(m: Model) -> np.ndarray:
    return m.muscle_metadata.numpy()


def muscle_length_info_np(d: Data) -> np.ndarray:
    return d.muscle_length_info.numpy()


def muscle_velocity_info_np(d: Data) -> np.ndarray:
    return d.muscle_velocity_info.numpy()


def site_positions(d: Data) -> torch.Tensor:
    return wp.to_torch(d.site_xpos)


def muscle_site_adr(m: Model) -> torch.Tensor:
    return wp.to_torch(m.muscle_pts_adr)


def muscle_site_num(m: Model) -> torch.Tensor:
    return wp.to_torch(m.muscle_pts_num)


# --- Actuators ---
def actuator_activations(d: Data) -> torch.Tensor:
    return wp.to_torch(d.a_act)


def actuator_excitations(d: Data) -> torch.Tensor:
    return wp.to_torch(d.a_excitations)


def actuator_metadata_np(m: Model) -> np.ndarray:
    return m.actuator_metadata.numpy()


# --- Visuals ---
def get_visual_positions(d: Data) -> torch.Tensor:
    return wp.to_torch(d.vis_X)


def get_visual_rotations(d: Data) -> torch.Tensor:
    return wp.to_torch(d.vis_xquat)


# --- Colliders ---
def get_collider_types(m: Model) -> torch.Tensor:
    return wp.to_torch(m.geom_type)


def get_collider_sizes(m: Model) -> torch.Tensor:
    return wp.to_torch(m.geom_size)


def collider_stiffness(m: Model) -> torch.Tensor:
    return wp.to_torch(m.geom_stiffness)


def collider_dissipation(m: Model) -> torch.Tensor:
    return wp.to_torch(m.geom_dissipation)


def collider_priority(m: Model) -> torch.Tensor:
    return wp.to_torch(m.geom_priority)


def collider_friction(m: Model) -> torch.Tensor:
    return wp.to_torch(m.geom_friction)


def collider_transition_velocity(m: Model) -> torch.Tensor:
    return wp.to_torch(m.geom_transition_velocity)


def get_collider_positions(d: Data) -> torch.Tensor:
    return wp.to_torch(d.geom_X)


def collider_forces(d: Data) -> torch.Tensor:
    return wp.to_torch(d.geom_cforce)


def get_collider_rotations(d: Data) -> torch.Tensor:
    return wp.to_torch(d.geom_xquat)


def grf(d: Data) -> torch.Tensor:
    return wp.to_torch(d.grf)


def joint_moments(d: Data) -> torch.Tensor:
    return wp.to_torch(d.joint_moments)
