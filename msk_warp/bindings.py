import json
import torch
import warp as wp
import numpy as np

import msk_warp._src.consts as consts
import msk_warp._src.forward as forward
import msk_warp._src.math as math
from msk_warp.render.renderer import Renderer, RendererType
from msk_warp.utils.load_utils import (
    exclusive_scan, to_warp_array, make_zero, make_full)
from msk_warp.utils.osim_converter import *
from msk_warp.utils.osim_parser import parse_osim_file


@dataclass
class ModelLoadResult:
    model: types.Model
    data: types.Data
    body_id_lookup: dict[str, int]
    dof_id_lookup: dict[str, tuple[int, int]]
    limit_id_lookup: dict[str, int]
    muscle_id_lookup: dict[str, int]
    actuator_id_lookup: dict[str, int]
    collider_id_lookup: dict[str, int]
    visuals: list[types.MeshLoadResult]


def prepare_contacts(
        geom_data: ColliderData,
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
    geom_types = geom_data.type
    geom_type_pair_count = np.bincount([
        math.upper_trid_index(len(types.GeomType),
                              int(geom_types[geom1[i]]),
                              int(geom_types[geom2[i]]))
        for i in np.arange(len(geom1))
        if nxn_pairid_contact[i] > -2 or nxn_pairid_collision[i] > -1
    ], minlength=len(types.GeomType) * (len(types.GeomType) + 1) // 2, )
    return geom_types, geom_type_pair_count, nxn_geom_pair_filtered, nxn_pairid_filtered


def get_num_scratch_states(integrator: types.IntegratorType) -> tuple[int, int]:
    """ Returns number of additional copies of state and state_dot for integration """
    if integrator == types.IntegratorType.RK4_ADAPTIVE:
        return 2, 5
    elif integrator == types.IntegratorType.EULER_ADAPTIVE:
        return 2, 1
    else:
        return 0, 0


def load_model(
        model_path: str,
        n_worlds: int,
        root_free: bool,
        integrator: types.IntegratorType,
        polynomial_data_path: str = None,
) -> ModelLoadResult:
    raw_osim_model = parse_osim_file(model_path)
    osim_model = to_checked_model(raw_osim_model, root_free=root_free)

    dof_id_lookup = get_dof_id_lookup(osim_model)

    nb = num_bodies(osim_model)
    jnt_qpos_num = get_joint_num_dofs(osim_model, vel_dofs=False)
    jnt_dof_num = get_joint_num_dofs(osim_model, vel_dofs=True)
    nv = sum(jnt_dof_num)
    nq = sum(jnt_qpos_num)
    nmuscle = num_muscles(osim_model)
    nactuators = num_actuators(osim_model)
    nfunctions = num_functions(osim_model)
    nz = nmuscle + nmuscle + nactuators  # muscle state, muscle activation, actuator activation

    joint_types = get_joint_types(osim_model)
    n_custom_jnts = len(list(filter(lambda jt: jt == types.JointType.CUSTOM, joint_types)))

    ngeom = num_colliders(osim_model)
    nvis = num_visuals(osim_model)
    site_data = get_site_data(osim_model)
    qpos0 = [0.0] * nq
    qpos_spring = [0.0] * len(qpos0)  # Placeholder for spring positions

    if root_free:
        qpos0[0:4] = [0.0, 0.0, 0.0, 1.0]  # Root orientation (quaternion)
        qpos0[4:7] = [0.0, 1.5, 0.0]  # Root pos

    qvel0 = [0.0] * nv  # Placeholder for initial velocities

    b_masses = body_masses(osim_model)
    inertias_OB_B = get_body_unit_inertias_OB_B(osim_model)
    body_mass_centers = get_body_mass_center(osim_model)
    body_parent_ids = get_body_parent_ids(osim_model)

    # Custom joints: compute address of joint -> custom joint
    is_custom_joint_mask = [1 if joint_types[i] == types.JointType.CUSTOM else 0
                            for i in range(len(joint_types))]
    custom_joint_indices = exclusive_scan(is_custom_joint_mask, True)
    assert (max(custom_joint_indices) == n_custom_jnts - 1)

    jnt_qpos_adr = exclusive_scan(jnt_qpos_num, False)
    jnt_dof_adr = exclusive_scan(jnt_dof_num, False)

    mob_X_PF = get_joint_rel_transform(osim_model, parent=True)
    mob_X_MB = get_joint_rel_transform(osim_model, parent=False)
    mob_extra_info = get_joint_extra_info(osim_model)

    geom_data = get_collider_data(osim_model)
    vis_data = get_visual_data(osim_model)

    muscle_pts_num = get_muscle_num_pts(osim_model)
    muscle_pts_adr = exclusive_scan(muscle_pts_num, False)

    # Muscle polynomial path
    use_fn_path = False
    poly_coeffs, poly_adr, poly_order = [], [], []
    poly_dep_dof_num, poly_dep_dof_adr = [], []
    poly_qpos_adr, poly_dof_adr = [], []
    max_dep_dof = 0
    if polynomial_data_path is not None:
        use_fn_path = True
        with open(polynomial_data_path, "r") as f:
            poly_data = json.load(f)

        # check that every muscle has polynomial data
        muscle_names = get_muscle_names(osim_model)
        total_poly_dofs = 0
        for muscle in muscle_names:
            assert muscle in poly_data, \
                f"Missing polynomial data for muscle: {muscle}"
            coeff = poly_data[muscle]["coeff"]
            order = poly_data[muscle]["order"]
            poly_dofs = poly_data[muscle]["dof"]

            poly_adr.append(len(poly_coeffs))
            poly_coeffs.extend(coeff)
            poly_order.append(order)

            poly_qpos_adr.extend([dof_id_lookup[d][0] for d in poly_dofs])
            poly_dof_adr.extend([dof_id_lookup[d][1] for d in poly_dofs])
            poly_dep_dof_num.append(len(poly_dofs))
            poly_dep_dof_adr.append(total_poly_dofs)
            total_poly_dofs += len(poly_dofs)
            max_dep_dof = max(max_dep_dof, len(poly_dofs))

    dof_damping = [0.1] * nv  # Placeholder for DOF
    jnt_stiffness = [0.0] * nb  # Placeholder for joint stiffness

    if root_free:
        dof_damping[0:6] = [0.0] * 6  # No damping for free joint
        jnt_stiffness[0] = 0.0  # No stiffness for free joint

    dof_limit_ranges, dof_limit_adr, dof_limit_qadr = get_dof_limits(osim_model)
    n_limits = len(dof_limit_ranges)
    dof_limit_forces = [(500.0, 500.0)] * n_limits  # Placeholder for limit forces
    dof_limit_shapes = [(0.1, 0.1)] * n_limits  # Placeholder for limit shapes

    body_tree = create_body_tree(osim_model)
    body_tree_warp = tuple([wp.array(bt, dtype=int) for bt in body_tree])

    # Create array for indices of children
    body_children = []
    for i in range(0, nb):
        if i == 0:  # ignore ground
            children = []
        else:
            children = [j for j, parent in enumerate(body_parent_ids) if parent == i]
        body_children.append(children)
    body_children_num = [len(children) for children in body_children]
    body_children_adr = exclusive_scan(body_children_num, False)
    # flatten
    body_children = [child for children in body_children for child in children]

    # Prepare contacts
    geom_types, geom_type_pair_count, nxn_geom_pair_filtered, nxn_pairid_filtered = prepare_contacts(
        geom_data, body_parent_ids, ngeom)

    # todo: don't hard code
    naconmax = max(512, n_worlds * 32)

    # needs shapes
    opt = types.Option(
        gravity=-9.80665,
        explicit_gravity=True,
        contact_type=types.ContactType.HUNT_CROSSLEY,
        limit_type=types.LimitType.EXPONENTIAL,
        activation_type=types.ActivationType.MILLARD,
        integrator=integrator,

        enable_drag=True,

        use_fn_path=use_fn_path,
        metabolic_options=types.MetabolicOptions(
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

    muscle_data = get_muscle_metadata(
        osim_model,
        max_pennation_angle=consts.M_MAX_PENNATION_ANGLE,
        min_norm_fiber_length=consts.MIN_NORM_FIBER_LENGTH,
        max_norm_fiber_length=consts.MAX_NORM_FIBER_LENGTH,
    )
    mm = wp.array(muscle_data, dtype=types.MuscleMetadata)

    actuator_data = get_actuator_metadata(osim_model)
    am = wp.array(actuator_data, dtype=types.ActuatorMetadata)

    nsite = site_data.nsite
    nsite_cond = site_data.nsite_cond
    dt = 1.0 / 500.0
    m = types.Model(
        nbody=nb,
        nv=nv,
        nq=nq,
        nmuscle=nmuscle,
        nz=nz,
        nactuator=nactuators,
        ndoflimit=n_limits,

        njnts_cst=n_custom_jnts,

        ngeom=ngeom,
        nvis=nvis,
        nsite=nsite,
        nsite_cond=nsite_cond,

        opt=opt,
        muscle_metadata=mm,
        muscle_data=muscle_data,

        actuator_metadata=am,

        # warp arrays
        qpos0=to_warp_array(qpos0, dtype=float),
        qpos_spring=to_warp_array(qpos_spring, dtype=float),

        body_mass=to_warp_array(b_masses, dtype=float),
        body_unit_inertia_OB_B=to_warp_array(inertias_OB_B, dtype=wp.mat33),
        body_mass_center=to_warp_array(body_mass_centers, dtype=wp.vec3),

        body_parentid=to_warp_array(body_parent_ids, dtype=int),
        jnt_type=to_warp_array(joint_types, dtype=int),
        jnt_stiffness=to_warp_array(jnt_stiffness, dtype=float),
        jnt_qposadr=to_warp_array(jnt_qpos_adr, dtype=int),
        jnt_dofnum=to_warp_array(jnt_dof_num, dtype=int),
        jnt_dofadr=to_warp_array(jnt_dof_adr, dtype=int),
        mob_X_PF=to_warp_array(mob_X_PF, dtype=wp.transform),
        mob_X_MB=to_warp_array(mob_X_MB, dtype=wp.transform),
        mob_extra_info=to_warp_array(mob_extra_info, dtype=wp.vec3),

        limit_dof_range=to_warp_array(dof_limit_ranges, dtype=wp.vec2),
        limit_dof_adr=to_warp_array(dof_limit_adr, dtype=int),
        limit_dof_qadr=to_warp_array(dof_limit_qadr, dtype=int),
        limit_dof_forces=to_warp_array(dof_limit_forces, dtype=wp.vec2),
        limit_dof_shapes=to_warp_array(dof_limit_shapes, dtype=wp.vec2),

        geom_type=to_warp_array(geom_types, dtype=int),
        geom_bodyid=to_warp_array(geom_data.body_id, dtype=int),
        geom_X_loc=to_warp_array(geom_data.transform, dtype=wp.transform),
        geom_size=to_warp_array(geom_data.size, dtype=wp.vec3),
        geom_friction=to_warp_array(geom_data.friction, dtype=wp.vec3),
        geom_stiffness=to_warp_array(geom_data.stiffness, dtype=float),
        geom_dissipation=to_warp_array(geom_data.dissipation, dtype=float),
        geom_transition_velocity=to_warp_array(geom_data.transition_velocity, dtype=float),
        geom_priority=to_warp_array(geom_data.priority, dtype=int),
        geom_aabb=to_warp_array(geom_data.aabb, dtype=wp.vec3),
        geom_rbound=to_warp_array(geom_data.rbound, dtype=float),

        geom_pair_type_count=tuple(geom_type_pair_count),
        nxn_geom_pair_filtered=wp.array(nxn_geom_pair_filtered, dtype=wp.vec2i),
        nxn_pairid_filtered=wp.array(nxn_pairid_filtered, dtype=wp.vec2i),

        vis_bodyid=to_warp_array(vis_data.body_id, dtype=int),
        vis_X_loc=to_warp_array(vis_data.transform, dtype=wp.transform),

        site_bodyid=to_warp_array(site_data.body_id, dtype=int),
        site_pos=to_warp_array(site_data.pos, dtype=wp.vec3),
        site_cond_id=to_warp_array(site_data.conditional_ids, dtype=int),
        site_cond_qadr=to_warp_array(site_data.conditional_qadr, dtype=int),
        site_cond_range=to_warp_array(site_data.conditional_range,
                                      dtype=wp.vec2),

        muscle_pts_num=to_warp_array(muscle_pts_num, dtype=int),
        muscle_pts_adr=to_warp_array(muscle_pts_adr, dtype=int),
        muscle_poly_coeffs=to_warp_array(poly_coeffs, dtype=float),
        muscle_poly_adr=to_warp_array(poly_adr, dtype=int),
        muscle_poly_order=to_warp_array(poly_order, dtype=int),
        muscle_poly_qpos_adr=to_warp_array(poly_qpos_adr, dtype=int),
        muscle_poly_dof_adr=to_warp_array(poly_dof_adr, dtype=int),
        muscle_dep_dof_num=to_warp_array(poly_dep_dof_num, dtype=int),
        muscle_dep_dof_adr=to_warp_array(poly_dep_dof_adr, dtype=int),

        dof_damping=to_warp_array(dof_damping, dtype=float),

        body_tree=body_tree_warp,
        body_children=to_warp_array(body_children, dtype=int),
        body_children_num=to_warp_array(body_children_num, dtype=int),
        body_children_adr=to_warp_array(body_children_adr, dtype=int),
        block_dim=types.TileBlockDim(),
    )

    n_int_states, n_int_dot_states = get_num_scratch_states(integrator)
    integrator_scratch = [
        types.IntegratorStateScratch(
            time=make_zero(n_worlds, dtype=float),
            qpos=make_zero((n_worlds, nq), dtype=float),
            qvel=make_zero((n_worlds, nv), dtype=float),
            m_state=make_zero((n_worlds, nmuscle), dtype=float),
            m_act=make_zero((n_worlds, nmuscle), dtype=float),
            a_act=make_zero((n_worlds, nactuators), dtype=float),
        ) for _ in range(n_int_states)
    ]

    integrator_dot_scratch = [
        types.IntegratorDotScratch(
            qvel=make_zero((n_worlds, nv), dtype=float),
            qacc=make_zero((n_worlds, nv), dtype=float),
            m_state_dot=make_zero((n_worlds, nmuscle), dtype=float),
            m_act_dot=make_zero((n_worlds, nmuscle), dtype=float),
            a_act_dot=make_zero((n_worlds, nactuators), dtype=float),
        ) for _ in range(n_int_dot_states)
    ]

    # Custom joints may need up to 6 additional vectors
    num_scratch = 3 if n_custom_jnts == 0 else 6

    d = types.Data(
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

        mob_X_GB=make_zero((n_worlds, nb), dtype=wp.transform),
        mob_X_FM=make_zero((n_worlds, nb), dtype=wp.transform),
        mob_X_PB=make_zero((n_worlds, nb), dtype=wp.transform),
        mob_scratch=make_zero((n_worlds, nb, num_scratch), dtype=wp.vec3),
        mob_phi=make_zero((n_worlds, nb), dtype=wp.vec3),
        mob_coriolis_acc=make_zero((n_worlds, nb), dtype=wp.spatial_vector),

        body_COM_G=make_zero((n_worlds, nb), dtype=wp.vec3),
        body_Mk_G=make_zero((n_worlds, nb), dtype=types.SpatialInertia),
        body_P=make_zero((n_worlds, nb), dtype=types.ArticulatedInertia),
        body_PPlus=make_zero((n_worlds, nb), dtype=types.ArticulatedInertia),
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

        muscle_length_info=make_zero((n_worlds, nmuscle),
                                     dtype=types.MuscleLengthInfo),
        muscle_velocity_info=make_zero((n_worlds, nmuscle),
                                       dtype=types.FiberVelocityInfo),
        muscle_dynamics_info=make_zero((n_worlds, nmuscle),
                                       dtype=types.MuscleDynamicsInfo),

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

        contact=types.Contact(
            dist=make_zero(naconmax, dtype=float),
            pos=make_zero(naconmax, dtype=wp.vec3),
            frame=make_zero(naconmax, dtype=wp.mat33),
            friction=make_zero(naconmax, dtype=types.vec5),
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
    for vis_idx in range(len(vis_data.file)):
        mesh_file = vis_data.file[vis_idx]
        mesh_scale = vis_data.scale[vis_idx]
        mesh_load_results.append(
            types.MeshLoadResult(
                file=mesh_file,
                scale=mesh_scale
            )
        )
    return ModelLoadResult(
        model=m,
        data=d,
        body_id_lookup=get_body_id_lookup(osim_model),
        dof_id_lookup=dof_id_lookup,
        limit_id_lookup=get_limit_id_lookup(osim_model),
        muscle_id_lookup=get_muscle_id_lookup(osim_model),
        actuator_id_lookup=get_actuator_id_lookup(osim_model),
        collider_id_lookup=get_collider_id_lookup(osim_model),
        visuals=mesh_load_results
    )


def reinitialize_model(
        m: types.Model,
        d: types.Data,
):
    """ Re-initialize the model (ie any parameters have changed). """
    # Ensure the muscle metadata is up to date
    mm = wp.array(m.muscle_data, dtype=types.MuscleMetadata)
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


def set_reset(d: types.Data, reset_worlds: torch.Tensor):
    d_reset_torch = wp.to_torch(d.world_reset)
    d_reset_torch[:] = reset_worlds.ravel()


# --- Model Fields ---
def damping(m: types.Model) -> torch.Tensor:
    return wp.to_torch(m.dof_damping)


def stiffness(m: types.Model) -> torch.Tensor:
    return wp.to_torch(m.jnt_stiffness)


def body_mass(m: types.Model) -> torch.Tensor:
    return wp.to_torch(m.body_mass)


def get_num_qpos(m: types.Model) -> int:
    return m.nq


def get_num_dofs(m: types.Model) -> int:
    return m.nv


def get_num_bodies(m: types.Model) -> int:
    return m.nbody


def get_num_visuals(m: types.Model) -> int:
    return m.nvis


def get_num_colliders(m: types.Model) -> int:
    return m.ngeom


def get_num_muscles(m: types.Model) -> int:
    return m.nmuscle


def get_num_actuators(m: types.Model) -> int:
    return m.nactuator


def get_num_limits(m: types.Model) -> int:
    return m.ndoflimit


def get_qpos_adr(m: types.Model, body_id: int) -> torch.Tensor:
    jnt_qpos_adr = wp.to_torch(m.jnt_qposadr)
    return jnt_qpos_adr[body_id]


def get_dof_adr(m: types.Model, body_id: int) -> torch.Tensor:
    jnt_dof_adr = wp.to_torch(m.jnt_dofadr)
    return jnt_dof_adr[body_id]


def get_qpos_num(m: types.Model, body_id: int) -> torch.Tensor:
    jnt_qpos_num = wp.to_torch(m.jnt_dofnum)
    return jnt_qpos_num[body_id]


def get_dof_num(m: types.Model, body_id: int) -> torch.Tensor:
    jnt_dof_num = wp.to_torch(m.jnt_dofnum)
    return jnt_dof_num[body_id]


def muscle_metadata(m: types.Model) -> list[types.MuscleMetadata]:
    return m.muscle_data


def gravity(m: types.Model) -> float:
    return m.opt.gravity


def set_drag_enabled(m: types.Model, enabled: bool):
    m.opt.enable_drag = enabled


def set_contact_type(m: types.Model, contact_type: types.ContactType):
    m.opt.contact_type = contact_type


def set_limit_type(m: types.Model, limit_type: types.LimitType):
    m.opt.limit_type = limit_type


def set_activation_type(m: types.Model, activation_type: types.ActivationType):
    m.opt.activation_type = activation_type


def steps_attempted(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.steps_attempted)


def set_integrator_accuracy(m: types.Model, accuracy: float):
    m.opt.accuracy = accuracy


def set_integrator_use_inf_norm(m: types.Model, use_inf_norm: bool):
    m.opt.use_inf_norm = use_inf_norm


def is_adaptive(integrator_type: types.IntegratorType) -> bool:
    return integrator_type in [
        types.IntegratorType.EULER_ADAPTIVE,
        types.IntegratorType.RK4_ADAPTIVE,
    ]


def joint_limit_ranges(m: types.Model) -> torch.Tensor:
    return wp.to_torch(m.limit_dof_range)


def joint_limit_qadr(m: types.Model) -> torch.Tensor:
    return wp.to_torch(m.limit_dof_qadr)


def exp_limit_forces(m: types.Model) -> torch.Tensor:
    return wp.to_torch(m.limit_dof_forces)


def exp_limit_shapes(m: types.Model) -> torch.Tensor:
    return wp.to_torch(m.limit_dof_shapes)


# --- Data Fields ---
def time(d: types.Data) -> torch.tensor:
    return wp.to_torch(d.time)


def body_positions(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.mob_X_GB)


def body_com_positions(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.body_COM_G)


def body_rotations(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.xquat)


def body_velocities(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.body_acc)


def body_com_velocities(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.xivel)


def body_user_forces(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.xfrc_applied)


def joint_positions(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.qpos)


def joint_velocities(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.qvel)


def joint_accelerations(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.qacc)


def qfrc_spring(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.qfrc_spring)


def qfrc_damper(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.qfrc_damper)


def qfrc_muscle(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.qfrc_muscle)


def qfrc_actuator(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.qfrc_actuator)


def qfrc_limit(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.qfrc_limit)


def subtree_com_positions(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.subtree_com)


# -- Muscles ---
def muscle_activations(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.m_act)


def muscle_activations_dot(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.m_act_dot)


def muscle_excitations(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.m_excitations)


def muscle_actuations(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.muscle_actuation)


def muscle_path_lengths(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.muscle_length)


def muscle_path_velocities(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.muscle_velocity)


def muscle_fiber_lengths(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.m_state)


def muscle_fiber_velocities(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.m_state_dot)


def muscle_powers(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.muscle_metabolic)


def muscle_moment_arms(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.muscle_moment_arm)


def muscle_metadata_np(m: types.Model) -> np.ndarray:
    return m.muscle_metadata.numpy()


def muscle_length_info_np(d: types.Data) -> np.ndarray:
    return d.muscle_length_info.numpy()


def muscle_velocity_info_np(d: types.Data) -> np.ndarray:
    return d.muscle_velocity_info.numpy()


def site_positions(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.site_xpos)


def muscle_site_adr(m: types.Model) -> torch.Tensor:
    return wp.to_torch(m.muscle_pts_adr)


def muscle_site_num(m: types.Model) -> torch.Tensor:
    return wp.to_torch(m.muscle_pts_num)


# --- Actuators ---
def actuator_activations(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.a_act)


def actuator_excitations(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.a_excitations)


def actuator_metadata_np(m: types.Model) -> np.ndarray:
    return m.actuator_metadata.numpy()


# --- Visuals ---
def get_visual_positions(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.vis_X)


def get_visual_rotations(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.vis_xquat)


# --- Colliders ---
def get_collider_types(m: types.Model) -> torch.Tensor:
    return wp.to_torch(m.geom_type)


def get_collider_sizes(m: types.Model) -> torch.Tensor:
    return wp.to_torch(m.geom_size)


def collider_stiffness(m: types.Model) -> torch.Tensor:
    return wp.to_torch(m.geom_stiffness)


def collider_dissipation(m: types.Model) -> torch.Tensor:
    return wp.to_torch(m.geom_dissipation)


def collider_priority(m: types.Model) -> torch.Tensor:
    return wp.to_torch(m.geom_priority)


def collider_friction(m: types.Model) -> torch.Tensor:
    return wp.to_torch(m.geom_friction)


def collider_transition_velocity(m: types.Model) -> torch.Tensor:
    return wp.to_torch(m.geom_transition_velocity)


def get_collider_positions(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.geom_X)


def collider_forces(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.geom_cforce)


def get_collider_rotations(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.geom_xquat)


def grf(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.grf)


def joint_moments(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.joint_moments)
