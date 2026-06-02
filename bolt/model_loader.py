from typing import Optional

import opensim as osim

from bolt.load_utils import *
from bolt.model_load_result import ModelLoadResult
from bolt.paths import get_geometry_dir
from bolt.types_consts import Model, Data, IntegratorType, Option, ActivationType, ContractionType, MetabolicOptions, \
    MuscleMetadata, ActuatorMetadata, IntegratorStateScratch, IntegratorDotScratch, IntegratorMidpointScratch, \
    MuscleLengthInfo, FiberVelocityInfo, MuscleDynamicsInfo, Contact, SpatialInertia, ArticulatedInertia, TileBlockDim, \
    SwingTwistLimit, CoordinateLimitForce, ExponentialContact, vec5, PolyInts


def get_num_scratch_states(integrator: IntegratorType) -> tuple[int, int]:
    """ Returns number of additional copies of state and state_dot required for integration """
    if integrator == IntegratorType.EULER_ADAPTIVE:
        return 2, 1
    elif integrator == IntegratorType.EULER_MIDPOINT_ADAPTIVE:
        return 2, 1
    elif integrator == IntegratorType.RK_MERSON_ADAPTIVE:
        return 2, 5
    return 0, 0


def load_model(
        model_path: str,
        n_worlds: int,
        integrator: IntegratorType,
        requires_visuals: bool,
        muscle_fn_path: Optional[str],
        render_kinematic_tree: bool,
) -> ModelLoadResult:
    # All the mesh files for visuals should be located here
    osim.ModelVisualizer.addDirToGeometrySearchPaths(get_geometry_dir())
    # Run OpenSim parser
    model = osim.Model(model_path)
    model.initSystem()

    # Check every body has a mobilizer
    if model.getNumBodies() != model.getNumJoints():
        raise ValueError(f"Num bodies ({model.getNumBodies()}) does not match num Joints ({model.getNumJoints()})")

    # Parse bodies, joints, collision geometry, visuals, etc.
    converted_bodies = [GROUND_BODY] + [body_helper.convert_body(body) for body in model.getBodyList()]
    converted_joints = [GROUND_JOINT] + [joint_helper.convert_joint(joint) for joint in model.getJointList()]
    converted_geoms = [GROUND_COLLIDER] + geom_helper.convert_geoms(model, include_body_components=False)
    converted_exp_contacts = exponential_contact_helper.convert_exponential_contacts(model)
    converted_visuals = visual_helper.convert_visuals(model) if requires_visuals else []
    converted_spatial_transforms = spatial_transform_helper.convert_spatial_transforms(model)
    converted_spring_gen_force = coordinate_force_helper.convert_spring_generalized_force(model)
    converted_limit_forces = coordinate_force_helper.convert_coordinate_limit_force(model)  # also a limit force
    converted_swing_twists = swing_twist_helper.convert_swing_twist_limits(model_path)
    converted_activation_actuators = actuator_helper.convert_activation_actuators(model)
    converted_muscles = muscle_helper.convert_muscles(model)
    # Gather all sites
    sites_mus = muscle_helper.flatten_sites(converted_muscles)
    sites_exp = exponential_contact_helper.flatten_sites(converted_exp_contacts)
    sites_rem = site_helper.convert_sites(model)
    converted_sites = sites_mus + sites_exp + sites_rem
    site_start_muscle, site_start_contact, site_start_rem = 0, len(sites_mus), len(sites_mus) + len(sites_exp)

    # Function-based paths
    if muscle_fn_path is not None:
        converted_function_paths = (
            function_based_path_helper.parse_function_based_paths(model_path, muscle_fn_path))
        assert len(converted_function_paths) == len(converted_muscles)
    else:
        converted_function_paths = [USE_POINT_PATH] * len(converted_muscles)

    # Create a lookup from body name -> body data. Needed for joint->body lookup
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

    # Using the kinematic tree, compute a forward ordering
    tree_ordering = tree.forward_ordering()
    # Re-order bodies, joints, spatial transforms
    ordered_bodies = [node.body for node in tree_ordering]
    ordered_joints = [node.joint for node in tree_ordering]
    joint_ordering = joint_helper.compute_joint_name_ordering(ordered_joints)
    ordered_spatial_transforms = spatial_transform_helper.order_spatial_transforms(
        converted_spatial_transforms, joint_ordering)
    ordered_bodies_names = [body.name for body in ordered_bodies]

    # Body name -> body id
    body_ordering = string_list_to_ordering(ordered_bodies_names)
    # Body id -> parent id
    body_parent_id = [joint_helper.get_joint_parent_id(joint, ordered_bodies_names) for joint in ordered_joints]

    # Create the "body-level" array (contains list of all bodies at level i)
    body_tree = tree.create_body_tree()
    body_tree_indices = [apply_map_to_list(level, body_ordering) for level in body_tree]
    body_tree_warp = tuple([to_warp_array(level, dtype=int) for level in body_tree_indices])

    # Get all the children of each body
    body_children = [node.get_children_no_roots() for node in tree_ordering]
    body_children_names = [[node.body.name for node in children] for children in body_children]
    body_children_indices = [apply_map_to_list(children, body_ordering) for children in body_children_names]
    # Flatten list, compute number of children, get address
    body_children_flattened = flatten_nested_list(body_children_indices)
    body_children_num = [len(children) for children in body_children_indices]
    body_children_adr = exclusive_scan(body_children_num)

    # Create the *global* ordering lookup for each coordinate in qpos, dof.
    qpos_ordering = joint_helper.get_global_qpos_ordering_lookup(ordered_joints)
    dof_ordering = joint_helper.get_global_dof_ordering_lookup(ordered_joints)
    # Starting address of joint's coordinates/speeds
    mob_qpos_adr, mob_dof_adr = joint_helper.compute_qpos_dof_adr(ordered_joints)
    # Ordering lookup for coordinates relative to each joint's starting address
    relative_qpos_ordering = joint_helper.get_relative_qpos_ordering_lookup(ordered_joints)
    relative_dof_ordering = joint_helper.get_relative_dof_ordering_lookup(ordered_joints)
    # Index of mobilizer -> index of custom joint (-1 if not custom)
    mob_to_cst_idx, cst_to_mob_idx = joint_helper.compute_mobilizer_index_of_type(ordered_joints, MobilizerType.CUSTOM)
    n_custom_jnts = joint_helper.compute_num_joints_of_type(ordered_joints, MobilizerType.CUSTOM)
    # Index of mobilizer -> index of beam joint
    mob_to_beam_idx, beam_to_mob_idx = joint_helper.compute_mobilizer_index_of_type(ordered_joints, MobilizerType.BEAM)
    n_beams = joint_helper.compute_num_joints_of_type(ordered_joints, MobilizerType.BEAM)
    n_beam_visuals = 5 if requires_visuals else 0

    # Spatial transforms: flatten all the axes
    ordered_transform_axes = spatial_transform_helper.get_flattened_transform_axes(ordered_spatial_transforms)
    # Collect all functions in the spatial transforms
    linear_fns, linear_fns_idx = function_helper.get_functions_of_type(ordered_transform_axes, cls=LinearFunctionData)
    const_fns, const_fns_idx = function_helper.get_functions_of_type(ordered_transform_axes, cls=ConstantFunctionData)
    poly_fns, poly_fns_idx = function_helper.get_functions_of_type(ordered_transform_axes, cls=PolynomialFunctionData)
    spline_fns, spline_fns_idx = function_helper.get_functions_of_type(ordered_transform_axes, cls=SimmSplineData)
    nlinearfn, nconstfn, npolyfn, nspline = len(linear_fns), len(const_fns), len(poly_fns), len(spline_fns)
    nfunctions = nlinearfn + nconstfn + npolyfn + nspline
    # Get all relative coordinate indices for each transform axis
    txfm_dofs = spatial_transform_helper.get_txfm_coordinate_names(ordered_transform_axes)
    txfm_qpos_relative_idx = apply_map_to_list(txfm_dofs, relative_dof_ordering)
    txfm_qpos_global_idx = apply_map_to_list(txfm_dofs, qpos_ordering)
    # Now, use gather to find the relative coordinate indices used for each function
    linear_fns_qpos_global_idx = gather(txfm_qpos_global_idx, linear_fns_idx)
    poly_fns_qpos_global_idx = gather(txfm_qpos_global_idx, poly_fns_idx)
    spline_fns_qpos_global_idx = gather(txfm_qpos_global_idx, spline_fns_idx)

    nq = sum([joint.num_coordinates for joint in ordered_joints])
    nv = sum([joint.num_speeds for joint in ordered_joints])
    nb = len(ordered_bodies)
    ngeom = len(converted_geoms)
    nexpcontact = len(converted_exp_contacts)
    nvis = len(converted_visuals)
    nsite = len(converted_sites)
    nlimitforce = len(converted_limit_forces)
    nswingtwist = len(converted_swing_twists)
    nmuscle = len(converted_muscles)
    nactuator = len(converted_activation_actuators)
    nz = 2 * nmuscle + nactuator  # need muscle activation & fiber state and actuator activation state

    # Flatten out entries of dataclasses into lists
    body_mass = body_helper.get_body_masses(ordered_bodies)
    body_mass_center = body_helper.get_body_center(ordered_bodies)
    body_unit_inertia_OB_B = body_helper.get_body_unit_inertia_OB_B(ordered_bodies)

    mob_type = joint_helper.get_mob_type(ordered_joints)
    mob_dofnum = joint_helper.get_mob_dofnum(ordered_joints)
    mob_X_PF = joint_helper.get_mob_X_PF(ordered_joints)
    mob_X_MB = joint_helper.get_mob_X_MB(ordered_joints)
    mob_extra_info = joint_helper.get_mob_extra_info(ordered_joints)

    site_body_id = apply_map_to_list(site_helper.get_site_body_name(converted_sites), body_ordering)
    site_offset = site_helper.get_site_offset(converted_sites)

    vis_body_names = visual_helper.get_vis_body_name(converted_visuals)
    vis_transforms = visual_helper.get_vis_transform(converted_visuals)
    vis_body_id = apply_map_to_list(vis_body_names, body_ordering)

    geom_type = geom_helper.get_geom_type(converted_geoms)
    geom_body_names = geom_helper.get_geom_body_name(converted_geoms)
    geom_body_id = apply_map_to_list(geom_body_names, body_ordering)
    geom_size = geom_helper.get_geom_size(converted_geoms)
    geom_transform = geom_helper.get_geom_transform(converted_geoms)
    geom_aabb = geom_helper.get_geom_aabb(converted_geoms)
    geom_rbound = geom_helper.get_geom_rbound(converted_geoms)
    geom_friction = geom_helper.get_geom_friction(converted_geoms)
    geom_stiffness = geom_helper.get_geom_stiffness(converted_geoms)
    geom_dissipation = geom_helper.get_geom_dissipation(converted_geoms)
    geom_transition_velocity = geom_helper.get_geom_transition_velocity(converted_geoms)
    geom_priority = geom_helper.get_geom_priority(converted_geoms)
    # Broadphase registration
    geom_type_pair_count, nxn_geom_pair_filtered, nxn_pairid_filtered = (
        geom_helper.prepare_contacts(geom_type, geom_body_id, body_parent_id, ngeom))

    # We need to reshape the transform data to be (num_custom_joints, 6)
    txfm_axes = spatial_transform_helper.get_txfm_axes(ordered_transform_axes)
    cst_txfm_axes = create_nested_list(txfm_axes, num_per_sublist=6)
    cst_txfm_dof = create_nested_list(txfm_qpos_relative_idx, num_per_sublist=6)
    # If these lists are empty, we should fill them with dummy data so that the shape is correct
    if not cst_txfm_axes:
        cst_txfm_axes = [[wp.vec3()] * 6]
    if not cst_txfm_dof:
        cst_txfm_dof = [[0] * 6]

    # Gather additional function metadata
    linear_fn_mb = function_helper.get_linear_fn_mb(linear_fns)
    const_fn_vals = function_helper.get_const_fn_vals(const_fns)
    poly_coeffs = function_helper.get_flattened_poly_coeffs(poly_fns)
    poly_coeffs_num, poly_coeffs_adr = function_helper.get_poly_coeffs_num_adr(poly_fns)
    spline_xy_y2s = function_helper.get_spline_xy_y2s(spline_fns)
    spline_xys_num, spline_xys_adr = function_helper.get_spline_xys_num_adr(spline_fns)

    # Joint damping, spring, and linear stops (limits)
    dof_stiffness, dof_damping = coordinate_force_helper.get_dof_stiffness_damping(
        converted_spring_gen_force, dof_ordering)
    qpos_spring_rest = coordinate_force_helper.get_qpos_spring_rest(qpos_ordering)

    coordinate_limit_forces = coordinate_force_helper.create_coordinate_limit_force(
        converted_limit_forces, qpos_ordering, dof_ordering)
    swing_twist_limits = swing_twist_helper.create_swing_twist_data(
        converted_swing_twists, joint_ordering, mob_qpos_adr, mob_dof_adr)

    # Exponential contact
    exp_contact_data = exponential_contact_helper.create_exp_contact_data(
        converted_exp_contacts, site_start_contact, body_ordering)

    # Actuators
    actuator_data = actuator_helper.create_actuator_metadata(converted_activation_actuators, dof_ordering)
    am = wp.array(actuator_data, dtype=ActuatorMetadata)

    # Muscles/sites
    muscle_pts_num = muscle_helper.get_muscle_pts_num(converted_muscles)
    muscle_pts_adr = exclusive_scan(muscle_pts_num)
    muscle_pts_adr = [adr + site_start_muscle for adr in muscle_pts_adr]  # shift by number of sites before muscles
    muscle_data = muscle_helper.create_muscle_metadata(converted_muscles)
    mm = wp.array(muscle_data, dtype=MuscleMetadata)

    # Muscle function-based paths
    fn_path_term_coeffs = function_based_path_helper.get_fn_path_term_coeffs(converted_function_paths)
    fn_path_term_start, fn_path_term_count = function_based_path_helper.compute_fn_path_term_start_and_count(
        converted_function_paths)
    fn_path_qpos_adr = function_based_path_helper.get_fn_term_adr(converted_function_paths, qpos_ordering)
    fn_path_dimension = function_based_path_helper.get_fn_path_dimension(converted_function_paths)
    fn_path_order = function_based_path_helper.get_fn_path_order(converted_function_paths)
    # Determine the path type for each muscle
    point_paths_group, function_paths_groups = function_based_path_helper.path_type_to_muscle(converted_function_paths)
    function_paths_groups_warp = tuple([to_warp_array(group, dtype=int) for group in function_paths_groups])

    naconmax = max(512, n_worlds * 64)  # we're capping it at 64 contacts per world. TODO(check if this is reasonable)

    # --- Create Options ---
    opt = Option(
        gravity=-9.80665,
        explicit_gravity=True,
        implicit_damping=True,
        enable_drag=True,
        visuals=requires_visuals,
        nbeam_visuals=n_beam_visuals,

        activation_type=ActivationType.MILLARD,
        contraction_type=ContractionType.DGF,
        integrator=integrator,

        use_linear_stop=False,

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
        min_step_size=1e-7,
        max_step_size=wp.inf,
        accuracy=0.01,
        use_inf_norm=False,
        qvel_weights=wp.full(nv, 1.0, dtype=float),
        z_weights=wp.full(nz, 1.0, dtype=float),
    )

    # --- Create Model ---
    m = Model(
        nbody=nb,
        nq=nq,
        nv=nv,
        nmuscle=nmuscle,
        nactuator=nactuator,
        nz=nz,
        njnts_cst=n_custom_jnts,
        nbeams=n_beams,
        ngeom=ngeom,
        nexpcontact=nexpcontact,
        nvis=nvis,
        nsite=nsite,
        nlimitforce=nlimitforce,
        nswingtwist=nswingtwist,

        nfunctions=nfunctions,
        nlinearfn=nlinearfn,
        nconstfn=nconstfn,
        npolyfn=npolyfn,
        nsplinefn=nspline,

        opt=opt,
        muscle_metadata=mm,
        muscle_data=muscle_data,

        actuator_metadata=am,
        actuator_data=actuator_data,

        body_mass=to_warp_array(body_mass, dtype=float),
        body_mass_center=to_warp_array(body_mass_center, dtype=wp.vec3),
        body_unit_inertia_OB_B=to_warp_array(body_unit_inertia_OB_B, dtype=wp.mat33),

        body_parentid=to_warp_array(body_parent_id, dtype=int),
        body_tree=body_tree_warp,
        body_children=to_warp_array(body_children_flattened, dtype=int),
        body_children_num=to_warp_array(body_children_num, dtype=int),
        body_children_adr=to_warp_array(body_children_adr, dtype=int),

        mob_type=to_warp_array(mob_type, dtype=int),
        mob_qposadr=to_warp_array(mob_qpos_adr, dtype=int),
        mob_dofadr=to_warp_array(mob_dof_adr, dtype=int),
        mob_dofnum=to_warp_array(mob_dofnum, dtype=int),
        mob_X_PF=to_warp_array(mob_X_PF, dtype=wp.transform),
        mob_X_MB=to_warp_array(mob_X_MB, dtype=wp.transform),
        mob_extra_info=to_warp_array(mob_extra_info, dtype=wp.vec3),

        mob_to_cst_id=to_warp_array(mob_to_cst_idx, dtype=int),
        cst_to_mob_id=to_warp_array(cst_to_mob_idx, dtype=int),
        cst_txfm_axes=to_warp_array(cst_txfm_axes, dtype=wp.vec3),
        cst_txfm_dof=to_warp_array(cst_txfm_dof, dtype=int),

        beam_to_mob_id=to_warp_array(beam_to_mob_idx, dtype=int),

        linear_fn_mb=to_warp_array(linear_fn_mb, dtype=wp.vec2),
        const_fn_c=to_warp_array(const_fn_vals, dtype=float),
        poly_fn_coeff=to_warp_array(poly_coeffs, dtype=float),
        poly_fn_coeff_adr=to_warp_array(poly_coeffs_adr, dtype=int),
        poly_fn_coeff_num=to_warp_array(poly_coeffs_num, dtype=int),
        spline_fn_xy_y2s=to_warp_array(spline_xy_y2s, dtype=wp.vec3),
        spline_fn_xys_adr=to_warp_array(spline_xys_adr, dtype=int),
        spline_fn_xys_num=to_warp_array(spline_xys_num, dtype=int),

        linear_fn_adr=to_warp_array(linear_fns_idx, dtype=int),
        const_fn_adr=to_warp_array(const_fns_idx, dtype=int),
        poly_fn_adr=to_warp_array(poly_fns_idx, dtype=int),
        spline_fn_adr=to_warp_array(spline_fns_idx, dtype=int),

        linear_fn_qpos_adr=to_warp_array(linear_fns_qpos_global_idx, dtype=int),
        poly_fn_qpos_adr=to_warp_array(poly_fns_qpos_global_idx, dtype=int),
        spline_fn_qpos_adr=to_warp_array(spline_fns_qpos_global_idx, dtype=int),

        dof_damping=to_warp_array(dof_damping, dtype=float),
        dof_armature=make_zero(nv, dtype=float),  # user-modified later
        dof_stiffness=to_warp_array(dof_stiffness, dtype=float),
        qpos_spring_rest=to_warp_array(qpos_spring_rest, dtype=float),

        coordinate_limit_force=wp.array(coordinate_limit_forces, dtype=CoordinateLimitForce),
        swing_twist_limit=wp.array(swing_twist_limits, dtype=SwingTwistLimit),

        geom_type=to_warp_array(geom_type, dtype=int),
        geom_bodyid=to_warp_array(geom_body_id, dtype=int),
        geom_X_loc=to_warp_array(geom_transform, dtype=wp.transform),
        geom_size=to_warp_array(geom_size, dtype=wp.vec3),
        geom_friction=to_warp_array(geom_friction, dtype=wp.vec3),
        geom_stiffness=to_warp_array(geom_stiffness, dtype=float),
        geom_dissipation=to_warp_array(geom_dissipation, dtype=float),
        geom_transition_velocity=to_warp_array(geom_transition_velocity, dtype=float),
        geom_priority=to_warp_array(geom_priority, dtype=int),
        geom_aabb=to_warp_array(geom_aabb, dtype=wp.vec3),
        geom_rbound=to_warp_array(geom_rbound, dtype=float),

        exp_contact=wp.array(exp_contact_data, dtype=ExponentialContact),

        geom_pair_type_count=tuple(geom_type_pair_count),
        nxn_geom_pair_filtered=wp.array(nxn_geom_pair_filtered, dtype=wp.vec2i),
        nxn_pairid_filtered=wp.array(nxn_pairid_filtered, dtype=wp.vec2i),

        vis_bodyid=to_warp_array(vis_body_id, dtype=int),
        vis_X_loc=to_warp_array(vis_transforms, dtype=wp.transform),

        site_bodyid=to_warp_array(site_body_id, dtype=int),
        site_offset=to_warp_array(site_offset, dtype=wp.vec3),

        muscle_pts_num=to_warp_array(muscle_pts_num, dtype=int),
        muscle_pts_adr=to_warp_array(muscle_pts_adr, dtype=int),

        muscle_pt_group=to_warp_array(point_paths_group, dtype=int),
        muscle_pt_group_tuple=tuple(point_paths_group),
        muscle_fn_groups=function_paths_groups_warp,

        fn_path_term_coeffs=to_warp_array(fn_path_term_coeffs, dtype=float),
        fn_path_term_start=to_warp_array(fn_path_term_start, dtype=int),
        fn_path_qpos_adr=to_warp_array(fn_path_qpos_adr, dtype=PolyInts),
        fn_path_dimension=to_warp_array(fn_path_dimension, dtype=int),
        fn_path_order=to_warp_array(fn_path_order, dtype=int),

        block_dim=TileBlockDim(),
    )

    # --- Determine scratch space (integrators, mobilizers) ---
    n_int_states, n_int_dot_states = get_num_scratch_states(integrator)
    integrator_scratch = [
        IntegratorStateScratch(
            time=make_zero(n_worlds, dtype=float),
            qpos=make_zero((n_worlds, nq), dtype=float),
            qvel=make_zero((n_worlds, nv), dtype=float),
            m_state=make_zero((n_worlds, nmuscle), dtype=float),
            m_act=make_zero((n_worlds, nmuscle), dtype=float),
            a_act=make_zero((n_worlds, nactuator), dtype=float),
            exp_contact_state=make_zero((n_worlds, nexpcontact), dtype=wp.vec4),
        ) for _ in range(n_int_states)
    ]

    integrator_dot_scratch = [
        IntegratorDotScratch(
            qvel=make_zero((n_worlds, nv), dtype=float),
            qacc=make_zero((n_worlds, nv), dtype=float),
            m_state_dot=make_zero((n_worlds, nmuscle), dtype=float),
            m_act_dot=make_zero((n_worlds, nmuscle), dtype=float),
            a_act_dot=make_zero((n_worlds, nactuator), dtype=float),
            exp_contact_state_dot=make_zero((n_worlds, nexpcontact), dtype=wp.vec4),
        ) for _ in range(n_int_dot_states)
    ]

    # Custom joints may need up to 6 additional vectors: [f(q), f'(q), f''(q)] for each 6 functions
    num_mob_scratch = 3 if n_custom_jnts == 0 else 6

    dt = 1.0 / 100.0  # This will be modified later by the user

    # --- Create Data ---
    d = Data(
        world_reset=make_full(True, n_worlds, dtype=bool),
        time=make_zero(n_worlds, dtype=float),
        rng_state=to_warp_array(wp.rand_init(0), dtype=wp.uint32),

        nworld=n_worlds,
        naconmax=naconmax,

        # mid point integrators
        integrator_midpoint_scratch=IntegratorMidpointScratch(
            qvel=make_zero((n_worlds, nv), dtype=float),
            qacc=make_zero((n_worlds, nv), dtype=float)
        ),

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

        qpos=make_zero((n_worlds, nq), dtype=float),
        qvel=make_zero((n_worlds, nv), dtype=float),
        m_act=make_zero((n_worlds, nmuscle), dtype=float),
        a_act=make_full(0.5, (n_worlds, nactuator), dtype=float),
        m_state=make_zero((n_worlds, nmuscle), dtype=float),
        exp_contact_state=make_zero((n_worlds, nexpcontact), dtype=wp.vec4),

        grf=make_zero((n_worlds,), dtype=wp.vec3),

        qdot=make_zero((n_worlds, nq), dtype=float),
        qacc=make_zero((n_worlds, nv), dtype=float),
        m_act_dot=make_zero((n_worlds, nmuscle), dtype=float),
        a_act_dot=make_zero((n_worlds, nactuator), dtype=float),
        m_excitations=make_full(0.5, (n_worlds, nmuscle), dtype=float),
        a_excitations=make_full(0.5, (n_worlds, nactuator), dtype=float),
        m_state_dot=make_zero((n_worlds, nmuscle), dtype=float),
        exp_contact_state_dot=make_zero((n_worlds, nexpcontact), dtype=wp.vec4),

        cst_fn_output=make_zero((n_worlds, nfunctions), dtype=wp.vec3),

        mob_X_GB=make_zero((n_worlds, nb), dtype=wp.transform),
        mob_X_FM=make_zero((n_worlds, nb), dtype=wp.transform),
        mob_X_PB=make_zero((n_worlds, nb), dtype=wp.transform),
        mob_scratch=make_zero((n_worlds, nb, num_mob_scratch), dtype=wp.vec3),
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
        body_zTmp=make_zero((n_worlds, nb), dtype=wp.spatial_vector),

        subtree_com=make_zero((n_worlds, nb), dtype=wp.vec3),
        subtree_mass=make_zero((n_worlds, nb), dtype=float),

        geom_X=make_zero((n_worlds, ngeom), dtype=wp.transform),
        geom_cforce=make_zero((n_worlds, ngeom), dtype=float),
        geom_self_cforce=make_zero((n_worlds, ngeom), dtype=float),
        body_self_cforce=make_zero((n_worlds, nb), dtype=float),
        joint_moments=make_zero((n_worlds, nv), dtype=float),

        vis_X=make_zero((n_worlds, nvis), dtype=wp.transform),
        vis_beam_pos=make_zero((n_worlds, n_beams, n_beam_visuals), dtype=wp.vec3),

        site_rel_pos_B=make_zero((n_worlds, nsite), dtype=wp.vec3),
        site_pos_G=make_zero((n_worlds, nsite), dtype=wp.vec3),
        site_vel_G=make_zero((n_worlds, nsite), dtype=wp.vec3),

        mob_H_FM=make_zero((n_worlds, nv), dtype=wp.spatial_vector),
        mob_H=make_zero((n_worlds, nv), dtype=wp.spatial_vector),
        mob_HDot_FM=make_zero((n_worlds, nv), dtype=wp.spatial_vector),
        mob_HDot=make_zero((n_worlds, nv), dtype=wp.spatial_vector),

        mob_G=make_zero((n_worlds, nv), dtype=wp.spatial_vector),
        mob_DI=make_zero((n_worlds, nv), dtype=wp.spatial_vector),

        muscle_length=make_zero((n_worlds, nmuscle), dtype=float),
        muscle_velocity=make_zero((n_worlds, nmuscle), dtype=float),
        muscle_moment_arm=make_zero((n_worlds, nmuscle, nq), dtype=float),
        muscle_actuation=make_zero((n_worlds, nmuscle), dtype=float),

        muscle_active_length_multiplier=make_zero((n_worlds, nmuscle), dtype=float),
        muscle_active_velocity_multiplier=make_zero((n_worlds, nmuscle), dtype=float),
        muscle_actuation_passive=make_zero((n_worlds, nmuscle), dtype=float),
        muscle_actuation_active=make_zero((n_worlds, nmuscle), dtype=float),
        muscle_metabolic=make_zero((n_worlds, nmuscle), dtype=float),

        muscle_length_info=make_zero((n_worlds, nmuscle), dtype=MuscleLengthInfo),
        muscle_velocity_info=make_zero((n_worlds, nmuscle), dtype=FiberVelocityInfo),
        muscle_dynamics_info=make_zero((n_worlds, nmuscle), dtype=MuscleDynamicsInfo),
        muscle_norm_fiber_length=make_zero((n_worlds, nmuscle), dtype=float),

        body_F=make_zero((n_worlds, nb), dtype=wp.spatial_vector),
        body_F_gravity=make_zero((n_worlds, nb), dtype=wp.spatial_vector),
        body_F_applied=make_zero((n_worlds, nb), dtype=wp.spatial_vector),
        body_F_contact=make_zero((n_worlds, nb), dtype=wp.spatial_vector),
        body_F_drag=make_zero((n_worlds, nb), dtype=wp.spatial_vector),
        body_F_muscle=make_zero((n_worlds, nb), dtype=wp.spatial_vector),

        qfrc_muscle=make_zero((n_worlds, nq), dtype=float),

        ufrc_applied=make_zero((n_worlds, nv), dtype=float),
        ufrc_spring=make_zero((n_worlds, nv), dtype=float),
        ufrc_damper=make_zero((n_worlds, nv), dtype=float),
        ufrc_muscle=make_zero((n_worlds, nv), dtype=float),
        ufrc_actuator=make_zero((n_worlds, nv), dtype=float),
        ufrc_limit=make_zero((n_worlds, nv), dtype=float),

        qfrc_muscle_passive=make_zero((n_worlds, nq), dtype=float),
        qfrc_muscle_passive_breakdown=make_zero((n_worlds, nq, nmuscle), dtype=float),
        qfrc_muscle_active_breakdown=make_zero((n_worlds, nq, nmuscle), dtype=float),
        ufrc_muscle_passive=make_zero((n_worlds, nv), dtype=float),

        ufrc_total=make_zero((n_worlds, nv), dtype=float),

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

        nacon=make_zero(n_worlds, dtype=int),

        # collision driver
        collision_pair=wp.zeros((naconmax,), dtype=wp.vec2i),
        collision_pairid=wp.zeros((naconmax,), dtype=wp.vec2i),
        collision_worldid=wp.zeros((naconmax,), dtype=int),
        ncollision=wp.zeros((1,), dtype=int),
    )

    muscle_ordering = muscle_helper.get_muscle_ordering(converted_muscles)
    actuator_ordering = actuator_helper.get_actuator_ordering(converted_activation_actuators)
    collider_ordering = geom_helper.get_geom_ordering(converted_geoms)
    limit_id_lookup = coordinate_force_helper.create_limit_id_lookup(converted_limit_forces, qpos_ordering)

    return ModelLoadResult(
        model=m,
        data=d,
        root_free=joint_helper.check_root_free(ordered_joints),
        body_id_lookup=body_ordering,
        qpos_id_lookup=qpos_ordering,
        dof_id_lookup=dof_ordering,
        limit_id_lookup=limit_id_lookup,
        muscle_id_lookup=muscle_ordering,
        actuator_id_lookup=actuator_ordering,
        collider_id_lookup=collider_ordering,
        mesh_load_results=visual_helper.create_mesh_load_results(converted_visuals),
        colliders=converted_geoms,
    )


def update_colliders(load_result: ModelLoadResult):
    m, d = load_result.model, load_result.data
    geom_data, body_id_lookup = load_result.colliders, load_result.body_id_lookup

    ngeom = len(geom_data)
    geom_type = geom_helper.get_geom_type(geom_data)
    geom_body_names = geom_helper.get_geom_body_name(geom_data)
    geom_body_id = apply_map_to_list(geom_body_names, body_id_lookup)
    geom_size = geom_helper.get_geom_size(geom_data)
    geom_transform = geom_helper.get_geom_transform(geom_data)
    geom_aabb = geom_helper.get_geom_aabb(geom_data)
    geom_rbound = geom_helper.get_geom_rbound(geom_data)
    geom_friction = geom_helper.get_geom_friction(geom_data)
    geom_stiffness = geom_helper.get_geom_stiffness(geom_data)
    geom_dissipation = geom_helper.get_geom_dissipation(geom_data)
    geom_transition_velocity = geom_helper.get_geom_transition_velocity(geom_data)
    geom_priority = geom_helper.get_geom_priority(geom_data)

    # Broadphase registration
    body_parent_id = m.body_parentid.list()
    geom_type_pair_count, nxn_geom_pair_filtered, nxn_pairid_filtered = (
        geom_helper.prepare_contacts(geom_type, geom_body_id, body_parent_id, ngeom))

    # Set the fields again
    m.ngeom = ngeom
    m.geom_type = to_warp_array(geom_type, dtype=int)
    m.geom_bodyid = to_warp_array(geom_body_id, dtype=int)
    m.geom_X_loc = to_warp_array(geom_transform, dtype=wp.transform)
    m.geom_size = to_warp_array(geom_size, dtype=wp.vec3)
    m.geom_friction = to_warp_array(geom_friction, dtype=wp.vec3)
    m.geom_stiffness = to_warp_array(geom_stiffness, dtype=float)
    m.geom_dissipation = to_warp_array(geom_dissipation, dtype=float)
    m.geom_transition_velocity = to_warp_array(geom_transition_velocity, dtype=float)
    m.geom_priority = to_warp_array(geom_priority, dtype=int)
    m.geom_aabb = to_warp_array(geom_aabb, dtype=wp.vec3)
    m.geom_rbound = to_warp_array(geom_rbound, dtype=float)
    m.geom_pair_type_count = tuple(geom_type_pair_count)
    m.nxn_geom_pair_filtered = wp.array(nxn_geom_pair_filtered, dtype=wp.vec2i)
    m.nxn_pairid_filtered = wp.array(nxn_pairid_filtered, dtype=wp.vec2i)

    # Need to update the load result's collider lookup
    load_result.collider_id_lookup = geom_helper.get_geom_ordering(geom_data)

    # Update Data fields
    n_worlds = d.nworld
    d.geom_X = make_zero((n_worlds, ngeom), dtype=wp.transform)
    d.geom_cforce = make_zero((n_worlds, ngeom), dtype=float)
    d.geom_self_cforce = make_zero((n_worlds, ngeom), dtype=float)
    return
