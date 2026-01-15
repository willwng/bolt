import json
import torch
import warp as wp

import msk_warp._src.consts as consts
import msk_warp._src.forward as forward
import msk_warp._src.init_model as init_model
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
    parent_child_collision = (
            ((bodyid1 == parentid2) & (bodyid1 != 0) & pc_filter1)
            | ((bodyid2 == parentid1) & (bodyid2 != 0) & pc_filter2)
    )
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


def load_model(
        model_path: str,
        n_worlds: int,
        polynomial_data_path: str = None,
) -> ModelLoadResult:
    raw_osim_model = parse_osim_file(model_path)
    osim_model = to_checked_model(raw_osim_model)

    # Lookups
    dof_id_lookup = get_dof_id_lookup(osim_model)

    nb = num_bodies(osim_model)
    joint_num_qdofs = get_joint_num_dofs(osim_model, vel_dofs=False)
    joint_num_vdofs = get_joint_num_dofs(osim_model, vel_dofs=True)

    nv = sum(joint_num_vdofs)
    nq = sum(joint_num_qdofs)
    nmuscle = num_muscles(osim_model)
    nactuators = num_actuators(osim_model)

    joint_types = get_joint_types(osim_model)
    n_conv_jnts, n_custom_jnts = 0, 0
    for jt in joint_types:
        if jt == types.JointType.CUSTOM:
            n_custom_jnts += 1
        else:
            n_conv_jnts += 1

    ngeom = num_colliders(osim_model)
    nvis = num_visuals(osim_model)
    site_data = get_site_data(osim_model)
    qpos0 = [0.0] * nq
    qpos0[0:3] = [0.0, 1.0, 0.0]  # Root pos
    qpos0[3] = 1  # root quat

    qvel0 = [0.0] * nv  # Placeholder for initial velocities
    qpos_spring = [0.0] * len(qpos0)  # Placeholder for spring positions

    b_masses = body_masses(osim_model)
    inertias = get_body_inertias(osim_model)
    body_local_com = get_local_body_com_pos(osim_model)
    body_local_rot = get_local_body_rot(osim_model)
    body_parent_ids = get_body_parent_ids(osim_model)

    # Custom joints: compute address of joint -> custom joint
    is_custom_joint_mask = [1 if joint_types[i] == types.JointType.CUSTOM else 0
                            for i in range(len(joint_types))]
    custom_joint_indices = exclusive_scan(is_custom_joint_mask, True)
    assert (max(custom_joint_indices) == n_custom_jnts - 1)

    jnt_qpos_adr = exclusive_scan(joint_num_qdofs, False)
    jnt_dof_adr = exclusive_scan(joint_num_vdofs, False)

    jnt_rel_parent = get_joint_rel_pos(osim_model, get_parent_rel=True)
    jnt_rel_child = get_joint_rel_pos(osim_model, get_parent_rel=False)
    jnt_rel_parent_rot = get_joint_rel_rot(osim_model, parent=True)
    jnt_rel_child_rot = get_joint_rel_rot(osim_model, parent=False)

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

    dof_armature = [0.03] * nv  # Placeholder for DOF armature
    dof_armature[0:6] = [0.0] * 6  # No armature for free joint
    dof_damping = [0.1] * nv  # Placeholder for DOF
    dof_damping[0:6] = [0.0] * 6  # No damping for free joint
    jnt_stiffness = [0.0] * nb  # Placeholder for joint stiffness

    dof_limit_ranges, dof_limit_adr, dof_limit_qadr = get_dof_limits(osim_model)
    n_limits = len(dof_limit_ranges)
    dof_limit_forces = [(50.0, 20.0)] * n_limits  # Placeholder for limit forces
    dof_limit_shapes = [(75.0, 25.0)] * n_limits  # Placeholder for limit shapes

    body_rootid = [1] * nb  # Placeholder for body root IDs
    body_tree = create_body_tree(osim_model)
    body_tree_warp = tuple([wp.array(bt, dtype=int) for bt in body_tree])

    dof_body_id = get_dof_body_ids(osim_model)
    dof_parent_id = compute_expanded_parent(osim_model, jnt_dof_adr)

    body_subtree_mass = get_subtree_mass(osim_model)
    tiles = make_tiles(osim_model, dof_parent_id)
    qM_tiles = tuple(
        types.TileSet(adr=wp.array(tiles[sz], dtype=int), size=sz) for sz in
        sorted(tiles.keys()))
    dof_tri_row, dof_tri_col = np.tril_indices(nv)

    linear_fns, const_fns = get_functions(osim_model)
    txfm_fn_type, txfm_fn_adr, txfm_qadr, txfm_dof_adr, txfm_axis = get_txfm_fns(
        osim_model)

    # Reshape
    txfm_fn_type = np.array(txfm_fn_type)
    txfm_fn_adr = np.array(txfm_fn_adr)
    txfm_qadr = np.array(txfm_qadr)
    txfm_dof_adr = np.array(txfm_dof_adr)
    txfm_axis = np.array(txfm_axis)

    txfm_fn_type = txfm_fn_type.reshape(n_custom_jnts, 6)
    txfm_fn_adr = txfm_fn_adr.reshape(n_custom_jnts, 6)
    txfm_qadr = txfm_qadr.reshape(n_custom_jnts, 6)
    txfm_dof_adr = txfm_dof_adr.reshape(n_custom_jnts, 6)
    txfm_axis = txfm_axis.reshape(n_custom_jnts, 6, 3)

    # Prepare contacts
    geom_types, geom_type_pair_count, nxn_geom_pair_filtered, nxn_pairid_filtered = prepare_contacts(
        geom_data, body_parent_ids, ngeom)

    # todo: don't hard code
    njmax = 128
    naconmax = max(512, n_worlds * 32)

    # needs shapes
    opt = types.Option(
        impratio=1.0,
        tolerance=1e-8,
        ls_tolerance=0.01,
        ccd_tolerance=1e-6,
        gravity=-9.80665,
        solver=types.SolverType.NEWTON,
        contact_type=types.ContactType.HUNT_CROSSLEY,
        limit_type=types.LimitType.EXPONENTIAL,
        integrator=types.IntegratorType.EULER_FIXED,
        iterations=50,
        ls_iterations=100,
        ccd_iterations=50,
        warm_start=True,

        enable_drag=True,

        muscle_dyn_substeps=30,
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
        use_inf_norm=True,

        solref=wp.vec2(0.02, 1.0),
        solimp=types.vec5(0.9, 0.95, 0.001, 0.5, 2.0),

        qvel_weights=wp.full(nv, 1.0, dtype=float),

        ls_parallel=False,
        ls_parallel_min_step=1e-8,
        graph_conditional=True,

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
        nactuator=nactuators,
        ndoflimit=n_limits,

        njnts_conv=n_conv_jnts,
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
        body_inertia=to_warp_array(inertias, dtype=wp.vec3),
        body_ipos=to_warp_array(body_local_com, dtype=wp.vec3),
        body_iquat=to_warp_array(body_local_rot, dtype=wp.quat),

        body_rootid=to_warp_array(body_rootid, dtype=int),
        body_parentid=to_warp_array(body_parent_ids, dtype=int),
        jnt_type=to_warp_array(joint_types, dtype=int),
        jnt_stiffness=to_warp_array(jnt_stiffness, dtype=float),
        jnt_qposadr=to_warp_array(jnt_qpos_adr, dtype=int),
        jnt_dofnum=to_warp_array(joint_num_vdofs, dtype=int),
        jnt_dofadr=to_warp_array(jnt_dof_adr, dtype=int),
        jnt_rel_parent=to_warp_array(jnt_rel_parent, dtype=wp.vec3),
        jnt_rel_child=to_warp_array(jnt_rel_child, dtype=wp.vec3),
        jnt_rel_parent_rot=to_warp_array(jnt_rel_parent_rot, dtype=wp.quat),
        jnt_rel_child_rot=to_warp_array(jnt_rel_child_rot, dtype=wp.quat),

        jnt_cst_adr=to_warp_array(custom_joint_indices, dtype=int),
        const_fns=to_warp_array(const_fns, dtype=float),
        linear_fns=to_warp_array(linear_fns, dtype=wp.vec2),
        cst_txfm_axis=wp.array(txfm_axis, dtype=wp.vec3),
        cst_txfm_fn=to_warp_array(txfm_fn_type, dtype=int),
        cst_txfm_fn_adr=to_warp_array(txfm_fn_adr, dtype=int),
        cst_txfm_qadr=to_warp_array(txfm_qadr, dtype=int),
        cst_txfm_dofadr=to_warp_array(txfm_dof_adr, dtype=int),

        limit_dof_range=to_warp_array(dof_limit_ranges, dtype=wp.vec2),
        limit_dof_adr=to_warp_array(dof_limit_adr, dtype=int),
        limit_dof_qadr=to_warp_array(dof_limit_qadr, dtype=int),
        limit_dof_forces=to_warp_array(dof_limit_forces, dtype=wp.vec2),
        limit_dof_shapes=to_warp_array(dof_limit_shapes, dtype=wp.vec2),

        geom_type=to_warp_array(geom_types, dtype=int),
        geom_bodyid=to_warp_array(geom_data.body_id, dtype=int),
        geom_size=to_warp_array(geom_data.size, dtype=wp.vec3),
        geom_pos=to_warp_array(geom_data.pos, dtype=wp.vec3),
        geom_quat=to_warp_array(geom_data.rot, dtype=wp.quat),
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

        vis_pos=to_warp_array(vis_data.pos, dtype=wp.vec3),
        vis_quat=to_warp_array(vis_data.rot, dtype=wp.quat),
        vis_bodyid=to_warp_array(vis_data.body_id, dtype=int),

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

        dof_armature=to_warp_array(dof_armature, dtype=float),
        dof_damping=to_warp_array(dof_damping, dtype=float),

        dof_bodyid=to_warp_array(dof_body_id, dtype=int),
        dof_parentid=to_warp_array(dof_parent_id, dtype=int),

        body_tree=body_tree_warp,
        body_subtreemass=to_warp_array(body_subtree_mass, dtype=float),
        qM_tiles=qM_tiles,
        block_dim=types.BlockDim(),
        dof_tri_row=to_warp_array(dof_tri_row, dtype=int),
        dof_tri_col=to_warp_array(dof_tri_col, dtype=int),

        # These are computed with the _model_init function
        mean_inertia=0.0,
        body_invweight0=to_warp_array([0.0, 0.0] * nb, dtype=wp.vec2),  # TODO
        dof_invweight0=to_warp_array([0.0] * nv, dtype=float),  # TODO
    )

    n_int_states = 2
    d = types.Data(
        world_reset=make_full(True, n_worlds, dtype=bool),

        solver_niter=make_zero(n_worlds, dtype=int),

        nl=make_zero(n_worlds, dtype=int),
        nefc=make_zero(n_worlds, dtype=int),
        needs_solve=make_zero(1, dtype=int),
        time=make_zero(n_worlds, dtype=float),
        time1=make_zero(n_worlds, dtype=float),
        next_time=make_zero(n_worlds, dtype=float),

        qpos=wp.array(np.tile(qpos0, (n_worlds, 1)), dtype=float),
        qvel=wp.array(np.tile(qvel0, (n_worlds, 1)), dtype=float),
        m_act=make_zero((n_worlds, nmuscle), dtype=float),
        a_act=make_zero((n_worlds, nactuators), dtype=float),
        m_state=make_zero((n_worlds, nmuscle), dtype=float),

        qacc_warmstart=make_zero((n_worlds, nv), dtype=float),
        qfrc_applied=make_zero((n_worlds, nv), dtype=float),
        xfrc_applied=make_zero((n_worlds, nb), dtype=wp.spatial_vector),
        xfrc_user=make_zero((n_worlds, nb), dtype=wp.spatial_vector),
        grf=make_zero((n_worlds,), dtype=wp.vec3),
        joint_moments=make_zero((n_worlds, nv), dtype=float),

        qacc=make_zero((n_worlds, nv), dtype=float),
        m_act_dot=make_zero((n_worlds, nmuscle), dtype=float),
        a_act_dot=make_zero((n_worlds, nactuators), dtype=float),
        m_excitations=make_zero((n_worlds, nmuscle), dtype=float),
        a_excitations=make_zero((n_worlds, nactuators), dtype=float),
        m_state_dot=make_zero((n_worlds, nmuscle), dtype=float),

        xpos=make_zero((n_worlds, nb), dtype=wp.vec3),
        xquat=make_zero((n_worlds, nb), dtype=wp.quat),
        xmat=make_zero((n_worlds, nb), dtype=wp.mat33),
        xipos=make_zero((n_worlds, nb), dtype=wp.vec3),
        ximat=make_zero((n_worlds, nb), dtype=wp.mat33),
        xanchor=make_zero((n_worlds, nb), dtype=wp.vec3),
        xaxis=make_zero((n_worlds, nb, 6), dtype=wp.vec3),

        geom_xpos=make_zero((n_worlds, ngeom), dtype=wp.vec3),
        geom_xquat=make_zero((n_worlds, ngeom), dtype=wp.quat),
        geom_xmat=make_zero((n_worlds, ngeom), dtype=wp.mat33),

        vis_xpos=make_zero((n_worlds, nvis), dtype=wp.vec3),
        vis_xquat=make_zero((n_worlds, nvis), dtype=wp.quat),

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

        subtree_com=make_zero((n_worlds, nb), dtype=wp.vec3),
        cdof=make_zero((n_worlds, nv), dtype=wp.spatial_vector),
        cdof_tmp=make_zero((n_worlds, n_custom_jnts, 6),
                           dtype=wp.spatial_vector),
        cinert=make_zero((n_worlds, nb), dtype=types.vec10),

        crb=make_zero((n_worlds, nb), dtype=types.vec10),
        qM=make_zero((n_worlds, nv, nv), dtype=float),
        qLD=make_zero((n_worlds, nv, nv), dtype=float),
        qLDiagInv=make_zero((n_worlds, nv), dtype=float),

        muscle_length=make_zero((n_worlds, nmuscle), dtype=float),
        muscle_velocity=make_zero((n_worlds, nmuscle), dtype=float),
        muscle_actuation=make_zero((n_worlds, nmuscle), dtype=float),
        muscle_metabolic=make_zero((n_worlds, nmuscle), dtype=float),
        muscle_length_prev=make_zero((n_worlds, nmuscle), dtype=float),
        muscle_velocity_prev=make_zero((n_worlds, nmuscle), dtype=float),

        muscle_length_info=make_zero((n_worlds, nmuscle),
                                     dtype=types.MuscleLengthInfo),
        muscle_velocity_info=make_zero((n_worlds, nmuscle),
                                       dtype=types.FiberVelocityInfo),
        muscle_dynamics_info=make_zero((n_worlds, nmuscle),
                                       dtype=types.MuscleDynamicsInfo),

        cvel=make_zero((n_worlds, nb), dtype=wp.spatial_vector),
        xvel=make_zero((n_worlds, nb), dtype=wp.spatial_vector),
        xivel=make_zero((n_worlds, nb), dtype=wp.spatial_vector),
        cdof_dot=make_zero((n_worlds, nv), dtype=wp.spatial_vector),

        qfrc_bias=make_zero((n_worlds, nv), dtype=float),
        qfrc_spring=make_zero((n_worlds, nv), dtype=float),
        qfrc_damper=make_zero((n_worlds, nv), dtype=float),
        qfrc_passive=make_zero((n_worlds, nv), dtype=float),

        subtree_linvel=make_zero((n_worlds, nb), dtype=wp.vec3),
        subtree_angmom=make_zero((n_worlds, nb), dtype=wp.vec3),

        qfrc_smooth=make_zero((n_worlds, nv), dtype=float),
        qacc_smooth=make_zero((n_worlds, nv), dtype=float),
        qfrc_constraint=make_zero((n_worlds, nv), dtype=float),
        qfrc_inverse=make_zero((n_worlds, nv), dtype=float),

        cacc=make_zero((n_worlds, nb), dtype=wp.spatial_vector),
        cfrc_int=make_zero((n_worlds, nb), dtype=wp.spatial_vector),
        cfrc_ext=make_zero((n_worlds, nb), dtype=wp.spatial_vector),

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
            efc_address=make_zero((naconmax, 4), dtype=int),
            # assuming condim_max = 3
            worldid=make_zero(naconmax, dtype=int),
            geomcollisionid=make_zero(naconmax, dtype=int),
        ),
        efc=types.Constraint(
            type=make_zero((n_worlds, njmax), dtype=int),
            id=make_zero((n_worlds, njmax), dtype=int),
            J=make_zero((n_worlds, njmax, nv), dtype=float),
            pos=make_zero((n_worlds, njmax), dtype=float),
            margin=make_zero((n_worlds, njmax), dtype=float),
            D=make_zero((n_worlds, njmax), dtype=float),
            vel=make_zero((n_worlds, njmax), dtype=float),
            aref=make_zero((n_worlds, njmax), dtype=float),
            frictionloss=make_zero((n_worlds, njmax), dtype=float),
            force=make_zero((n_worlds, njmax), dtype=float),
            Jaref=make_zero((n_worlds, njmax), dtype=float),
            Ma=make_zero((n_worlds, nv), dtype=float),
            grad=make_zero((n_worlds, nv), dtype=float),
            cholesky_L_tmp=make_zero((n_worlds, nv, nv), dtype=float),
            cholesky_y_tmp=make_zero((n_worlds, nv), dtype=float),
            grad_dot=make_zero(n_worlds, dtype=float),
            Mgrad=make_zero((n_worlds, nv), dtype=float),
            search=make_zero((n_worlds, nv), dtype=float),
            search_dot=make_zero(n_worlds, dtype=float),
            gauss=make_zero(n_worlds, dtype=float),
            cost=make_zero(n_worlds, dtype=float),
            prev_cost=make_zero(n_worlds, dtype=float),
            state=make_zero((n_worlds, njmax), dtype=int),
            mv=make_zero((n_worlds, nv), dtype=float),
            jv=make_zero((n_worlds, njmax), dtype=float),
            quad=make_zero((n_worlds, njmax), dtype=wp.vec3f),
            quad_gauss=make_zero(n_worlds, dtype=wp.vec3f),
            h=make_zero((n_worlds, nv, nv), dtype=float),
            alpha=make_zero(n_worlds, dtype=float),
            prev_grad=make_zero((n_worlds, nv), dtype=float),
            prev_Mgrad=make_zero((n_worlds, nv), dtype=float),
            beta=make_zero(n_worlds, dtype=float),
            done=make_zero(n_worlds, dtype=bool),
        ),
        dof_lim_efc_address=make_zero((n_worlds, n_limits), dtype=int),
        dof_lim_torque=make_zero((n_worlds, n_limits), dtype=float),

        nworld=n_worlds,
        naconmax=naconmax,
        njmax=njmax,
        nacon=make_zero(n_worlds, dtype=int),
        nsolving=make_zero(1, dtype=int),
        subtree_bodyvel=make_zero((n_worlds, nb), dtype=wp.vec3),

        # Variable-step integrator
        integrator_state=types.IntegratorState(
            time=make_zero((n_worlds, n_int_states), dtype=float),
            qpos=make_zero((n_worlds, n_int_states, nq), dtype=float),
            qvel=make_zero((n_worlds, n_int_states, nv), dtype=float),
            mstate=make_zero((n_worlds, n_int_states, nmuscle), dtype=float),
            act=make_zero((n_worlds, n_int_states, nmuscle), dtype=float),
        ),
        nintegrating=make_zero(1, dtype=int),
        step_size=make_full(dt / 10.0, (n_worlds,), dtype=float),
        actual_step_size=make_full(dt, (n_worlds,), dtype=float),
        artificially_limited=make_zero((n_worlds,), dtype=bool),
        error=make_zero((n_worlds,), dtype=float),
        qvel_scales=make_zero((n_worlds, nv), dtype=float),
        qpos_diff=make_zero((n_worlds, nq), dtype=float),
        qpos_diff_scaled=make_zero((n_worlds, nq), dtype=float),
        qvel_diff=make_zero((n_worlds, nq), dtype=float),
        mstate_diff=make_zero((n_worlds, nmuscle), dtype=float),
        act_diff=make_zero((n_worlds, nmuscle), dtype=float),
        qpos_error=make_zero((n_worlds,), dtype=float),
        qvel_error=make_zero((n_worlds,), dtype=float),
        ninv_dq_tmp=make_zero((n_worlds, nv), dtype=float),

        step_accepted=make_zero((n_worlds,), dtype=bool),
        integration_done=make_zero((n_worlds,), dtype=bool),

        # collision driver
        collision_pair=wp.zeros((naconmax,), dtype=wp.vec2i),
        collision_pairid=wp.zeros((naconmax,), dtype=wp.vec2i),
        collision_worldid=wp.zeros((naconmax,), dtype=int),
        ncollision=wp.zeros((1,), dtype=int),
    )

    init_model._model_init(m, d)
    forward.reset(m, d)

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

    init_model._model_init(m, d)
    d.world_reset.fill_(True)
    forward.reset(m, d)


def create_renderer(
        load_result: ModelLoadResult,
        renderer_type: RendererType,
        draw_colliders: bool,
        draw_visuals: bool,
        draw_muscles: bool
):
    viewer = Renderer(
        m=load_result.model,
        renderer_type=renderer_type,
        draw_colliders=draw_colliders,
        draw_visuals=draw_visuals,
        draw_muscles=draw_muscles
    )
    viewer.load_meshes(load_result.visuals)
    return viewer


def set_reset(d: types.Data, reset_worlds: torch.Tensor):
    d_reset_torch = wp.to_torch(d.world_reset)
    d_reset_torch[:] = reset_worlds.ravel()


# --- Model Fields ---
def damping(m: types.Model) -> torch.Tensor:
    return wp.to_torch(m.dof_damping)


def armature(m: types.Model) -> torch.Tensor:
    return wp.to_torch(m.dof_armature)


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


def subtree_mass(m: types.Model) -> torch.Tensor:
    return wp.to_torch(m.body_subtreemass)


def gravity(m: types.Model) -> float:
    return m.opt.gravity


def set_drag_enabled(m: types.Model, enabled: bool):
    m.opt.enable_drag = enabled


def set_solver_type(m: types.Model, solver_type: types.SolverType):
    m.opt.solver = solver_type


def set_contact_type(m: types.Model, contact_type: types.ContactType):
    m.opt.contact_type = contact_type


def use_exponential_limit(m: types.Model):
    m.opt.limit_type = types.LimitType.EXPONENTIAL


def set_limit_type(m: types.Model, limit_type: types.LimitType):
    m.opt.limit_type = limit_type


def set_integrator_type(m: types.Model, integrator_type: types.IntegratorType):
    m.opt.integrator = integrator_type


def set_muscle_dynamics_substeps(m: types.Model, substeps: int):
    m.opt.muscle_dyn_substeps = substeps


def set_solref(m: types.Model, solref: tuple[float, float]):
    m.opt.solref = wp.vec2(solref[0], solref[1])


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
    return wp.to_torch(d.xpos)


def body_com_positions(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.xipos)


def body_rotations(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.xquat)


def body_velocities(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.xvel)


def body_com_velocities(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.xivel)


def body_user_forces(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.xfrc_user)


def joint_positions(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.qpos)


def joint_velocities(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.qvel)


def joint_accelerations(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.qacc)


def joint_accelerations(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.qacc)


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


def muscle_metadata_np(m: types.Model) -> np.ndarray:
    return m.muscle_metadata.numpy()


def muscle_length_info_np(d: types.Data) -> np.ndarray:
    return d.muscle_length_info.numpy()


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
    return wp.to_torch(d.vis_xpos)


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
    return wp.to_torch(d.geom_xpos)


def get_collider_rotations(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.geom_xquat)


def grf(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.grf)


def limit_torques(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.dof_lim_torque)


def joint_moments(d: types.Data) -> torch.Tensor:
    return wp.to_torch(d.joint_moments)
