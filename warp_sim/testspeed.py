import warp_sim._src.init_model as init_model
import warp_sim._src.forward as forward
import warp_sim._src.math as math

import warp as wp
import numpy as np

from warp_sim.utils.osim_parser import parse_osim_file
from warp_sim.utils.osim_converter import *
from warp_sim.render.renderer import Viewer, ViewerType
from warp_sim._src.benchmark import benchmark

import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--recompile", action="store_true")
arg_parser.add_argument("--debug", action="store_true")


def _print_trace(trace, indent, steps):
    if indent == 0:
        print("\nEvent trace:\n")
    for k, v in trace.items():
        times, sub_trace = v
        if len(times) == 1:
            print("  " * indent + f"{k}: {1e6 * times[0] / steps:.2f}")
        else:
            print("  " * indent + f"{k}: [ ", end="")
            for i in range(len(times)):
                print(f"{1e6 * times[i] / steps:.2f}", end="")
                print(", " if i < len(times) - 1 else " ", end="")
            print("]")
        _print_trace(sub_trace, indent + 1, steps)


def exclusive_scan(v, mark_empty: bool):
    result = [0] * (len(v) + 1)
    for i in range(1, len(result)):
        result[i] = result[i - 1] + v[i - 1]
    # Remove the last element to return the exclusive scan
    result = result[:-1]

    if mark_empty:
        for i in range(len(v)):
            if v[i] == 0:
                result[i] = -1

    return result


def to_warp_array(lst, dtype):
    arr = np.array(lst)
    # remove 2nd dimension if it exists
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.squeeze(axis=1)
    return wp.from_numpy(arr, dtype=dtype)


def make_zero(shape, dtype):
    return wp.zeros(shape, dtype=dtype)


def main():
    raw_osim_model = parse_osim_file("data/osim/model.osim")
    checked_osim_model = to_checked_model(raw_osim_model)
    osim_model = convert_y_up_z_up(checked_osim_model)

    nb = num_bodies(osim_model)

    joint_num_qdofs = get_joint_num_dofs(osim_model, vel_dofs=False)
    joint_num_vdofs = get_joint_num_dofs(osim_model, vel_dofs=True)

    nv = sum(joint_num_vdofs)
    nq = sum(joint_num_qdofs)
    nmuscle = num_muscles(osim_model)

    joint_types = get_joint_types(osim_model)
    n_conv_jnts, n_custom_jnts = 0, 0
    for jt in joint_types:
        if jt == types.JointType.CUSTOM:
            n_custom_jnts += 1
        else:
            n_conv_jnts += 1

    ngeom = num_colliders(osim_model)
    nsite = num_sites(osim_model)
    # qpos0 = get_default_positions(osim_model)
    # print(qpos0)
    qpos0 = [0.0] * nq
    qpos0[0:3] = [0.0, 0.0, 1.5]  # Root pos
    qpos0[3] = 1  # root quat

    qpos_spring = [0.0] * len(qpos0)  # Placeholder for spring positions

    b_masses = body_masses(osim_model)
    inertias = get_body_inertias(osim_model)
    body_local_com = get_local_body_com_pos(osim_model)
    body_local_rot = get_local_body_rot(osim_model)
    body_num_colliders = get_body_num_colliders(osim_model)
    body_collider_offset = exclusive_scan(body_num_colliders, True)

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

    site_body_ids = get_site_body_ids(osim_model)
    site_pos = get_site_pos(osim_model)

    muscle_pts_num = get_muscle_num_pts(osim_model)
    muscle_pts_adr = exclusive_scan(muscle_pts_num, False)

    dof_armature = [0.01] * nv  # Placeholder for DOF armature
    dof_armature[0:6] = [0.0] * 6  # No armature for free joint
    dof_damping = [1.0] * nv  # Placeholder for DOF
    dof_damping[0:6] = [0.0] * 6  # No damping for free joint
    jnt_stiffness = [0.0] * nb  # Placeholder for joint stiffness

    dof_limit_ranges, dof_limit_adr, dof_limit_qadr = get_dof_limits(osim_model)
    n_limits = len(dof_limit_ranges)

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

    #
    # jnt_limited_slide_hinge_adr = wp.array(
    #   np.nonzero(
    #     mjm.jnt_limited & ((mjm.jnt_type == mujoco.mjtJoint.mjJNT_SLIDE) | (mjm.jnt_type == mujoco.mjtJoint.mjJNT_HINGE))
    #   )[0],
    #   dtype=int,
    # ),
    # jnt_limited_ball_adr=wp.array(
    #   np.nonzero(mjm.jnt_limited & (mjm.jnt_type == mujoco.mjtJoint.mjJNT_BALL))[0],
    #   dtype=int,
    # ),

    n_worlds = 8192

    # precalculated geom pairs
    geom1, geom2 = np.triu_indices(ngeom, k=1)
    nxn_geom_pair = np.stack((geom1, geom2), axis=1)

    # Contact pair id: -1 if not pre-defined, -2 if skipped, id otherwise
    nxn_pairid_contact = -1 * np.ones(len(geom1), dtype=int)

    # filter out parent-child collisions and self-collisions
    geom_bodyid = np.array(geom_data.body_id)
    body_parentid = np.array(body_parent_ids)
    bodyid1, bodyid2 = geom_bodyid[geom1], geom_bodyid[geom2]
    parentid1, parentid2 = body_parentid[bodyid1], body_parentid[bodyid2]

    self_collision = (bodyid1 == bodyid2)
    parent_child_collision = (
            ((bodyid1 == parentid2) & (bodyid1 != 0)) |
            ((bodyid2 == parentid1) & (bodyid2 != 0)))
    nxn_pairid_contact[parent_child_collision | self_collision] = -2

    nxn_pairid_collision = -1 * np.ones(len(geom1), dtype=int)

    include = (nxn_pairid_contact > -2) | (nxn_pairid_collision >= 0)
    nxn_pairid = np.hstack([nxn_pairid_contact.reshape((-1, 1)),
                            nxn_pairid_collision.reshape((-1, 1))])
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

    naconmax = 262144
    njmax = 128

    # needs shapes
    opt = types.Option(
        timestep=0.002,
        impratio=1.0,
        tolerance=1e-8,
        ls_tolerance=0.01,
        ccd_tolerance=1e-6,
        gravity=-9.81,
        solver=types.SolverType.NEWTON,
        iterations=50,
        ls_iterations=100,
        ccd_iterations=50,
        warm_start=True,
        ls_parallel=False,
        ls_parallel_min_step=1e-8,
        graph_conditional=True
    )

    m = types.Model(
        nbody=nb,
        nv=nv,
        nq=nq,
        nmuscle=nmuscle,
        ndoflimit=n_limits,

        njnts_conv=n_conv_jnts,
        njnts_cst=n_custom_jnts,

        ngeom=ngeom,
        nsite=nsite,

        opt=opt,

        # warp arrays
        qpos0=to_warp_array(qpos0, dtype=float),
        qpos_spring=to_warp_array(qpos_spring, dtype=float),

        body_mass=to_warp_array(b_masses, dtype=float),
        body_inertia=to_warp_array(inertias, dtype=wp.vec3),
        body_ipos=to_warp_array(body_local_com, dtype=wp.vec3),
        body_iquat=to_warp_array(body_local_rot, dtype=wp.quat),

        body_geomnum=to_warp_array(body_num_colliders, dtype=int),
        body_geomadr=to_warp_array(body_collider_offset, dtype=int),

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

        geom_type=to_warp_array(geom_types, dtype=int),
        geom_bodyid=to_warp_array(geom_data.body_id, dtype=int),
        geom_size=to_warp_array(geom_data.size, dtype=wp.vec3),
        geom_pos=to_warp_array(geom_data.pos, dtype=wp.vec3),
        geom_quat=to_warp_array(geom_data.rot, dtype=wp.quat),
        geom_friction=to_warp_array(geom_data.friction, dtype=wp.vec3),
        geom_aabb=to_warp_array(geom_data.aabb, dtype=wp.vec3),
        geom_rbound=to_warp_array(geom_data.rbound, dtype=float),
        geom_margin=make_zero(ngeom, dtype=float),

        geom_pair_type_count=tuple(geom_type_pair_count),
        nxn_geom_pair_filtered=wp.array(nxn_geom_pair_filtered, dtype=wp.vec2i),
        nxn_pairid_filtered=wp.array(nxn_pairid_filtered, dtype=wp.vec2i),

        site_bodyid=to_warp_array(site_body_ids, dtype=int),
        site_pos=to_warp_array(site_pos, dtype=wp.vec3),

        muscle_pts_num=to_warp_array(muscle_pts_num, dtype=int),
        muscle_pts_adr=to_warp_array(muscle_pts_adr, dtype=int),

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

    d = types.Data(
        solver_niter=make_zero(n_worlds, dtype=int),

        nl=make_zero(n_worlds, dtype=int),
        nefc=make_zero(n_worlds, dtype=int),
        time=make_zero(n_worlds, dtype=float),

        qpos=wp.array(np.tile(qpos0, (n_worlds, 1)), dtype=float),
        qvel=make_zero((n_worlds, nv), dtype=float),
        act=make_zero((n_worlds, nmuscle), dtype=float),

        qacc_warmstart=make_zero((n_worlds, nv), dtype=float),
        qfrc_applied=make_zero((n_worlds, nv), dtype=float),
        xfrc_applied=make_zero((n_worlds, nb), dtype=wp.spatial_vector),

        qacc=make_zero((n_worlds, nv), dtype=float),
        act_dot=make_zero((n_worlds, nmuscle), dtype=float),

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

        site_rpos=make_zero((n_worlds, nsite), dtype=wp.vec3),
        site_xpos=make_zero((n_worlds, nsite), dtype=wp.vec3),
        site_xvel=make_zero((n_worlds, nsite), dtype=wp.vec3),
        site_diff_vec=make_zero((n_worlds, nsite-1), dtype=wp.vec3),
        site_diff_len=make_zero((n_worlds, nsite-1), dtype=float),
        site_diff_vel=make_zero((n_worlds, nsite - 1), dtype=float),

        subtree_com=make_zero((n_worlds, nb), dtype=wp.vec3),
        cdof=make_zero((n_worlds, nv), dtype=wp.spatial_vector),
        cinert=make_zero((n_worlds, nb), dtype=types.vec10),

        crb=make_zero((n_worlds, nb), dtype=types.vec10),
        qM=make_zero((n_worlds, nv, nv), dtype=float),
        qLD=make_zero((n_worlds, nv, nv), dtype=float),
        qLDiagInv=make_zero((n_worlds, nv), dtype=float),

        muscle_length=make_zero((n_worlds, nmuscle), dtype=float),
        muscle_velocity=make_zero((n_worlds, nmuscle), dtype=float),

        cvel=make_zero((n_worlds, nb), dtype=wp.spatial_vector),
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
        nworld=n_worlds,
        naconmax=naconmax,
        njmax=njmax,
        nacon=make_zero(n_worlds, dtype=int),
        nsolving=make_zero(1, dtype=int),
        subtree_bodyvel=make_zero((n_worlds, nb), dtype=wp.vec3),

        # collision driver
        collision_pair=wp.zeros((naconmax,), dtype=wp.vec2i),
        collision_pairid=wp.zeros((naconmax,), dtype=wp.vec2i),
        collision_worldid=wp.zeros((naconmax,), dtype=int),
        ncollision=wp.zeros((1,), dtype=int),
    )

    if args.recompile:
        wp.clear_kernel_cache()
    if args.debug:
        wp.config.mode = "debug"

    init_model._model_init(m, d)

    # viewer = Viewer(viewer_type=ViewerType.USD)
    # for _ in range(1000):
    #     forward.step(m, d)
    #     viewer.render(m, d)
    # viewer.close()

    n_steps = 1500
    res = benchmark(fn=forward.step, m=m, d=d, nstep=n_steps, event_trace=True,
                    measure_alloc=True, measure_solver_niter=True)
    jit_time, run_time, trace, nacon, nefc, solver_niter, nsuccess = res
    steps = n_worlds * n_steps

    print(f"""
    Summary for {n_worlds} parallel rollouts

    Total JIT time: {jit_time:.2f} s
    Total simulation time: {run_time:.2f} s
    Total steps per second: {steps / run_time:,.0f}
    Total realtime factor: {steps * m.opt.timestep / run_time:,.2f} x
    Total time per step: {1e9 * run_time / steps:.2f} ns
    Total converged worlds: {nsuccess} / {d.nworld}""")

    _print_trace(trace, 0, steps)


if __name__ == "__main__":
    args = arg_parser.parse_args()
    main()
