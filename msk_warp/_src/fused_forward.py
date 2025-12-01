"""
Experimental: giving each world a thread block
"""
import numpy as np
import warp as wp

from . import math
from . import mobilizers
from .types import Data
from .types import Model
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})

# Such a hack
snippet = """ __syncthreads(); """


@wp.func_native(snippet)
def sync(tid: int): ...


@wp.func
def kinematics_root(
        xpos_out: wp.array(dtype=wp.vec3),
        xquat_out: wp.array(dtype=wp.quat),
        xipos_out: wp.array(dtype=wp.vec3),
        xmat_out: wp.array(dtype=wp.mat33),
        ximat_out: wp.array(dtype=wp.mat33),
):
    xpos_out[0] = wp.vec3(0.0)
    xquat_out[0] = wp.quat(1.0, 0.0, 0.0, 0.0)
    xipos_out[0] = wp.vec3(0.0)
    xmat_out[0] = wp.identity(n=3, dtype=wp.float32)
    ximat_out[0] = wp.identity(n=3, dtype=wp.float32)
    return


@wp.func
def _kinematics_level(
        # Model:
        body_parentid: wp.array(dtype=int),
        body_ipos: wp.array(dtype=wp.vec3),
        body_iquat: wp.array(dtype=wp.quat),
        jnt_type: wp.array(dtype=int),
        jnt_qposadr: wp.array(dtype=int),
        jnt_rel_parent: wp.array(dtype=wp.vec3),
        jnt_rel_child: wp.array(dtype=wp.vec3),
        jnt_rel_parent_rot: wp.array(dtype=wp.quat),
        jnt_rel_child_rot: wp.array(dtype=wp.quat),
        jnt_cst_adr: wp.array(dtype=int),  # start custom joints
        const_fns: wp.array(dtype=float),
        linear_fns: wp.array(dtype=wp.vec2),
        cst_txfm_axis: wp.array2d(dtype=wp.vec3),
        cst_txfm_fn: wp.array2d(dtype=int),
        cst_txfm_fn_adr: wp.array2d(dtype=int),
        cst_txfm_qadr: wp.array2d(dtype=int),
        # Data in:
        qpos_in: wp.array2d(dtype=float),
        xpos_in: wp.array2d(dtype=wp.vec3),
        xquat_in: wp.array2d(dtype=wp.quat),
        # Data out:
        xpos_out: wp.array2d(dtype=wp.vec3),
        xquat_out: wp.array2d(dtype=wp.quat),
        xmat_out: wp.array2d(dtype=wp.mat33),
        xipos_out: wp.array2d(dtype=wp.vec3),
        ximat_out: wp.array2d(dtype=wp.mat33),
        xanchor_out: wp.array2d(dtype=wp.vec3),
        xaxis_out: wp.array3d(dtype=wp.vec3),
        #
        worldid: int,
        bodyid: int
):
    jnt_type_ = jnt_type[bodyid]

    cst_adr = jnt_cst_adr[bodyid]
    xpos, xquat, xanchor = mobilizers.fk_joint(
        jnt_type_, jnt_qposadr[bodyid], qpos_in[worldid],
        body_parentid[bodyid], xpos_in[worldid], xquat_in[worldid],
        jnt_rel_parent[bodyid], jnt_rel_parent_rot[bodyid],
        jnt_rel_child[bodyid], jnt_rel_child_rot[bodyid],
        cst_txfm_axis[cst_adr], cst_txfm_fn[cst_adr],
        cst_txfm_fn_adr[cst_adr], cst_txfm_qadr[cst_adr],
        const_fns, linear_fns,
        xaxis_out[worldid, bodyid]
    )

    xpos_out[worldid, bodyid] = xpos
    xquat_out[worldid, bodyid] = wp.normalize(xquat)
    xmat_out[worldid, bodyid] = math.quat_to_mat(xquat)
    xanchor_out[worldid, bodyid] = xanchor

    # inertial frame
    xipos_out[worldid, bodyid] = (
            xpos + math.rot_vec_quat(body_ipos[bodyid], xquat))
    ximat_out[worldid, bodyid] = (
        math.quat_to_mat(math.mul_quat(xquat, body_iquat[bodyid])))


@wp.func
def geom_local_to_global(
        geom_bodyid: wp.array(dtype=int),
        geom_pos: wp.array(dtype=wp.vec3),
        geom_quat: wp.array(dtype=wp.quat),
        xpos_in: wp.array(dtype=wp.vec3),
        xquat_in: wp.array(dtype=wp.quat),
        geom_xpos_out: wp.array(dtype=wp.vec3),
        geom_xquat_out: wp.array(dtype=wp.quat),
        geom_xmat_out: wp.array(dtype=wp.mat33),
        geomid: int
):
    bodyid = geom_bodyid[geomid]

    xpos = xpos_in[bodyid]
    xquat = xquat_in[bodyid]

    geom_xpos_out[geomid] = (
            xpos + math.rot_vec_quat(geom_pos[geomid], xquat))
    geom_xquat_out[geomid] = (
        math.mul_quat(xquat, geom_quat[geomid]))
    geom_xmat_out[geomid] = (
        math.quat_to_mat(geom_xquat_out[geomid]))


@wp.func
def site_local_to_global(
        # Model:
        site_bodyid: wp.array(dtype=int),
        site_pos: wp.array(dtype=wp.vec3),
        # Data in:
        xpos_in: wp.array(dtype=wp.vec3),
        xquat_in: wp.array(dtype=wp.quat),
        # Data out:
        site_rpos_out: wp.array(dtype=wp.vec3),
        site_xpos_out: wp.array(dtype=wp.vec3),
        siteid: int,
):
    bodyid = site_bodyid[siteid]
    xpos = xpos_in[bodyid]
    xquat = xquat_in[bodyid]
    # Relative to body and world positions
    site_rpos_out[siteid] = math.rot_vec_quat(site_pos[siteid], xquat)
    site_xpos_out[siteid] = xpos + site_rpos_out[siteid]


@wp.func
def subtree_com_init(
        # Model:
        body_mass: wp.array(dtype=float),
        # Data in:
        xipos_in: wp.array(dtype=wp.vec3),
        # Data out:
        subtree_com_out: wp.array(dtype=wp.vec3),
        # In:
        bodyid: int
):
    subtree_com_out[bodyid] = xipos_in[bodyid] * body_mass[bodyid]


@wp.kernel
def fused_fwd(
        # Model:
        nbody: int,
        ngeom: int,
        nsite: int,

        body_mass: wp.array(dtype=float),

        body_parentid: wp.array(dtype=int),
        body_ipos: wp.array(dtype=wp.vec3),
        body_iquat: wp.array(dtype=wp.quat),

        jnt_type: wp.array(dtype=int),
        jnt_qposadr: wp.array(dtype=int),
        jnt_rel_parent: wp.array(dtype=wp.vec3),
        jnt_rel_child: wp.array(dtype=wp.vec3),
        jnt_rel_parent_rot: wp.array(dtype=wp.quat),
        jnt_rel_child_rot: wp.array(dtype=wp.quat),
        jnt_cst_adr: wp.array(dtype=int),  # start custom joints
        const_fns: wp.array(dtype=float),
        linear_fns: wp.array(dtype=wp.vec2),
        cst_txfm_axis: wp.array2d(dtype=wp.vec3),
        cst_txfm_fn: wp.array2d(dtype=int),
        cst_txfm_fn_adr: wp.array2d(dtype=int),
        cst_txfm_qadr: wp.array2d(dtype=int),

        geom_bodyid: wp.array(dtype=int),
        geom_pos: wp.array(dtype=wp.vec3),
        geom_quat: wp.array(dtype=wp.quat),

        site_bodyid: wp.array(dtype=int),
        site_pos: wp.array(dtype=wp.vec3),
        # Data in:
        # Data:
        d_qpos: wp.array2d(dtype=float),
        d_xpos: wp.array2d(dtype=wp.vec3),
        d_xquat: wp.array2d(dtype=wp.quat),
        d_xipos: wp.array2d(dtype=wp.vec3),
        d_xmat: wp.array2d(dtype=wp.mat33),
        d_ximat: wp.array2d(dtype=wp.mat33),
        d_xanchor_out: wp.array2d(dtype=wp.vec3),
        d_xaxis_out: wp.array3d(dtype=wp.vec3),

        d_subtree_com_out: wp.array2d(dtype=wp.vec3),

        d_geom_xpos: wp.array2d(dtype=wp.vec3),
        d_geom_xquat: wp.array2d(dtype=wp.quat),
        d_geom_xmat: wp.array2d(dtype=wp.mat33),

        d_site_rpos: wp.array2d(dtype=wp.vec3),
        d_site_xpos: wp.array2d(dtype=wp.vec3),
):
    worldid, tid = wp.tid()
    block_dim = wp.block_dim()

    xpos = d_xpos[worldid]
    xquat = d_xquat[worldid]
    xipos = d_xipos[worldid]
    xmat = d_xmat[worldid]
    ximat = d_ximat[worldid]

    # Kinematics
    if tid == 0:
        kinematics_root(
            xpos_out=xpos,
            xquat_out=xquat,
            xipos_out=xipos,
            xmat_out=xmat,
            ximat_out=ximat,
        )

    # sync(tid)
    if tid == 0:
        for bodyid in range(nbody):
            _kinematics_level(
                body_parentid=body_parentid,
                body_ipos=body_ipos,
                body_iquat=body_iquat,
                jnt_type=jnt_type,
                jnt_qposadr=jnt_qposadr,
                jnt_rel_parent=jnt_rel_parent,
                jnt_rel_child=jnt_rel_child,
                jnt_rel_parent_rot=jnt_rel_parent_rot,
                jnt_rel_child_rot=jnt_rel_child_rot,
                jnt_cst_adr=jnt_cst_adr,
                const_fns=const_fns,
                linear_fns=linear_fns,
                cst_txfm_axis=cst_txfm_axis,
                cst_txfm_fn=cst_txfm_fn,
                cst_txfm_fn_adr=cst_txfm_fn_adr,
                cst_txfm_qadr=cst_txfm_qadr,
                qpos_in=d_qpos,
                xpos_in=d_xpos,
                xquat_in=d_xquat,
                xpos_out=d_xpos,
                xquat_out=d_xquat,
                xmat_out=d_xmat,
                xipos_out=d_xipos,
                ximat_out=d_ximat,
                xanchor_out=d_xanchor_out,
                xaxis_out=d_xaxis_out,
                worldid=worldid,
                bodyid=bodyid,
            )

    # sync(tid)
    for geomid in range(tid, ngeom, block_dim):
        geom_local_to_global(
            geom_bodyid=geom_bodyid,
            geom_pos=geom_pos,
            geom_quat=geom_quat,
            xpos_in=d_xpos[worldid],
            xquat_in=d_xquat[worldid],
            geom_xpos_out=d_geom_xpos[worldid],
            geom_xquat_out=d_geom_xquat[worldid],
            geom_xmat_out=d_geom_xmat[worldid],
            geomid=geomid,
        )

    for siteid in range(tid, nsite, block_dim):
        site_local_to_global(
            site_bodyid=site_bodyid,
            site_pos=site_pos,
            xpos_in=d_xpos[worldid],
            xquat_in=d_xquat[worldid],
            site_rpos_out=d_site_rpos[worldid],
            site_xpos_out=d_site_xpos[worldid],
            siteid=siteid,
        )

    for bodyid in range(tid, nbody, block_dim):
        subtree_com_init(
            body_mass=body_mass,
            xipos_in=d_xipos[worldid],
            subtree_com_out=d_subtree_com_out[worldid],
            bodyid=bodyid,
        )
    return


@event_scope
def step_fused(m: Model, d: Data):
    wp.launch(
        fused_fwd,
        dim=(d.nworld, 32),
        inputs=[m.nbody, m.ngeom, m.nsite,
                m.body_mass, m.body_parentid, m.body_ipos, m.body_iquat,
                m.jnt_type, m.jnt_qposadr, m.jnt_rel_parent,
                m.jnt_rel_child, m.jnt_rel_parent_rot,
                m.jnt_rel_child_rot, m.jnt_cst_adr,
                m.const_fns, m.linear_fns,
                m.cst_txfm_axis, m.cst_txfm_fn,
                m.cst_txfm_fn_adr, m.cst_txfm_qadr,
                m.geom_bodyid, m.geom_pos, m.geom_quat,
                m.site_bodyid, m.site_pos,
                d.qpos, d.xpos, d.xquat, d.xipos, d.xmat, d.ximat,
                d.xanchor, d.xaxis,
                d.subtree_com,
                d.geom_xpos, d.geom_xquat, d.geom_xmat,
                d.site_rpos, d.site_xpos
                ],
        max_blocks=1,
        block_dim=32,
    )
    return


@wp.kernel
def _set_fixed_step_size(
        # In:
        dt: float,
        # Data out:
        actual_step_size_out: wp.array(dtype=float),
):
    worldid = wp.tid()
    actual_step_size_out[worldid] = dt


from time import perf_counter


@event_scope
def step_to(m: Model, d: Data, dt: float, dt_sim: float):
    start = perf_counter()
    num_substeps = np.ceil(dt / dt_sim)
    wp.launch(
        _set_fixed_step_size,
        dim=d.nworld,
        inputs=[dt_sim],
        outputs=[d.actual_step_size],
    )

    # We are off to the races now
    for _step in range(int(num_substeps)):
        step_fused(m, d)

    end = perf_counter()
    print("FPS: ", 1.0 / (end - start))
