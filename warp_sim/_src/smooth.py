# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import warp as wp

from . import math
from . import support
from .types import MJ_MINVAL
from .types import Data
from .types import JointType
from .types import Model
from .types import TileSet
from .types import CustomFnType
from .types import vec5
from .types import vec10
from .types import vec11
from .warp_util import cache_kernel
from .warp_util import event_scope
from .warp_util import kernel as nested_kernel

wp.set_module_options({"enable_backward": False})


# returns f(x) and df(x)
@wp.func
def evaluate_txfm(
        # data in
        qpos_in: wp.array(dtype=float),
        # in
        txfm_fn_type: int,
        txfm_fn_adr: int,
        txfm_qadr: int,
        # model in
        const_fns: wp.array(dtype=float),
        linear_fns: wp.array(dtype=wp.vec2),
) -> wp.vec2:
    if txfm_fn_type == CustomFnType.CONSTANT:
        return wp.vec2(const_fns[txfm_fn_adr], 0.0)
    elif txfm_fn_type == CustomFnType.LINEAR:
        if txfm_qadr == -1:
            return wp.vec2(0.0, 0.0)
        lin_fn = linear_fns[txfm_fn_adr]
        return wp.vec2(lin_fn[0] * qpos_in[txfm_qadr] + lin_fn[1], lin_fn[0])
    return wp.vec2(0.0, 0.0)


@wp.kernel
def _kinematics_root(
        # Data out:
        xpos_out: wp.array2d(dtype=wp.vec3),
        xquat_out: wp.array2d(dtype=wp.quat),
        xmat_out: wp.array2d(dtype=wp.mat33),
        xipos_out: wp.array2d(dtype=wp.vec3),
        ximat_out: wp.array2d(dtype=wp.mat33),
):
    worldid = wp.tid()
    xpos_out[worldid, 0] = wp.vec3(0.0)
    xquat_out[worldid, 0] = wp.quat(1.0, 0.0, 0.0, 0.0)
    xipos_out[worldid, 0] = wp.vec3(0.0)
    xmat_out[worldid, 0] = wp.identity(n=3, dtype=wp.float32)
    ximat_out[worldid, 0] = wp.identity(n=3, dtype=wp.float32)


@wp.kernel
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
        # In:
        body_tree_: wp.array(dtype=int),
        # Data out:
        xpos_out: wp.array2d(dtype=wp.vec3),
        xquat_out: wp.array2d(dtype=wp.quat),
        xmat_out: wp.array2d(dtype=wp.mat33),
        xipos_out: wp.array2d(dtype=wp.vec3),
        ximat_out: wp.array2d(dtype=wp.mat33),
        xanchor_out: wp.array2d(dtype=wp.vec3),
        xaxis_out: wp.array3d(dtype=wp.vec3),
):
    worldid, nodeid = wp.tid()
    bodyid = body_tree_[nodeid]
    qpos = qpos_in[worldid]
    jnt_type_ = jnt_type[bodyid]

    if jnt_type_ == JointType.FREE:
        # free joint: (x,y,z) + (qw,qx,qy,qz)
        qadr = jnt_qposadr[bodyid]
        xpos = wp.vec3(qpos[qadr], qpos[qadr + 1], qpos[qadr + 2])
        xquat = wp.quat(qpos[qadr + 3], qpos[qadr + 4], qpos[qadr + 5],
                        qpos[qadr + 6])
        xquat = wp.normalize(xquat)

        xanchor_out[worldid, bodyid] = xpos
    else:
        # Grab parent frame information
        pid = body_parentid[bodyid]
        p_pos = xpos_in[worldid, pid]
        p_rot = xquat_in[worldid, pid]

        # compute the joint frame
        p_to_jnt = math.rot_vec_quat(jnt_rel_parent[bodyid], p_rot)
        p_to_jnt_rot = jnt_rel_parent_rot[bodyid]
        jnt_pos = p_pos + p_to_jnt
        jnt_rot = math.mul_quat(p_rot, p_to_jnt_rot)

        # joint to child information
        jnt_to_c_rot = jnt_rel_child_rot[bodyid]
        jnt_to_c = jnt_rel_child[bodyid]

        # local joint transformation
        qadr = jnt_qposadr[bodyid]
        qloc_ = wp.quat(1.0, 0.0, 0.0, 0.0)
        xloc_ = wp.vec3(0.0, 0.0, 0.0)
        if jnt_type_ == JointType.HINGE:
            hinge_axis = wp.vec3(0.0, -1.0, 0.0)
            qloc_ = math.axis_angle_to_quat(hinge_axis, qpos[qadr])
            xaxis_out[worldid, bodyid, 0] = math.rot_vec_quat(
                hinge_axis, jnt_rot)

        if jnt_type_ == JointType.BALL:
            qloc_ = wp.quat(qpos[qadr + 0], qpos[qadr + 1], qpos[qadr + 2],
                            qpos[qadr + 3])
            qloc_ = wp.normalize(qloc_)

        elif jnt_type_ == JointType.SLIDE:
            slide_axis = wp.vec3(1.0, 0.0, 0.0)
            xloc_ = qpos[qadr] * slide_axis
            xaxis_out[worldid, bodyid, 0] = math.rot_vec_quat(
                slide_axis, jnt_rot)

        elif jnt_type_ == JointType.UNIVERSAL:
            axis1 = wp.vec3(1.0, 0.0, 0.0)
            axis2 = wp.vec3(0.0, 0.0, 1.0)

            qloc1 = math.axis_angle_to_quat(axis1, qpos[qadr + 0])
            qloc2 = math.axis_angle_to_quat(axis2, qpos[qadr + 1])
            qloc_ = math.mul_quat(qloc1, qloc2)

            # Keep track of first rotation
            xaxis_out[worldid, bodyid, 0] = (
                math.rot_vec_quat(axis1, jnt_rot))
            xaxis_out[worldid, bodyid, 1] = (
                math.rot_vec_quat(axis2, math.mul_quat(jnt_rot, qloc1)))

        elif jnt_type_ == JointType.CUSTOM:
            cst_adr = jnt_cst_adr[bodyid]
            txfm_axes = cst_txfm_axis[cst_adr]
            txfm_fn = cst_txfm_fn[cst_adr]

            # First 3 are rotation
            for i in range(3):
                fn_eval = evaluate_txfm(
                    qpos,
                    txfm_fn[i],
                    cst_txfm_fn_adr[cst_adr, i],
                    cst_txfm_qadr[cst_adr, i],
                    const_fns,
                    linear_fns,
                )
                # store intermediate rotated axes
                xaxis_out[worldid, bodyid, i] = fn_eval[1] * math.rot_vec_quat(
                    txfm_axes[i], math.mul_quat(jnt_rot, qloc_))

                qloc_ = math.mul_quat(
                    qloc_, math.axis_angle_to_quat(txfm_axes[i], fn_eval[0]))

            # Next 3 are translation
            for i in range(3, 6):
                fn_eval = evaluate_txfm(
                    qpos,
                    txfm_fn[i],
                    cst_txfm_fn_adr[cst_adr, i],
                    cst_txfm_qadr[cst_adr, i],
                    const_fns,
                    linear_fns,
                )
                xloc_ += fn_eval[0] * txfm_axes[i]
                # store intermediate rotated axes, with derivative
                xaxis_out[worldid, bodyid, i] = fn_eval[1] * math.rot_vec_quat(
                    txfm_axes[i], jnt_rot)

        # world coordinates
        xquat = math.mul_quat(jnt_rot, math.mul_quat(qloc_, jnt_to_c_rot))
        xpos = (jnt_pos + math.rot_vec_quat(xloc_, jnt_rot) +
                math.rot_vec_quat(jnt_to_c, xquat))

        xanchor_out[worldid, bodyid] = jnt_pos

    xpos_out[worldid, bodyid] = xpos
    xquat_out[worldid, bodyid] = wp.normalize(xquat)
    xmat_out[worldid, bodyid] = math.quat_to_mat(xquat)

    # inertial frame
    xipos_out[worldid, bodyid] = (
            xpos + math.rot_vec_quat(body_ipos[bodyid], xquat))
    ximat_out[worldid, bodyid] = (
        math.quat_to_mat(math.mul_quat(xquat, body_iquat[bodyid])))


@wp.kernel
def _geom_local_to_global(
        # Model:
        geom_bodyid: wp.array(dtype=int),
        geom_pos: wp.array(dtype=wp.vec3),
        geom_quat: wp.array(dtype=wp.quat),
        # Data in:
        xpos_in: wp.array2d(dtype=wp.vec3),
        xquat_in: wp.array2d(dtype=wp.quat),
        # Data out:
        geom_xpos_out: wp.array2d(dtype=wp.vec3),
        geom_xquat_out: wp.array2d(dtype=wp.quat),
        geom_xmat_out: wp.array2d(dtype=wp.mat33),
):
    worldid, geomid = wp.tid()
    bodyid = geom_bodyid[geomid]

    xpos = xpos_in[worldid, bodyid]
    xquat = xquat_in[worldid, bodyid]

    geom_xpos_out[worldid, geomid] = (
            xpos + math.rot_vec_quat(geom_pos[geomid], xquat))
    geom_xquat_out[worldid, geomid] = (
        math.mul_quat(xquat, geom_quat[geomid]))
    geom_xmat_out[worldid, geomid] = (
        math.quat_to_mat(geom_xquat_out[worldid, geomid]))


@wp.kernel
def _site_local_to_global(
        # Model:
        site_bodyid: wp.array(dtype=int),
        site_pos: wp.array(dtype=wp.vec3),
        # Data in:
        xpos_in: wp.array2d(dtype=wp.vec3),
        xquat_in: wp.array2d(dtype=wp.quat),
        # Data out:
        site_rpos_out: wp.array2d(dtype=wp.vec3),
        site_xpos_out: wp.array2d(dtype=wp.vec3),
):
    worldid, siteid = wp.tid()
    bodyid = site_bodyid[siteid]
    xpos = xpos_in[worldid, bodyid]
    xquat = xquat_in[worldid, bodyid]
    # Relative to body and world positions
    site_rpos_out[worldid, siteid] = math.rot_vec_quat(site_pos[siteid], xquat)
    site_xpos_out[worldid, siteid] = xpos + site_rpos_out[worldid, siteid]


@wp.kernel
def _site_consecutive_diff_len(
        # Data in:
        site_xpos_in: wp.array2d(dtype=wp.vec3),
        # Data out:
        site_diff_vec_out: wp.array2d(dtype=wp.vec3),
        site_diff_len_out: wp.array2d(dtype=float),
):
    worldid, site_diff_id = wp.tid()
    p1 = site_xpos_in[worldid, site_diff_id]
    p2 = site_xpos_in[worldid, site_diff_id + 1]
    vec, length = math.normalize_with_norm(p2 - p1)
    site_diff_vec_out[worldid, site_diff_id] = vec
    site_diff_len_out[worldid, site_diff_id] = length


@wp.kernel
def _site_consecutive_diff_vel(
        # Data in:
        site_vel_in: wp.array2d(dtype=wp.vec3),
        site_diff_vec_in: wp.array2d(dtype=wp.vec3),
        site_diff_len_in: wp.array2d(dtype=float),
        # Data out:
        site_diff_vel_out: wp.array2d(dtype=float),
):
    worldid, site_diff_id = wp.tid()
    v1 = site_vel_in[worldid, site_diff_id]
    v2 = site_vel_in[worldid, site_diff_id + 1]

    vec = site_diff_vec_in[worldid, site_diff_id]
    length = site_diff_len_in[worldid, site_diff_id]
    if length > 1e-8:
        site_diff_vel_out[worldid, site_diff_id] = wp.dot((v2 - v1), vec)
    else:
        site_diff_vel_out[worldid, site_diff_id] = 0.0


@event_scope
def kinematics(m: Model, d: Data):
    """ Computes forward kinematics for all bodies, sites, geoms. """
    # World body
    wp.launch(_kinematics_root, dim=(d.nworld), inputs=[],
              outputs=[d.xpos, d.xquat, d.xmat, d.xipos, d.ximat])

    for i in range(1, len(m.body_tree)):
        body_tree = m.body_tree[i]
        wp.launch(
            _kinematics_level,
            dim=(d.nworld, body_tree.size),
            inputs=[
                m.body_parentid,
                m.body_ipos,
                m.body_iquat,
                m.jnt_type,
                m.jnt_qposadr,
                m.jnt_rel_parent,
                m.jnt_rel_child,
                m.jnt_rel_parent_rot,
                m.jnt_rel_child_rot,
                m.jnt_cst_adr,
                m.const_fns,
                m.linear_fns,
                m.cst_txfm_axis,
                m.cst_txfm_fn,
                m.cst_txfm_fn_adr,
                m.cst_txfm_qadr,
                d.qpos,
                d.xpos,
                d.xquat,
                body_tree,
            ],
            outputs=[d.xpos, d.xquat, d.xmat, d.xipos, d.ximat, d.xanchor,
                     d.xaxis],
        )
    wp.launch(
        _geom_local_to_global,
        dim=(d.nworld, m.ngeom),
        inputs=[m.geom_bodyid, m.geom_pos, m.geom_quat, d.xpos, d.xquat],
        outputs=[d.geom_xpos, d.geom_xquat, d.geom_xmat],
    )


@wp.kernel
def _subtree_com_init(
        # Model:
        body_mass: wp.array(dtype=float),
        # Data in:
        xipos_in: wp.array2d(dtype=wp.vec3),
        # Data out:
        subtree_com_out: wp.array2d(dtype=wp.vec3),
):
    worldid, bodyid = wp.tid()
    subtree_com_out[worldid, bodyid] = xipos_in[worldid, bodyid] * body_mass[
        bodyid]


@wp.kernel
def _subtree_com_acc(
        # Model:
        body_parentid: wp.array(dtype=int),
        # Data in:
        subtree_com_in: wp.array2d(dtype=wp.vec3),
        # In:
        body_tree_: wp.array(dtype=int),
        # Data out:
        subtree_com_out: wp.array2d(dtype=wp.vec3),
):
    worldid, nodeid = wp.tid()
    bodyid = body_tree_[nodeid]
    pid = body_parentid[bodyid]
    if bodyid != 0:
        wp.atomic_add(subtree_com_out, worldid, pid,
                      subtree_com_in[worldid, bodyid])


@wp.kernel
def _subtree_div(
        # Model:
        body_subtreemass: wp.array(dtype=float),
        # Data in:
        subtree_com_in: wp.array2d(dtype=wp.vec3),
        # Data out:
        subtree_com_out: wp.array2d(dtype=wp.vec3),
):
    worldid, bodyid = wp.tid()
    com = subtree_com_in[worldid, bodyid]
    mass = body_subtreemass[bodyid]
    if mass != 0.0:
        subtree_com_out[worldid, bodyid] = com / mass


@wp.kernel
def _cinert(
        # Model:
        body_rootid: wp.array(dtype=int),
        body_mass: wp.array(dtype=float),
        body_inertia: wp.array(dtype=wp.vec3),
        # Data in:
        xipos_in: wp.array2d(dtype=wp.vec3),
        ximat_in: wp.array2d(dtype=wp.mat33),
        subtree_com_in: wp.array2d(dtype=wp.vec3),
        # Data out:
        cinert_out: wp.array2d(dtype=vec10),
):
    # express inertia in com-based frame
    worldid, bodyid = wp.tid()
    mat = ximat_in[worldid, bodyid]
    inert = body_inertia[bodyid]
    mass = body_mass[bodyid]
    # offset from "origin" to body com
    dif = xipos_in[worldid, bodyid] - subtree_com_in[
        worldid, body_rootid[bodyid]]

    res = vec10()
    # res_rot = mat * diag(inert) * mat'
    inertia_wf = mat @ wp.diag(inert) @ wp.transpose(mat)
    res[0] = inertia_wf[0, 0]
    res[1] = inertia_wf[1, 1]
    res[2] = inertia_wf[2, 2]
    res[3] = inertia_wf[0, 1]
    res[4] = inertia_wf[0, 2]
    res[5] = inertia_wf[1, 2]
    # res_rot -= mass * dif_cross * dif_cross
    res[0] += mass * (dif[1] * dif[1] + dif[2] * dif[2])
    res[1] += mass * (dif[0] * dif[0] + dif[2] * dif[2])
    res[2] += mass * (dif[0] * dif[0] + dif[1] * dif[1])
    res[3] -= mass * dif[0] * dif[1]
    res[4] -= mass * dif[0] * dif[2]
    res[5] -= mass * dif[1] * dif[2]
    # res_tran = mass * dif
    res[6] = mass * dif[0]
    res[7] = mass * dif[1]
    res[8] = mass * dif[2]
    # res_mass = mass
    res[9] = mass

    cinert_out[worldid, bodyid] = res


@wp.kernel
def _cdof(
        # Model:
        body_rootid: wp.array(dtype=int),
        jnt_type: wp.array(dtype=int),
        jnt_dofadr: wp.array(dtype=int),
        jnt_dofnum: wp.array(dtype=int),
        jnt_cst_adr: wp.array(dtype=int),  # start custom joints
        cst_txfm_dofadr: wp.array2d(dtype=int),
        # Data in:
        xmat_in: wp.array2d(dtype=wp.mat33),
        xanchor_in: wp.array2d(dtype=wp.vec3),
        xaxis_in: wp.array3d(dtype=wp.vec3),
        subtree_com_in: wp.array2d(dtype=wp.vec3),
        # Data out:
        cdof_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, bodyid = wp.tid()
    if bodyid == 0:
        return
    dofid = jnt_dofadr[bodyid]
    jnt_type_ = jnt_type[bodyid]
    xmat = wp.transpose(xmat_in[worldid, bodyid])

    joint_pos = xanchor_in[worldid, bodyid]

    # compute com-anchor vector
    root_com = subtree_com_in[worldid, body_rootid[bodyid]]
    offset = root_com - joint_pos

    res = cdof_out[worldid]
    if jnt_type_ == JointType.FREE:
        res[dofid + 0] = wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        res[dofid + 1] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        res[dofid + 2] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        # I_3 rotation in child frame (assume no subsequent rotations)
        res[dofid + 3] = wp.spatial_vector(xmat[0], wp.cross(xmat[0], offset))
        res[dofid + 4] = wp.spatial_vector(xmat[1], wp.cross(xmat[1], offset))
        res[dofid + 5] = wp.spatial_vector(xmat[2], wp.cross(xmat[2], offset))
    elif jnt_type_ == JointType.BALL:  # ball
        # I_3 rotation in child frame (assume no subsequent rotations)
        res[dofid + 0] = wp.spatial_vector(xmat[0], wp.cross(xmat[0], offset))
        res[dofid + 1] = wp.spatial_vector(xmat[1], wp.cross(xmat[1], offset))
        res[dofid + 2] = wp.spatial_vector(xmat[2], wp.cross(xmat[2], offset))
    elif jnt_type_ == JointType.SLIDE:
        xaxis = xaxis_in[worldid, bodyid, 0]
        res[dofid] = wp.spatial_vector(wp.vec3(0.0), xaxis)
    elif jnt_type_ == JointType.HINGE:  # hinge
        xaxis = xaxis_in[worldid, bodyid, 0]
        res[dofid] = wp.spatial_vector(xaxis, wp.cross(xaxis, offset))
    elif jnt_type_ == JointType.UNIVERSAL:
        xaxis1 = xaxis_in[worldid, bodyid, 0]
        xaxis2 = xaxis_in[worldid, bodyid, 1]

        res[dofid + 0] = wp.spatial_vector(xaxis1, wp.cross(xaxis1, offset))
        res[dofid + 1] = wp.spatial_vector(xaxis2, wp.cross(xaxis2, offset))
    elif jnt_type_ == JointType.CUSTOM:
        # initialize to zero
        for i in range(jnt_dofnum[bodyid]):
            res[dofid + i] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # accumulate over all spatial txfm
        for i in range(6):
            cst_jnt_adr = jnt_cst_adr[bodyid]
            dof_adr = cst_txfm_dofadr[cst_jnt_adr, i]
            if dof_adr == -1:  # not attached to a dof
                continue

            xaxis = xaxis_in[worldid, bodyid, i]
            if i < 3:  # rotation
                res[dof_adr] += wp.spatial_vector(xaxis,
                                                  wp.cross(xaxis, offset))
            else:  # translation
                res[dof_adr] += wp.spatial_vector(wp.vec3(0.0), xaxis)


@event_scope
def com_pos(m: Model, d: Data):
    """ Computes subtree center of mass positions. """

    # Initialize to (current body com * mass)
    wp.launch(_subtree_com_init,
              dim=(d.nworld, m.nbody),
              inputs=[m.body_mass, d.xipos],
              outputs=[d.subtree_com])

    # Backward pass to propagate subtree com * mass
    for i in reversed(range(len(m.body_tree))):
        body_tree = m.body_tree[i]
        wp.launch(
            _subtree_com_acc,
            dim=(d.nworld, body_tree.size),
            inputs=[m.body_parentid, d.subtree_com, body_tree],
            outputs=[d.subtree_com],
        )

    # Compute the subtree com
    wp.launch(
        _subtree_div,
        dim=(d.nworld, m.nbody),
        inputs=[m.body_subtreemass, d.subtree_com],
        outputs=[d.subtree_com])

    # Spatial inertia
    wp.launch(
        _cinert,
        dim=(d.nworld, m.nbody),
        inputs=[m.body_rootid, m.body_mass, m.body_inertia, d.xipos, d.ximat,
                d.subtree_com],
        outputs=[d.cinert],
    )
    # Phi: todo
    wp.launch(
        _cdof,
        dim=(d.nworld, m.nbody),
        inputs=[m.body_rootid, m.jnt_type, m.jnt_dofadr, m.jnt_dofnum,
                m.jnt_cst_adr, m.cst_txfm_dofadr, d.xmat, d.xanchor, d.xaxis,
                d.subtree_com],
        outputs=[d.cdof],
    )


@wp.kernel
def _crb_accumulate(
        # Model:
        body_parentid: wp.array(dtype=int),
        # Data in:
        crb_in: wp.array2d(dtype=vec10),
        # In:
        body_tree_: wp.array(dtype=int),
        # Data out:
        crb_out: wp.array2d(dtype=vec10),
):
    worldid, nodeid = wp.tid()
    bodyid = body_tree_[nodeid]
    pid = body_parentid[bodyid]
    if pid == 0:
        return
    wp.atomic_add(crb_out, worldid, pid, crb_in[worldid, bodyid])


@wp.kernel
def _qM_dense(
        # Model:
        dof_bodyid: wp.array(dtype=int),
        dof_parentid: wp.array(dtype=int),
        dof_armature: wp.array(dtype=float),
        # Data in:
        cdof_in: wp.array2d(dtype=wp.spatial_vector),
        crb_in: wp.array2d(dtype=vec10),
        # Data out:
        qM_out: wp.array3d(dtype=float),
):
    worldid, dofid = wp.tid()
    bodyid = dof_bodyid[dofid]
    # init M(i,i) with armature inertia.
    M = dof_armature[dofid]

    # precompute buf = crb_body_i * cdof_i
    buf = math.inert_vec(crb_in[worldid, bodyid], cdof_in[worldid, dofid])
    M += wp.dot(cdof_in[worldid, dofid], buf)

    qM_out[worldid, dofid, dofid] = M

    # sparse backward pass over ancestors
    dofidi = dofid
    dofid = dof_parentid[dofid]
    while dofid >= 0:
        qMij = wp.dot(cdof_in[worldid, dofid], buf)
        qM_out[worldid, dofidi, dofid] += qMij
        qM_out[worldid, dofid, dofidi] += qMij
        dofid = dof_parentid[dofid]


@event_scope
def crb(m: Model, d: Data):
    """Computes composite rigid body inertias for each body and the joint-space inertia matrix.

    Accumulates composite rigid body inertias up the kinematic tree and computes the
    joint-space inertia matrix in dense format, depending on model options.
    """
    wp.copy(d.crb, d.cinert)

    for i in reversed(range(len(m.body_tree))):
        body_tree = m.body_tree[i]
        wp.launch(_crb_accumulate,
                  dim=(d.nworld, body_tree.size),
                  inputs=[m.body_parentid, d.crb, body_tree],
                  outputs=[d.crb])

    d.qM.zero_()
    wp.launch(
        _qM_dense,
        dim=(d.nworld, m.nv),
        inputs=[m.dof_bodyid, m.dof_parentid, m.dof_armature, d.cdof, d.crb],
        outputs=[d.qM]
    )


@cache_kernel
def _tile_cholesky_factorize(tile: TileSet):
    """Returns a kernel for dense Cholesky factorization of a tile."""

    @nested_kernel(module="unique", enable_backward=False)
    def cholesky_factorize(
            # Data In:
            qM_in: wp.array3d(dtype=float),
            # In:
            adr: wp.array(dtype=int),
            # Out:
            L_out: wp.array3d(dtype=float),
    ):
        worldid, nodeid = wp.tid()
        TILE_SIZE = wp.static(tile.size)

        dofid = adr[nodeid]
        M_tile = wp.tile_load(qM_in[worldid], shape=(TILE_SIZE, TILE_SIZE),
                              offset=(dofid, dofid))
        L_tile = wp.tile_cholesky(M_tile)
        wp.tile_store(L_out[worldid], L_tile, offset=(dofid, dofid))

    return cholesky_factorize


def _factor_i_dense(m: Model, d: Data, M: wp.array, L: wp.array):
    """Dense Cholesky factorization of inertia-like matrix M, assumed spd."""
    for tile in m.qM_tiles:
        wp.launch_tiled(
            _tile_cholesky_factorize(tile),
            dim=(d.nworld, tile.adr.size),
            inputs=[M, tile.adr],
            outputs=[L],
            block_dim=m.block_dim.cholesky_factorize,
        )


@event_scope
def factor_m(m: Model, d: Data):
    """Factorization of inertia-like matrix M, assumed spd."""
    _factor_i_dense(m, d, d.qM, d.qLD)


@wp.kernel
def _cacc_world(
        # In:
        gravity: float,
        # Data out:
        cacc_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid = wp.tid()
    cacc_out[worldid, 0] = (
        wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0, 0.0, -gravity)))


def _rne_cacc_world(m: Model, d: Data):
    wp.launch(_cacc_world, dim=[d.nworld], inputs=[m.opt.gravity],
              outputs=[d.cacc])


@wp.kernel
def _cacc(
        # Model:
        body_parentid: wp.array(dtype=int),
        jnt_dofnum: wp.array(dtype=int),
        jnt_dofadr: wp.array(dtype=int),
        # Data in:
        qvel_in: wp.array2d(dtype=float),
        qacc_in: wp.array2d(dtype=float),
        cdof_in: wp.array2d(dtype=wp.spatial_vector),
        cdof_dot_in: wp.array2d(dtype=wp.spatial_vector),
        cacc_in: wp.array2d(dtype=wp.spatial_vector),
        # In:
        body_tree_: wp.array(dtype=int),
        flg_acc: bool,
        # Data out:
        cacc_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, nodeid = wp.tid()

    bodyid = body_tree_[nodeid]
    dofnum = jnt_dofnum[bodyid]
    dofadr = jnt_dofadr[bodyid]

    pid = body_parentid[bodyid]
    local_cacc = cacc_in[worldid, pid]
    for i in range(dofnum):
        local_cacc += cdof_dot_in[worldid, dofadr + i] * qvel_in[
            worldid, dofadr + i]
        if flg_acc:
            local_cacc += cdof_in[worldid, dofadr + i] * qacc_in[
                worldid, dofadr + i]
    cacc_out[worldid, bodyid] = local_cacc


def _rne_cacc_forward(m: Model, d: Data, flg_acc: bool = False):
    for body_tree in m.body_tree:
        wp.launch(
            _cacc,
            dim=(d.nworld, body_tree.size),
            inputs=[m.body_parentid, m.jnt_dofnum, m.jnt_dofadr, d.qvel,
                    d.qacc, d.cdof, d.cdof_dot, d.cacc,
                    body_tree, flg_acc],
            outputs=[d.cacc],
        )


@wp.kernel
def _cfrc(
        # Data in:
        cinert_in: wp.array2d(dtype=vec10),
        cvel_in: wp.array2d(dtype=wp.spatial_vector),
        cacc_in: wp.array2d(dtype=wp.spatial_vector),
        cfrc_ext_in: wp.array2d(dtype=wp.spatial_vector),
        # In:
        flg_cfrc_ext: bool,
        # Data out:
        cfrc_int_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, bodyid = wp.tid()
    bodyid += 1  # skip world body
    cacc = cacc_in[worldid, bodyid]
    cinert = cinert_in[worldid, bodyid]
    cvel = cvel_in[worldid, bodyid]
    frc = math.inert_vec(cinert, cacc)
    frc += math.motion_cross_force(cvel, math.inert_vec(cinert, cvel))
    if flg_cfrc_ext:
        frc -= cfrc_ext_in[worldid, bodyid]

    cfrc_int_out[worldid, bodyid] = frc


def _rne_cfrc(m: Model, d: Data, flg_cfrc_ext: bool = False):
    wp.launch(
        _cfrc, dim=[d.nworld, m.nbody - 1],
        inputs=[d.cinert, d.cvel, d.cacc, d.cfrc_ext, flg_cfrc_ext],
        outputs=[d.cfrc_int]
    )


@wp.kernel
def _cfrc_backward(
        # Model:
        body_parentid: wp.array(dtype=int),
        # Data in:
        cfrc_int_in: wp.array2d(dtype=wp.spatial_vector),
        # In:
        body_tree_: wp.array(dtype=int),
        # Data out:
        cfrc_int_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, nodeid = wp.tid()
    bodyid = body_tree_[nodeid]
    pid = body_parentid[bodyid]
    if bodyid != 0:
        wp.atomic_add(cfrc_int_out[worldid], pid, cfrc_int_in[worldid, bodyid])


def _rne_cfrc_backward(m: Model, d: Data):
    for body_tree in reversed(m.body_tree):
        wp.launch(
            _cfrc_backward, dim=[d.nworld, body_tree.size],
            inputs=[m.body_parentid, d.cfrc_int, body_tree],
            outputs=[d.cfrc_int]
        )


@wp.kernel
def _qfrc_bias(
        # Model:
        dof_bodyid: wp.array(dtype=int),
        # Data in:
        cdof_in: wp.array2d(dtype=wp.spatial_vector),
        cfrc_int_in: wp.array2d(dtype=wp.spatial_vector),
        # Data out:
        qfrc_bias_out: wp.array2d(dtype=float),
):
    worldid, dofid = wp.tid()
    bodyid = dof_bodyid[dofid]
    qfrc_bias_out[worldid, dofid] = wp.dot(cdof_in[worldid, dofid],
                                           cfrc_int_in[worldid, bodyid])


@event_scope
def rne(m: Model, d: Data, flg_acc: bool = False):
    """Computes inverse dynamics using the recursive Newton-Euler algorithm.

    Computes the bias forces (`qfrc_bias`) and internal forces (`cfrc_int`) for the current state,
    including the effects of gravity and optionally joint accelerations.

    Args:
      m: The model containing kinematic and dynamic information.
      d: The data object containing the current state and output arrays.
      flg_acc: If True, includes joint accelerations in the computation.
    """
    _rne_cacc_world(m, d)
    _rne_cacc_forward(m, d, flg_acc=flg_acc)
    _rne_cfrc(m, d)
    _rne_cfrc_backward(m, d)
    wp.launch(_qfrc_bias, dim=[d.nworld, m.nv],
              inputs=[m.dof_bodyid, d.cdof, d.cfrc_int], outputs=[d.qfrc_bias])


@wp.kernel
def _cfrc_ext(
        # Model:
        body_rootid: wp.array(dtype=int),
        # Data in:
        xfrc_applied_in: wp.array2d(dtype=wp.spatial_vector),
        xipos_in: wp.array2d(dtype=wp.vec3),
        subtree_com_in: wp.array2d(dtype=wp.vec3),
        # Data out:
        cfrc_ext_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, bodyid = wp.tid()
    if bodyid == 0:
        cfrc_ext_out[worldid, 0] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0,
                                                     0.0)
    else:
        xfrc_applied = xfrc_applied_in[worldid, bodyid]
        subtree_com = subtree_com_in[worldid, body_rootid[bodyid]]
        xipos = xipos_in[worldid, bodyid]
        cfrc_ext_out[worldid, bodyid] = support.transform_force(
            xfrc_applied, subtree_com - xipos)


@wp.kernel
def _cfrc_ext_equality(
        # Model:
        body_rootid: wp.array(dtype=int),
        site_bodyid: wp.array(dtype=int),
        site_pos: wp.array2d(dtype=wp.vec3),
        eq_obj1id: wp.array(dtype=int),
        eq_obj2id: wp.array(dtype=int),
        eq_objtype: wp.array(dtype=int),
        eq_data: wp.array2d(dtype=vec11),
        # Data in:
        xpos_in: wp.array2d(dtype=wp.vec3),
        xmat_in: wp.array2d(dtype=wp.mat33),
        subtree_com_in: wp.array2d(dtype=wp.vec3),
        efc_id_in: wp.array2d(dtype=int),
        efc_force_in: wp.array2d(dtype=float),
        ne_connect_in: wp.array(dtype=int),
        ne_weld_in: wp.array(dtype=int),
        # Data out:
        cfrc_ext_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, eqid = wp.tid()

    ne_connect = ne_connect_in[worldid]
    ne_weld = ne_weld_in[worldid]
    num_connect = ne_connect // 3

    if eqid >= num_connect + ne_weld // 6:
        return

    is_connect = eqid < num_connect
    if is_connect:
        efcid = 3 * eqid
        cfrc_torque = wp.vec3(0.0, 0.0, 0.0)  # no torque from connect
    else:
        efcid = 6 * eqid - ne_connect
        cfrc_torque = wp.vec3(efc_force_in[worldid, efcid + 3],
                              efc_force_in[worldid, efcid + 4],
                              efc_force_in[worldid, efcid + 5])

    cfrc_force = wp.vec3(
        efc_force_in[worldid, efcid + 0],
        efc_force_in[worldid, efcid + 1],
        efc_force_in[worldid, efcid + 2],
    )

    id = efc_id_in[worldid, efcid]
    eq_data_ = eq_data[worldid, id]
    # body_semantic = eq_objtype[id] == ObjType.BODY
    body_semantic = True  # ?

    obj1 = eq_obj1id[id]
    obj2 = eq_obj2id[id]

    if body_semantic:
        bodyid1 = obj1
        bodyid2 = obj2
    else:
        bodyid1 = site_bodyid[obj1]
        bodyid2 = site_bodyid[obj2]

    # body 1
    if bodyid1:
        if body_semantic:
            if is_connect:
                offset = wp.vec3(eq_data_[0], eq_data_[1], eq_data_[2])
            else:
                offset = wp.vec3(eq_data_[3], eq_data_[4], eq_data_[5])
        else:
            offset = site_pos[worldid, obj1]

        # transform point on body1: local -> global
        pos = xmat_in[worldid, bodyid1] @ offset + xpos_in[worldid, bodyid1]

        # subtree CoM-based torque_force vector
        newpos = subtree_com_in[worldid, body_rootid[bodyid1]]

        dif = newpos - pos
        cfrc_com = wp.spatial_vector(cfrc_torque - wp.cross(dif, cfrc_force),
                                     cfrc_force)

        # apply (opposite for body 1)
        wp.atomic_add(cfrc_ext_out[worldid], bodyid1, cfrc_com)

    # body 2
    if bodyid2:
        if body_semantic:
            if is_connect:
                offset = wp.vec3(eq_data_[3], eq_data_[4], eq_data_[5])
            else:
                offset = wp.vec3(eq_data_[0], eq_data_[1], eq_data_[2])
        else:
            offset = site_pos[worldid, obj2]

        # transform point on body2: local -> global
        pos = xmat_in[worldid, bodyid2] @ offset + xpos_in[worldid, bodyid2]

        # subtree CoM-based torque_force vector
        newpos = subtree_com_in[worldid, body_rootid[bodyid2]]

        dif = newpos - pos
        cfrc_com = wp.spatial_vector(cfrc_torque - wp.cross(dif, cfrc_force),
                                     cfrc_force)

        # apply
        wp.atomic_sub(cfrc_ext_out[worldid], bodyid2, cfrc_com)


@wp.func
def transform_force(force: wp.vec3, torque: wp.vec3,
                    offset: wp.vec3) -> wp.spatial_vector:
    torque -= wp.cross(offset, force)
    return wp.spatial_vector(torque, force)


@wp.kernel
def _cfrc_ext_contact(
        # Model:
        opt_cone: int,
        body_rootid: wp.array(dtype=int),
        geom_bodyid: wp.array(dtype=int),
        # Data in:
        subtree_com_in: wp.array2d(dtype=wp.vec3),
        contact_pos_in: wp.array(dtype=wp.vec3),
        contact_frame_in: wp.array(dtype=wp.mat33),
        contact_friction_in: wp.array(dtype=vec5),
        contact_dim_in: wp.array(dtype=int),
        contact_geom_in: wp.array(dtype=wp.vec2i),
        contact_efc_address_in: wp.array2d(dtype=int),
        contact_worldid_in: wp.array(dtype=int),
        efc_force_in: wp.array2d(dtype=float),
        njmax_in: int,
        nacon_in: wp.array(dtype=int),
        # Data out:
        cfrc_ext_out: wp.array2d(dtype=wp.spatial_vector),
):
    contactid = wp.tid()

    if contactid >= nacon_in[0]:
        return

    geom = contact_geom_in[contactid]
    id1 = geom_bodyid[geom[0]]
    id2 = geom_bodyid[geom[1]]

    if id1 == 0 and id2 == 0:
        return

    worldid = contact_worldid_in[contactid]

    # contact force in world frame
    force = support.contact_force_fn(
        opt_cone,
        contact_frame_in,
        contact_friction_in,
        contact_dim_in,
        contact_efc_address_in,
        efc_force_in,
        njmax_in,
        nacon_in,
        worldid,
        contactid,
        to_world_frame=True,
    )

    pos = contact_pos_in[contactid]

    # contact force on bodies
    if id1:
        com1 = subtree_com_in[worldid, body_rootid[id1]]
        wp.atomic_sub(cfrc_ext_out[worldid], id1,
                      support.transform_force(force, com1 - pos))

    if id2:
        com2 = subtree_com_in[worldid, body_rootid[id2]]
        wp.atomic_add(cfrc_ext_out[worldid], id2,
                      support.transform_force(force, com2 - pos))


@event_scope
def rne_postconstraint(m: Model, d: Data):
    """Computes the recursive Newton-Euler algorithm after constraints are applied.

    Computes `cacc`, `cfrc_ext`, and `cfrc_int`, including the effects of applied forces, equality
    constraints, and contacts.
    """
    # cfrc_ext = perturb
    wp.launch(
        _cfrc_ext,
        dim=(d.nworld, m.nbody),
        inputs=[m.body_rootid, d.xfrc_applied, d.xipos, d.subtree_com],
        outputs=[d.cfrc_ext],
    )

    wp.launch(
        _cfrc_ext_equality,
        dim=(d.nworld, m.neq),
        inputs=[
            m.body_rootid,
            m.site_bodyid,
            m.site_pos,
            m.eq_obj1id,
            m.eq_obj2id,
            m.eq_objtype,
            m.eq_data,
            d.xpos,
            d.xmat,
            d.subtree_com,
            d.efc.id,
            d.efc.force,
            d.ne_connect,
            d.ne_weld,
        ],
        outputs=[d.cfrc_ext],
    )

    # cfrc_ext += contacts
    wp.launch(
        _cfrc_ext_contact,
        dim=(d.naconmax,),
        inputs=[
            m.opt.cone,
            m.body_rootid,
            m.geom_bodyid,
            d.subtree_com,
            d.contact.pos,
            d.contact.frame,
            d.contact.friction,
            d.contact.dim,
            d.contact.geom,
            d.contact.efc_address,
            d.contact.worldid,
            d.efc.force,
            d.njmax,
            d.nacon,
        ],
        outputs=[d.cfrc_ext],
    )

    # forward pass over bodies: compute cacc, cfrc_int
    _rne_cacc_world(m, d)
    _rne_cacc_forward(m, d, flg_acc=True)

    # cfrc_body = cinert * cacc + cvel x (cinert * cvel)
    _rne_cfrc(m, d, flg_cfrc_ext=True)

    # backward pass over bodies: accumulate cfrc_int from children
    _rne_cfrc_backward(m, d)


@wp.kernel
def _comvel_root(cvel_out: wp.array2d(dtype=wp.spatial_vector)):
    worldid, elementid = wp.tid()
    cvel_out[worldid, 0][elementid] = 0.0


@wp.kernel
def _comvel_level(
        # Model:
        body_parentid: wp.array(dtype=int),
        jnt_dofadr: wp.array(dtype=int),
        jnt_type: wp.array(dtype=int),
        # Data in:
        qvel_in: wp.array2d(dtype=float),
        cdof_in: wp.array2d(dtype=wp.spatial_vector),
        cvel_in: wp.array2d(dtype=wp.spatial_vector),
        # In:
        body_tree_: wp.array(dtype=int),
        # Data out:
        cvel_out: wp.array2d(dtype=wp.spatial_vector),
        cdof_dot_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, nodeid = wp.tid()
    bodyid = body_tree_[nodeid]

    # parent velocity
    pid = body_parentid[bodyid]
    cvel = cvel_in[worldid, pid]

    qvel = qvel_in[worldid]
    cdof = cdof_in[worldid]
    dofid = jnt_dofadr[bodyid]
    jnttype = jnt_type[bodyid]
    if jnttype == JointType.FREE:
        cvel += cdof[dofid + 0] * qvel[dofid + 0]
        cvel += cdof[dofid + 1] * qvel[dofid + 1]
        cvel += cdof[dofid + 2] * qvel[dofid + 2]

        cdof_dot_out[worldid, dofid + 3] = math.motion_cross(cvel,
                                                             cdof[dofid + 3])
        cdof_dot_out[worldid, dofid + 4] = math.motion_cross(cvel,
                                                             cdof[dofid + 4])
        cdof_dot_out[worldid, dofid + 5] = math.motion_cross(cvel,
                                                             cdof[dofid + 5])

        cvel += cdof[dofid + 3] * qvel[dofid + 3]
        cvel += cdof[dofid + 4] * qvel[dofid + 4]
        cvel += cdof[dofid + 5] * qvel[dofid + 5]

        dofid += 6
    elif jnttype == JointType.BALL:
        cdof_dot_out[worldid, dofid + 0] = math.motion_cross(cvel,
                                                             cdof[dofid + 0])
        cdof_dot_out[worldid, dofid + 1] = math.motion_cross(cvel,
                                                             cdof[dofid + 1])
        cdof_dot_out[worldid, dofid + 2] = math.motion_cross(cvel,
                                                             cdof[dofid + 2])

        cvel += cdof[dofid + 0] * qvel[dofid + 0]
        cvel += cdof[dofid + 1] * qvel[dofid + 1]
        cvel += cdof[dofid + 2] * qvel[dofid + 2]

        dofid += 3
    else:
        cdof_dot_out[worldid, dofid] = math.motion_cross(cvel, cdof[dofid])
        cvel += cdof[dofid] * qvel[dofid]

        dofid += 1

    cvel_out[worldid, bodyid] = cvel


@event_scope
def com_vel(m: Model, d: Data):
    """Computes the spatial velocities (cvel) and the derivative `cdof_dot` for all bodies.

    Propagates velocities down the kinematic tree, updating the spatial velocity and
    derivative for each body.
    """
    wp.launch(_comvel_root, dim=(d.nworld, 6), inputs=[], outputs=[d.cvel])

    for i in range(1, len(m.body_tree)):
        body_tree = m.body_tree[i]
        wp.launch(
            _comvel_level,
            dim=(d.nworld, body_tree.size),
            inputs=[m.body_parentid, m.jnt_dofadr, m.jnt_type, d.qvel, d.cdof,
                    d.cvel,
                    body_tree],
            outputs=[d.cvel, d.cdof_dot],
        )


@wp.kernel
def _site_velocity(
        # Model:
        site_bodyid: wp.array(dtype=int),
        body_rootid: wp.array(dtype=int),
        # Data in:
        site_xpos_in: wp.array2d(dtype=wp.vec3),
        subtree_com_in: wp.array2d(dtype=wp.vec3),
        cvel_in: wp.array2d(dtype=wp.spatial_vector),
        # Data out:
        site_xvel: wp.array2d(dtype=wp.vec3),
):
    worldid, siteid = wp.tid()
    # Body COM-velocity
    bodyid = site_bodyid[siteid]
    cvel = cvel_in[worldid, bodyid]
    ang = wp.spatial_top(cvel)
    lin = wp.spatial_bottom(cvel)
    # Transform to site
    pos = site_xpos_in[worldid, siteid]
    subtree_com = subtree_com_in[worldid, body_rootid[bodyid]]
    dif = pos - subtree_com

    site_xvel[worldid, siteid] = lin - wp.cross(dif, ang)


@wp.kernel
def _compute_path_velocity(
        # Model:
        muscle_pts_adr: wp.array(dtype=int),
        muscle_pts_num: wp.array(dtype=int),
        # Data in:
        site_diff_vel_in: wp.array2d(dtype=float),
        # Data out:
        muscle_velocity_out: wp.array2d(dtype=float),
):
    worldid, muscle_id = wp.tid()

    n_sites = muscle_pts_num[muscle_id]
    pts_adr = muscle_pts_adr[muscle_id]
    for i in range(n_sites - 1):
        muscle_velocity_out[worldid, muscle_id] += site_diff_vel_in[worldid, pts_adr + i]


@cache_kernel
def _tile_cholesky_solve(tile: TileSet):
    """Returns a kernel for dense Cholesky backsubstitution of a tile."""

    @nested_kernel(module="unique", enable_backward=False)
    def cholesky_solve(
            # In:
            L: wp.array3d(dtype=float),
            y: wp.array2d(dtype=float),
            adr: wp.array(dtype=int),
            # Out:
            x: wp.array2d(dtype=float),
    ):
        worldid, nodeid = wp.tid()
        TILE_SIZE = wp.static(tile.size)

        dofid = adr[nodeid]
        y_slice = wp.tile_load(y[worldid], shape=(TILE_SIZE,), offset=(dofid,))
        L_tile = wp.tile_load(L[worldid], shape=(TILE_SIZE, TILE_SIZE),
                              offset=(dofid, dofid))
        x_slice = wp.tile_cholesky_solve(L_tile, y_slice)
        wp.tile_store(x[worldid], x_slice, offset=(dofid,))

    return cholesky_solve


def _solve_LD_dense(m: Model, d: Data, L: wp.array3d(dtype=float),
                    x: wp.array2d(dtype=float),
                    y: wp.array2d(dtype=float)):
    """Computes dense backsubstitution: x = inv(L'*L)*y."""
    for tile in m.qM_tiles:
        wp.launch_tiled(
            _tile_cholesky_solve(tile),
            dim=(d.nworld, tile.adr.size),
            inputs=[L, y, tile.adr],
            outputs=[x],
            block_dim=m.block_dim.cholesky_solve,
        )


def solve_LD(
        m: Model,
        d: Data,
        L: wp.array3d(dtype=float),
        x: wp.array2d(dtype=float),
        y: wp.array2d(dtype=float),
):
    """Computes backsubstitution to solve a linear system of the form x = inv(L'*D*L) * y.

    L and D are the factors from the Cholesky factorization of the inertia matrix.

    Args:
      m: The model containing factorization and sparsity information.
      d: The data object containing workspace and factorization results.
      L: Lower-triangular factor from the factorization (dense).
      x: Output array for the solution.
      y: Input right-hand side array.
    """
    _solve_LD_dense(m, d, L, x, y)


@event_scope
def solve_m(m: Model, d: Data, x: wp.array2d(dtype=float),
            y: wp.array2d(dtype=float)):
    """Computes backsubstitution: x = qLD * y.

    Args:
      m: The model containing inertia and factorization information.
      d: The data object containing factorization results.
      x: Output array for the solution.
      y: Input right-hand side array.
    """
    solve_LD(m, d, d.qLD, x, y)


@cache_kernel
def _tile_cholesky_factorize_solve(tile: TileSet):
    """Returns a kernel for dense Cholesky factorization and backsubstitution of a tile."""

    @nested_kernel(module="unique", enable_backward=False)
    def cholesky_factorize_solve(
            # In:
            M: wp.array3d(dtype=float),
            y: wp.array2d(dtype=float),
            adr: wp.array(dtype=int),
            # Out:
            x: wp.array2d(dtype=float),
            L: wp.array3d(dtype=float),
    ):
        worldid, nodeid = wp.tid()
        TILE_SIZE = wp.static(tile.size)

        dofid = adr[nodeid]
        M_tile = wp.tile_load(M[worldid], shape=(TILE_SIZE, TILE_SIZE),
                              offset=(dofid, dofid))
        y_slice = wp.tile_load(y[worldid], shape=(TILE_SIZE,), offset=(dofid,))

        L_tile = wp.tile_cholesky(M_tile)
        wp.tile_store(L[worldid], L_tile, offset=(dofid, dofid))
        x_slice = wp.tile_cholesky_solve(L_tile, y_slice)
        wp.tile_store(x[worldid], x_slice, offset=(dofid,))

    return cholesky_factorize_solve


def _factor_solve_i_dense(
        m: Model,
        d: Data,
        M: wp.array3d(dtype=float),
        x: wp.array2d(dtype=float),
        y: wp.array2d(dtype=float),
        L: wp.array3d(dtype=float),
):
    for tile in m.qM_tiles:
        wp.launch_tiled(
            _tile_cholesky_factorize_solve(tile),
            dim=(d.nworld, tile.adr.size),
            inputs=[M, y, tile.adr],
            outputs=[x, L],
            block_dim=m.block_dim.cholesky_factorize_solve,
        )


def factor_solve_i(m, d, M, L, x, y):
    """Factorizes and solves the linear system: x = inv(L'*D*L) * y or x = inv(L'*L) * y.

    M is an inertia-like matrix and L, D are its Cholesky-like factors.

    This function first factorizes the matrix M, then solves the system
    for x given right-hand side y.

    Args:
      m: The model containing factorization and sparsity information.
      d: The data object containing workspace and factorization results.
      M: The inertia-like matrix to factorize.
      L: Output lower-triangular factor from the factorization (dense).
      x: Output array for the solution.
      y: Input right-hand side array.
    """
    _factor_solve_i_dense(m, d, M, x, y, L)


@wp.kernel
def _subtree_vel_forward(
        # Model:
        body_rootid: wp.array(dtype=int),
        body_mass: wp.array2d(dtype=float),
        body_inertia: wp.array2d(dtype=wp.vec3),
        # Data in:
        xipos_in: wp.array2d(dtype=wp.vec3),
        ximat_in: wp.array2d(dtype=wp.mat33),
        subtree_com_in: wp.array2d(dtype=wp.vec3),
        cvel_in: wp.array2d(dtype=wp.spatial_vector),
        # Data out:
        subtree_linvel_out: wp.array2d(dtype=wp.vec3),
        subtree_angmom_out: wp.array2d(dtype=wp.vec3),
        subtree_bodyvel_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, bodyid = wp.tid()
    body_mass_id = worldid % body_mass.shape[0]
    body_inertia_id = worldid % body_inertia.shape[0]

    cvel = cvel_in[worldid, bodyid]
    ang = wp.spatial_top(cvel)
    lin = wp.spatial_bottom(cvel)
    xipos = xipos_in[worldid, bodyid]
    ximat = ximat_in[worldid, bodyid]
    subtree_com_root = subtree_com_in[worldid, body_rootid[bodyid]]

    # update linear velocity
    lin -= wp.cross(xipos - subtree_com_root, ang)

    subtree_linvel_out[worldid, bodyid] = body_mass[body_mass_id, bodyid] * lin
    dv = wp.transpose(ximat) @ ang
    dv[0] *= body_inertia[body_inertia_id, bodyid][0]
    dv[1] *= body_inertia[body_inertia_id, bodyid][1]
    dv[2] *= body_inertia[body_inertia_id, bodyid][2]
    subtree_angmom_out[worldid, bodyid] = ximat @ dv
    subtree_bodyvel_out[worldid, bodyid] = wp.spatial_vector(ang, lin)


@wp.kernel
def _linear_momentum(
        # Model:
        body_parentid: wp.array(dtype=int),
        body_subtreemass: wp.array2d(dtype=float),
        # Data in:
        subtree_linvel_in: wp.array2d(dtype=wp.vec3),
        # In:
        body_tree_: wp.array(dtype=int),
        # Data out:
        subtree_linvel_out: wp.array2d(dtype=wp.vec3),
):
    worldid, nodeid = wp.tid()
    bodyid = body_tree_[nodeid]
    if bodyid:
        pid = body_parentid[bodyid]
        wp.atomic_add(subtree_linvel_out[worldid], pid,
                      subtree_linvel_in[worldid, bodyid])
    subtree_linvel_out[worldid, bodyid] /= wp.max(MJ_MINVAL,
                                                  body_subtreemass[worldid %
                                                                   body_subtreemass.shape[
                                                                       0], bodyid])


@wp.kernel
def _angular_momentum(
        # Model:
        body_parentid: wp.array(dtype=int),
        body_mass: wp.array2d(dtype=float),
        body_subtreemass: wp.array2d(dtype=float),
        # Data in:
        xipos_in: wp.array2d(dtype=wp.vec3),
        subtree_com_in: wp.array2d(dtype=wp.vec3),
        subtree_linvel_in: wp.array2d(dtype=wp.vec3),
        subtree_bodyvel_in: wp.array2d(dtype=wp.spatial_vector),
        # In:
        body_tree_: wp.array(dtype=int),
        # Data out:
        subtree_angmom_out: wp.array2d(dtype=wp.vec3),
):
    worldid, nodeid = wp.tid()
    bodyid = body_tree_[nodeid]

    if bodyid == 0:
        return

    pid = body_parentid[bodyid]

    xipos = xipos_in[worldid, bodyid]
    com = subtree_com_in[worldid, bodyid]
    com_parent = subtree_com_in[worldid, pid]
    vel = subtree_bodyvel_in[worldid, bodyid]
    linvel = subtree_linvel_in[worldid, bodyid]
    linvel_parent = subtree_linvel_in[worldid, pid]  # Data field
    mass = body_mass[worldid % body_mass.shape[0], bodyid]
    subtreemass = body_subtreemass[worldid % body_subtreemass.shape[0], bodyid]

    # momentum wrt body i
    dx = xipos - com
    dv = wp.spatial_bottom(vel) - linvel
    dp = dv * mass
    dL = wp.cross(dx, dp)

    # add to subtree i
    subtree_angmom_out[worldid, bodyid] += dL

    # add to parent
    wp.atomic_add(subtree_angmom_out[worldid], pid,
                  subtree_angmom_out[worldid, bodyid])

    # momentum wrt parent
    dx = com - com_parent
    dv = linvel - linvel_parent
    dv *= subtreemass
    dL = wp.cross(dx, dv)
    wp.atomic_add(subtree_angmom_out[worldid], pid, dL)


def subtree_vel(m: Model, d: Data):
    """Computes subtree linear velocity and angular momentum.

    Computes the linear momentum and angular momentum for each subtree, accumulating
    contributions up the kinematic tree.
    """
    # bodywise quantities
    wp.launch(
        _subtree_vel_forward,
        dim=(d.nworld, m.nbody),
        inputs=[m.body_rootid, m.body_mass, m.body_inertia, d.xipos, d.ximat,
                d.subtree_com, d.cvel],
        outputs=[d.subtree_linvel, d.subtree_angmom, d.subtree_bodyvel],
    )

    # sum body linear momentum recursively up the kinematic tree
    for body_tree in reversed(m.body_tree):
        wp.launch(
            _linear_momentum,
            dim=[d.nworld, body_tree.size],
            inputs=[m.body_parentid, m.body_subtreemass, d.subtree_linvel,
                    body_tree],
            outputs=[d.subtree_linvel],
        )

    for body_tree in reversed(m.body_tree):
        wp.launch(
            _angular_momentum,
            dim=[d.nworld, body_tree.size],
            inputs=[
                m.body_parentid,
                m.body_mass,
                m.body_subtreemass,
                d.xipos,
                d.subtree_com,
                d.subtree_linvel,
                d.subtree_bodyvel,
                body_tree,
            ],
            outputs=[d.subtree_angmom],
        )


@wp.kernel
def _compute_path_length(
        # Model:
        muscle_pts_adr: wp.array(dtype=int),
        muscle_pts_num: wp.array(dtype=int),
        # Data in:
        site_diff_len_in: wp.array2d(dtype=float),
        # Data out:
        muscle_length_out: wp.array2d(dtype=float),
):
    worldid, muscle_id = wp.tid()

    n_sites = muscle_pts_num[muscle_id]
    pts_adr = muscle_pts_adr[muscle_id]

    for i in range(n_sites - 1):
        muscle_length_out[worldid, muscle_id] += site_diff_len_in[worldid, pts_adr + i]


@event_scope
def muscle_path_length(m: Model, d: Data):
    """Computes the muscle path lengths. """
    if not m.nmuscle:
        return
    d.muscle_length.zero_()

    # Compute global site positions
    wp.launch(
        _site_local_to_global,
        dim=(d.nworld, m.nsite),
        inputs=[m.site_bodyid, m.site_pos, d.xpos, d.xquat],
        outputs=[d.site_rpos, d.site_xpos],
    )

    # Vector between consecutive muscle points
    wp.launch(
        _site_consecutive_diff_len,
        dim=(d.nworld, m.nsite - 1),
        inputs=[d.site_xpos, ],
        outputs=[d.site_diff_vec, d.site_diff_len],
    )

    # Compute path length of muscles
    wp.launch(
        _compute_path_length,
        dim=(d.nworld, m.nmuscle),
        inputs=[m.muscle_pts_adr, m.muscle_pts_num, d.site_diff_len, ],
        outputs=[d.muscle_length],
    )


@event_scope
def muscle_path_velocity(m: Model, d: Data):
    """Computes the muscle path velocities. """
    if not m.nmuscle:
        return

    d.muscle_length.zero_()

    wp.launch(
        _site_velocity,
        dim=(d.nworld, m.nsite),
        inputs=[m.site_bodyid, m.body_rootid, d.site_xpos, d.subtree_com,
                d.cvel],
        outputs=[d.site_xvel]
    )

    wp.launch(
        _site_consecutive_diff_vel,
        dim=(d.nworld, m.nsite - 1),
        inputs=[d.site_xvel, d.site_diff_vec, d.site_diff_len],
        outputs=[d.site_diff_vel, ]
    )

    wp.launch(
        _compute_path_velocity,
        dim=(d.nworld, m.nmuscle),
        inputs=[
            m.muscle_pts_adr,
            m.muscle_pts_num,
            d.site_diff_vel,
        ],
        outputs=[d.muscle_velocity],
    )
