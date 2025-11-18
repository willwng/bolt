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
from .types import Data
from .types import JointType
from .types import Model
from .types import TileSet
from .types import CustomFnType
from .types import vec10
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

    wp.launch(
        _site_local_to_global,
        dim=(d.nworld, m.nsite),
        inputs=[m.site_bodyid, m.site_pos, d.xpos, d.xquat],
        outputs=[d.site_rpos, d.site_xpos],
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
