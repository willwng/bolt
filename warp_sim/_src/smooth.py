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
from . import util_misc
from .types import MJ_MINVAL
from .types import Data
from .types import JointType
from .types import Model
from .types import TileSet
from .types import WrapType
from .types import vec5
from .types import vec10
from .types import vec11
from .warp_util import cache_kernel
from .warp_util import event_scope
from .warp_util import kernel as nested_kernel

wp.set_module_options({"enable_backward": False})


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
        qpos0: wp.array2d(dtype=float),
        body_parentid: wp.array(dtype=int),
        body_jntnum: wp.array(dtype=int),
        body_jntadr: wp.array(dtype=int),
        body_pos: wp.array2d(dtype=wp.vec3),
        body_quat: wp.array2d(dtype=wp.quat),
        body_ipos: wp.array2d(dtype=wp.vec3),
        body_iquat: wp.array2d(dtype=wp.quat),
        jnt_type: wp.array(dtype=int),
        jnt_qposadr: wp.array(dtype=int),
        jnt_pos: wp.array2d(dtype=wp.vec3),
        jnt_axis: wp.array2d(dtype=wp.vec3),
        # Data in:
        qpos_in: wp.array2d(dtype=float),
        xpos_in: wp.array2d(dtype=wp.vec3),
        xquat_in: wp.array2d(dtype=wp.quat),
        xmat_in: wp.array2d(dtype=wp.mat33),
        # In:
        body_tree_: wp.array(dtype=int),
        # Data out:
        xpos_out: wp.array2d(dtype=wp.vec3),
        xquat_out: wp.array2d(dtype=wp.quat),
        xmat_out: wp.array2d(dtype=wp.mat33),
        xipos_out: wp.array2d(dtype=wp.vec3),
        ximat_out: wp.array2d(dtype=wp.mat33),
        xanchor_out: wp.array2d(dtype=wp.vec3),
        xaxis_out: wp.array2d(dtype=wp.vec3),
):
    worldid, nodeid = wp.tid()
    bodyid = body_tree_[nodeid]
    jntadr = body_jntadr[bodyid]
    jntnum = body_jntnum[bodyid]
    qpos = qpos_in[worldid]
    body_pos_id = worldid % body_pos.shape[0]
    body_quat_id = worldid % body_quat.shape[0]
    jnt_axis_id = worldid % jnt_axis.shape[0]

    if jntnum == 0:
        # no joints - apply fixed translation and rotation relative to parent
        pid = body_parentid[bodyid]
        xpos = (xmat_in[worldid, pid] * body_pos[body_pos_id, bodyid]) + xpos_in[worldid, pid]
        xquat = math.mul_quat(xquat_in[worldid, pid], body_quat[body_quat_id, bodyid])
    elif jntnum == 1 and jnt_type[jntadr] == JointType.FREE:
        # free joint
        qadr = jnt_qposadr[jntadr]
        xpos = wp.vec3(qpos[qadr], qpos[qadr + 1], qpos[qadr + 2])
        xquat = wp.quat(qpos[qadr + 3], qpos[qadr + 4], qpos[qadr + 5], qpos[qadr + 6])
        xquat = wp.normalize(xquat)
        xanchor_out[worldid, jntadr] = xpos
        xaxis_out[worldid, jntadr] = jnt_axis[jnt_axis_id, jntadr]
    else:
        # regular or no joints
        # apply fixed translation and rotation relative to parent
        qpos0_id = worldid % qpos0.shape[0]
        jnt_pos_id = worldid % jnt_pos.shape[0]
        pid = body_parentid[bodyid]
        xpos = (xmat_in[worldid, pid] * body_pos[body_pos_id, bodyid]) + xpos_in[worldid, pid]
        xquat = math.mul_quat(xquat_in[worldid, pid], body_quat[body_quat_id, bodyid])

        for _ in range(jntnum):
            qadr = jnt_qposadr[jntadr]
            jnt_type_ = jnt_type[jntadr]
            jnt_axis_ = jnt_axis[jnt_axis_id, jntadr]
            xanchor = math.rot_vec_quat(jnt_pos[jnt_pos_id, jntadr], xquat) + xpos
            xaxis = math.rot_vec_quat(jnt_axis_, xquat)

            if jnt_type_ == JointType.BALL:
                qloc = wp.quat(qpos[qadr + 0], qpos[qadr + 1], qpos[qadr + 2], qpos[qadr + 3])
                qloc = wp.normalize(qloc)
                xquat = math.mul_quat(xquat, qloc)
                # correct for off-center rotation
                xpos = xanchor - math.rot_vec_quat(jnt_pos[jnt_pos_id, jntadr], xquat)
            elif jnt_type_ == JointType.SLIDE:
                xpos += xaxis * (qpos[qadr] - qpos0[qpos0_id, qadr])
            elif jnt_type_ == JointType.HINGE:
                qpos0_ = qpos0[qpos0_id, qadr]
                qloc_ = math.axis_angle_to_quat(jnt_axis_, qpos[qadr] - qpos0_)
                xquat = math.mul_quat(xquat, qloc_)
                # correct for off-center rotation
                xpos = xanchor - math.rot_vec_quat(jnt_pos[jnt_pos_id, jntadr], xquat)

            xanchor_out[worldid, jntadr] = xanchor
            xaxis_out[worldid, jntadr] = xaxis
            jntadr += 1

    xpos_out[worldid, bodyid] = xpos
    xquat_out[worldid, bodyid] = wp.normalize(xquat)
    xmat_out[worldid, bodyid] = math.quat_to_mat(xquat)
    xipos_out[worldid, bodyid] = xpos + math.rot_vec_quat(
        body_ipos[worldid % body_ipos.shape[0], bodyid], xquat)
    ximat_out[worldid, bodyid] = math.quat_to_mat(
        math.mul_quat(xquat, body_iquat[worldid % body_iquat.shape[0], bodyid]))


@wp.kernel
def _geom_local_to_global(
        # Model:
        geom_bodyid: wp.array(dtype=int),
        geom_pos: wp.array2d(dtype=wp.vec3),
        geom_quat: wp.array2d(dtype=wp.quat),
        # Data in:
        xpos_in: wp.array2d(dtype=wp.vec3),
        xquat_in: wp.array2d(dtype=wp.quat),
        # Data out:
        geom_xpos_out: wp.array2d(dtype=wp.vec3),
        geom_xmat_out: wp.array2d(dtype=wp.mat33),
):
    worldid, geomid = wp.tid()
    bodyid = geom_bodyid[geomid]

    xpos = xpos_in[worldid, bodyid]
    xquat = xquat_in[worldid, bodyid]
    geom_xpos_out[worldid, geomid] = xpos + math.rot_vec_quat(geom_pos[worldid % geom_pos.shape[0], geomid], xquat)
    geom_xmat_out[worldid, geomid] = math.quat_to_mat(
        math.mul_quat(xquat, geom_quat[worldid % geom_quat.shape[0], geomid]))


@wp.kernel
def _site_local_to_global(
        # Model:
        site_bodyid: wp.array(dtype=int),
        site_pos: wp.array2d(dtype=wp.vec3),
        site_quat: wp.array2d(dtype=wp.quat),
        # Data in:
        xpos_in: wp.array2d(dtype=wp.vec3),
        xquat_in: wp.array2d(dtype=wp.quat),
        # Data out:
        site_xpos_out: wp.array2d(dtype=wp.vec3),
        site_xmat_out: wp.array2d(dtype=wp.mat33),
):
    worldid, siteid = wp.tid()
    bodyid = site_bodyid[siteid]
    xpos = xpos_in[worldid, bodyid]
    xquat = xquat_in[worldid, bodyid]
    site_xpos_out[worldid, siteid] = xpos + math.rot_vec_quat(site_pos[worldid % site_pos.shape[0], siteid], xquat)
    site_xmat_out[worldid, siteid] = math.quat_to_mat(
        math.mul_quat(xquat, site_quat[worldid % site_quat.shape[0], siteid]))


@event_scope
def kinematics(m: Model, d: Data):
    """ Computes forward kinematics for all bodies, sites, geoms. """
    wp.launch(_kinematics_root, dim=(d.nworld), inputs=[], outputs=[d.xpos, d.xquat, d.xmat, d.xipos, d.ximat])

    for i in range(1, len(m.body_tree)):
        body_tree = m.body_tree[i]
        wp.launch(
            _kinematics_level,
            dim=(d.nworld, body_tree.size),
            inputs=[
                m.qpos0,
                m.body_parentid,
                m.body_jntnum,
                m.body_jntadr,
                m.body_pos,
                m.body_quat,
                m.body_ipos,
                m.body_iquat,
                m.jnt_type,
                m.jnt_qposadr,
                m.jnt_pos,
                m.jnt_axis,
                d.qpos,
                d.xpos,
                d.xquat,
                d.xmat,
                body_tree,
            ],
            outputs=[d.xpos, d.xquat, d.xmat, d.xipos, d.ximat, d.xanchor, d.xaxis],
        )

    wp.launch(
        _geom_local_to_global,
        dim=(d.nworld, m.ngeom),
        inputs=[m.geom_bodyid, m.geom_pos, m.geom_quat, d.xpos, d.xquat],
        outputs=[d.geom_xpos, d.geom_xmat],
    )

    wp.launch(
        _site_local_to_global,
        dim=(d.nworld, m.nsite),
        inputs=[m.site_bodyid, m.site_pos, m.site_quat, d.xpos, d.xquat],
        outputs=[d.site_xpos, d.site_xmat],
    )


@wp.kernel
def _subtree_com_init(
        # Model:
        body_mass: wp.array2d(dtype=float),
        # Data in:
        xipos_in: wp.array2d(dtype=wp.vec3),
        # Data out:
        subtree_com_out: wp.array2d(dtype=wp.vec3),
):
    worldid, bodyid = wp.tid()
    subtree_com_out[worldid, bodyid] = xipos_in[worldid, bodyid] * body_mass[worldid % body_mass.shape[0], bodyid]


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
        wp.atomic_add(subtree_com_out, worldid, pid, subtree_com_in[worldid, bodyid])


@wp.kernel
def _subtree_div(
        # Model:
        body_subtreemass: wp.array2d(dtype=float),
        # Data in:
        subtree_com_in: wp.array2d(dtype=wp.vec3),
        # Data out:
        subtree_com_out: wp.array2d(dtype=wp.vec3),
):
    worldid, bodyid = wp.tid()
    com = subtree_com_in[worldid, bodyid]
    mass = body_subtreemass[worldid % body_subtreemass.shape[0], bodyid]
    if mass != 0.0:
        subtree_com_out[worldid, bodyid] = com / mass


@wp.kernel
def _cinert(
        # Model:
        body_rootid: wp.array(dtype=int),
        body_mass: wp.array2d(dtype=float),
        body_inertia: wp.array2d(dtype=wp.vec3),
        # Data in:
        xipos_in: wp.array2d(dtype=wp.vec3),
        ximat_in: wp.array2d(dtype=wp.mat33),
        subtree_com_in: wp.array2d(dtype=wp.vec3),
        # Data out:
        cinert_out: wp.array2d(dtype=vec10),
):
    worldid, bodyid = wp.tid()
    mat = ximat_in[worldid, bodyid]
    inert = body_inertia[worldid % body_inertia.shape[0], bodyid]
    mass = body_mass[worldid % body_mass.shape[0], bodyid]
    dif = xipos_in[worldid, bodyid] - subtree_com_in[worldid, body_rootid[bodyid]]
    # express inertia in com-based frame (mju_inertCom)

    res = vec10()
    # res_rot = mat * diag(inert) * mat'
    tmp = mat @ wp.diag(inert) @ wp.transpose(mat)
    res[0] = tmp[0, 0]
    res[1] = tmp[1, 1]
    res[2] = tmp[2, 2]
    res[3] = tmp[0, 1]
    res[4] = tmp[0, 2]
    res[5] = tmp[1, 2]
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
        jnt_bodyid: wp.array(dtype=int),
        # Data in:
        xmat_in: wp.array2d(dtype=wp.mat33),
        xanchor_in: wp.array2d(dtype=wp.vec3),
        xaxis_in: wp.array2d(dtype=wp.vec3),
        subtree_com_in: wp.array2d(dtype=wp.vec3),
        # Data out:
        cdof_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, jntid = wp.tid()
    bodyid = jnt_bodyid[jntid]
    dofid = jnt_dofadr[jntid]
    jnt_type_ = jnt_type[jntid]
    xaxis = xaxis_in[worldid, jntid]
    xmat = wp.transpose(xmat_in[worldid, bodyid])

    # compute com-anchor vector
    offset = subtree_com_in[worldid, body_rootid[bodyid]] - xanchor_in[worldid, jntid]

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
        res[dofid] = wp.spatial_vector(wp.vec3(0.0), xaxis)
    elif jnt_type_ == JointType.HINGE:  # hinge
        res[dofid] = wp.spatial_vector(xaxis, wp.cross(xaxis, offset))


@event_scope
def com_pos(m: Model, d: Data):
    """
    Computes subtree center of mass positions.

    Transforms inertia and motion to global frame centered at subtree CoM. Accumulates the
    mass-weighted positions up the kinematic tree, divides by total mass, and computes composite
    inertias and motion degrees of freedom in the subtree CoM frame.
    """
    wp.launch(_subtree_com_init, dim=(d.nworld, m.nbody), inputs=[m.body_mass, d.xipos], outputs=[d.subtree_com])

    for i in reversed(range(len(m.body_tree))):
        body_tree = m.body_tree[i]
        wp.launch(
            _subtree_com_acc,
            dim=(d.nworld, body_tree.size),
            inputs=[m.body_parentid, d.subtree_com, body_tree],
            outputs=[d.subtree_com],
        )

    wp.launch(_subtree_div, dim=(d.nworld, m.nbody), inputs=[m.body_subtreemass, d.subtree_com],
              outputs=[d.subtree_com])
    wp.launch(
        _cinert,
        dim=(d.nworld, m.nbody),
        inputs=[m.body_rootid, m.body_mass, m.body_inertia, d.xipos, d.ximat, d.subtree_com],
        outputs=[d.cinert],
    )
    wp.launch(
        _cdof,
        dim=(d.nworld, m.njnt),
        inputs=[m.body_rootid, m.jnt_type, m.jnt_dofadr, m.jnt_bodyid, d.xmat, d.xanchor, d.xaxis, d.subtree_com],
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
def _qM_sparse(
        # Model:
        dof_bodyid: wp.array(dtype=int),
        dof_parentid: wp.array(dtype=int),
        dof_Madr: wp.array(dtype=int),
        dof_armature: wp.array2d(dtype=float),
        # Data in:
        cdof_in: wp.array2d(dtype=wp.spatial_vector),
        crb_in: wp.array2d(dtype=vec10),
        # Data out:
        qM_out: wp.array3d(dtype=float),
):
    worldid, dofid = wp.tid()
    madr_ij = dof_Madr[dofid]  # dof_Madr is not batched
    bodyid = dof_bodyid[dofid]

    # init M(i,i) with armature inertia
    qM_out[worldid, 0, madr_ij] = dof_armature[worldid, dofid]

    # precompute buf = crb_body_i * cdof_i
    buf = math.inert_vec(crb_in[worldid, bodyid], cdof_in[worldid, dofid])

    # sparse backward pass over ancestors
    while dofid >= 0:
        qM_out[worldid, 0, madr_ij] += wp.dot(cdof_in[worldid, dofid], buf)
        madr_ij += 1
        dofid = dof_parentid[dofid]


@wp.kernel
def _qM_dense(
        # Model:
        dof_bodyid: wp.array(dtype=int),
        dof_parentid: wp.array(dtype=int),
        dof_armature: wp.array2d(dtype=float),
        # Data in:
        cdof_in: wp.array2d(dtype=wp.spatial_vector),
        crb_in: wp.array2d(dtype=vec10),
        # Data out:
        qM_out: wp.array3d(dtype=float),
):
    worldid, dofid = wp.tid()
    bodyid = dof_bodyid[dofid]
    # init M(i,i) with armature inertia.
    M = dof_armature[worldid % dof_armature.shape[0], dofid]

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
    joint-space inertia matrix in either sparse or dense format, depending on model options.
    """
    wp.copy(d.crb, d.cinert)

    for i in reversed(range(len(m.body_tree))):
        body_tree = m.body_tree[i]
        wp.launch(_crb_accumulate, dim=(d.nworld, body_tree.size), inputs=[m.body_parentid, d.crb, body_tree],
                  outputs=[d.crb])

    d.qM.zero_()
    if m.opt.is_sparse:
        wp.launch(
            _qM_sparse,
            dim=(d.nworld, m.nv),
            inputs=[m.dof_bodyid, m.dof_parentid, m.dof_Madr, m.dof_armature, d.cdof, d.crb],
            outputs=[d.qM],
        )
    else:
        wp.launch(
            _qM_dense, dim=(d.nworld, m.nv), inputs=[m.dof_bodyid, m.dof_parentid, m.dof_armature, d.cdof, d.crb],
            outputs=[d.qM]
        )


@wp.kernel
def _copy_CSR(
        # Model:
        mapM2M: wp.array(dtype=int),
        # In:
        M_in: wp.array3d(dtype=float),
        # Out:
        L_out: wp.array3d(dtype=float),
):
    worldid, ind = wp.tid()
    L_out[worldid, 0, ind] = M_in[worldid, 0, mapM2M[ind]]


@wp.kernel
def _qLD_acc(
        # Model:
        M_rownnz: wp.array(dtype=int),
        M_rowadr: wp.array(dtype=int),
        # In:
        qLD_updates_: wp.array(dtype=wp.vec3i),
        L_in: wp.array3d(dtype=float),
        # Out:
        L_out: wp.array3d(dtype=float),
):
    worldid, nodeid = wp.tid()
    update = qLD_updates_[nodeid]
    i, k, Madr_ki = update[0], update[1], update[2]
    Madr_i = M_rowadr[i]  # Address of row being updated
    diag_k = M_rowadr[k] + M_rownnz[k] - 1  # Address of diagonal element of k
    # tmp = M(k,i) / M(k,k)
    tmp = L_out[worldid, 0, Madr_ki] / L_out[worldid, 0, diag_k]
    for j in range(M_rownnz[i]):
        # M(i,j) -= M(k,j) * tmp
        wp.atomic_sub(L_out[worldid, 0], Madr_i + j, L_in[worldid, 0, M_rowadr[k] + j] * tmp)
    # M(k,i) = tmp
    L_out[worldid, 0, Madr_ki] = tmp


@wp.kernel
def _qLDiag_div(
        # Model:
        M_rownnz: wp.array(dtype=int),
        M_rowadr: wp.array(dtype=int),
        # In:
        L_in: wp.array3d(dtype=float),
        # Out:
        D_out: wp.array2d(dtype=float),
):
    worldid, dofid = wp.tid()
    diag_i = M_rowadr[dofid] + M_rownnz[dofid] - 1  # Address of diagonal element of i
    D_out[worldid, dofid] = 1.0 / L_in[worldid, 0, diag_i]


def _factor_i_sparse(m: Model, d: Data, M: wp.array3d(dtype=float), L: wp.array3d(dtype=float),
                     D: wp.array2d(dtype=float)):
    """Sparse L'*D*L factorization of inertia-like matrix M, assumed spd."""
    wp.launch(_copy_CSR, dim=(d.nworld, m.nC), inputs=[m.mapM2M, M], outputs=[L])

    for i in reversed(range(len(m.qLD_updates))):
        qLD_updates = m.qLD_updates[i]
        wp.launch(_qLD_acc, dim=(d.nworld, qLD_updates.size), inputs=[m.M_rownnz, m.M_rowadr, qLD_updates, L],
                  outputs=[L])

    wp.launch(_qLDiag_div, dim=(d.nworld, m.nv), inputs=[m.M_rownnz, m.M_rowadr, L], outputs=[D])


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
        M_tile = wp.tile_load(qM_in[worldid], shape=(TILE_SIZE, TILE_SIZE), offset=(dofid, dofid))
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
    if m.opt.is_sparse:
        _factor_i_sparse(m, d, d.qM, d.qLD, d.qLDiagInv)
    else:
        _factor_i_dense(m, d, d.qM, d.qLD)


@wp.kernel
def _cacc_world(
        # In:
        gravity: wp.array(dtype=wp.vec3),
        # Data out:
        cacc_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid = wp.tid()
    cacc_out[worldid, 0] = wp.spatial_vector(wp.vec3(0.0), -gravity[worldid % gravity.shape[0]])


def _rne_cacc_world(m: Model, d: Data):
    wp.launch(_cacc_world, dim=[d.nworld], inputs=[m.opt.gravity], outputs=[d.cacc])


@wp.kernel
def _cacc(
        # Model:
        body_parentid: wp.array(dtype=int),
        body_dofnum: wp.array(dtype=int),
        body_dofadr: wp.array(dtype=int),
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
    dofnum = body_dofnum[bodyid]
    pid = body_parentid[bodyid]
    dofadr = body_dofadr[bodyid]
    local_cacc = cacc_in[worldid, pid]
    for i in range(dofnum):
        local_cacc += cdof_dot_in[worldid, dofadr + i] * qvel_in[worldid, dofadr + i]
        if flg_acc:
            local_cacc += cdof_in[worldid, dofadr + i] * qacc_in[worldid, dofadr + i]
    cacc_out[worldid, bodyid] = local_cacc


def _rne_cacc_forward(m: Model, d: Data, flg_acc: bool = False):
    for body_tree in m.body_tree:
        wp.launch(
            _cacc,
            dim=(d.nworld, body_tree.size),
            inputs=[m.body_parentid, m.body_dofnum, m.body_dofadr, d.qvel, d.qacc, d.cdof, d.cdof_dot, d.cacc,
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
        _cfrc, dim=[d.nworld, m.nbody - 1], inputs=[d.cinert, d.cvel, d.cacc, d.cfrc_ext, flg_cfrc_ext],
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
            _cfrc_backward, dim=[d.nworld, body_tree.size], inputs=[m.body_parentid, d.cfrc_int, body_tree],
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
    qfrc_bias_out[worldid, dofid] = wp.dot(cdof_in[worldid, dofid], cfrc_int_in[worldid, bodyid])


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
    wp.launch(_qfrc_bias, dim=[d.nworld, m.nv], inputs=[m.dof_bodyid, d.cdof, d.cfrc_int], outputs=[d.qfrc_bias])


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
        cfrc_ext_out[worldid, 0] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        xfrc_applied = xfrc_applied_in[worldid, bodyid]
        subtree_com = subtree_com_in[worldid, body_rootid[bodyid]]
        xipos = xipos_in[worldid, bodyid]
        cfrc_ext_out[worldid, bodyid] = support.transform_force(xfrc_applied, subtree_com - xipos)


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
        cfrc_torque = wp.vec3(efc_force_in[worldid, efcid + 3], efc_force_in[worldid, efcid + 4],
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
        cfrc_com = wp.spatial_vector(cfrc_torque - wp.cross(dif, cfrc_force), cfrc_force)

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
        cfrc_com = wp.spatial_vector(cfrc_torque - wp.cross(dif, cfrc_force), cfrc_force)

        # apply
        wp.atomic_sub(cfrc_ext_out[worldid], bodyid2, cfrc_com)


@wp.func
def transform_force(force: wp.vec3, torque: wp.vec3, offset: wp.vec3) -> wp.spatial_vector:
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
        wp.atomic_sub(cfrc_ext_out[worldid], id1, support.transform_force(force, com1 - pos))

    if id2:
        com2 = subtree_com_in[worldid, body_rootid[id2]]
        wp.atomic_add(cfrc_ext_out[worldid], id2, support.transform_force(force, com2 - pos))


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
        body_jntnum: wp.array(dtype=int),
        body_jntadr: wp.array(dtype=int),
        body_dofadr: wp.array(dtype=int),
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
    dofid = body_dofadr[bodyid]
    jntid = body_jntadr[bodyid]
    jntnum = body_jntnum[bodyid]
    pid = body_parentid[bodyid]

    if jntnum == 0:
        cvel_out[worldid, bodyid] = cvel_in[worldid, pid]
        return

    cvel = cvel_in[worldid, pid]
    qvel = qvel_in[worldid]
    cdof = cdof_in[worldid]

    for j in range(jntid, jntid + jntnum):
        jnttype = jnt_type[j]

        if jnttype == JointType.FREE:
            cvel += cdof[dofid + 0] * qvel[dofid + 0]
            cvel += cdof[dofid + 1] * qvel[dofid + 1]
            cvel += cdof[dofid + 2] * qvel[dofid + 2]

            cdof_dot_out[worldid, dofid + 3] = math.motion_cross(cvel, cdof[dofid + 3])
            cdof_dot_out[worldid, dofid + 4] = math.motion_cross(cvel, cdof[dofid + 4])
            cdof_dot_out[worldid, dofid + 5] = math.motion_cross(cvel, cdof[dofid + 5])

            cvel += cdof[dofid + 3] * qvel[dofid + 3]
            cvel += cdof[dofid + 4] * qvel[dofid + 4]
            cvel += cdof[dofid + 5] * qvel[dofid + 5]

            dofid += 6
        elif jnttype == JointType.BALL:
            cdof_dot_out[worldid, dofid + 0] = math.motion_cross(cvel, cdof[dofid + 0])
            cdof_dot_out[worldid, dofid + 1] = math.motion_cross(cvel, cdof[dofid + 1])
            cdof_dot_out[worldid, dofid + 2] = math.motion_cross(cvel, cdof[dofid + 2])

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

    for body_tree in m.body_tree:
        wp.launch(
            _comvel_level,
            dim=(d.nworld, body_tree.size),
            inputs=[m.body_parentid, m.body_jntnum, m.body_jntadr, m.body_dofadr, m.jnt_type, d.qvel, d.cdof, d.cvel,
                    body_tree],
            outputs=[d.cvel, d.cdof_dot],
        )


@wp.kernel
def _solve_LD_sparse_x_acc_up(
        # In:
        L: wp.array3d(dtype=float),
        qLD_updates_: wp.array(dtype=wp.vec3i),
        # Out:
        x: wp.array2d(dtype=float),
):
    worldid, nodeid = wp.tid()
    update = qLD_updates_[nodeid]
    i, k, Madr_ki = update[0], update[1], update[2]
    wp.atomic_sub(x[worldid], i, L[worldid, 0, Madr_ki] * x[worldid, k])


@wp.kernel
def _solve_LD_sparse_qLDiag_mul(
        # In:
        D: wp.array2d(dtype=float),
        # Out:
        out: wp.array2d(dtype=float),
):
    worldid, dofid = wp.tid()
    out[worldid, dofid] *= D[worldid, dofid]


@wp.kernel
def _solve_LD_sparse_x_acc_down(
        # In:
        L: wp.array3d(dtype=float),
        qLD_updates_: wp.array(dtype=wp.vec3i),
        # Out:
        x: wp.array2d(dtype=float),
):
    worldid, nodeid = wp.tid()
    update = qLD_updates_[nodeid]
    i, k, Madr_ki = update[0], update[1], update[2]
    wp.atomic_sub(x[worldid], k, L[worldid, 0, Madr_ki] * x[worldid, i])


def _solve_LD_sparse(
        m: Model,
        d: Data,
        L: wp.array3d(dtype=float),
        D: wp.array2d(dtype=float),
        x: wp.array2d(dtype=float),
        y: wp.array2d(dtype=float),
):
    """Computes sparse backsubstitution: x = inv(L'*D*L)*y."""
    wp.copy(x, y)
    for qLD_updates in reversed(m.qLD_updates):
        wp.launch(_solve_LD_sparse_x_acc_up, dim=(d.nworld, qLD_updates.size), inputs=[L, qLD_updates], outputs=[x])

    wp.launch(_solve_LD_sparse_qLDiag_mul, dim=(d.nworld, m.nv), inputs=[D], outputs=[x])

    for qLD_updates in m.qLD_updates:
        wp.launch(_solve_LD_sparse_x_acc_down, dim=(d.nworld, qLD_updates.size), inputs=[L, qLD_updates], outputs=[x])


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
        L_tile = wp.tile_load(L[worldid], shape=(TILE_SIZE, TILE_SIZE), offset=(dofid, dofid))
        x_slice = wp.tile_cholesky_solve(L_tile, y_slice)
        wp.tile_store(x[worldid], x_slice, offset=(dofid,))

    return cholesky_solve


def _solve_LD_dense(m: Model, d: Data, L: wp.array3d(dtype=float), x: wp.array2d(dtype=float),
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
        D: wp.array2d(dtype=float),
        x: wp.array2d(dtype=float),
        y: wp.array2d(dtype=float),
):
    """Computes backsubstitution to solve a linear system of the form x = inv(L'*D*L) * y.

    L and D are the factors from the Cholesky factorization of the inertia matrix.

    This function dispatches to either a sparse or dense solver depending on Model options.

    Args:
      m: The model containing factorization and sparsity information.
      d: The data object containing workspace and factorization results.
      L: Lower-triangular factor from the factorization (sparse or dense).
      D: Diagonal factor from the factorization (only used for sparse).
      x: Output array for the solution.
      y: Input right-hand side array.
    """
    if m.opt.is_sparse:
        _solve_LD_sparse(m, d, L, D, x, y)
    else:
        _solve_LD_dense(m, d, L, x, y)


@event_scope
def solve_m(m: Model, d: Data, x: wp.array2d(dtype=float), y: wp.array2d(dtype=float)):
    """Computes backsubstitution: x = qLD * y.

    Args:
      m: The model containing inertia and factorization information.
      d: The data object containing factorization results.
      x: Output array for the solution.
      y: Input right-hand side array.
    """
    solve_LD(m, d, d.qLD, d.qLDiagInv, x, y)


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
        M_tile = wp.tile_load(M[worldid], shape=(TILE_SIZE, TILE_SIZE), offset=(dofid, dofid))
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


def factor_solve_i(m, d, M, L, D, x, y):
    """Factorizes and solves the linear system: x = inv(L'*D*L) * y or x = inv(L'*L) * y.

    M is an inertia-like matrix and L, D are its Cholesky-like factors.

    This function first factorizes the matrix M (sparse or dense depending on model options),
    then solves the system for x given right-hand side y.

    Args:
      m: The model containing factorization and sparsity information.
      d: The data object containing workspace and factorization results.
      M: The inertia-like matrix to factorize.
      L: Output lower-triangular factor from the factorization (sparse or dense).
      D: Output diagonal factor from the factorization (only used for sparse).
      x: Output array for the solution.
      y: Input right-hand side array.
    """
    if m.opt.is_sparse:
        _factor_i_sparse(m, d, M, L, D)
        _solve_LD_sparse(m, d, L, D, x, y)
    else:
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
        wp.atomic_add(subtree_linvel_out[worldid], pid, subtree_linvel_in[worldid, bodyid])
    subtree_linvel_out[worldid, bodyid] /= wp.max(MJ_MINVAL,
                                                  body_subtreemass[worldid % body_subtreemass.shape[0], bodyid])


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
    wp.atomic_add(subtree_angmom_out[worldid], pid, subtree_angmom_out[worldid, bodyid])

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
        inputs=[m.body_rootid, m.body_mass, m.body_inertia, d.xipos, d.ximat, d.subtree_com, d.cvel],
        outputs=[d.subtree_linvel, d.subtree_angmom, d.subtree_bodyvel],
    )

    # sum body linear momentum recursively up the kinematic tree
    for body_tree in reversed(m.body_tree):
        wp.launch(
            _linear_momentum,
            dim=[d.nworld, body_tree.size],
            inputs=[m.body_parentid, m.body_subtreemass, d.subtree_linvel, body_tree],
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
def _spatial_site_tendon(
        # Model:
        nv: int,
        body_parentid: wp.array(dtype=int),
        body_rootid: wp.array(dtype=int),
        dof_bodyid: wp.array(dtype=int),
        site_bodyid: wp.array(dtype=int),
        wrap_objid: wp.array(dtype=int),
        tendon_site_pair_adr: wp.array(dtype=int),
        wrap_site_pair_adr: wp.array(dtype=int),
        wrap_pulley_scale: wp.array(dtype=float),
        # Data in:
        site_xpos_in: wp.array2d(dtype=wp.vec3),
        subtree_com_in: wp.array2d(dtype=wp.vec3),
        cdof_in: wp.array2d(dtype=wp.spatial_vector),
        # Data out:
        ten_J_out: wp.array3d(dtype=float),
        ten_length_out: wp.array2d(dtype=float),
):
    worldid, elementid = wp.tid()

    # site pairs
    site_pair_adr = wrap_site_pair_adr[elementid]
    ten_adr = tendon_site_pair_adr[elementid]

    # pulley scaling
    pulley_scale = wrap_pulley_scale[site_pair_adr]

    id0 = wrap_objid[site_pair_adr + 0]
    id1 = wrap_objid[site_pair_adr + 1]

    pnt0 = site_xpos_in[worldid, id0]
    pnt1 = site_xpos_in[worldid, id1]
    dif = pnt1 - pnt0
    vec, length = math.normalize_with_norm(dif)
    wp.atomic_add(ten_length_out[worldid], ten_adr, length * pulley_scale)

    if length < MJ_MINVAL:
        vec = wp.vec3(1.0, 0.0, 0.0)

    body0 = site_bodyid[id0]
    body1 = site_bodyid[id1]
    if body0 != body1:
        # TODO(team): parallelize
        for i in range(nv):
            jacp1, _ = support.jac(body_parentid, body_rootid, dof_bodyid, subtree_com_in, cdof_in, pnt0, body0, i,
                                   worldid)
            jacp2, _ = support.jac(body_parentid, body_rootid, dof_bodyid, subtree_com_in, cdof_in, pnt1, body1, i,
                                   worldid)

            J = wp.dot(jacp2 - jacp1, vec)
            if J:
                wp.atomic_add(ten_J_out[worldid, ten_adr], i, J * pulley_scale)


def tendon(m: Model, d: Data):
    """Computes tendon lengths and moments.

    Updates the tendon length and moment arrays for all tendons in the model, including joint,
    site, and geom tendons.
    """
    if not m.nmuscle:
        return

    d.ten_length.zero_()
    d.ten_J.zero_()

    spatial_site = m.wrap_site_pair_adr.size > 0
    spatial_geom = m.wrap_geom_adr.size > 0

    if spatial_site or spatial_geom:
        d.wrap_xpos.zero_()
        d.wrap_obj.zero_()

    # process spatial site tendons
    wp.launch(
        _spatial_site_tendon,
        dim=(d.nworld, m.wrap_site_pair_adr.size),
        inputs=[
            m.nv,
            m.body_parentid,
            m.body_rootid,
            m.dof_bodyid,
            m.site_bodyid,
            m.wrap_objid,
            m.tendon_site_pair_adr,
            m.wrap_site_pair_adr,
            m.wrap_pulley_scale,
            d.site_xpos,
            d.subtree_com,
            d.cdof,
        ],
        outputs=[d.ten_J, d.ten_length],
    )
