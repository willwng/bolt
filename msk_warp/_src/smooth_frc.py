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
from .types import Data
from .types import JointType
from .types import Model
from .types import vec10
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _cacc_world(
        # In:
        gravity: float,
        # Data out:
        cacc_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid = wp.tid()
    cacc_out[worldid, 0] = (
        wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0, -gravity, 0.0)))


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
        cdof_dot_in: wp.array2d(dtype=wp.spatial_vector),
        cacc_in: wp.array2d(dtype=wp.spatial_vector),
        # In:
        body_tree_: wp.array(dtype=int),
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
    cacc_out[worldid, bodyid] = local_cacc


def _rne_cacc_forward(m: Model, d: Data):
    for body_tree in m.body_tree:
        wp.launch(
            _cacc,
            dim=(d.nworld, body_tree.size),
            inputs=[m.body_parentid, m.jnt_dofnum, m.jnt_dofadr, d.qvel,
                    d.cdof_dot, d.cacc, body_tree],
            outputs=[d.cacc],
        )


@wp.kernel
def _cfrc(
        # Data in:
        cinert_in: wp.array2d(dtype=vec10),
        cvel_in: wp.array2d(dtype=wp.spatial_vector),
        cacc_in: wp.array2d(dtype=wp.spatial_vector),
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

    cfrc_int_out[worldid, bodyid] = frc


def _rne_cfrc(m: Model, d: Data):
    wp.launch(
        _cfrc, dim=[d.nworld, m.nbody - 1],
        inputs=[d.cinert, d.cvel, d.cacc],
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
def rne(m: Model, d: Data):
    """Computes inverse dynamics using the recursive Newton-Euler algorithm.

    Computes the bias forces (`qfrc_bias`) and internal forces (`cfrc_int`)
    for the current state, including the effects of gravity.

    Args:
      m: The model containing kinematic and dynamic information.
      d: The data object containing the current state and output arrays.
    """
    _rne_cacc_world(m, d)
    _rne_cacc_forward(m, d)
    _rne_cfrc(m, d)
    _rne_cfrc_backward(m, d)
    wp.launch(_qfrc_bias, dim=[d.nworld, m.nv],
              inputs=[m.dof_bodyid, d.cdof, d.cfrc_int], outputs=[d.qfrc_bias])


@wp.kernel
def _spring_damper_dof_passive(
        # Model:
        qpos_spring: wp.array(dtype=float),
        jnt_type: wp.array(dtype=int),
        jnt_qposadr: wp.array(dtype=int),
        jnt_dofadr: wp.array(dtype=int),
        jnt_stiffness: wp.array(dtype=float),
        dof_damping: wp.array(dtype=float),
        # Data in:
        qpos_in: wp.array2d(dtype=float),
        qvel_in: wp.array2d(dtype=float),
        # Data out:
        qfrc_spring_out: wp.array2d(dtype=float),
        qfrc_damper_out: wp.array2d(dtype=float),
):
    worldid, jntid = wp.tid()
    dofid = jnt_dofadr[jntid]
    stiffness = jnt_stiffness[jntid]
    damping = dof_damping[dofid]

    has_stiffness = stiffness != 0.0
    has_damping = damping != 0.0

    if not has_stiffness:
        qfrc_spring_out[worldid, dofid] = 0.0

    if not has_damping:
        qfrc_damper_out[worldid, dofid] = 0.0

    if not (has_stiffness or has_damping):
        return

    jnttype = jnt_type[jntid]
    qposid = jnt_qposadr[jntid]

    if jnttype == JointType.FREE:
        # spring
        if has_stiffness:
            dif = wp.vec3(
                qpos_in[worldid, qposid + 0] - qpos_spring[qposid + 0],
                qpos_in[worldid, qposid + 1] - qpos_spring[qposid + 1],
                qpos_in[worldid, qposid + 2] - qpos_spring[qposid + 2],
            )
            qfrc_spring_out[worldid, dofid + 0] = -stiffness * dif[0]
            qfrc_spring_out[worldid, dofid + 1] = -stiffness * dif[1]
            qfrc_spring_out[worldid, dofid + 2] = -stiffness * dif[2]
            rot = wp.quat(
                qpos_in[worldid, qposid + 3],
                qpos_in[worldid, qposid + 4],
                qpos_in[worldid, qposid + 5],
                qpos_in[worldid, qposid + 6],
            )
            rot = wp.normalize(rot)
            ref = wp.quat(
                qpos_spring[qposid + 3],
                qpos_spring[qposid + 4],
                qpos_spring[qposid + 5],
                qpos_spring[qposid + 6],
            )
            dif = math.quat_sub(rot, ref)
            qfrc_spring_out[worldid, dofid + 3] = -stiffness * dif[0]
            qfrc_spring_out[worldid, dofid + 4] = -stiffness * dif[1]
            qfrc_spring_out[worldid, dofid + 5] = -stiffness * dif[2]

        # damper
        if has_damping:
            qfrc_damper_out[worldid, dofid + 0] = -damping * qvel_in[
                worldid, dofid + 0]
            qfrc_damper_out[worldid, dofid + 1] = -damping * qvel_in[
                worldid, dofid + 1]
            qfrc_damper_out[worldid, dofid + 2] = -damping * qvel_in[
                worldid, dofid + 2]
            qfrc_damper_out[worldid, dofid + 3] = -damping * qvel_in[
                worldid, dofid + 3]
            qfrc_damper_out[worldid, dofid + 4] = -damping * qvel_in[
                worldid, dofid + 4]
            qfrc_damper_out[worldid, dofid + 5] = -damping * qvel_in[
                worldid, dofid + 5]
    elif jnttype == JointType.BALL:
        # spring
        if has_stiffness:
            rot = wp.quat(
                qpos_in[worldid, qposid + 0],
                qpos_in[worldid, qposid + 1],
                qpos_in[worldid, qposid + 2],
                qpos_in[worldid, qposid + 3],
            )
            rot = wp.normalize(rot)
            ref = wp.quat(
                qpos_spring[qposid + 0],
                qpos_spring[qposid + 1],
                qpos_spring[qposid + 2],
                qpos_spring[qposid + 3],
            )
            dif = math.quat_sub(rot, ref)
            qfrc_spring_out[worldid, dofid + 0] = -stiffness * dif[0]
            qfrc_spring_out[worldid, dofid + 1] = -stiffness * dif[1]
            qfrc_spring_out[worldid, dofid + 2] = -stiffness * dif[2]

        # damper
        if has_damping:
            qfrc_damper_out[worldid, dofid + 0] = -damping * qvel_in[
                worldid, dofid + 0]
            qfrc_damper_out[worldid, dofid + 1] = -damping * qvel_in[
                worldid, dofid + 1]
            qfrc_damper_out[worldid, dofid + 2] = -damping * qvel_in[
                worldid, dofid + 2]
    else:  # mjJNT_SLIDE, mjJNT_HINGE
        # spring
        if has_stiffness:
            fdif = qpos_in[worldid, qposid] - qpos_spring[qposid]
            qfrc_spring_out[worldid, dofid] = -stiffness * fdif

        # damper
        if has_damping:
            qfrc_damper_out[worldid, dofid] = -damping * qvel_in[worldid, dofid]


@wp.kernel
def _qfrc_passive(
        # Data in:
        qfrc_spring_in: wp.array2d(dtype=float),
        qfrc_damper_in: wp.array2d(dtype=float),
        # Data out:
        qfrc_passive_out: wp.array2d(dtype=float),
):
    worldid, dofid = wp.tid()
    qfrc_passive = qfrc_spring_in[worldid, dofid]
    qfrc_passive += qfrc_damper_in[worldid, dofid]

    qfrc_passive_out[worldid, dofid] = qfrc_passive


@event_scope
def apply_passive_forces(m: Model, d: Data):
    """Adds all passive forces."""
    wp.launch(
        _spring_damper_dof_passive,
        dim=(d.nworld, m.nbody),
        inputs=[
            m.qpos_spring,
            m.jnt_type,
            m.jnt_qposadr,
            m.jnt_dofadr,
            m.jnt_stiffness,
            m.dof_damping,
            d.qpos,
            d.qvel,
        ],
        outputs=[d.qfrc_spring, d.qfrc_damper],
    )

    wp.launch(
        _qfrc_passive,
        dim=(d.nworld, m.nv),
        inputs=[
            d.qfrc_spring,
            d.qfrc_damper,
        ],
        outputs=[
            d.qfrc_passive,
        ],
    )


@wp.kernel
def _qfrc_smooth(
        # Data in:
        qfrc_applied_in: wp.array2d(dtype=float),
        qfrc_bias_in: wp.array2d(dtype=float),
        qfrc_passive_in: wp.array2d(dtype=float),
        # Data out:
        qfrc_smooth_out: wp.array2d(dtype=float),
):
    worldid, dofid = wp.tid()
    qfrc_smooth_out[worldid, dofid] = (
            qfrc_passive_in[worldid, dofid]
            - qfrc_bias_in[worldid, dofid]
            + qfrc_applied_in[worldid, dofid]
    )


@event_scope
def reset_applied_forces(m: Model, d: Data):
    """ Compute all applied forces """
    d.xfrc_applied.zero_()
    d.qfrc_applied.zero_()
    # dof actuators would go here


@event_scope
def accumulate_forces(m: Model, d: Data):
    """ Accumulate all forces into qfrc_applied smooth"""
    wp.launch(
        _qfrc_smooth,
        dim=(d.nworld, m.nv),
        inputs=[d.qfrc_applied, d.qfrc_bias, d.qfrc_passive],
        outputs=[d.qfrc_smooth],
    )
    support.xfrc_accumulate(m, d, d.qfrc_smooth)
