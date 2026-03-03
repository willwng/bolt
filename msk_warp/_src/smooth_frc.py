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
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        # Data out:
        cacc_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid = wp.tid()
    if integration_done_in[worldid]:
        return
    cacc_out[worldid, 0] = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0, -gravity, 0.0))


def _rne_cacc_world(m: Model, d: Data):
    wp.launch(
        _cacc_world,
        dim=[d.nworld],
        inputs=[m.opt.gravity, d.integration_done],
        outputs=[d.cacc]
    )


@wp.kernel
def _cacc(
        # Model:
        body_parentid: wp.array(dtype=int),
        jnt_dofnum: wp.array(dtype=int),
        jnt_dofadr: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qvel_in: wp.array2d(dtype=float),
        cdof_dot_in: wp.array2d(dtype=wp.spatial_vector),
        cacc_in: wp.array2d(dtype=wp.spatial_vector),
        # In:
        body_tree_: wp.array(dtype=int),
        # Data out:
        cacc_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, nodeid = wp.tid()
    if integration_done_in[worldid]:
        return

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
            inputs=[m.body_parentid, m.jnt_dofnum, m.jnt_dofadr,
                    d.integration_done, d.qvel, d.cdof_dot, d.cacc, body_tree],
            outputs=[d.cacc],
        )


@wp.kernel
def _cfrc(
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        cinert_in: wp.array2d(dtype=vec10),
        cvel_in: wp.array2d(dtype=wp.spatial_vector),
        cacc_in: wp.array2d(dtype=wp.spatial_vector),
        # Data out:
        cfrc_int_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, bodyid = wp.tid()
    if integration_done_in[worldid]:
        return
    bodyid += 1  # skip world body

    cacc = cacc_in[worldid, bodyid]
    cinert = cinert_in[worldid, bodyid]
    cvel = cvel_in[worldid, bodyid]
    frc = math.inert_vec(cinert, cacc)
    frc += math.motion_cross_force(cvel, math.inert_vec(cinert, cvel))

    cfrc_int_out[worldid, bodyid] = frc


def _rne_cfrc(m: Model, d: Data):
    wp.launch(
        _cfrc,
        dim=[d.nworld, m.nbody - 1],
        inputs=[d.integration_done, d.cinert, d.cvel, d.cacc],
        outputs=[d.cfrc_int]
    )


@wp.kernel
def _cfrc_backward(
        # Model:
        body_parentid: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        cfrc_int_in: wp.array2d(dtype=wp.spatial_vector),
        # In:
        body_tree_: wp.array(dtype=int),
        # Data out:
        cfrc_int_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, nodeid = wp.tid()
    if integration_done_in[worldid]:
        return
    bodyid = body_tree_[nodeid]
    pid = body_parentid[bodyid]
    if bodyid != 0:
        wp.atomic_add(cfrc_int_out[worldid], pid, cfrc_int_in[worldid, bodyid])


def _rne_cfrc_backward(m: Model, d: Data):
    for body_tree in reversed(m.body_tree):
        wp.launch(
            _cfrc_backward,
            dim=[d.nworld, body_tree.size],
            inputs=[m.body_parentid, d.integration_done, d.cfrc_int, body_tree],
            outputs=[d.cfrc_int]
        )


@wp.kernel
def _qfrc_bias(
        # Model:
        dof_bodyid: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        cdof_in: wp.array2d(dtype=wp.spatial_vector),
        cfrc_int_in: wp.array2d(dtype=wp.spatial_vector),
        # Data out:
        qfrc_bias_out: wp.array2d(dtype=float),
):
    worldid, dofid = wp.tid()
    if integration_done_in[worldid]:
        return
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
    wp.launch(
        _qfrc_bias,
        dim=[d.nworld, m.nv],
        inputs=[m.dof_bodyid, d.integration_done, d.cdof, d.cfrc_int],
        outputs=[d.qfrc_bias]
    )


@wp.kernel
def _spring_jnt_passive(
        # Model:
        qpos_spring: wp.array(dtype=float),
        jnt_type: wp.array(dtype=int),
        jnt_qposadr: wp.array(dtype=int),
        jnt_dofadr: wp.array(dtype=int),
        jnt_dofnum: wp.array(dtype=int),
        jnt_stiffness: wp.array(dtype=float),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qpos_in: wp.array2d(dtype=float),
        # Data out:
        qfrc_spring_out: wp.array2d(dtype=float),
):
    worldid, jntid = wp.tid()
    if integration_done_in[worldid]:
        return
    dofid = jnt_dofadr[jntid]
    stiffness = jnt_stiffness[jntid]

    has_stiffness = stiffness != 0.0
    if not has_stiffness:
        qfrc_spring_out[worldid, dofid] = 0.0

    jnttype = jnt_type[jntid]
    qposid = jnt_qposadr[jntid]

    if jnttype == JointType.FREE:  # no spring forces on free joints
        return
    elif jnttype == JointType.BALL:  # quaternion target
        return # todo!
    else:
        if has_stiffness:
            for i in range(jnt_dofnum[jntid]):
                fdif = qpos_in[worldid, qposid + i] - qpos_spring[qposid + i]
                qfrc_spring_out[worldid, dofid + i] = -stiffness * fdif


@wp.kernel
def _damper_dof_passive(
        # Model:
        dof_damping: wp.array(dtype=float),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qvel_in: wp.array2d(dtype=float),
        # Data out:
        qfrc_damper_out: wp.array2d(dtype=float),
):
    worldid, dofid = wp.tid()
    if integration_done_in[worldid]:
        return
    damping = dof_damping[dofid]
    qfrc_damper_out[worldid, dofid] = -damping * qvel_in[worldid, dofid]


@event_scope
def apply_spring_damper(m: Model, d: Data):
    """Adds all passive forces."""
    wp.launch(
        _spring_jnt_passive,
        dim=(d.nworld, m.nbody),
        inputs=[
            m.qpos_spring,
            m.jnt_type,
            m.jnt_qposadr,
            m.jnt_dofadr,
            m.jnt_dofnum,
            m.jnt_stiffness,
            d.integration_done,
            d.qpos,
        ],
        outputs=[d.qfrc_spring],
    )

    wp.launch(
        _damper_dof_passive,
        dim=(d.nworld, m.nv),
        inputs=[
            m.dof_damping,
            d.integration_done,
            d.qvel,
        ],
        outputs=[d.qfrc_damper],
    )


@wp.kernel
def _qfrc_smooth(
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qfrc_applied_in: wp.array2d(dtype=float),
        qfrc_bias_in: wp.array2d(dtype=float),
        qfrc_muscle_in: wp.array2d(dtype=float),
        qfrc_actuator_in: wp.array2d(dtype=float),
        qfrc_limit_in: wp.array2d(dtype=float),
        qfrc_contact_in: wp.array2d(dtype=float),
        qfrc_spring_in: wp.array2d(dtype=float),
        qfrc_damper_in: wp.array2d(dtype=float),
        qfrc_drag_in: wp.array2d(dtype=float),
        # Data out:
        qfrc_smooth_out: wp.array2d(dtype=float),
):
    worldid, dofid = wp.tid()
    if integration_done_in[worldid]:
        return
    qfrc_smooth_out[worldid, dofid] = (
            qfrc_applied_in[worldid, dofid]
            - qfrc_bias_in[worldid, dofid]
            + qfrc_muscle_in[worldid, dofid]
            + qfrc_actuator_in[worldid, dofid]
            + qfrc_limit_in[worldid, dofid]
            + qfrc_contact_in[worldid, dofid]
            + qfrc_spring_in[worldid, dofid]
            + qfrc_damper_in[worldid, dofid]
            + qfrc_drag_in[worldid, dofid]
    )


@event_scope
def reset_forces(m: Model, d: Data):
    """ Compute all applied forces """
    d.xfrc_contact.zero_()
    d.xfrc_drag.zero_()
    d.xfrc_muscle.zero_()

    d.qfrc_applied.zero_()
    d.qfrc_bias.zero_()
    d.qfrc_spring.zero_()
    d.qfrc_damper.zero_()
    d.qfrc_drag.zero_()
    d.qfrc_muscle.zero_()
    d.qfrc_actuator.zero_()
    d.qfrc_contact.zero_()
    d.qfrc_limit.zero_()

    d.geom_cforce.zero_()


@event_scope
def accumulate_forces(m: Model, d: Data):
    """ Accumulate all forces into qfrc_applied smooth"""
    support.apply_ft(m, d, d.xfrc_drag, d.qfrc_drag, True)
    support.apply_ft(m, d, d.xfrc_contact, d.qfrc_contact, True)
    support.apply_ft(m, d, d.xfrc_applied, d.qfrc_applied, True)
    if not wp.static(m.opt.use_fn_path):
        support.apply_ft(m, d, d.xfrc_muscle, d.qfrc_muscle, True)

    wp.launch(
        _qfrc_smooth,
        dim=(d.nworld, m.nv),
        inputs=[d.integration_done,
                d.qfrc_applied, d.qfrc_bias, d.qfrc_muscle, d.qfrc_actuator, d.qfrc_limit,
                d.qfrc_contact, d.qfrc_spring, d.qfrc_damper, d.qfrc_drag],
        outputs=[d.qfrc_smooth],
    )
