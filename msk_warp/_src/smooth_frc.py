# Copyright 2025 The Newton Developers
# Modified for MSKWarp by Will Wang
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
def _link_frc(
        # Model:
        body_inertia: wp.array(dtype=wp.spatial_matrix),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        body_X_com_in: wp.array2d(dtype=wp.transform),
        body_vel_in: wp.array2d(dtype=wp.spatial_vector),
        body_acc_in: wp.array2d(dtype=wp.spatial_vector),
        # Out:
        body_f_s_out: wp.array2d(dtype=wp.spatial_vector),
        body_I_s_out: wp.array2d(dtype=wp.spatial_matrix),
):
    worldid, bodyid = wp.tid()
    if integration_done_in[worldid]:
        return

    X_body_com = body_X_com_in[worldid, bodyid]
    I_body = body_inertia[bodyid]

    I_s = math.transform_spatial_inertia(X_body_com, I_body)
    v_s, a_s = body_vel_in[worldid, bodyid], body_acc_in[worldid, bodyid]

    f_body = I_s * a_s + wp.spatial_cross_dual(v_s, I_s * v_s)
    body_f_s_out[worldid, bodyid] = f_body
    body_I_s_out[worldid, bodyid] = I_s


@event_scope
def link_frc(m: Model, d: Data):
    wp.launch(
        _link_frc,
        dim=(d.nworld, m.nbody),
        inputs=[m.body_inertia, d.integration_done, d.body_X_com, d.body_vel, d.body_acc],
        outputs=[d.body_f_s, d.body_I_s],
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
        return  # todo!
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
    qfrc_damper_out[worldid, dofid] = -dof_damping[dofid] * qvel_in[worldid, dofid]


@event_scope
def spring(m: Model, d: Data):
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

@event_scope
def damping(m: Model, d: Data):
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
