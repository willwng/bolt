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

from .types import Data
from .types import JointType
from .types import Model
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


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
def _qfrc_accumulate(
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
        qfrc_tau_out: wp.array2d(dtype=float),
):
    worldid, dofid = wp.tid()
    if integration_done_in[worldid]:
        return
    qfrc_tau_out[worldid, dofid] = (
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


@wp.kernel
def _xfrc_accumulate(
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        xfrc_applied_in: wp.array2d(dtype=wp.spatial_vector),
        xfrc_contact_in: wp.array2d(dtype=wp.spatial_vector),
        xfrc_drag_in: wp.array2d(dtype=wp.spatial_vector),
        xfrc_muscle_in: wp.array2d(dtype=wp.spatial_vector),
        # Data out:
        xfrc_smooth_out: wp.array2d(dtype=wp.spatial_vector)
):
    worldid, bodyid = wp.tid()
    if integration_done_in[worldid]:
        return
    xfrc_smooth_out[worldid, bodyid] = (
            xfrc_applied_in[worldid, bodyid] +
            xfrc_contact_in[worldid, bodyid] +
            xfrc_drag_in[worldid, bodyid] +
            xfrc_muscle_in[worldid, bodyid]
    )


@wp.kernel
def _gravity(
        # Model in:
        body_mass_in: wp.array(dtype=float),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        # In:
        gravity: float,
        # Data out:
        xfrc_gravity_out: wp.array2d(dtype=wp.spatial_vector)
):
    worldid, bodyid = wp.tid()
    if integration_done_in[worldid]:
        return
    m = body_mass_in[bodyid]
    # TODO: this should be applied to COM Frame
    xfrc_gravity_out[worldid, bodyid] = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0, m * gravity, 0.0))
    return


@event_scope
def apply_gravity(m: Model, d: Data):
    """
    Compute gravity forces. Note: this applies an external force.
    It is more efficient to set the acceleration to -g in the articulated body algorithm
    """
    wp.launch(
        _gravity,
        dim=(d.nworld, m.nbody),
        inputs=[m.body_mass, d.integration_done, m.opt.gravity],
        outputs=[d.xfrc_gravity],
    )


@event_scope
def reset_forces(m: Model, d: Data):
    """ Compute all applied forces """
    d.xfrc_gravity.zero_()
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
    wp.launch(
        _qfrc_accumulate,
        dim=(d.nworld, m.nv),
        inputs=[d.integration_done,
                d.qfrc_applied, d.qfrc_bias, d.qfrc_muscle, d.qfrc_actuator, d.qfrc_limit,
                d.qfrc_contact, d.qfrc_spring, d.qfrc_damper, d.qfrc_drag],
        outputs=[d.qfrc_tau],
    )
