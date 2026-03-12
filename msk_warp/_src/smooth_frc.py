import warp as wp

from . import math
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

    jnttype = jnt_type[jntid]
    qposadr = jnt_qposadr[jntid]
    dofadr = jnt_dofadr[jntid]
    stiffness = jnt_stiffness[jntid]
    dofnum = jnt_dofnum[jntid]

    if jnttype == JointType.FREE:  # no spring forces on free joints
        return
    elif jnttype == JointType.BALL:  # quaternion target
        return  # todo!

    for i in range(dofnum):
        dif = qpos_in[worldid, qposadr + i] - qpos_spring[qposadr + i]
        qfrc_spring_out[worldid, dofadr + i] = -stiffness * dif


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
def _gravity(
        # Model in:
        body_mass_in: wp.array(dtype=float),
        body_mass_center: wp.array(dtype=wp.vec3),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        X_GB_in: wp.array2d(dtype=wp.transform),
        # In:
        gravity: float,
        # Data out:
        body_F_gravity: wp.array2d(dtype=wp.spatial_vector)
):
    worldid, bodyid = wp.tid()
    if integration_done_in[worldid]:
        return
    m = body_mass_in[bodyid]
    com_local = body_mass_center[bodyid]
    X_GB = X_GB_in[worldid, bodyid]
    frc = wp.vec3(0.0, m * gravity, 0.0)
    body_F_gravity[worldid, bodyid] = math.apply_force_to_body_point(X_GB, com_local, frc)
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
        inputs=[m.body_mass, m.body_mass_center, d.integration_done, d.mob_X_GB, m.opt.gravity],
        outputs=[d.body_F_gravity],
    )


@wp.kernel
def _mob_f_accumulate(
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
        mob_f_out: wp.array2d(dtype=float),
):
    worldid, dofid = wp.tid()
    if integration_done_in[worldid]:
        return
    mob_f_out[worldid, dofid] = (
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
def _body_frc_accumulate(
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        body_F_gravity_in: wp.array2d(dtype=wp.spatial_vector),
        body_F_contact_in: wp.array2d(dtype=wp.spatial_vector),
        # Data out:
        body_F_out: wp.array2d(dtype=wp.spatial_vector)
):
    worldid, bodyid = wp.tid()
    if integration_done_in[worldid]:
        return
    body_F_out[worldid, bodyid] = (
            body_F_gravity_in[worldid, bodyid] +
            body_F_contact_in[worldid, bodyid]
    )


@event_scope
def reset_forces(m: Model, d: Data):
    """ Compute all applied forces """
    d.body_F_gravity.zero_()
    d.body_F_contact.zero_()
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
    """ Accumulate all forces into d.body_F and d.mob_f smooth"""
    wp.launch(
        _body_frc_accumulate,
        dim=(d.nworld, m.nbody),
        inputs=[d.integration_done, d.body_F_gravity, d.body_F_contact],
        outputs=[d.body_F]
    )

    wp.launch(
        _mob_f_accumulate,
        dim=(d.nworld, m.nv),
        inputs=[d.integration_done,
                d.qfrc_applied, d.qfrc_bias, d.qfrc_muscle, d.qfrc_actuator, d.qfrc_limit,
                d.qfrc_contact, d.qfrc_spring, d.qfrc_damper, d.qfrc_drag],
        outputs=[d.qfrc_total],
    )
