import warp as wp

from . import math
from . import mobilizers
from .types import Data
from .types import MobilizerType
from .types import Model
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _spring_jnt_passive(
        # Model:
        qpos_spring: wp.array(dtype=float),
        mob_type: wp.array(dtype=int),
        mob_qposadr: wp.array(dtype=int),
        mob_dofadr: wp.array(dtype=int),
        mob_dofnum: wp.array(dtype=int),
        dof_stiffness: wp.array(dtype=float),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qpos_in: wp.array2d(dtype=float),
        # Data out:
        ufrc_spring_out: wp.array2d(dtype=float),
):
    worldid, jntid = wp.tid()
    if integration_done_in[worldid]:
        return

    mobtype = mob_type[jntid]
    qposadr = mob_qposadr[jntid]
    dofadr = mob_dofadr[jntid]
    dofnum = mob_dofnum[jntid]

    if mobtype == MobilizerType.FREE:  # no spring forces on free joints
        return
    elif mobtype == MobilizerType.BALL:  # quaternion target
        return  # todo!

    for i in range(dofnum):
        stiffness = dof_stiffness[dofadr + i]
        dif = qpos_in[worldid, qposadr + i] - qpos_spring[qposadr + i]
        ufrc_spring_out[worldid, dofadr + i] = -stiffness * dif
    return


@wp.kernel
def _damper_dof_passive(
        # Model:
        dof_damping: wp.array(dtype=float),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qvel_in: wp.array2d(dtype=float),
        # Data out:
        ufrc_damper_out: wp.array2d(dtype=float),
):
    worldid, dofid = wp.tid()
    if integration_done_in[worldid]:
        return
    ufrc_damper_out[worldid, dofid] = -dof_damping[dofid] * qvel_in[worldid, dofid]


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


@wp.kernel
def _mob_f_accumulate(
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        ufrc_applied_in: wp.array2d(dtype=float),
        ufrc_muscle_in: wp.array2d(dtype=float),
        ufrc_actuator_in: wp.array2d(dtype=float),
        ufrc_limit_in: wp.array2d(dtype=float),
        ufrc_spring_in: wp.array2d(dtype=float),
        ufrc_damper_in: wp.array2d(dtype=float),
        # Data out:
        mob_f_out: wp.array2d(dtype=float),
):
    worldid, dofid = wp.tid()
    if integration_done_in[worldid]:
        return
    mob_f_out[worldid, dofid] = (
            ufrc_applied_in[worldid, dofid]
            + ufrc_muscle_in[worldid, dofid]
            + ufrc_actuator_in[worldid, dofid]
            + ufrc_limit_in[worldid, dofid]
            + ufrc_spring_in[worldid, dofid]
            + ufrc_damper_in[worldid, dofid]
    )


@wp.kernel
def _body_frc_accumulate(
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        body_F_gravity_in: wp.array2d(dtype=wp.spatial_vector),
        body_F_contact_in: wp.array2d(dtype=wp.spatial_vector),
        body_F_muscle_in: wp.array2d(dtype=wp.spatial_vector),
        body_F_drag_in: wp.array2d(dtype=wp.spatial_vector),
        # Data out:
        body_F_out: wp.array2d(dtype=wp.spatial_vector)
):
    worldid, bodyid = wp.tid()
    if integration_done_in[worldid]:
        return
    body_F_out[worldid, bodyid] = (
            body_F_gravity_in[worldid, bodyid] +
            body_F_contact_in[worldid, bodyid] +
            body_F_muscle_in[worldid, bodyid] +
            body_F_drag_in[worldid, bodyid]
    )


@wp.kernel
def _calc_ufrc_from_qfrc(
        # Model:
        mob_type: wp.array(dtype=int),
        mob_qposadr: wp.array(dtype=int),
        mob_dofadr: wp.array(dtype=int),
        mob_dofnum: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qpos_in: wp.array2d(dtype=float),
        qfrc_in: wp.array2d(dtype=float),
        # Data out:
        ufrc_out: wp.array2d(dtype=float),
):
    worldid, bodyid = wp.tid()
    if integration_done_in[worldid]:
        return

    mob_type_ = mob_type[bodyid]
    qposadr = mob_qposadr[bodyid]
    dofadr = mob_dofadr[bodyid]
    dofnum = mob_dofnum[bodyid]
    qpos = qpos_in[worldid]
    qfrc = qfrc_in[worldid]
    mobilizers.multiply_by_N_transpose(mob_type_, qpos, qfrc, qposadr, dofadr, dofnum, ufrc_out[worldid])
    return


@event_scope
def reset_forces(m: Model, d: Data):
    """ Compute all applied forces """
    d.body_F_gravity.zero_()
    d.body_F_contact.zero_()
    d.body_F_drag.zero_()
    d.body_F_muscle.zero_()

    d.ufrc_spring.zero_()
    d.ufrc_damper.zero_()
    d.qfrc_muscle.zero_()
    d.ufrc_muscle.zero_()
    d.ufrc_actuator.zero_()
    d.ufrc_limit.zero_()

    d.grf.zero_()
    d.geom_cforce.zero_()


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


@event_scope
def spring(m: Model, d: Data):
    """Adds all passive forces."""
    wp.launch(
        _spring_jnt_passive,
        dim=(d.nworld, m.nbody),
        inputs=[
            m.qpos_spring_rest,
            m.mob_type,
            m.mob_qposadr,
            m.mob_dofadr,
            m.mob_dofnum,
            m.dof_stiffness,
            d.integration_done,
            d.qpos,
        ],
        outputs=[d.ufrc_spring],
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
        outputs=[d.ufrc_damper],
    )


@event_scope
def qfrc_to_ufrc(m: Model, d: Data):
    """ Converts all forces in qpos space to u space """
    def qfrc_to_ufrc_helper(qfrc_in, ufrc_out):
        wp.launch(
            _calc_ufrc_from_qfrc,
            dim=(d.nworld, m.nbody),
            inputs=[
                m.mob_type, m.mob_qposadr, m.mob_dofadr, m.mob_dofnum,
                d.integration_done, d.qpos, qfrc_in
            ],
            outputs=[ufrc_out],
        )

    qfrc_to_ufrc_helper(d.qfrc_muscle, d.ufrc_muscle)


@event_scope
def accumulate_forces(m: Model, d: Data):
    """ Accumulate all forces into d.body_F and d.mob_f smooth"""
    wp.launch(
        _body_frc_accumulate,
        dim=(d.nworld, m.nbody),
        inputs=[d.integration_done, d.body_F_gravity, d.body_F_contact, d.body_F_muscle, d.body_F_drag],
        outputs=[d.body_F]
    )

    wp.launch(
        _mob_f_accumulate,
        dim=(d.nworld, m.nv),
        inputs=[d.integration_done,
                d.ufrc_applied, d.ufrc_muscle, d.ufrc_actuator, d.ufrc_limit,
                d.ufrc_spring, d.ufrc_damper],
        outputs=[d.ufrc_total],
    )
