import warp as wp

from bolt.types import Data
from bolt.types import MobilizerType
from bolt.types import Model

from .. import math
from ..kinematics import mobilizers
from ..warp_util import event_scope
from ..warp_util import launch

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
        gravity: wp.vec3,
        # Data out:
        body_F_gravity: wp.array2d(dtype=wp.spatial_vector)
):
    worldid, bodyid = wp.tid()
    if integration_done_in[worldid]:
        return
    m = body_mass_in[bodyid]
    com_local = body_mass_center[bodyid]
    X_GB = X_GB_in[worldid, bodyid]
    frc = m * gravity
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
        body_F_applied: wp.array2d(dtype=wp.spatial_vector),
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
            body_F_applied[worldid, bodyid]
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


@wp.kernel
def _calc_qfrc_from_ufrc(
        # Model:
        mob_type: wp.array(dtype=int),
        mob_qposadr: wp.array(dtype=int),
        mob_dofadr: wp.array(dtype=int),
        mob_dofnum: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qpos_in: wp.array2d(dtype=float),
        ufrc_in: wp.array2d(dtype=float),
        # Data out:
        qfrc_out: wp.array2d(dtype=float),
):
    worldid, bodyid = wp.tid()
    if integration_done_in[worldid]:
        return

    mob_type_ = mob_type[bodyid]
    qposadr = mob_qposadr[bodyid]
    dofadr = mob_dofadr[bodyid]
    dofnum = mob_dofnum[bodyid]
    qpos = qpos_in[worldid]
    ufrc = ufrc_in[worldid]
    mobilizers.multiply_by_N_inv_transpose(mob_type_, qpos, ufrc, qposadr, dofadr, dofnum, qfrc_out[worldid])
    return


@wp.kernel
def _reset_forces(
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        # Data out:
        body_F_gravity_out: wp.array2d(dtype=wp.spatial_vector),
        body_F_contact_out: wp.array2d(dtype=wp.spatial_vector),
        body_F_muscle_out: wp.array2d(dtype=wp.spatial_vector),
        ufrc_spring_out: wp.array2d(dtype=float),
        ufrc_damper_out: wp.array2d(dtype=float),
        qfrc_muscle_out: wp.array2d(dtype=float),
        ufrc_muscle_out: wp.array2d(dtype=float),
        ufrc_actuator_out: wp.array2d(dtype=float),
        ufrc_limit_out: wp.array2d(dtype=float),
        grf_out: wp.array(dtype=wp.vec3),
        geom_cforce_out: wp.array2d(dtype=float),
        geom_self_cforce_out: wp.array2d(dtype=float),
        body_self_cforce_out: wp.array2d(dtype=float),
):
    worldid, i = wp.tid()
    # Worlds that finished integrating keep their forces, which describe their final state
    if integration_done_in[worldid]:
        return

    if i < body_F_gravity_out.shape[1]:
        body_F_gravity_out[worldid, i] = wp.spatial_vector()
        body_F_contact_out[worldid, i] = wp.spatial_vector()
        body_F_muscle_out[worldid, i] = wp.spatial_vector()
        body_self_cforce_out[worldid, i] = 0.0
    if i < ufrc_spring_out.shape[1]:
        ufrc_spring_out[worldid, i] = 0.0
        ufrc_damper_out[worldid, i] = 0.0
        ufrc_muscle_out[worldid, i] = 0.0
        ufrc_actuator_out[worldid, i] = 0.0
        ufrc_limit_out[worldid, i] = 0.0
    if i < qfrc_muscle_out.shape[1]:
        qfrc_muscle_out[worldid, i] = 0.0
    if i < geom_cforce_out.shape[1]:
        geom_cforce_out[worldid, i] = 0.0
        geom_self_cforce_out[worldid, i] = 0.0
    if i == 0:
        grf_out[worldid] = wp.vec3()


@event_scope
def reset_forces(m: Model, d: Data):
    """ Zeros the force accumulators of the worlds that are still integrating """
    launch(
        _reset_forces,
        dim=(d.nworld, max(m.nbody, m.nv, m.nq, m.ngeom, 1)),
        inputs=[d.integration_done],
        outputs=[
            d.body_F_gravity, d.body_F_contact, d.body_F_muscle,
            d.ufrc_spring, d.ufrc_damper, d.qfrc_muscle, d.ufrc_muscle, d.ufrc_actuator, d.ufrc_limit,
            d.grf, d.geom_cforce, d.geom_self_cforce, d.body_self_cforce,
        ],
    )
    if m.opt.implicit_friction_damping:
        d.body_C_contact.zero_()


@event_scope
def apply_gravity(m: Model, d: Data):
    """
    Compute gravity forces. Note: this applies an external force.
    It is more efficient to set the acceleration to -g in the articulated body algorithm
    """
    launch(
        _gravity,
        dim=(d.nworld, m.nbody),
        inputs=[m.body_mass, m.body_mass_center, d.integration_done, d.mob_X_GB, m.opt.gravity],
        outputs=[d.body_F_gravity],
    )


@event_scope
def spring(m: Model, d: Data):
    """Adds all passive forces."""
    launch(
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
    launch(
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
def qfrc_to_ufrc(m: Model, d: Data, qfrc_in: wp.array, ufrc_out: wp.array):
    """ Converts all forces in qpos space to u space """
    launch(
        _calc_ufrc_from_qfrc,
        dim=(d.nworld, m.nbody),
        inputs=[
            m.mob_type, m.mob_qposadr, m.mob_dofadr, m.mob_dofnum,
            d.integration_done, d.qpos, qfrc_in
        ],
        outputs=[ufrc_out],
    )


@event_scope
def ufrc_to_qfrc(m: Model, d: Data, ufrc_in: wp.array, qfrc_out: wp.array):
    """ Converts all forces in u space to q space """
    launch(
        _calc_qfrc_from_ufrc,
        dim=(d.nworld, m.nbody),
        inputs=[
            m.mob_type, m.mob_qposadr, m.mob_dofadr, m.mob_dofnum,
            d.integration_done, d.qpos, ufrc_in
        ],
        outputs=[qfrc_out],
    )


@event_scope
def accumulate_forces(m: Model, d: Data):
    """ Accumulate all forces into d.body_F and d.mob_f smooth"""
    launch(
        _body_frc_accumulate,
        dim=(d.nworld, m.nbody),
        inputs=[d.integration_done, d.body_F_gravity, d.body_F_contact, d.body_F_muscle, d.body_F_applied],
        outputs=[d.body_F]
    )

    launch(
        _mob_f_accumulate,
        dim=(d.nworld, m.nv),
        inputs=[d.integration_done,
                d.ufrc_applied, d.ufrc_muscle, d.ufrc_actuator, d.ufrc_limit,
                d.ufrc_spring, d.ufrc_damper],
        outputs=[d.ufrc_total],
    )
