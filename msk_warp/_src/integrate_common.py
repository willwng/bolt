import warp as wp

from . import mobilizers
from .types import ActuatorMetadata
from .types import Data
from .types import Model
from .types import MuscleMetadata
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _next_position(
        # Model:
        jnt_type: wp.array(dtype=int),
        jnt_qposadr: wp.array(dtype=int),
        jnt_dofadr: wp.array(dtype=int),
        jnt_dofnum: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qpos_in: wp.array2d(dtype=float),
        qvel_in: wp.array2d(dtype=float),
        actual_step_size_in: wp.array(dtype=float),
        # In:
        scale: float,
        # Data out:
        qpos_out: wp.array2d(dtype=float),
):
    worldid, bodyid = wp.tid()
    if integration_done_in[worldid]:
        return
    timestep = actual_step_size_in[worldid] * scale

    jnttype = jnt_type[bodyid]
    qpos_adr = jnt_qposadr[bodyid]
    dof_adr = jnt_dofadr[bodyid]
    dof_num = jnt_dofnum[bodyid]

    qpos = qpos_in[worldid]
    qvel = qvel_in[worldid]
    qpos_next = qpos_out[worldid]

    mobilizers.integrate(
        jnttype, qpos, qvel, qpos_adr, dof_adr, timestep, dof_num, qpos_next
    )


@wp.kernel
def _next_velocity(
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qvel_in: wp.array2d(dtype=float),
        qacc_in: wp.array2d(dtype=float),
        actual_step_size_in: wp.array(dtype=float),
        # In:
        scale: float,
        # Data out:
        qvel_out: wp.array2d(dtype=float),
):
    worldid, dofid = wp.tid()
    if integration_done_in[worldid]:
        return
    timestep = actual_step_size_in[worldid] * scale
    qvel_out[worldid, dofid] = (qvel_in[worldid, dofid] +
                                qacc_in[worldid, dofid] * timestep)


@wp.kernel
def _next_muscle_activation(
        # Model:
        muscle_metadata: wp.array(dtype=MuscleMetadata),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        act_in: wp.array2d(dtype=float),
        act_dot_in: wp.array2d(dtype=float),
        actual_step_size_in: wp.array(dtype=float),
        # In:
        scale: float,
        # Data out:
        act_out: wp.array2d(dtype=float),
):
    worldid, muscle_id = wp.tid()
    if integration_done_in[worldid]:
        return
    mm = muscle_metadata[muscle_id]
    step_size = actual_step_size_in[worldid] * scale

    # advance muscle activation
    act = act_in[worldid, muscle_id] + act_dot_in[worldid, muscle_id] * step_size
    act_out[worldid, muscle_id] = (wp.clamp(act, mm.min_activation, mm.max_activation))


@wp.kernel
def _next_muscle_state(
        # Model:
        muscle_metadata: wp.array(dtype=MuscleMetadata),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        m_state_in: wp.array2d(dtype=float),
        m_state_dot_in: wp.array2d(dtype=float),
        actual_step_size_in: wp.array(dtype=float),
        # In:
        scale: float,
        # Data out:
        m_state_out: wp.array2d(dtype=float),
):
    worldid, muscle_id = wp.tid()
    if integration_done_in[worldid]:
        return
    mm = muscle_metadata[muscle_id]
    step_size = actual_step_size_in[worldid] * scale

    norm_fiber_length = m_state_in[worldid, muscle_id]
    norm_fiber_length += step_size * m_state_dot_in[worldid, muscle_id]
    norm_fiber_length = wp.clamp(
        norm_fiber_length, mm.min_norm_fiber_length, mm.max_norm_fiber_length)
    m_state_out[worldid, muscle_id] = norm_fiber_length


@wp.kernel
def _next_actuator_activation(
        # Model:
        actuator_metadata: wp.array(dtype=ActuatorMetadata),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        act_in: wp.array2d(dtype=float),
        act_dot_in: wp.array2d(dtype=float),
        actual_step_size_in: wp.array(dtype=float),
        # In:
        scale: float,
        # Data out:
        act_out: wp.array2d(dtype=float),
):
    worldid, actuator_id = wp.tid()
    if integration_done_in[worldid]:
        return
    am = actuator_metadata[actuator_id]
    step_size = actual_step_size_in[worldid] * scale

    # advance muscle activation
    act = act_in[worldid, actuator_id] + act_dot_in[
        worldid, actuator_id] * step_size
    act_out[worldid, actuator_id] = (
        wp.clamp(act, am.min_activation, am.max_activation))


@wp.kernel
def _next_time(
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        time_in: wp.array(dtype=float),
        actual_step_size_in: wp.array(dtype=float),
        # In:
        scale: float,
        # Data out:
        time_out: wp.array(dtype=float),
):
    worldid = wp.tid()
    if integration_done_in[worldid]:
        return
    step_size = actual_step_size_in[worldid] * scale
    time_out[worldid] = time_in[worldid] + step_size


@wp.kernel
def _update_fixed_step_size(
        # Data in:
        time_in: wp.array(dtype=float),
        next_time_in: wp.array(dtype=float),
        # Data out:
        actual_step_size_out: wp.array(dtype=float),
):
    worldid = wp.tid()
    actual_step_size_out[worldid] = next_time_in[worldid] - time_in[worldid]
    return


@event_scope
def advance(m: Model, d: Data, qacc: wp.array, qvel: wp.array, scale: float, symplectic: bool = True):
    """Advance state and time given state derivatives"""
    if m.nmuscle:
        wp.launch(
            _next_muscle_activation,
            dim=(d.nworld, m.nmuscle),
            inputs=[m.muscle_metadata, d.integration_done, d.m_act, d.m_act_dot, d.actual_step_size, scale],
            outputs=[d.m_act],
        )
        wp.launch(
            _next_muscle_state,
            dim=(d.nworld, m.nmuscle),
            inputs=[m.muscle_metadata, d.integration_done, d.m_state, d.m_state_dot, d.actual_step_size, scale],
            outputs=[d.m_state],
        )

    if m.nactuator:
        wp.launch(
            _next_actuator_activation,
            dim=(d.nworld, m.nactuator),
            inputs=[m.actuator_metadata, d.integration_done, d.a_act, d.a_act_dot, d.actual_step_size, scale],
            outputs=[d.a_act],
        )

    if symplectic:
        wp.launch(
            _next_velocity,
            dim=(d.nworld, m.nv),
            inputs=[d.integration_done, d.qvel, qacc, d.actual_step_size, scale],
            outputs=[d.qvel],
        )
        wp.launch(
            _next_position,
            dim=(d.nworld, m.nbody),
            inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, m.jnt_dofnum,
                    d.integration_done, d.qpos, qvel, d.actual_step_size, scale],
            outputs=[d.qpos],
        )
    else:
        wp.launch(
            _next_position,
            dim=(d.nworld, m.nbody),
            inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, m.jnt_dofnum,
                    d.integration_done, d.qpos, qvel, d.actual_step_size, scale],
            outputs=[d.qpos],
        )
        wp.launch(
            _next_velocity,
            dim=(d.nworld, m.nv),
            inputs=[d.integration_done, d.qvel, qacc, d.actual_step_size, scale],
            outputs=[d.qvel],
        )
    wp.launch(
        _next_time,
        dim=d.nworld,
        inputs=[d.integration_done, d.time, d.actual_step_size, scale],
        outputs=[d.time],
    )


@event_scope
def update_step_size(m: Model, d: Data):
    """ For fixed time-stepping. Updates the actual step size to match the desired dt """
    wp.launch(
        _update_fixed_step_size,
        dim=d.nworld,
        inputs=[d.time, d.next_time],
        outputs=[d.actual_step_size],
    )
