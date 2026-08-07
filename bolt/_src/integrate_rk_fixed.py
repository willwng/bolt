import warp as wp

from . import forward
from . import integrate_common
from .types import Data
from .types import Model
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _rk_accumulate_velocity_acceleration(
        # Data in:
        qvel_in: wp.array2d(dtype=float),
        qacc_in: wp.array2d(dtype=float),
        # In:
        scale: float,
        # Data out:
        qvel_out: wp.array2d(dtype=float),
        qacc_out: wp.array2d(dtype=float),
):
    worldid, dofid = wp.tid()
    qvel_out[worldid, dofid] += scale * qvel_in[worldid, dofid]
    qacc_out[worldid, dofid] += scale * qacc_in[worldid, dofid]


@wp.kernel
def _rk_accumulate_muscle(
        # Data in:
        m_act_dot_in: wp.array2d(dtype=float),
        m_state_dot_in: wp.array2d(dtype=float),
        # In:
        scale: float,
        # Data out:
        m_act_dot_out: wp.array2d(dtype=float),
        m_state_dot_out: wp.array2d(dtype=float),
):
    worldid, muscle_id = wp.tid()
    m_act_dot_out[worldid, muscle_id] += scale * m_act_dot_in[worldid, muscle_id]
    m_state_dot_out[worldid, muscle_id] += scale * m_state_dot_in[worldid, muscle_id]


@wp.kernel
def _rk_accumulate_actuator(
        # Data in:
        a_act_dot_in: wp.array2d(dtype=float),
        # In:
        scale: float,
        # Data out:
        a_act_dot_out: wp.array2d(dtype=float),
):
    worldid, actuator_id = wp.tid()
    a_act_dot_out[worldid, actuator_id] += scale * a_act_dot_in[worldid, actuator_id]


def _rk_accumulate(
        m: Model,
        d: Data,
        scale: float,
        qvel_rk: wp.array2d(dtype=float),
        qacc_rk: wp.array2d(dtype=float),
        m_act_dot_rk: wp.array2d(dtype=float),
        m_state_dot_rk: wp.array2d(dtype=float),
        a_act_dot_rk: wp.array2d(dtype=float),
):
    """Computes one term of 1/6 k_1 + 1/3 k_2 + 1/3 k_3 + 1/6 k_4."""
    wp.launch(
        _rk_accumulate_velocity_acceleration,
        dim=(d.nworld, m.nv),
        inputs=[d.qvel, d.qacc, scale],
        outputs=[qvel_rk, qacc_rk],
    )

    if m.nmuscle:
        wp.launch(
            _rk_accumulate_muscle,
            dim=(d.nworld, m.nmuscle),
            inputs=[d.m_act_dot, d.m_state_dot, scale],
            outputs=[m_act_dot_rk, m_state_dot_rk],
        )
    if m.nactuator:
        wp.launch(
            _rk_accumulate_actuator,
            dim=(d.nworld, m.nactuator),
            inputs=[d.a_act_dot, scale],
            outputs=[a_act_dot_rk],
        )


def _rk_perturb_state(
        m: Model,
        d: Data,
        scale: float,
        qpos_t0: wp.array2d(dtype=float),
        qvel_t0: wp.array2d(dtype=float),
        m_act_t0: wp.array2d(dtype=float),
        m_state_t0: wp.array2d(dtype=float),
        a_act_t0: wp.array2d(dtype=float)
):
    # position
    wp.launch(
        integrate_common._next_position,
        dim=(d.nworld, m.nbody),
        inputs=[m.mob_type, m.mob_qposadr, m.mob_dofadr, m.mob_dofnum,
                d.integration_done, qpos_t0, d.qvel, d.actual_step_size, scale],
        outputs=[d.qpos],
    )
    # velocity
    wp.launch(
        integrate_common._next_velocity,
        dim=(d.nworld, m.nv),
        inputs=[d.integration_done, qvel_t0, d.qacc, d.actual_step_size, scale],
        outputs=[d.qvel],
    )

    # muscles
    if m.nmuscle:
        wp.launch(
            integrate_common._next_muscle_activation,
            dim=(d.nworld, m.nmuscle),
            inputs=[m.muscle_metadata, d.integration_done, m_act_t0, d.m_act_dot, d.actual_step_size, scale],
            outputs=[d.m_act],
        )
        wp.launch(
            integrate_common._next_muscle_state,
            dim=(d.nworld, m.nmuscle),
            inputs=[m.muscle_metadata, d.integration_done, m_state_t0, d.m_state_dot, d.actual_step_size, scale],
            outputs=[d.m_state],
        )

    if m.nactuator:
        wp.launch(
            integrate_common._next_actuator_activation,
            dim=(d.nworld, m.nactuator),
            inputs=[m.actuator_metadata, d.integration_done, a_act_t0, d.a_act_dot, d.actual_step_size, scale],
            outputs=[d.a_act],
        )

    if m.nstlcontact:
        wp.launch(
            integrate_common._next_stl_contact_state,
            dim=(d.nworld, m.nstlcontact),
            inputs=[m.stl_contact, d.integration_done, d.exp_contact_state_dot, d.actual_step_size, scale],
            outputs=[d.exp_contact_state],
        )

@event_scope
def rungekutta4(m: Model, d: Data):
    """Runge-Kutta explicit order 4 integrator."""
    # RK4 tableau
    A = [0.5, 0.5, 1.0]
    B = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]

    # Initial state y_0 and derivative accumulators y'_rk
    qpos_t0 = wp.clone(d.qpos)
    qvel_t0 = wp.clone(d.qvel)
    qvel_rk = wp.zeros((d.nworld, m.nv), dtype=float)
    qacc_rk = wp.zeros((d.nworld, m.nv), dtype=float)
    if m.nmuscle:
        m_act_t0 = wp.clone(d.m_act)
        m_state_t0 = wp.clone(d.m_state)
        m_act_dot_rk = wp.zeros((d.nworld, m.nmuscle), dtype=float)
        m_state_dot_rk = wp.zeros((d.nworld, m.nmuscle), dtype=float)
    else:
        m_act_t0, m_state_t0 = None, None
        m_act_dot_rk, m_state_dot_rk = None, None
    if m.nactuator:
        a_act_t0 = wp.clone(d.a_act)
        a_act_dot_rk = wp.zeros((d.nworld, m.nactuator), dtype=float)
    else:
        a_act_t0 = None
        a_act_dot_rk = None

    # Compute 1/6 k_1
    _rk_accumulate(m, d, B[0], qvel_rk, qacc_rk, m_act_dot_rk, m_state_dot_rk, a_act_dot_rk)
    # Compute k_2, k_3, k_4
    for i in range(3):
        a, b = float(A[i]), B[i + 1]
        # Realize state, compute next derivative
        _rk_perturb_state(m, d, a, qpos_t0, qvel_t0, m_act_t0, m_state_t0, a_act_t0)
        forward.fwd(m, d)
        _rk_accumulate(m, d, b, qvel_rk, qacc_rk, m_act_dot_rk, m_state_dot_rk, a_act_dot_rk)

    # Restore initial state, set accumulated derivatives
    wp.copy(d.qpos, qpos_t0)
    wp.copy(d.qvel, qvel_t0)
    if m.nmuscle:
        wp.copy(d.m_act, m_act_t0)
        wp.copy(d.m_act_dot, m_act_dot_rk)
        wp.copy(d.m_state, m_state_t0)
        wp.copy(d.m_state_dot, m_state_dot_rk)
    if m.nactuator:
        wp.copy(d.a_act, a_act_t0)
        wp.copy(d.a_act_dot, a_act_dot_rk)
    integrate_common.advance(m, d, qacc_rk, qvel_rk, scale=1.0, time_scale=1.0, symplectic=False)
    wp.copy(d.qacc, qacc_rk)  # copy acceleration for post-step analysis
    return


@event_scope
def integrate(m: Model, d: Data):
    """Steps from d.time to d.next_time using RK4 """
    integrate_common.update_step_size(m, d)
    rungekutta4(m, d)
    forward.fwd(m, d)  # realize for next step
