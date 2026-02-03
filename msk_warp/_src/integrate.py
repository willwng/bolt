import numpy as np
import warp as wp

from . import forward
from . import mobilizers
from .types import ActuatorMetadata
from .types import Data
from .types import Model
from .types import MuscleMetadata
from .types import TileSet
from .warp_util import cache_kernel
from .warp_util import event_scope
from .warp_util import kernel as nested_kernel

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _next_position(
        # Model:
        jnt_type: wp.array(dtype=int),
        jnt_qposadr: wp.array(dtype=int),
        jnt_dofadr: wp.array(dtype=int),
        jnt_dofnum: wp.array(dtype=int),
        # Data in:
        qpos_in: wp.array2d(dtype=float),
        qvel_in: wp.array2d(dtype=float),
        actual_step_size_in: wp.array(dtype=float),
        # In:
        scale: float,
        # Data out:
        qpos_out: wp.array2d(dtype=float),
):
    worldid, bodyid = wp.tid()
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
        qvel_in: wp.array2d(dtype=float),
        qacc_in: wp.array2d(dtype=float),
        actual_step_size_in: wp.array(dtype=float),
        # In:
        scale: float,
        # Data out:
        qvel_out: wp.array2d(dtype=float),
):
    worldid, dofid = wp.tid()
    timestep = actual_step_size_in[worldid] * scale
    qvel_out[worldid, dofid] = (qvel_in[worldid, dofid] +
                                qacc_in[worldid, dofid] * timestep)


@wp.kernel
def _next_muscle_activation(
        # Model:
        muscle_metadata: wp.array(dtype=MuscleMetadata),
        # Data in:
        act_in: wp.array2d(dtype=float),
        act_dot_in: wp.array2d(dtype=float),
        actual_step_size_in: wp.array(dtype=float),
        # In:
        scale: float,
        # Data out:
        act_out: wp.array2d(dtype=float),
):
    worldid, muscle_id = wp.tid()
    mm = muscle_metadata[muscle_id]
    step_size = actual_step_size_in[worldid] * scale

    # advance muscle activation
    act = act_in[worldid, muscle_id] + act_dot_in[
        worldid, muscle_id] * step_size
    act_out[worldid, muscle_id] = (
        wp.clamp(act, mm.min_activation, mm.max_activation))


@wp.kernel
def _next_muscle_state(
        # Model:
        muscle_metadata: wp.array(dtype=MuscleMetadata),
        # Data in:
        m_state_in: wp.array2d(dtype=float),
        m_state_dot_in: wp.array2d(dtype=float),
        actual_step_size_in: wp.array(dtype=float),
        # In:
        scale: float,
        # Data out:
        m_state_out: wp.array2d(dtype=float),
):
    worldid, muscle_id = wp.tid()
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
        act_in: wp.array2d(dtype=float),
        act_dot_in: wp.array2d(dtype=float),
        actual_step_size_in: wp.array(dtype=float),
        # In:
        scale: float,
        # Data out:
        act_out: wp.array2d(dtype=float),
):
    worldid, actuator_id = wp.tid()
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
        time_in: wp.array(dtype=float),
        actual_step_size_in: wp.array(dtype=float),
        # In:
        scale: float,
        # Data out:
        time_out: wp.array(dtype=float),
):
    worldid = wp.tid()
    step_size = actual_step_size_in[worldid] * scale
    time_out[worldid] = time_in[worldid] + step_size


def _advance(m: Model, d: Data, qacc: wp.array, qvel: wp.array, scale: float):
    """Advance state and time given state derivatives"""
    if m.nmuscle:
        wp.launch(
            _next_muscle_activation,
            dim=(d.nworld, m.nmuscle),
            inputs=[m.muscle_metadata, d.m_act, d.m_act_dot,
                    d.actual_step_size, scale],
            outputs=[d.m_act],
        )
        # If we didn't sub-step, advance here
        if wp.static(m.opt.muscle_dyn_substeps) == 0:
            wp.launch(
                _next_muscle_state,
                dim=(d.nworld, m.nmuscle),
                inputs=[m.muscle_metadata, d.m_state, d.m_state_dot,
                        d.actual_step_size, scale],
                outputs=[d.m_state],
            )

    if m.nactuator:
        wp.launch(
            _next_actuator_activation,
            dim=(d.nworld, m.nactuator),
            inputs=[m.actuator_metadata, d.a_act, d.a_act_dot,
                    d.actual_step_size, scale],
            outputs=[d.a_act],
        )

    wp.launch(
        _next_velocity,
        dim=(d.nworld, m.nv),
        inputs=[d.qvel, qacc, d.actual_step_size, scale],
        outputs=[d.qvel],
    )
    wp.launch(
        _next_position,
        dim=(d.nworld, m.nbody),
        inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, m.jnt_dofnum,
                d.qpos, qvel, d.actual_step_size, scale],
        outputs=[d.qpos],
    )
    wp.launch(
        _next_time,
        dim=d.nworld,
        inputs=[d.time, d.actual_step_size, scale],
        outputs=[d.time],
    )

    wp.copy(d.qacc_warmstart, d.qacc)


@cache_kernel
def _tile_euler_dense(tile: TileSet):
    @nested_kernel(module="unique", enable_backward=False)
    def euler_dense(
            # Model:
            dof_damping: wp.array(dtype=float),
            opt_timestep: float,
            # Data in:
            qM_in: wp.array3d(dtype=float),
            efc_Ma_in: wp.array2d(dtype=float),
            # In:
            adr_in: wp.array(dtype=int),
            # Out:
            qacc_out: wp.array2d(dtype=float),
    ):
        worldid, nodeid = wp.tid()
        timestep = opt_timestep
        TILE_SIZE = wp.static(tile.size)

        dofid = adr_in[nodeid]
        M_tile = wp.tile_load(qM_in[worldid], shape=(TILE_SIZE, TILE_SIZE),
                              offset=(dofid, dofid))
        damping_tile = wp.tile_load(dof_damping, shape=(TILE_SIZE,),
                                    offset=(dofid,))
        damping_scaled = damping_tile * timestep
        qm_integration_tile = wp.tile_diag_add(M_tile, damping_scaled)

        Ma_tile = wp.tile_load(efc_Ma_in[worldid], shape=(TILE_SIZE,),
                               offset=(dofid,))
        L_tile = wp.tile_cholesky(qm_integration_tile)
        qacc_tile = wp.tile_cholesky_solve(L_tile, Ma_tile)
        wp.tile_store(qacc_out[worldid], qacc_tile, offset=(dofid))

    return euler_dense


@event_scope
def euler(m: Model, d: Data, dt: float):
    """
    Euler integrator, semi-implicit in velocity.
    Requires state derivative is set already
    """
    qacc = wp.empty((d.nworld, m.nv), dtype=float)
    for tile in m.qM_tiles:
        wp.launch_tiled(
            _tile_euler_dense(tile),
            dim=(d.nworld, tile.adr.size),
            inputs=[m.dof_damping, dt, d.qM, d.efc.Ma, tile.adr],
            outputs=[qacc],
            block_dim=m.block_dim.euler_dense,
        )
    _advance(m, d, qacc, d.qvel, 1.0)


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
        _next_position,
        dim=(d.nworld, m.nbody),
        inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, m.jnt_dofnum,
                qpos_t0, d.qvel, d.actual_step_size, scale],
        outputs=[d.qpos],
    )
    # velocity
    wp.launch(
        _next_velocity,
        dim=(d.nworld, m.nv),
        inputs=[qvel_t0, d.qacc, d.actual_step_size, scale],
        outputs=[d.qvel],
    )

    # muscles
    if m.nmuscle:
        wp.launch(
            _next_muscle_activation,
            dim=(d.nworld, m.nmuscle),
            inputs=[m.muscle_metadata, m_act_t0, d.m_act_dot,
                    d.actual_step_size, scale],
            outputs=[d.m_act],
        )
        if wp.static(m.opt.muscle_dyn_substeps) == 0:
            wp.launch(
                _next_muscle_state,
                dim=(d.nworld, m.nmuscle),
                inputs=[m.muscle_metadata, m_state_t0, d.m_state_dot,
                        d.actual_step_size, scale],
                outputs=[d.m_state],
            )

    if m.nactuator:
        wp.launch(
            _next_actuator_activation,
            dim=(d.nworld, m.nactuator),
            inputs=[m.actuator_metadata, a_act_t0, d.a_act_dot,
                    d.actual_step_size, scale],
            outputs=[d.a_act],
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
        forward.fwd(m, d, run_post=False)
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
    _advance(m, d, qacc_rk, qvel_rk, 1.0)
    wp.copy(d.qacc, qacc_rk)  # copy acceleration for post-step analysis
    return


@wp.kernel
def _set_fixed_step_size(
        # In:
        dt: float,
        # Data out:
        actual_step_size_out: wp.array(dtype=float),
):
    worldid = wp.tid()
    actual_step_size_out[worldid] = dt
    return


@event_scope
def euler_fixed(m: Model, d: Data, dt_sim: float):
    """Steps to [t + dt] using a fixed time step of [dt_sim] with Euler integrator."""
    wp.launch(
        _set_fixed_step_size,
        dim=d.nworld,
        inputs=[dt_sim],
        outputs=[d.actual_step_size],
    )
    euler(m, d, dt_sim)
    forward.fwd(m, d)  # realize state for next step


@event_scope
def rk4_fixed(m: Model, d: Data, dt_sim: float):
    """Steps to [t + dt] using a fixed time step of [dt_sim] with RK4."""
    wp.launch(
        _set_fixed_step_size,
        dim=d.nworld,
        inputs=[dt_sim],
        outputs=[d.actual_step_size],
    )
    rungekutta4(m, d)
    forward.fwd(m, d)  # realize for next step
