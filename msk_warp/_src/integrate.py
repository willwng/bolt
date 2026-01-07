import numpy as np
import warp as wp

from . import forward
from . import math
from . import mobilizers
from .consts import MJ_MINVAL
from .types import Data
from .types import Model
from .types import MuscleMetadata
from .types import ActuatorMetadata
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
def set_target_time(
        # Data in:
        time_in: wp.array(dtype=float),
        next_time_in: wp.array(dtype=float),
        step_size_in: wp.array(dtype=float),
        integration_done: wp.array(dtype=bool),
        # Data out:
        time1_out: wp.array(dtype=float),
        actual_step_size_out: wp.array(dtype=float),
        artificially_limited_out: wp.array(dtype=bool),
):
    worldid = wp.tid()
    if integration_done[worldid]:
        return

    t0 = time_in[worldid]
    t_max = next_time_in[worldid]
    current_step_size = step_size_in[worldid]
    artificially_limited_out[worldid] = False

    # If we lose more than a small fraction of the step size we wanted
    # to take (due to a need to stop at next_time/t_max), make a note so the
    # step size adjuster won't try to grow
    if t_max < t0 + 0.95 * current_step_size:
        artificially_limited_out[worldid] = True
        time1_out[worldid] = t_max  # t_max is much smaller than step size
    elif t_max > t0 + 1.001 * current_step_size:
        time1_out[worldid] = t0 + current_step_size  # t_max too big
    else:
        time1_out[worldid] = t_max  # roughly fits in a step, try for it

    # h = t1 - t0
    actual_step_size_out[worldid] = time1_out[worldid] - t0
    return


def _adjust_scales(m: Model, d: Data):
    @wp.func
    def calc_relative_scaling(abs_v: float, w: float) -> float:
        """
        Choose the current value as its scale when it is large enough,
        otherwise use absolute scale
        """
        return (1.0 / abs_v) if abs_v * w > 1.0 else w

    @wp.kernel
    def adjust_scales(
            # Model:
            qvel_weights: wp.array(dtype=float),
            # Data in:
            qvel_in: wp.array2d(dtype=float),
            # Out:
            qvel_scales_out: wp.array2d(dtype=float),
    ):
        worldid = wp.tid()
        nv = wp.static(m.nv)
        qvel_tile = wp.tile_load(qvel_in[worldid], shape=nv)
        qvel_weight_tile = wp.tile_load(qvel_weights, shape=nv)

        qvel_abs_tile = wp.tile_map(wp.abs, qvel_tile)
        qvel_scale_tile = wp.tile_map(calc_relative_scaling,
                                      qvel_abs_tile, qvel_weight_tile)

        wp.tile_store(qvel_scales_out[worldid], qvel_scale_tile)
        return

    wp.launch_tiled(
        adjust_scales,
        dim=d.nworld,
        inputs=[m.opt.qvel_weights, d.qvel],
        outputs=[d.qvel_scales],
        block_dim=m.block_dim.adjust_scales,
    )


@wp.kernel
def _check_done_integrating(
        # Data in:
        step_accepted_in: wp.array(dtype=bool),
        time1_in: wp.array(dtype=float),
        next_time_in: wp.array(dtype=float),
        # Data out:
        integration_done: wp.array(dtype=bool),
        nintegrating_out: wp.array(dtype=int),
):
    worldid = wp.tid()
    if not step_accepted_in[worldid] or integration_done[worldid]:
        return

    # Reached target time
    if time1_in[worldid] >= next_time_in[worldid]:
        integration_done[worldid] = True
        wp.atomic_add(nintegrating_out, 0, -1)


@wp.kernel
def _adjust_step_size(
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        step_size_in: wp.array(dtype=float),
        error_in: wp.array(dtype=float),
        artificially_limited_in: wp.array(dtype=bool),
        # In:
        safety: float,
        min_shrink: float,
        max_grow: float,
        hysteresis_low: float,
        hysteresis_high: float,
        accuracy: float,
        err_order: float,
        # Data out:
        step_size_out: wp.array(dtype=float),
        step_accepted_out: wp.array(dtype=bool),
):
    worldid = wp.tid()
    if integration_done_in[worldid]:
        step_accepted_out[worldid] = False
        return

    # Start with the actual step size taken
    curr_step_size = step_size_in[worldid]
    error = error_in[worldid]
    if wp.isinf(error) or wp.isnan(error):
        new_step_size = curr_step_size * min_shrink
    elif wp.abs(error) < MJ_MINVAL:
        new_step_size = curr_step_size * max_grow
    else:
        new_step_size = (safety * curr_step_size *
                         wp.pow(accuracy / error, 1.0 / err_order))
    # If the new step is bigger than the old, don't make the change if the
    #  old one was artificially limited or if the change
    #  would be very small
    if new_step_size > curr_step_size:
        if (artificially_limited_in[worldid] or
                new_step_size < hysteresis_high * curr_step_size):
            new_step_size = curr_step_size

    # If we're supposed to shrink the step size but the one we have actually
    # achieved the desired accuracy last time, we won't change the step now.
    # Otherwise, if we are going to shrink the step
    if new_step_size < curr_step_size:
        if error <= accuracy:
            new_step_size = curr_step_size
        else:
            new_step_size = min(new_step_size, hysteresis_low * curr_step_size)

    # Keep the size change within the allowable bounds
    new_step_size = min(new_step_size, max_grow * curr_step_size)
    new_step_size = max(new_step_size, min_shrink * curr_step_size)
    step_size_out[worldid] = new_step_size
    # This is an odd definition of success
    step_accepted_out[worldid] = (new_step_size >= curr_step_size)
    return


def _save_state(m: Model, d: Data, save_id: int, ):
    """ Saves current state into [integrator_state] at index [save_id] """

    @wp.kernel
    def save_state(
            # Data In:
            time_in: wp.array(dtype=float),
            qpos_in: wp.array2d(dtype=float),
            qvel_in: wp.array2d(dtype=float),
            mstate_in: wp.array2d(dtype=float),
            act_in: wp.array2d(dtype=float),
            # In:
            sid: int,
            # Out:
            time_out: wp.array2d(dtype=float),
            qpos_out: wp.array3d(dtype=float),
            qvel_out: wp.array3d(dtype=float),
            mstate_out: wp.array3d(dtype=float),
            act_out: wp.array3d(dtype=float),
    ):
        worldid = wp.tid()
        nq, nv, nmuscle = wp.static(m.nq), wp.static(m.nv), wp.static(m.nmuscle)

        time_out[worldid, sid] = time_in[worldid]

        qpos_tile = wp.tile_load(qpos_in[worldid], shape=nq)
        wp.tile_store(qpos_out[worldid, sid], qpos_tile)

        qvel_tile = wp.tile_load(qvel_in[worldid], shape=nv)
        wp.tile_store(qvel_out[worldid, sid], qvel_tile)

        if nmuscle:
            mstate_tile = wp.tile_load(mstate_in[worldid], shape=nmuscle)
            wp.tile_store(mstate_out[worldid, sid], mstate_tile)

            act_tile = wp.tile_load(act_in[worldid], shape=nmuscle)
            wp.tile_store(act_out[worldid, sid], act_tile)
        return

    wp.launch_tiled(
        save_state,
        dim=d.nworld,
        inputs=[d.time, d.qpos, d.qvel, d.m_state, d.m_act, save_id],
        outputs=[d.integrator_state.time, d.integrator_state.qpos,
                 d.integrator_state.qvel, d.integrator_state.mstate,
                 d.integrator_state.act],
        block_dim=m.block_dim.error_step,
    )


def _restore_state(m: Model, d: Data, restore_id: int, reject_only: bool):
    """ Restores state from [integrator_state] at index [restore_id]"""

    # @nested_kernel(module="unique", enable_backward=False)
    @wp.kernel
    def restore_state(
            # Data in
            time_in: wp.array2d(dtype=float),
            qpos_in: wp.array3d(dtype=float),
            qvel_in: wp.array3d(dtype=float),
            mstate_in: wp.array3d(dtype=float),
            act_in: wp.array3d(dtype=float),
            step_accepted_in: wp.array(dtype=bool),
            # In:
            lid: int,
            # Data out:
            time_out: wp.array(dtype=float),
            qpos_out: wp.array2d(dtype=float),
            qvel_out: wp.array2d(dtype=float),
            mstate_out: wp.array2d(dtype=float),
            act_out: wp.array2d(dtype=float),
    ):
        worldid = wp.tid()
        if wp.static(reject_only) and step_accepted_in[worldid]:
            return
        nq, nv, nmuscle = wp.static(m.nq), wp.static(m.nv), wp.static(m.nmuscle)

        time_out[worldid] = time_in[worldid, lid]

        qpos_tile = wp.tile_load(qpos_in[worldid, lid], shape=nq)
        wp.tile_store(qpos_out[worldid], qpos_tile)

        qvel_tile = wp.tile_load(qvel_in[worldid, lid], shape=nv)
        wp.tile_store(qvel_out[worldid], qvel_tile)

        if nmuscle:
            mstate_tile = wp.tile_load(mstate_in[worldid, lid], shape=nmuscle)
            wp.tile_store(mstate_out[worldid], mstate_tile)

            act_tile = wp.tile_load(act_in[worldid, lid], shape=nmuscle)
            wp.tile_store(act_out[worldid], act_tile)

    wp.launch_tiled(
        restore_state,
        dim=d.nworld,
        inputs=[d.integrator_state.time, d.integrator_state.qpos,
                d.integrator_state.qvel, d.integrator_state.mstate,
                d.integrator_state.act, d.step_accepted, restore_id],
        outputs=[d.time, d.qpos, d.qvel, d.m_state, d.m_act],
        block_dim=m.block_dim.error_step,
    )


def _compute_error(m: Model, d: Data, compare_id: int):
    """
    Computes error between current state and stored state at [compare_id].
    Then decides whether to accept the step
    """

    @wp.kernel
    def compute_diffs(
            # Data in:
            qpos_in: wp.array2d(dtype=float),
            qvel_in: wp.array2d(dtype=float),
            qpos_store_in: wp.array3d(dtype=float),
            qvel_store_in: wp.array3d(dtype=float),
            # In
            cid: int,
            # Out:
            qpos_diff_out: wp.array2d(dtype=float),
            qvel_diff_out: wp.array2d(dtype=float),
    ):
        worldid = wp.tid()
        nq, nv, nmuscle = wp.static(m.nq), wp.static(m.nv), wp.static(m.nmuscle)

        # q_curr - q_stored
        qpos_tile = wp.tile_load(qpos_in[worldid], nq)
        qpos_s_tile = wp.tile_load(qpos_store_in[worldid, cid], nq)
        q_diff_tile = wp.tile_map(wp.sub, qpos_tile, qpos_s_tile)
        wp.tile_store(qpos_diff_out[worldid], q_diff_tile)

        # qvel_curr - qvel_stored
        qvel_tile = wp.tile_load(qvel_in[worldid], nv)
        qvel_s_tile = wp.tile_load(qvel_store_in[worldid, cid], nv)
        qvel_diff_tile = wp.tile_map(wp.sub, qvel_tile, qvel_s_tile)
        wp.tile_store(qvel_diff_out[worldid], qvel_diff_tile)
        return

    @wp.kernel
    def compute_qpos_error(
            # Data in:
            qpos_diff_in: wp.array2d(dtype=float),
            # Out:
            qpos_error_out: wp.array(dtype=float),
    ):
        worldid = wp.tid()
        nq = wp.static(m.nq)

        # qpos is already scaled
        qpos_diff_tile = wp.tile_load(qpos_diff_in[worldid], nq)

        # Error
        if wp.static(m.opt.use_inf_norm):
            qpos_scaled_diff_abs = wp.tile_map(wp.abs, qpos_diff_tile)
            q_err = wp.tile_max(qpos_scaled_diff_abs)[0]
        else:
            qpos_scaled_diff_sq = wp.tile_map(math.sqr, qpos_diff_tile)
            q_err = wp.sqrt(wp.tile_sum(qpos_scaled_diff_sq)[0] / float(nq))

        qpos_error_out[worldid] = q_err
        return

    # @nested_kernel(module="unique", enable_backward=False)
    @wp.kernel
    def compute_qvel_error(
            # Data in:
            qvel_diff_in: wp.array2d(dtype=float),
            qvel_scales_in: wp.array2d(dtype=float),
            # Out:
            qvel_error_out: wp.array(dtype=float),
    ):
        worldid = wp.tid()
        nv = wp.static(m.nv)

        # Multiply (qvel_diff) by scales
        qvel_diff_tile = wp.tile_load(qvel_diff_in[worldid], nv)
        qvel_scales_tile = wp.tile_load(qvel_scales_in[worldid], nv)
        qvel_scaled_diff_tile = wp.tile_map(
            wp.mul, qvel_diff_tile, qvel_scales_tile)

        # Error
        if wp.static(m.opt.use_inf_norm):
            qvel_scaled_diff_abs = wp.tile_map(wp.abs, qvel_scaled_diff_tile)
            qv_err = wp.tile_max(qvel_scaled_diff_abs)[0]
        else:
            qvel_scaled_diff_sq = wp.tile_map(math.sqr, qvel_scaled_diff_tile)
            qv_err = wp.sqrt(wp.tile_sum(qvel_scaled_diff_sq)[0] / float(nv))

        qvel_error_out[worldid] = qv_err
        return

    @wp.kernel
    def compute_error(
            # Data in:
            qpos_error_in: wp.array(dtype=float),
            qvel_error_in: wp.array(dtype=float),
            # Out:
            error_out: wp.array(dtype=float),
    ):
        worldid = wp.tid()
        error_out[worldid] = wp.max(qpos_error_in[worldid],
                                    qvel_error_in[worldid])

    wp.launch_tiled(
        compute_diffs,
        dim=d.nworld,
        inputs=[d.qpos, d.qvel,
                d.integrator_state.qpos, d.integrator_state.qvel,
                compare_id],
        outputs=[d.qpos_diff, d.qvel_diff],
        block_dim=m.block_dim.error_step,
    )

    mobilizers.scale_dq(m, d, d.qpos_diff, d.qpos_diff_scaled)
    wp.launch_tiled(
        compute_qpos_error,
        dim=d.nworld,
        inputs=[d.qpos_diff_scaled],
        outputs=[d.qpos_error],
        block_dim=m.block_dim.error_step,
    )

    wp.launch_tiled(
        compute_qvel_error,
        dim=d.nworld,
        inputs=[d.qvel_diff, d.qvel_scales],
        outputs=[d.qvel_error],
        block_dim=m.block_dim.error_step,
    )

    wp.launch(
        compute_error,
        dim=d.nworld,
        inputs=[d.qpos_error, d.qvel_error],
        outputs=[d.error],
    )


@event_scope
def attempt_step(m: Model, d: Data):
    # Set the target time to integrate to
    wp.launch(
        set_target_time,
        dim=d.nworld,
        inputs=[d.time, d.next_time, d.step_size, d.integration_done],
        outputs=[d.time1, d.actual_step_size, d.artificially_limited],
    )

    # Adjust scales for error computation
    _adjust_scales(m, d)

    # Save state y_0, note the derivative y_0' is already available
    _save_state(m, d, save_id=0)

    # Big step using full current step size, store y_1
    _advance(m, d, d.qacc, d.qvel, 1.0)
    _save_state(m, d, save_id=1)

    # Restore y_0. Note: advance doesn't modify state derivatives (except qvel)
    # so we can reuse y_0'
    _restore_state(m, d, restore_id=0, reject_only=False)
    _advance(m, d, d.qacc, d.qvel, 0.5)
    # Half-step 2: y_1* = y_{1/2} + dt/2 * y_{1/2}'
    forward.fwd(m, d)
    _advance(m, d, d.qacc, d.qvel, 0.5)

    # Compute error with y_1 from big step
    _compute_error(m, d, compare_id=1)

    # Compute new step size and reject step here if accuracy isn't good enough
    wp.launch(
        _adjust_step_size,
        dim=d.nworld,
        inputs=[
            d.integration_done,
            d.step_size,
            d.error,
            d.artificially_limited,
            m.opt.safety,
            m.opt.min_shrink,
            m.opt.max_grow,
            m.opt.hysteresis_low,
            m.opt.hysteresis_high,
            m.opt.accuracy,
            2.0,  # err_order
        ],
        outputs=[d.step_size, d.step_accepted],
    )

    # Restore state if step was rejected, otherwise keep y_1*
    _restore_state(m, d, restore_id=0, reject_only=True)

    wp.launch(
        _check_done_integrating,
        dim=d.nworld,
        inputs=[d.step_accepted, d.time1, d.next_time],
        outputs=[d.integration_done, d.nintegrating],
    )

    # Prepare derivatives for next attempt
    forward.fwd(m, d)
    return


@wp.kernel
def _start_integrating(
        # In:
        dt: float,
        # Data out:
        next_time_out: wp.array(dtype=float),
        integration_done_out: wp.array(dtype=bool),
):
    worldid = wp.tid()
    next_time_out[worldid] += dt
    integration_done_out[worldid] = False


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
def step_to_adaptive(m: Model, d: Data, dt: float):
    wp.launch(
        _start_integrating,
        dim=d.nworld,
        inputs=[dt],
        outputs=[d.next_time, d.integration_done],
    )

    d.nintegrating.fill_(d.nworld)
    wp.capture_while(
        d.nintegrating,
        while_body=attempt_step,
        m=m,
        d=d,
    )


@event_scope
def euler_fixed(m: Model, d: Data, dt: float, dt_sim: float):
    """Steps to [t + dt] using a fixed time step of [dt_sim] with Euler integrator."""
    num_substeps = np.ceil(dt / dt_sim)
    wp.launch(
        _set_fixed_step_size,
        dim=d.nworld,
        inputs=[dt_sim],
        outputs=[d.actual_step_size],
    )
    for _step in range(int(num_substeps)):
        euler(m, d, dt)
        forward.post(m, d)
        forward.fwd(m, d)


@event_scope
def rk4_fixed(m: Model, d: Data, dt: float, dt_sim: float):
    """Steps to [t + dt] using a fixed time step of [dt_sim] with RK4."""
    num_substeps = np.ceil(dt / dt_sim)
    wp.launch(
        _set_fixed_step_size,
        dim=d.nworld,
        inputs=[dt_sim],
        outputs=[d.actual_step_size],
    )
    for _step in range(int(num_substeps)):
        rungekutta4(m, d)
        forward.post(m, d)
        forward.fwd(m, d)  # realize for next step
