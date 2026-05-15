import warp as wp

from . import math
from . import mobilizers
from .consts import BOLT_MINVAL
from .types import Data
from .types import IntegratorStateScratch
from .types import Model
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _determine_current_target_time(
        # Data in:
        time_in: wp.array(dtype=float),
        next_time_in: wp.array(dtype=float),
        step_size_in: wp.array(dtype=float),
        integration_done: wp.array(dtype=bool),
        # Data out:
        time1_out: wp.array(dtype=float),
        actual_step_size_out: wp.array(dtype=float),
        artificially_limited_out: wp.array(dtype=bool),
        steps_attempted_out: wp.array(dtype=int),
):
    worldid = wp.tid()
    if integration_done[worldid]:
        actual_step_size_out[worldid] = 0.0
        return

    t0 = time_in[worldid]
    t_max = next_time_in[worldid]
    current_step_size = step_size_in[worldid]
    artificially_limited_out[worldid] = False
    steps_attempted_out[worldid] += 1

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


@event_scope
def choose_target_time(m: Model, d: Data):
    wp.launch(
        _determine_current_target_time,
        dim=d.nworld,
        inputs=[d.time, d.next_time, d.step_size, d.integration_done],
        outputs=[d.time1, d.actual_step_size, d.artificially_limited, d.steps_attempted],
    )


def adjust_err_scales(m: Model, d: Data):
    @wp.func
    def calc_relative_scaling(abs_v: float, w: float) -> float:
        """
        Choose the current value as its scale when it is large enough,
        otherwise use absolute scale
        """
        return (1.0 / abs_v) if abs_v * w > 1.0 else w

    @wp.kernel
    def adjust_qvel_scales(
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
        qvel_scale_tile = wp.tile_map(calc_relative_scaling, qvel_abs_tile, qvel_weight_tile)
        wp.tile_store(qvel_scales_out[worldid], qvel_scale_tile)
        return

    @wp.kernel
    def adjust_z_scales(
            # Model:
            z_weights: wp.array(dtype=float),
            # Data in:
            m_state_in: wp.array2d(dtype=float),
            m_act_in: wp.array2d(dtype=float),
            a_act_in: wp.array2d(dtype=float),
            # Out:
            z_scales_out: wp.array2d(dtype=float),
    ):
        worldid = wp.tid()
        nm, na = wp.static(m.nmuscle), wp.static(m.nactuator)
        if nm:
            # muscle state
            m_state_tile = wp.tile_load(m_state_in[worldid], shape=nm)
            m_state_weight_tile = wp.tile_load(z_weights, shape=nm, offset=0)
            m_state_abs_tile = wp.tile_map(wp.abs, m_state_tile)
            m_state_scale_tile = wp.tile_map(calc_relative_scaling, m_state_abs_tile, m_state_weight_tile)
            wp.tile_store(z_scales_out[worldid], m_state_scale_tile, offset=(0,))
            # muscle activation
            m_act_tile = wp.tile_load(m_act_in[worldid], shape=nm)
            m_act_weight_tile = wp.tile_load(z_weights, shape=nm, offset=nm)
            m_act_abs_tile = wp.tile_map(wp.abs, m_act_tile)
            m_act_scale_tile = wp.tile_map(calc_relative_scaling, m_act_abs_tile, m_act_weight_tile)
            wp.tile_store(z_scales_out[worldid], m_act_scale_tile, offset=nm)
        if na:
            # actuator activation
            a_act_tile = wp.tile_load(a_act_in[worldid], shape=na)
            a_act_weight_tile = wp.tile_load(z_weights, shape=na, offset=nm + nm)
            a_act_abs_tile = wp.tile_map(wp.abs, a_act_tile)
            a_act_scale_tile = wp.tile_map(calc_relative_scaling, a_act_abs_tile, a_act_weight_tile)
            wp.tile_store(z_scales_out[worldid], a_act_scale_tile, offset=nm + nm)
        return

    wp.launch_tiled(
        adjust_qvel_scales,
        dim=d.nworld,
        inputs=[m.opt.qvel_weights, d.qvel],
        outputs=[d.qvel_scales],
        block_dim=m.block_dim.adjust_scales,
    )
    wp.launch_tiled(
        adjust_z_scales,
        dim=d.nworld,
        inputs=[m.opt.z_weights, d.m_state, d.m_act, d.a_act],
        outputs=[d.z_scales],
        block_dim=m.block_dim.adjust_scales,
    )


@wp.kernel
def _check_done_integrating(
        # Data in:
        step_accepted_in: wp.array(dtype=bool),
        time1_in: wp.array(dtype=float),
        next_time_in: wp.array(dtype=float),
        # Data out:
        time_out: wp.array(dtype=float),
        integration_done: wp.array(dtype=bool),
        nintegrating_out: wp.array(dtype=int),
):
    worldid = wp.tid()
    if not step_accepted_in[worldid] or integration_done[worldid]:
        return

    # Reached target time. Need to compare time1 (in case of floating point imprecision)
    if time1_in[worldid] >= next_time_in[worldid]:
        time_out[worldid] = next_time_in[worldid]  # this shouldn't be needed, but good for precision
        integration_done[worldid] = True
        wp.atomic_add(nintegrating_out, 0, -1)


@event_scope
def check_done_integrating(m: Model, d: Data):
    wp.launch(
        _check_done_integrating,
        dim=d.nworld,
        inputs=[d.step_accepted, d.time1, d.next_time],
        outputs=[d.time, d.integration_done, d.nintegrating],
    )


def compute_error(m: Model, d: Data, scratch: IntegratorStateScratch, scale: float = 1.0):
    """ Computes error of current state with given state. Stores error in d.error. """

    @wp.kernel
    def compute_diffs(
            # Data in:
            qpos_in: wp.array2d(dtype=float),
            qvel_in: wp.array2d(dtype=float),
            m_state_in: wp.array2d(dtype=float),
            m_act_in: wp.array2d(dtype=float),
            a_act_in: wp.array2d(dtype=float),

            qpos_store_in: wp.array2d(dtype=float),
            qvel_store_in: wp.array2d(dtype=float),
            m_state_store_in: wp.array2d(dtype=float),
            m_act_store_in: wp.array2d(dtype=float),
            a_act_store_in: wp.array2d(dtype=float),
            # Out:
            qpos_diff_out: wp.array2d(dtype=float),
            qvel_diff_out: wp.array2d(dtype=float),
            z_diff_out: wp.array2d(dtype=float),
    ):
        worldid = wp.tid()
        nq, nv, nm, na = wp.static(m.nq), wp.static(m.nv), wp.static(m.nmuscle), wp.static(m.nactuator)

        # q_curr - q_stored
        qpos_tile = wp.tile_load(qpos_in[worldid], nq)
        qpos_s_tile = wp.tile_load(qpos_store_in[worldid], nq)
        q_diff_tile = scale * wp.tile_map(wp.sub, qpos_tile, qpos_s_tile)
        wp.tile_store(qpos_diff_out[worldid], q_diff_tile)

        # qvel_curr - qvel_stored
        qvel_tile = wp.tile_load(qvel_in[worldid], nv)
        qvel_s_tile = wp.tile_load(qvel_store_in[worldid], nv)
        qvel_diff_tile = scale * wp.tile_map(wp.sub, qvel_tile, qvel_s_tile)
        wp.tile_store(qvel_diff_out[worldid], qvel_diff_tile)
        if nm:
            # m_state_curr - m_state_stored
            m_state_tile = wp.tile_load(m_state_in[worldid], nm)
            m_state_s_tile = wp.tile_load(m_state_store_in[worldid], nm)
            m_state_diff_tile = scale * wp.tile_map(wp.sub, m_state_tile, m_state_s_tile)
            wp.tile_store(z_diff_out[worldid], m_state_diff_tile, offset=(0,))
            # m_act_curr - m_act_stored
            m_act_tile = wp.tile_load(m_act_in[worldid], nm)
            m_act_s_tile = wp.tile_load(m_act_store_in[worldid], nm)
            m_act_diff_tile = scale * wp.tile_map(wp.sub, m_act_tile, m_act_s_tile)
            wp.tile_store(z_diff_out[worldid], m_act_diff_tile, offset=(nm,))
        if na:
            # a_act_curr - a_act_stored
            a_act_tile = wp.tile_load(a_act_in[worldid], na)
            a_act_s_tile = wp.tile_load(a_act_store_in[worldid], na)
            a_act_diff_tile = scale * wp.tile_map(wp.sub, a_act_tile, a_act_s_tile)
            wp.tile_store(z_diff_out[worldid], a_act_diff_tile, offset=(nm + nm,))
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
        qpos_diff_tile = wp.tile_load(qpos_diff_in[worldid], nq)
        if wp.static(m.opt.use_inf_norm):
            qpos_scaled_diff_abs = wp.tile_map(wp.abs, qpos_diff_tile)
            q_err = wp.tile_max(qpos_scaled_diff_abs)[0]
        else:
            qpos_scaled_diff_sq = wp.tile_map(math.sqr, qpos_diff_tile)
            q_err = wp.sqrt(wp.tile_sum(qpos_scaled_diff_sq)[0] / float(nq))
        qpos_error_out[worldid] = q_err
        return

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
        # Multiply qvel_diff by scales
        qvel_diff_tile = wp.tile_load(qvel_diff_in[worldid], nv)
        qvel_scales_tile = wp.tile_load(qvel_scales_in[worldid], nv)
        qvel_scaled_diff_tile = wp.tile_map(wp.mul, qvel_diff_tile, qvel_scales_tile)
        # inf-norm or L2 norm error
        if wp.static(m.opt.use_inf_norm):
            qvel_scaled_diff_abs = wp.tile_map(wp.abs, qvel_scaled_diff_tile)
            qv_err = wp.tile_max(qvel_scaled_diff_abs)[0]
        else:
            qvel_scaled_diff_sq = wp.tile_map(math.sqr, qvel_scaled_diff_tile)
            qv_err = wp.sqrt(wp.tile_sum(qvel_scaled_diff_sq)[0] / float(nv))
        qvel_error_out[worldid] = qv_err
        return

    @wp.kernel
    def compute_z_error(
            # Data in:
            z_diff_in: wp.array2d(dtype=float),
            z_scales_in: wp.array2d(dtype=float),
            # Out:
            z_error_out: wp.array(dtype=float),
    ):
        worldid = wp.tid()
        nz = wp.static(m.nz)
        if nz:
            # Multiply qvel_diff by scales
            z_diff_tile = wp.tile_load(z_diff_in[worldid], nz)
            z_scales_tile = wp.tile_load(z_scales_in[worldid], nz)
            z_scaled_diff_tile = wp.tile_map(wp.mul, z_diff_tile, z_scales_tile)
            # Error
            if wp.static(m.opt.use_inf_norm):
                z_diff_abs = wp.tile_map(wp.abs, z_scaled_diff_tile)
                z_err = wp.tile_max(z_diff_abs)[0]
            else:
                z_diff_sq = wp.tile_map(math.sqr, z_scaled_diff_tile)
                z_err = wp.sqrt(wp.tile_sum(z_diff_sq)[0] / float(nz))
            z_error_out[worldid] = z_err
        return

    @wp.kernel
    def aggregate_errors(
            # Data in:
            integration_done: wp.array(dtype=bool),
            qpos_error_in: wp.array(dtype=float),
            qvel_error_in: wp.array(dtype=float),
            z_error_in: wp.array(dtype=float),
            # Out:
            error_out: wp.array(dtype=float),
    ):
        worldid = wp.tid()
        if integration_done[worldid]:
            error_out[worldid] = 0.0
            return
        error = qpos_error_in[worldid]
        error = math.max_err(error, qvel_error_in[worldid])
        error = math.max_err(error, z_error_in[worldid])
        error_out[worldid] = error
        return

    wp.launch_tiled(
        compute_diffs,
        dim=d.nworld,
        inputs=[
            d.qpos, d.qvel, d.m_state, d.m_act, d.a_act,
            scratch.qpos, scratch.qvel, scratch.m_state, scratch.m_act, scratch.a_act,
        ],
        outputs=[d.qpos_diff, d.qvel_diff, d.z_diff, ],
        block_dim=m.block_dim.error_step,
    )

    # qpos error needs to be scaled by the qvel weights
    mobilizers.scale_dq(m, d, d.qpos_diff, d.qpos_diff_scaled)
    wp.launch_tiled(
        compute_qpos_error,
        dim=d.nworld,
        inputs=[d.qpos_diff_scaled],
        outputs=[d.qpos_err],
        block_dim=m.block_dim.error_step,
    )
    wp.launch_tiled(
        compute_qvel_error,
        dim=d.nworld,
        inputs=[d.qvel_diff, d.qvel_scales],
        outputs=[d.qvel_err],
        block_dim=m.block_dim.error_step,
    )
    wp.launch_tiled(
        compute_z_error,
        dim=d.nworld,
        inputs=[d.z_diff, d.z_scales],
        outputs=[d.z_err],
        block_dim=m.block_dim.error_step,
    )
    wp.launch(
        aggregate_errors,
        dim=d.nworld,
        inputs=[d.integration_done, d.qpos_err, d.qvel_err, d.z_err],
        outputs=[d.error],
    )
    return


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
        min_step_size: float,
        max_step_size: float,
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
        step_accepted_out[worldid] = True
        return

    # Start with the actual step size taken
    curr_step_size = step_size_in[worldid]
    error = error_in[worldid]
    if wp.isinf(error) or wp.isnan(error):
        new_step_size = curr_step_size * min_shrink
    elif wp.abs(error) < BOLT_MINVAL:
        new_step_size = curr_step_size * max_grow
    else:
        new_step_size = (safety * curr_step_size * wp.pow(accuracy / error, 1.0 / err_order))
    # If the new step is bigger than the old, don't make the change if the
    #  old one was artificially limited or if the change
    #  would be very small
    if new_step_size > curr_step_size:
        if artificially_limited_in[worldid] or new_step_size < hysteresis_high * curr_step_size:
            new_step_size = curr_step_size

    # If we're supposed to shrink the step size but the one we have actually
    # achieved the desired accuracy last time, we won't change the step now.
    # Otherwise, if we are going to shrink the step
    if new_step_size < curr_step_size:
        if error <= accuracy:
            new_step_size = curr_step_size
        else:
            new_step_size = wp.min(new_step_size, hysteresis_low * curr_step_size)

    # Keep the size change within the allowable bounds
    new_step_size = wp.min(new_step_size, max_grow * curr_step_size)
    new_step_size = wp.max(new_step_size, min_shrink * curr_step_size)
    new_step_size = wp.clamp(new_step_size, min_step_size, max_step_size)
    step_size_out[worldid] = new_step_size
    # This is an odd definition of success
    step_accepted_out[worldid] = (new_step_size >= curr_step_size)
    return


@event_scope
def adjust_step_size(m: Model, d: Data, err_order: float):
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
            m.opt.min_step_size,
            m.opt.max_step_size,
            m.opt.hysteresis_low,
            m.opt.hysteresis_high,
            m.opt.accuracy,
            err_order
        ],
        outputs=[d.step_size, d.step_accepted],
    )


@event_scope
def save_state(
        m: Model, d: Data,
        time_dest: wp.array(dtype=float),
        qpos_dest: wp.array2d(dtype=float),
        qvel_dest: wp.array2d(dtype=float),
        m_state_dest: wp.array2d(dtype=float),
        m_act_dest: wp.array2d(dtype=float),
        a_act_dest: wp.array2d(dtype=float),
        exp_contact_state_dest: wp.array2d(dtype=wp.vec4),
):
    wp.copy(time_dest, d.time)
    wp.copy(qpos_dest, d.qpos)
    wp.copy(qvel_dest, d.qvel)
    if m.nmuscle:
        wp.copy(m_act_dest, d.m_act)
        wp.copy(m_state_dest, d.m_state)
    if m.nactuator:
        wp.copy(a_act_dest, d.a_act)
    if m.nexpcontact:
        wp.copy(exp_contact_state_dest, d.exp_contact_state)


@event_scope
def save_state_dot(
        m: Model, d: Data,
        qvel_dest: wp.array2d(dtype=float),
        qacc_dest: wp.array2d(dtype=float),
        m_state_dot_dest: wp.array2d(dtype=float),
        m_act_dot_dest: wp.array2d(dtype=float),
        a_act_dot_dest: wp.array2d(dtype=float)
):
    wp.copy(qvel_dest, d.qvel)
    wp.copy(qacc_dest, d.qacc)
    if m.nmuscle:
        wp.copy(m_state_dot_dest, d.m_state_dot)
        wp.copy(m_act_dot_dest, d.m_act_dot)
    if m.nactuator:
        wp.copy(a_act_dot_dest, d.a_act_dot)


@event_scope
def restore_state_dot(
        m: Model, d: Data,
        qvel_src: wp.array2d(dtype=float),
        qacc_src: wp.array2d(dtype=float),
        m_state_dot_src: wp.array2d(dtype=float),
        m_act_dot_src: wp.array2d(dtype=float),
        a_act_dot_src: wp.array2d(dtype=float),
        only_on_reject: bool
):
    @wp.kernel
    def restore_state_dot_conditional(
            # Data in
            done_integrating_in: wp.array(dtype=bool),
            step_accepted_in: wp.array(dtype=bool),
            qvel_in: wp.array2d(dtype=float),
            qacc_in: wp.array2d(dtype=float),
            m_state_dot_in: wp.array2d(dtype=float),
            m_act_dot_in: wp.array2d(dtype=float),
            a_act_dot_in: wp.array2d(dtype=float),
            # Data out:
            qvel_out: wp.array2d(dtype=float),
            qacc_out: wp.array2d(dtype=float),
            m_state_dot_out: wp.array2d(dtype=float),
            m_act_dot_out: wp.array2d(dtype=float),
            a_act_dot_out: wp.array2d(dtype=float),
    ):
        worldid = wp.tid()
        if step_accepted_in[worldid] or done_integrating_in[worldid]:
            return
        nv, nm, na = wp.static(m.nv), wp.static(m.nmuscle), wp.static(m.nactuator)
        wp.tile_store(qvel_out[worldid], wp.tile_load(qvel_in[worldid], shape=(nv,)))
        wp.tile_store(qacc_out[worldid], wp.tile_load(qacc_in[worldid], shape=(nv,)))
        if nm:
            wp.tile_store(m_state_dot_out[worldid], wp.tile_load(m_state_dot_in[worldid], shape=(nm,)))
            wp.tile_store(m_act_dot_out[worldid], wp.tile_load(m_act_dot_in[worldid], shape=(nm,)))
        if na:
            wp.tile_store(a_act_dot_out[worldid], wp.tile_load(a_act_dot_in[worldid], shape=(na,)))
        return

    if only_on_reject:
        wp.launch_tiled(
            restore_state_dot_conditional,
            dim=d.nworld,
            inputs=[d.integration_done, d.step_accepted,
                    qvel_src, qacc_src, m_state_dot_src, m_act_dot_src, a_act_dot_src],
            outputs=[d.qvel, d.qacc, d.m_state_dot, d.m_act_dot, d.a_act_dot],
            block_dim=m.block_dim.restore_state,
        )
    else:
        wp.copy(d.qvel, qvel_src)
        wp.copy(d.qacc, qacc_src)
        if m.nmuscle:
            wp.copy(d.m_state_dot, m_state_dot_src)
            wp.copy(d.m_act_dot, m_act_dot_src)
        if m.nactuator:
            wp.copy(d.a_act_dot, a_act_dot_src)


@event_scope
def restore_state(
        m: Model,
        d: Data,
        time_src: wp.array,
        qpos_src: wp.array2d,
        qvel_src: wp.array2d,
        m_state_src: wp.array2d,
        m_act_src: wp.array2d,
        a_act_src: wp.array2d,
        exp_contact_state_src: wp.array2d(dtype=wp.vec4),
        only_on_reject: bool
):
    @wp.kernel
    def restore_state_conditional(
            # Data in
            done_integrating_in: wp.array(dtype=bool),
            step_accepted_in: wp.array(dtype=bool),
            time_in: wp.array(dtype=float),
            qpos_in: wp.array2d(dtype=float),
            qvel_in: wp.array2d(dtype=float),
            m_state_in: wp.array2d(dtype=float),
            m_act_in: wp.array2d(dtype=float),
            a_act_in: wp.array2d(dtype=float),
            exp_contact_state_in: wp.array2d(dtype=wp.vec4),
            # Data out:
            time_out: wp.array(dtype=float),
            qpos_out: wp.array2d(dtype=float),
            qvel_out: wp.array2d(dtype=float),
            m_state_out: wp.array2d(dtype=float),
            m_act_out: wp.array2d(dtype=float),
            a_act_out: wp.array2d(dtype=float),
            exp_contact_state_out: wp.array2d(dtype=wp.vec4),
    ):
        worldid = wp.tid()
        if step_accepted_in[worldid] or done_integrating_in[worldid]:
            return
        nq, nv, nm, na = wp.static(m.nq), wp.static(m.nv), wp.static(m.nmuscle), wp.static(m.nactuator)
        nexp = wp.static(m.nexpcontact)
        time_out[worldid] = time_in[worldid]

        wp.tile_store(qpos_out[worldid], wp.tile_load(qpos_in[worldid], shape=(nq,)))
        wp.tile_store(qvel_out[worldid], wp.tile_load(qvel_in[worldid], shape=(nv,)))
        if nm:
            wp.tile_store(m_state_out[worldid], wp.tile_load(m_state_in[worldid], shape=(nm,)))
            wp.tile_store(m_act_out[worldid], wp.tile_load(m_act_in[worldid], shape=(nm,)))
        if na:
            wp.tile_store(a_act_out[worldid], wp.tile_load(a_act_in[worldid], shape=(na,)))
        if nexp:
            wp.tile_store(exp_contact_state_out[worldid], wp.tile_load(exp_contact_state_in[worldid], shape=(nexp,)))

    if only_on_reject:
        wp.launch_tiled(
            restore_state_conditional,
            dim=d.nworld,
            inputs=[d.integration_done, d.step_accepted,
                    time_src, qpos_src, qvel_src, m_state_src, m_act_src, a_act_src, exp_contact_state_src],
            outputs=[d.time, d.qpos, d.qvel, d.m_state, d.m_act, d.a_act, d.exp_contact_state],
            block_dim=m.block_dim.restore_state,
        )
    else:  # everyone gets restored!
        wp.copy(d.time, time_src)
        wp.copy(d.qpos, qpos_src)
        wp.copy(d.qvel, qvel_src)
        if m.nmuscle:
            wp.copy(d.m_act, m_act_src)
            wp.copy(d.m_state, m_state_src)
        if m.nactuator:
            wp.copy(d.a_act, a_act_src)
        if m.nexpcontact:
            wp.copy(d.exp_contact_state, exp_contact_state_src)


@event_scope
def add_to_state_dot(
        m: Model, d: Data,
        scale: float,
        qvel_add: wp.array2d,
        qacc_add: wp.array2d,
        m_state_dot_add: wp.array2d,
        m_act_dot_add: wp.array2d,
        a_act_dot_add: wp.array2d,
):
    @wp.kernel
    def _add_to_state_dot(
            # Data in
            done_integrating_in: wp.array(dtype=bool),
            qvel_in: wp.array2d(dtype=float),
            qacc_in: wp.array2d(dtype=float),
            m_state_dot_in: wp.array2d(dtype=float),
            m_act_dot_in: wp.array2d(dtype=float),
            a_act_dot_in: wp.array2d(dtype=float),
            # Data out:
            qvel_out: wp.array2d(dtype=float),
            qacc_out: wp.array2d(dtype=float),
            m_state_dot_out: wp.array2d(dtype=float),
            m_act_dot_out: wp.array2d(dtype=float),
            a_act_dot_out: wp.array2d(dtype=float),
    ):
        worldid = wp.tid()
        if done_integrating_in[worldid]:
            return
        nv, nm, na = wp.static(m.nv), wp.static(m.nmuscle), wp.static(m.nactuator)

        qv_og = wp.tile_load(qvel_out[worldid], shape=(nv,))
        qv_add = scale * wp.tile_load(qvel_in[worldid], shape=(nv,))
        wp.tile_store(qvel_out[worldid], wp.tile_map(wp.add, qv_og, qv_add))

        qacc_og = wp.tile_load(qacc_out[worldid], shape=(nv,))
        qacc_add_scaled = scale * wp.tile_load(qacc_in[worldid], shape=(nv,))
        wp.tile_store(qacc_out[worldid], wp.tile_map(wp.add, qacc_og, qacc_add_scaled))

        if nm:
            ms_dot_og = wp.tile_load(m_state_dot_out[worldid], shape=(nm,))
            ms_dot_add_scaled = scale * wp.tile_load(m_state_dot_in[worldid], shape=(nm,))
            wp.tile_store(m_state_dot_out[worldid], wp.tile_map(wp.add, ms_dot_og, ms_dot_add_scaled))

            ma_dot_og = wp.tile_load(m_act_dot_out[worldid], shape=(nm,))
            ma_dot_add_scaled = scale * wp.tile_load(m_act_dot_in[worldid], shape=(nm,))
            wp.tile_store(m_act_dot_out[worldid], wp.tile_map(wp.add, ma_dot_og, ma_dot_add_scaled))

        if na:
            aa_dot_og = wp.tile_load(a_act_dot_out[worldid], shape=(na,))
            aa_dot_add_scaled = scale * wp.tile_load(a_act_dot_in[worldid], shape=(na,))
            wp.tile_store(a_act_dot_out[worldid], wp.tile_map(wp.add, aa_dot_og, aa_dot_add_scaled))
        return

    wp.launch_tiled(
        _add_to_state_dot,
        dim=d.nworld,
        inputs=[d.integration_done, qvel_add, qacc_add, m_state_dot_add, m_act_dot_add, a_act_dot_add],
        outputs=[d.qvel_buffer, d.qacc, d.m_state_dot, d.m_act_dot, d.a_act_dot],
        block_dim=m.block_dim.restore_state,
    )
    return


def get_state_at_idx(d: Data, idx: int):
    scratch = d.integrator_scratch[idx]
    return (scratch.time, scratch.qpos, scratch.qvel, scratch.m_state,
            scratch.m_act, scratch.a_act, scratch.exp_contact_state)


def get_state_dot_at_idx(d: Data, idx: int):
    scratch = d.integrator_dot_scratch[idx]
    return scratch.qvel, scratch.qacc, scratch.m_state_dot, scratch.m_act_dot, scratch.a_act_dot


def save_state_idx(m: Model, d: Data, save_idx: int, ):
    time_dest, qpos_dest, qvel_dest, m_state_dest, m_act_dest, a_act_dest, exp_contact_state_dest = (
        get_state_at_idx(d, save_idx))
    save_state(m, d, time_dest, qpos_dest, qvel_dest, m_state_dest, m_act_dest, a_act_dest, exp_contact_state_dest)


def save_state_dot_idx(m: Model, d: Data, save_idx: int, ):
    qvel_dest, qacc_dest, m_state_dot_dest, m_act_dot_dest, a_act_dot_dest = get_state_dot_at_idx(d, save_idx)
    save_state_dot(m, d, qvel_dest, qacc_dest, m_state_dot_dest, m_act_dot_dest, a_act_dot_dest)


def restore_state_idx(m: Model, d: Data, restore_idx: int, only_on_reject: bool):
    time_src, qpos_src, qvel_src, m_state_src, m_act_src, a_act_src, exp_contact_state_src \
        = get_state_at_idx(d, restore_idx)
    restore_state(m, d, time_src, qpos_src, qvel_src, m_state_src, m_act_src, a_act_src, exp_contact_state_src,
                  only_on_reject=only_on_reject)


def restore_state_dot_idx(m: Model, d: Data, restore_idx: int, only_on_reject: bool):
    qvel_src, qacc_src, m_state_dot_src, m_act_dot_src, a_act_dot_src = get_state_dot_at_idx(d, restore_idx)
    restore_state_dot(m, d, qvel_src, qacc_src, m_state_dot_src, m_act_dot_src, a_act_dot_src,
                      only_on_reject=only_on_reject)


def add_to_state_dot_from_idx(m: Model, d: Data, scale: float, add_idx: int):
    qvel_add, qacc_add, m_state_dot_add, m_act_dot_add, a_act_dot_add = get_state_dot_at_idx(d, add_idx)
    add_to_state_dot(m, d, scale, qvel_add, qacc_add, m_state_dot_add, m_act_dot_add, a_act_dot_add)
