import warp as wp

from ..pipeline import forward
from ..types import Data
from ..types import Model
from ..warp_util import event_scope
from . import adaptive_common
from . import common

wp.set_module_options({"enable_backward": False})



@event_scope
def restore_0(m: Model, d: Data, only_on_reject: bool):
    adaptive_common.restore_state_idx(m, d, 0, only_on_reject=only_on_reject)
    adaptive_common.restore_state_dot_idx(m, d, 0, only_on_reject=only_on_reject)
    wp.copy(d.qvel_buffer, d.qvel)


@event_scope
def attempt_adaptive_step(m: Model, d: Data):
    # Set the target time to integrate to, set actual step size
    adaptive_common.choose_target_time(m, d)

    # Adjust scales for error computation
    adaptive_common.adjust_err_scales(m, d)

    # Save y_0, k_0 (y_0')
    adaptive_common.save_state_idx(m, d, 0)
    adaptive_common.save_state_dot_idx(m, d, 0)

    # k_1 = f(y_0 + (h/3) * k_0)
    common.advance(m, d, d.qacc, d.qvel, scale=1.0 / 3.0, time_scale=1.0 / 3.0, symplectic=False)
    forward.fwd(m, d)
    adaptive_common.save_state_dot_idx(m, d, 1)

    # k_2 = f(y_0 + (h/6) * (k_0 + k_1))
    restore_0(m, d, only_on_reject=False)
    adaptive_common.add_to_state_dot_from_idx(m, d, 1.0, 1)
    common.advance(m, d, d.qacc, d.qvel_buffer, scale=1.0 / 6.0, time_scale=1.0 / 3.0, symplectic=False)
    forward.fwd(m, d)
    adaptive_common.save_state_dot_idx(m, d, 2)

    # k_3 = f(y_0 + (h/8) * (k_0 + 3*k_2))
    restore_0(m, d, only_on_reject=False)
    adaptive_common.add_to_state_dot_from_idx(m, d, 3.0, 2)
    common.advance(m, d, d.qacc, d.qvel_buffer, scale=1.0 / 8.0, time_scale=1.0 / 2.0, symplectic=False)
    forward.fwd(m, d)
    adaptive_common.save_state_dot_idx(m, d, 3)

    # k_4 = f(y_0 + (h/2) * (k_0 - 3*k_2 + 4*k_3))
    restore_0(m, d, only_on_reject=False)
    adaptive_common.add_to_state_dot_from_idx(m, d, -3.0, 2)
    adaptive_common.add_to_state_dot_from_idx(m, d, 4.0, 3)
    common.advance(m, d, d.qacc, d.qvel_buffer, scale=1.0 / 2.0, time_scale=1.0, symplectic=False)
    forward.fwd(m, d)
    adaptive_common.save_state_dot_idx(m, d, 4)

    adaptive_common.save_state_idx(m, d, 1)  # y_save for error estimate

    # k = (1/6) * (k_0 + 4*k_3 + k_4)
    restore_0(m, d, only_on_reject=False)
    adaptive_common.add_to_state_dot_from_idx(m, d, 4.0, 3)
    adaptive_common.add_to_state_dot_from_idx(m, d, 1.0, 4)
    common.advance(m, d, d.qacc, d.qvel_buffer, scale=1.0 / 6.0, time_scale=1.0, symplectic=False)

    # Compute error against y_save: (1/5) * (y - y_save)
    adaptive_common.compute_error(m, d, d.integrator_scratch[1], scale=0.2)

    # Reject step if accuracy isn't good, compute new step size
    adaptive_common.adjust_step_size(m, d, err_order=4.0)

    # Restore state for worlds where the step was rejected
    restore_0(m, d, only_on_reject=True)

    # Check if we've reached the target time
    adaptive_common.check_done_integrating(m, d)

    # Prepare derivatives for next attempt
    forward.fwd(m, d)
    return


@event_scope
def integrate(m: Model, d: Data):
    """ Steps from d.time to d.next_time using RK Merson. """
    d.integration_done.zero_()
    d.steps_attempted.zero_()

    # take adaptive steps until target time is reached
    d.nintegrating.fill_(d.nworld)
    wp.capture_while(
        d.nintegrating,
        while_body=attempt_adaptive_step,
        m=m,
        d=d,
    )

    # One more forward pass to realize state
    d.integration_done.zero_()
    forward.fwd(m, d)
    return
