import warp as wp

from . import forward
from . import integrate_adaptive_common
from . import integrate_common
from .types import Data
from .types import Model
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})



@event_scope
def restore_0(m: Model, d: Data, only_on_reject: bool):
    integrate_adaptive_common.restore_state_idx(m, d, 0, only_on_reject=only_on_reject)
    integrate_adaptive_common.restore_state_dot_idx(m, d, 0, only_on_reject=only_on_reject)
    wp.copy(d.qvel_buffer, d.qvel)


@event_scope
def attempt_adaptive_step(m: Model, d: Data):
    # Set the target time to integrate to, set actual step size
    integrate_adaptive_common.choose_target_time(m, d)

    # Adjust scales for error computation
    integrate_adaptive_common.adjust_err_scales(m, d)

    # Save y_0, k_0 (y_0')
    integrate_adaptive_common.save_state_idx(m, d, 0)
    integrate_adaptive_common.save_state_dot_idx(m, d, 0)

    # k_1 = f(y_0 + (h/3) * k_0)
    integrate_common.advance(m, d, d.qacc, d.qvel, 1.0 / 3.0)
    forward.fwd(m, d)
    integrate_adaptive_common.save_state_dot_idx(m, d, 1)

    # k_2 = f(y_0 + (h/6) * (k_0 + k_1))
    restore_0(m, d, only_on_reject=False)
    integrate_adaptive_common.add_to_state_dot_from_idx(m, d, 1.0, 1)
    integrate_common.advance(m, d, d.qacc, d.qvel_buffer, 1.0 / 6.0)
    forward.fwd(m, d)
    integrate_adaptive_common.save_state_dot_idx(m, d, 2)

    # k_3 = f(y_0 + (h/8) * (k_0 + 3*k_2))
    restore_0(m, d, only_on_reject=False)
    integrate_adaptive_common.add_to_state_dot_from_idx(m, d, 3.0, 2)
    integrate_common.advance(m, d, d.qacc, d.qvel_buffer, 1.0 / 8.0)
    forward.fwd(m, d)
    integrate_adaptive_common.save_state_dot_idx(m, d, 3)

    # k_4 = f(y_0 + (h/2) * (k_0 - 3*k_2 + 4*k_3))
    restore_0(m, d, only_on_reject=False)
    integrate_adaptive_common.add_to_state_dot_from_idx(m, d, -3.0, 2)
    integrate_adaptive_common.add_to_state_dot_from_idx(m, d, 4.0, 3)
    integrate_common.advance(m, d, d.qacc, d.qvel_buffer, 1.0 / 2.0)
    forward.fwd(m, d)
    integrate_adaptive_common.save_state_dot_idx(m, d, 4)

    integrate_adaptive_common.save_state_idx(m, d, 1)  # y_save for error estimate

    # k = (1/6) * (k_0 + 4*k_3 + k_4)
    restore_0(m, d, only_on_reject=False)
    integrate_adaptive_common.add_to_state_dot_from_idx(m, d, 4.0, 3)
    integrate_adaptive_common.add_to_state_dot_from_idx(m, d, 1.0, 4)
    integrate_common.advance(m, d, d.qacc, d.qvel_buffer, 1.0 / 6.0)

    # Compute error against y_save: (1/5) * (y - y_save)
    ss = d.integrator_scratch[1]
    integrate_adaptive_common.compute_error(m, d, ss.qpos, ss.qvel, ss.m_act, ss.m_state, ss.a_act, scale=0.2)

    # Reject step if accuracy isn't good, compute new step size
    integrate_adaptive_common.adjust_step_size(m, d, err_order=4.0)

    # Restore state for worlds where the step was rejected
    restore_0(m, d, only_on_reject=True)

    # Check if we've reached the target time
    integrate_adaptive_common.check_done_integrating(m, d)

    # Prepare derivatives for next attempt
    forward.fwd(m, d)
    return


@event_scope
def integrate(m: Model, d: Data):
    """ Steps from d.time to d.next_time using second-order Euler. """
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
