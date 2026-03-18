import warp as wp

from . import forward
from . import integrate_adaptive_common
from . import integrate_common
from .types import Data
from .types import Model
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@event_scope
def attempt_adaptive_step(m: Model, d: Data):
    # Set the target time to integrate to, set actual step size
    integrate_adaptive_common.choose_target_time(m, d)

    # Adjust scales for error computation
    integrate_adaptive_common.adjust_err_scales(m, d)

    # Save state y_0
    integrate_adaptive_common.save_state_idx(m, d, 0)

    # Big step using full current step size, store y_1
    integrate_common.advance(m, d, d.qacc, d.qvel, 1.0)
    integrate_adaptive_common.save_state_idx(m, d, 1)

    # Restore y_0 (note that y_0' is unmodified after restore). Take two half steps
    integrate_adaptive_common.restore_state_idx(m, d, 0, only_on_reject=False)
    integrate_common.advance(m, d, d.qacc, d.qvel, 0.5)
    forward.fwd(m, d)  # realize for mid-point
    integrate_common.advance(m, d, d.qacc, d.qvel, 0.5)

    # Compute error between y_1* and y_1
    integrate_adaptive_common.compute_error(m, d, d.integrator_scratch[1], scale=1.0)

    # Reject step if accuracy isn't good, compute new step size
    integrate_adaptive_common.adjust_step_size(m, d, 2.0)

    # Restore state for worlds where the step was rejected
    integrate_adaptive_common.restore_state_idx(m, d, 0, only_on_reject=True)

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
