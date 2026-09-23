import warp as wp

from bolt.types import Data
from bolt.types import Model

from ..pipeline import forward
from ..warp_util import event_scope
from . import adaptive_common
from . import common

wp.set_module_options({"enable_backward": False})


@event_scope
def attempt_adaptive_step(m: Model, d: Data):
    # Set the target time to integrate to, set actual step size
    adaptive_common.choose_target_time(m, d)

    # Adjust scales for error computation
    adaptive_common.adjust_err_scales(m, d)

    # Save state y_0
    adaptive_common.save_state(m, d, 0)

    # Big step using full current step size, store y_1
    common.advance(m, d, d.qacc, d.qvel, scale=1.0, time_scale=1.0)
    adaptive_common.save_state(m, d, 1)

    # Restore y_0 (note that y_0' is unmodified after restore). Take two half steps
    adaptive_common.restore_state(m, d, 0, only_on_reject=False)
    common.advance(m, d, d.qacc, d.qvel, 0.5, time_scale=0.5)
    forward.fwd(m, d)  # realize for mid-point
    common.advance(m, d, d.qacc, d.qvel, 0.5, time_scale=0.5)

    # Compute error between y_1* and y_1
    adaptive_common.compute_error(m, d, 1, scale=1.0)

    # Reject step if accuracy isn't good, compute new step size
    adaptive_common.adjust_step_size(m, d, 2.0)

    # Restore state for worlds where the step was rejected
    adaptive_common.restore_state(m, d, 0, only_on_reject=True)

    # Check if we've reached the target time
    adaptive_common.check_done_integrating(m, d)

    # Prepare derivatives for next attempt
    forward.fwd(m, d)
    return


@event_scope
def integrate(m: Model, d: Data):
    """ Steps from d.time to d.next_time using adaptive Euler. """
    adaptive_common.integrate_adaptive(m, d, attempt_adaptive_step)
