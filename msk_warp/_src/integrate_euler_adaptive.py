import warp as wp

from . import forward
from . import integrate_common
from . import integrate_adaptive_common
# from . import integrate_euler_fixed
from . import math
from . import mobilizers
from .consts import MJ_MINVAL
from .types import ActuatorMetadata
from .types import Data
from .types import Model
from .types import MuscleMetadata
from .types import TileSet
from .warp_util import cache_kernel
from .warp_util import event_scope
from .warp_util import kernel as nested_kernel

wp.set_module_options({"enable_backward": False})


@event_scope
def attempt_adaptive_step(m: Model, d: Data):
    # Set the target time to integrate to, set actual step size
    integrate_adaptive_common.choose_target_time(m, d)

    # Adjust scales for error computation
    integrate_adaptive_common.adjust_err_scales(m, d)

    # Save state y_0
    integrate_adaptive_common.save_state(m, d, d.time_0, d.qpos_0, d.qvel_0, d.m_state_0, d.m_act_0, d.a_act_0)

    # Big step using full current step size, store y_1
    integrate_common.advance(m, d, d.qacc, d.qvel, 1.0)
    integrate_adaptive_common.save_state(m, d, d.time_1, d.qpos_1, d.qvel_1, d.m_state_1, d.m_act_1, d.a_act_1)

    # Restore y_0 (note that y_0' is unmodified after restore). Take two half steps
    integrate_adaptive_common.restore_state(m, d, d.time_0, d.qpos_0, d.qvel_0, d.m_state_0, d.m_act_0, d.a_act_0, only_on_reject=False)
    integrate_common.advance(m, d, d.qacc, d.qvel, 0.5)
    forward.fwd(m, d)  # realize for mid-point
    integrate_common.advance(m, d, d.qacc, d.qvel, 0.5)

    # Compute error between y_1* and y_1
    integrate_adaptive_common.compute_error(m, d, d.qpos_1, d.qvel_1, d.m_act_1, d.m_state_1, d.a_act_1)

    # Reject step if accuracy isn't good, compute new step size
    integrate_adaptive_common.adjust_step_size(m, d, 2.0)

    # Restore state for worlds where the step was rejected
    integrate_adaptive_common.restore_state(m, d, d.time_0, d.qpos_0, d.qvel_0, d.m_state_0, d.m_act_0, d.a_act_0, only_on_reject=True)

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
