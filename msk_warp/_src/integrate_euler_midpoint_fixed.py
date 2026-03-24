import warp as wp

from . import forward
from . import integrate_common
from .types import Data
from .types import Model
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@event_scope
def integrate(m: Model, d: Data, scale: float = 1.0, update_step_size: bool = True, realize_state: bool = True):
    """
    Steps from d.time to d.next_time using Midpoint Euler
    Note that we assume that only position is integrated as a second order ODE
    Everything else (muscle state, activation, etc) is integrated as a first order ODE
    """
    if update_step_size:
        integrate_common.update_step_size(m, d)

    # Store the current velocity
    midpoint_scratch = d.integrator_midpoint_scratch
    wp.copy(midpoint_scratch.qvel, d.qvel)
    # Advance only velocity
    integrate_common.advance_velocity(m, d, d.qacc, scale=scale)
    # Advance position using the midpoint velocity, don't touch velocity
    qacc = midpoint_scratch.qacc  # this will be zero
    hscale = 0.5 * scale
    integrate_common.advance(m, d, qacc, midpoint_scratch.qvel, scale=hscale, time_scale=hscale)
    integrate_common.advance(m, d, qacc, d.qvel, scale=hscale, time_scale=hscale)

    if realize_state:
        forward.fwd(m, d)
    return
