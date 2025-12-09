import warp as wp

from . import integrate
from .types import Data
from .types import IntegratorType
from .types import Model
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@event_scope
def step_to(m: Model, d: Data, dt: float, dt_sim: float):
    if wp.static(m.opt.integrator) == IntegratorType.EULER_FIXED:
        integrate.step_to_fixed(m, d, dt, dt_sim)
    elif wp.static(m.opt.integrator) == IntegratorType.EULER_ADAPTIVE:
        integrate.step_to_adaptive(m, d, dt, dt)
    else:
        raise RuntimeError("Unknown integrator type")
