import warp as wp

from . import forward
from . import integrate_common
from .types import Data
from .types import Model
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@event_scope
def integrate(m: Model, d: Data):
    """Steps from d.time to d.next_time using RK4 """
    integrate_common.update_step_size(m, d)
    integrate_common.advance(m, d, d.qacc, d.qvel, 1.0)
    forward.fwd(m, d)  # realize state for next step
