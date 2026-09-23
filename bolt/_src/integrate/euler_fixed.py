import warp as wp

from bolt.types import Data
from bolt.types import Model

from ..pipeline import forward
from ..warp_util import event_scope
from . import common

wp.set_module_options({"enable_backward": False})


@event_scope
def integrate(m: Model, d: Data):
    """Steps from d.time to d.next_time using Symplectic Euler """
    common.update_step_size(m, d)
    common.advance(m, d, d.qacc, d.qvel, scale=1.0, time_scale=1.0, symplectic=True)
    forward.fwd(m, d)  # realize state for next step
