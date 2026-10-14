import warp as wp

from bolt.types import Data
from bolt.types import IntegratorType
from bolt.types import Model

from ..integrate import euler_adaptive
from ..integrate import euler_fixed
from ..integrate import rk_adaptive
from ..integrate import rk_fixed
from ..warp_util import event_scope
from ..warp_util import launch
from . import forward

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _increment_next_time(
        # Data in:
        time_in: wp.array(dtype=float),
        # In:
        dt: float,
        # Data out:
        next_time_out: wp.array(dtype=float),
):
    worldid = wp.tid()
    next_time_out[worldid] = time_in[worldid] + dt
    return


@event_scope
def step(m: Model, d: Data, realize_state: bool = False):
    """
    Steps from d.time to d.next_time.

    The previous step leaves the state realized, so by default only input-dependent quantities
    (excitations, applied forces) are recomputed. Pass realize_state=True if qpos, qvel or other
    state was modified directly since the last step without going through bolt.reset.
    """
    if realize_state:
        forward.fwd(m, d)
    else:
        forward.realize_inputs(m, d)

    if wp.static(m.opt.integrator) == IntegratorType.EULER_FIXED:
        euler_fixed.integrate(m, d)
    elif wp.static(m.opt.integrator) == IntegratorType.RK4_FIXED:
        rk_fixed.integrate(m, d)
    elif wp.static(m.opt.integrator) == IntegratorType.EULER_ADAPTIVE:
        euler_adaptive.integrate(m, d)
    elif wp.static(m.opt.integrator) == IntegratorType.RK_MERSON_ADAPTIVE:
        rk_adaptive.integrate(m, d)
    else:
        raise RuntimeError("Unknown integrator type")


def increment_next_time(m: Model, d: Data, dt: float):
    """ Updates d.next_time to d.time + dt """
    launch(
        _increment_next_time,
        dim=d.nworld,
        inputs=[d.time, dt],
        outputs=[d.next_time],
    )
