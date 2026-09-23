import warp as wp

from bolt.types import Data
from bolt.types import IntegratorType
from bolt.types import Model

from ..integrate import euler_adaptive
from ..integrate import euler_fixed
from ..integrate import rk_adaptive
from ..integrate import rk_fixed
from ..warp_util import event_scope
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
def step(m: Model, d: Data):
    """ Steps from d.time to d.next_time """
    forward.fwd(m, d)  # Realize state (needed if for example excitation changes after a step)

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
    wp.launch(
        _increment_next_time,
        dim=d.nworld,
        inputs=[d.time, dt],
        outputs=[d.next_time],
    )
