import warp as wp

from . import integrate_euler_fixed
from . import integrate_euler_midpoint_fixed
from . import integrate_rk_fixed
from . import integrate_euler_adaptive
from . import integrate_euler_midpoint_adaptive
from . import integrate_rk_adaptive
from .types import Data
from .types import IntegratorType
from .types import Model
from .warp_util import event_scope
import torch

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
    if wp.static(m.opt.integrator) == IntegratorType.EULER_FIXED:
        integrate_euler_fixed.integrate(m, d)
    if wp.static(m.opt.integrator) == IntegratorType.EULER_MIDPOINT_FIXED:
        integrate_euler_midpoint_fixed.integrate(m, d)
    elif wp.static(m.opt.integrator) == IntegratorType.RK4_FIXED:
        integrate_rk_fixed.integrate(m, d)
    elif wp.static(m.opt.integrator) == IntegratorType.EULER_ADAPTIVE:
        integrate_euler_adaptive.integrate(m, d)
    elif wp.static(m.opt.integrator) == IntegratorType.EULER_MIDPOINT_ADAPTIVE:
        integrate_euler_midpoint_adaptive.integrate(m, d)
    elif wp.static(m.opt.integrator) == IntegratorType.RK_MERSON_ADAPTIVE:
        integrate_rk_adaptive.integrate(m, d)
    else:
        raise RuntimeError("Unknown integrator type")


def set_next_time(m: Model, d: Data, next_time: torch.Tensor):
    wp.copy(d.next_time, wp.from_torch(next_time))


def increment_next_time(m: Model, d: Data, dt: float):
    wp.launch(
        _increment_next_time,
        dim=d.nworld,
        inputs=[d.time, dt],
        outputs=[d.next_time],
    )


def increment_next_time_tensor(m: Model, d: Data, dt: torch.Tensor):
    curr_time = wp.to_torch(d.time)
    dt = dt.to(curr_time.device)
    wp.copy(d.next_time, wp.from_torch(curr_time + dt))
