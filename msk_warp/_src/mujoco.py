import warp as wp

from . import consts
from .types import vec6

wp.set_module_options({"enable_backward": False})


@wp.func
def bump(l: float, a: float, mid: float, b: float) -> float:
    left = 0.5 * (a + mid)
    right = 0.5 * (mid + b)

    if (l <= a) or (l >= b):
        y = 0.0
    elif l < left:
        x = (l - a) / (left - a)
        y = 0.5 * x * x
    elif l < mid:
        x = (mid - l) / (mid - left)
        y = 1.0 - 0.5 * x * x
    elif l < right:
        x = (l - mid) / (right - mid)
        y = 1.0 - 0.5 * x * x
    else:
        x = (b - l) / (b - right)
        y = 0.5 * x * x
    return y


@wp.func
def calc_passive_force_multiplier(norm_fiber_length: float) -> float:
    lmax = 1.6
    fpmax = 1.3
    b = 0.5 * (1.0 + lmax)
    if norm_fiber_length <= 1.0:
        return 0.0
    elif norm_fiber_length <= b:
        x = (norm_fiber_length - 1.0) / (b - 1.0)
        return 0.25 * fpmax * x * x * x
    else:
        x = (norm_fiber_length - b) / (b - 1.0)
        return 0.25 * fpmax * (1.0 + 3.0 * x)


@wp.func
def calc_active_force_length_multiplier(norm_fiber_length: float) -> float:
    lmin = 0.5
    lmax = 1.6
    return bump(norm_fiber_length, lmin, 1.0, lmax) + 0.15 * bump(norm_fiber_length, lmin, 0.5 * (lmin + 0.95), 0.95)


@wp.func
def calc_force_velocity_multiplier(norm_fiber_velocity: float) -> float:
    fvmax = 1.2
    c = fvmax - 1.0
    if norm_fiber_velocity <= -1.0:
        return 0.0
    elif norm_fiber_velocity <= 0.0:
        return (norm_fiber_velocity + 1.0) * (norm_fiber_velocity + 1.0)
    elif norm_fiber_velocity <= c:
        return fvmax - (c - norm_fiber_velocity) * (c - norm_fiber_velocity) / c
    else:
        return fvmax
