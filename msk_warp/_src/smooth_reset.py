import warp as wp

from . import mobilizers
from .types import Data
from .types import Model
from .types import CoordinateLinearStop
from .types import CoordinateLimitForce
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _fix_limits_linear_stop_kernel(
        # Model in:
        coordinate_linear_stop: wp.array(dtype=CoordinateLinearStop),
        # Data in:
        world_reset_in: wp.array(dtype=bool),
        qpos_in: wp.array2d(dtype=float),
        # Data out:
        qpos_out: wp.array2d(dtype=float),
):
    worldid, limitid = wp.tid()
    stop = coordinate_linear_stop[limitid]
    if world_reset_in[worldid]:
        qpos_range = stop.qpos_range
        qpos_adr = stop.qpos_adr
        qpos = qpos_in[worldid, qpos_adr]

        qpos_clamped = wp.clamp(qpos, qpos_range[0], qpos_range[1])
        qpos_out[worldid, qpos_adr] = qpos_clamped
    return


@wp.kernel
def _fix_limits_lf_kernel(
        # Model in:
        coordinate_limit_force: wp.array(dtype=CoordinateLimitForce),
        # Data in:
        world_reset_in: wp.array(dtype=bool),
        qpos_in: wp.array2d(dtype=float),
        # Data out:
        qpos_out: wp.array2d(dtype=float),
):
    worldid, limitid = wp.tid()
    lf = coordinate_limit_force[limitid]
    if world_reset_in[worldid]:
        qpos_range = lf.qpos_range
        qpos_adr = lf.qpos_adr
        qpos = qpos_in[worldid, qpos_adr]

        qpos_clamped = wp.clamp(qpos, qpos_range[0], qpos_range[1])
        qpos_out[worldid, qpos_adr] = qpos_clamped
    return


@wp.kernel
def _fix_quaternions_kernel(
        # Model in:
        mob_type: wp.array(dtype=int),
        mob_qposadr: wp.array(dtype=int),
        # Data in:
        world_reset_in: wp.array(dtype=bool),
        # Data out:
        qpos_out: wp.array2d(dtype=float),
):
    worldid, bodyid = wp.tid()
    if world_reset_in[worldid]:
        mob_type_ = mob_type[bodyid]
        qadr = mob_qposadr[bodyid]
        qpos = qpos_out[worldid]
        mobilizers.ensure_valid_qpos(mob_type_, qadr, qpos)
    return


@event_scope
def fix_qpos_limits(m: Model, d: Data):
    """Clamps qpos values to joint limits."""
    wp.launch(
        _fix_limits_linear_stop_kernel,
        dim=(d.nworld, m.nlinearstop),
        inputs=[m.coordinate_linear_stop, d.world_reset, d.qpos, ],
        outputs=[d.qpos, ],
    )
    wp.launch(
        _fix_limits_lf_kernel,
        dim=(d.nworld, m.nlimitforce),
        inputs=[m.coordinate_limit_force, d.world_reset, d.qpos, ],
        outputs=[d.qpos, ],
    )
    return


@event_scope
def fix_quaternions(m: Model, d: Data):
    """Ensures quaternion qpos values are normalized."""
    wp.launch(
        _fix_quaternions_kernel,
        dim=(d.nworld, m.nbody),
        inputs=[m.mob_type, m.mob_qposadr, d.world_reset, ],
        outputs=[d.qpos, ],
    )
    return
