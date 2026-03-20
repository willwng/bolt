import warp as wp

from . import mobilizers
from .types import Data
from .types import Model
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _fix_limits_kernel(
        # Model in:
        limit_dof_range: wp.array(dtype=wp.vec2),
        limit_dof_qadr: wp.array(dtype=int),
        # Data in:
        world_reset_in: wp.array(dtype=bool),
        qpos_in: wp.array2d(dtype=float),
        # Data out:
        qpos_out: wp.array2d(dtype=float),
):
    worldid, limitdofid = wp.tid()
    if world_reset_in[worldid]:
        dof_range = limit_dof_range[limitdofid]
        dof_qadr = limit_dof_qadr[limitdofid]
        qpos = qpos_in[worldid, dof_qadr]

        qpos_clamped = wp.clamp(qpos, dof_range[0], dof_range[1])
        qpos_out[worldid, dof_qadr] = qpos_clamped
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
        _fix_limits_kernel,
        dim=(d.nworld, m.nlinearstop),
        inputs=[m.stop_qpos_range, m.stop_qpos_adr, d.world_reset, d.qpos, ],
        outputs=[d.qpos, ],
    )
    wp.launch(
        _fix_limits_kernel,
        dim=(d.nworld, m.nlimitforce),
        inputs=[m.lf_qpos_range, m.lf_qpos_adr, d.world_reset, d.qpos, ],
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
