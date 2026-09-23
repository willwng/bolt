"""
Exponential coordinate limit forces:
    f = k_lower * exp(-s_lower * (q - q_min)) - k_upper * exp(s_upper * (q - q_max))

Not part of the forward pipeline; call exponential_limit_forces explicitly.
"""
import warp as wp

from ..types import Data
from ..types import Model
from ..warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.struct
class ExponentialLimit:
    qpos_adr: int  # qpos address of the limited coordinate
    dof_adr: int  # dof address the generalized force is applied to
    qpos_range: wp.vec2  # (q_min, q_max)
    stiffness: wp.vec2  # (k_lower, k_upper)
    shape: wp.vec2  # (s_lower, s_upper)


@wp.kernel
def _exponential_limit_forces(
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qpos_in: wp.array2d(dtype=float),
        # In:
        limits: wp.array(dtype=ExponentialLimit),
        # Out:
        ufrc_out: wp.array2d(dtype=float),
):
    worldid, limitid = wp.tid()
    if integration_done_in[worldid]:
        return

    limit = limits[limitid]
    q = qpos_in[worldid, limit.qpos_adr]
    force = (limit.stiffness[0] * wp.exp(-limit.shape[0] * (q - limit.qpos_range[0])) -
             limit.stiffness[1] * wp.exp(limit.shape[1] * (q - limit.qpos_range[1])))
    wp.atomic_add(ufrc_out, worldid, limit.dof_adr, force)


@event_scope
def exponential_limit_forces(m: Model, d: Data, limits: wp.array, ufrc_out: wp.array2d):
    """ Adds the exponential limit generalized forces (in dof space) to ufrc_out (nworld, nv) """
    if limits.size == 0:
        return
    wp.launch(
        _exponential_limit_forces,
        dim=(d.nworld, limits.size),
        inputs=[d.integration_done, d.qpos, limits],
        outputs=[ufrc_out],
    )
