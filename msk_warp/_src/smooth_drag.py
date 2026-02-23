import warp as wp

from . import consts
from . import math
from .types import Data
from .types import Model
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _drag_force(
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        xivel_in: wp.array2d(dtype=wp.spatial_vector),
        # In:
        body_tree: wp.array(dtype=int),
        # Data out:
        xfrc_drag_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, nodeid = wp.tid()
    if integration_done_in[worldid]:
        return
    bodyid = body_tree[nodeid]
    body_vel = wp.spatial_bottom(xivel_in[worldid, bodyid])

    drag_x = -consts.A_AFK * math.sqr(body_vel[0]) * wp.sign(body_vel[0])
    drag_frc = wp.spatial_vector(drag_x, 0.0, 0.0, 0.0, 0.0, 0.0)
    xfrc_drag_out[worldid, bodyid] += drag_frc
    return


@event_scope
def apply_drag(m: Model, d: Data):
    # Only apply to root bodies
    body_roots = m.body_tree[1]
    wp.launch(
        _drag_force,
        dim=(d.nworld, body_roots.size),
        inputs=[d.integration_done, d.xivel, body_roots],
        outputs=[d.xfrc_drag],
    )
