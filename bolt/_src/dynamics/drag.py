"""
Quadratic aerodynamic drag on a body's center of mass:
    F = -drag_factor * |v| * v,  drag_factor = 0.5 * rho * A * Cd

Not part of the forward pipeline; call drag_forces explicitly.
"""
import warp as wp

from .. import math
from ..types import Data
from ..types import Model
from ..warp_util import event_scope

wp.set_module_options({"enable_backward": False})

# Whole-body running drag (Pugh, 1971): air density (kg/m^3), frontal area (m^2), drag coefficient
AIR_DENSITY = 1.20474061
FRONTAL_AREA = 0.50641133
DRAG_COEFFICIENT = 0.9
DEFAULT_DRAG_FACTOR = 0.5 * AIR_DENSITY * FRONTAL_AREA * DRAG_COEFFICIENT


@wp.struct
class Drag:
    bodyid: int  # body the drag acts on (at its center of mass)
    drag_factor: float  # 0.5 * rho * A * Cd


@wp.kernel
def _drag_forces(
        # Model:
        body_mass_center: wp.array(dtype=wp.vec3),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        mob_X_GB_in: wp.array2d(dtype=wp.transform),
        body_V_GB_in: wp.array2d(dtype=wp.spatial_vector),
        # In:
        drags: wp.array(dtype=Drag),
        # Out:
        body_F_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, dragid = wp.tid()
    if integration_done_in[worldid]:
        return

    drag = drags[dragid]
    X_GB = mob_X_GB_in[worldid, drag.bodyid]
    V_GB = body_V_GB_in[worldid, drag.bodyid]
    com_B = body_mass_center[drag.bodyid]

    # Velocity of the center of mass, in ground
    com_offset_G = wp.quat_rotate(wp.transform_get_rotation(X_GB), com_B)
    v_com_G = wp.spatial_bottom(V_GB) + wp.cross(wp.spatial_top(V_GB), com_offset_G)

    force_G = -drag.drag_factor * wp.length(v_com_G) * v_com_G
    wp.atomic_add(body_F_out, worldid, drag.bodyid, math.apply_force_to_body_point(X_GB, com_B, force_G))


@event_scope
def drag_forces(m: Model, d: Data, drags: wp.array, body_F_out: wp.array2d):
    """
    Adds drag spatial forces (torque about the body origin, force; in ground) to body_F_out (nworld, nbody).
    Requires positions and velocities to be realized.
    """
    if drags.size == 0:
        return
    wp.launch(
        _drag_forces,
        dim=(d.nworld, drags.size),
        inputs=[m.body_mass_center, d.integration_done, d.mob_X_GB, d.body_V_GB, drags],
        outputs=[body_F_out],
    )
