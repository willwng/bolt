import warp as wp

from . import math
from .types import Data
from .types import Model
from .types import vec5
from .consts import MSK_MINVAL
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _process_contacts_hc(
        # Model:
        geom_bodyid: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        mob_X_GB_in: wp.array2d(dtype=wp.transform),
        body_V_GB_in: wp.array2d(dtype=wp.spatial_vector),
        nacon_in: wp.array(dtype=int),
        # In:
        dist_in: wp.array(dtype=float),
        curvature_in: wp.array(dtype=float),
        stiffness_in: wp.array(dtype=float),
        dissipation_in: wp.array(dtype=float),
        transition_velocity_in: wp.array(dtype=float),
        worldid_in: wp.array(dtype=int),
        geom_in: wp.array(dtype=wp.vec2i),
        pos_in: wp.array(dtype=wp.vec3),
        frame_in: wp.array(dtype=wp.mat33),
        friction_in: wp.array(dtype=vec5),
        # Data out:
        body_F_contact_out: wp.array2d(dtype=wp.spatial_vector),
        grf_out: wp.array(dtype=wp.vec3),
        geom_cforce_out: wp.array2d(dtype=float),
        geom_self_cforce_out: wp.array2d(dtype=float)
):
    conid = wp.tid()
    if conid >= nacon_in[0]:
        return

    depth = -dist_in[conid]
    if depth < 0.0:
        return

    worldid = worldid_in[conid]
    if integration_done_in[worldid]:
        return

    geom = geom_in[conid]
    body1 = geom_bodyid[geom[0]]
    body2 = geom_bodyid[geom[1]]
    cpos = pos_in[conid]
    frame = frame_in[conid]
    radius = curvature_in[conid]
    friction = friction_in[conid]
    stiffness = stiffness_in[conid]
    dissipation = dissipation_in[conid]
    transition_velocity = transition_velocity_in[conid]
    normal = frame[0]

    us, ud, uv = friction[0], friction[1], friction[2]

    # Adjust the contact location based on the relative stiffness
    location = cpos

    # Calculate the Hertz force.
    k = 0.5 * stiffness
    c = dissipation
    fH = (4.0 / 3.0) * k * depth * wp.sqrt(radius * k * depth)

    # Calculate the relative velocity of the two bodies at the contact point
    X_GB_1, X_GB_2 = mob_X_GB_in[worldid, body1], mob_X_GB_in[worldid, body2]
    V_GB_1, V_GB_2 = body_V_GB_in[worldid, body1], body_V_GB_in[worldid, body2]
    station1 = math.find_station_at_ground_point(X_GB_1, location)
    station2 = math.find_station_at_ground_point(X_GB_2, location)
    v1 = math.find_station_velocity_in_ground(X_GB_1, V_GB_1, station1)
    v2 = math.find_station_velocity_in_ground(X_GB_2, V_GB_2, station2)

    # Compute relative velocities of the bodies
    v = v1 - v2
    # Project into contact frame
    v_n = wp.dot(v, normal)
    v_t = v - (v_n * normal)

    # Hunt-Crossley correction forces
    f = fH * (1.0 + 1.5 * c * v_n)
    if f <= 0:
        return
    force = f * normal

    # Friction force
    v_slip = wp.length(v_t)
    if v_slip > MSK_MINVAL:
        v_rel = v_slip / transition_velocity
        f_friction = f * (wp.min(v_rel, 1.0) * (ud + 2.0 * (us - ud) / (1.0 + v_rel * v_rel)) + uv * v_slip)

        force += f_friction * v_t / v_slip

    # Apply forces to bodies
    wp.atomic_add(body_F_contact_out[worldid], body1,
                  math.apply_force_to_body_point(X_GB_1, station1, -1.0 * force))
    wp.atomic_add(body_F_contact_out[worldid], body2,
                  math.apply_force_to_body_point(X_GB_2, station2, 1.0 * force))

    # Keep track of contact forces on the geom for output
    wp.atomic_add(geom_cforce_out[worldid], geom[0], wp.length(force))
    wp.atomic_add(geom_cforce_out[worldid], geom[1], wp.length(force))

    if body1 == body2:
        wp.atomic_add(geom_self_cforce_out[worldid], geom[0], wp.length(force))
        wp.atomic_add(geom_self_cforce_out[worldid], geom[1], wp.length(force))

    # Keep track of ground reaction forces
    if body1 == 0:
        wp.atomic_add(grf_out, worldid, force)
    elif body2 == 0:
        wp.atomic_add(grf_out, worldid, -force)


@event_scope
def contact_forces_hc(m: Model, d: Data):
    wp.launch(
        _process_contacts_hc,
        dim=(d.naconmax),
        inputs=[m.geom_bodyid,
                d.integration_done,
                d.mob_X_GB,
                d.body_V_GB,
                d.nacon,
                d.contact.dist,
                d.contact.curvature,
                d.contact.stiffness,
                d.contact.dissipation,
                d.contact.transition_velocity,
                d.contact.worldid,
                d.contact.geom,
                d.contact.pos,
                d.contact.frame,
                d.contact.friction,
                ],
        outputs=[
            d.body_F_contact,
            d.grf,
            d.geom_cforce,
            d.geom_self_cforce
        ],
    )
    return
