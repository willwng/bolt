import warp as wp

from . import support
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
        xfrc_contact_out: wp.array2d(dtype=wp.spatial_vector),
        grf_out: wp.array(dtype=wp.vec3),
        geom_cforce_out: wp.array2d(dtype=float)
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
    p1 = wp.transform_get_translation(mob_X_GB_in[worldid, body1])
    p2 = wp.transform_get_translation(mob_X_GB_in[worldid, body2])
    dif1 = location - p1
    dif2 = location - p2

    body_v_s1, body_v_s2 = body_V_GB_in[worldid, body1], body_V_GB_in[worldid, body2]
    vel1 = support.transform_velocity(body_v_s1, dif1)
    vel2 = support.transform_velocity(body_v_s2, dif2)

    # Compute relative velocities of the bodies
    v = wp.spatial_bottom(vel1 - vel2)
    # Project into contact frame
    v_n = wp.dot(v, normal)
    v_t = v - (v_n * normal)

    # Hunt-Crossley correction forces
    f = fH * (1.0 + 1.5 * c * v_n)
    if f <= 0:
        return
    force = f * normal

    # Friction cone
    v_slip = wp.length(v_t)
    if v_slip > MSK_MINVAL:
        v_rel = v_slip / transition_velocity
        f_friction = f * (wp.min(v_rel, 1.0) * (ud + 2.0 * (us - ud) / (1.0 + v_rel * v_rel)) + uv * v_slip)

        force += f_friction * v_t / v_slip

    # Apply forces to bodies
    wp.atomic_add(xfrc_contact_out[worldid], body1, support.force_at_point(-1.0 * force, dif1))
    wp.atomic_add(xfrc_contact_out[worldid], body2, support.force_at_point(1.0 * force, dif2))

    # Keep track of contact forces on the geom for output
    wp.atomic_add(geom_cforce_out[worldid], geom[0], wp.length(force))
    wp.atomic_add(geom_cforce_out[worldid], geom[1], wp.length(force))

    # Keep track of ground reaction forces
    if body1 == 0:
        wp.atomic_add(grf_out, worldid, force)
    elif body2 == 0:
        wp.atomic_add(grf_out, worldid, -force)


@event_scope
def contact_forces(m: Model, d: Data):
    d.grf.zero_()
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
            d.xfrc_contact,
            d.grf,
            d.geom_cforce
        ],
    )
    return
