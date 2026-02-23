import warp as wp

from . import support
from .types import Data
from .types import Model
from .types import vec5
from .consts import MJ_MINVAL
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _process_contacts_hc(
        # Model:
        body_rootid: wp.array(dtype=int),
        geom_bodyid: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        xipos_in: wp.array2d(dtype=wp.vec3),
        cvel_in: wp.array2d(dtype=wp.spatial_vector),
        subtree_com_in: wp.array2d(dtype=wp.vec3),
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
    cvel1 = cvel_in[worldid, body1]
    cvel2 = cvel_in[worldid, body2]
    subtree_com1 = subtree_com_in[worldid, body_rootid[body1]]
    subtree_com2 = subtree_com_in[worldid, body_rootid[body2]]
    dif1 = location - subtree_com1
    dif2 = location - subtree_com2
    vel1 = support.transform_velocity(cvel1, dif1)
    vel2 = support.transform_velocity(cvel2, dif2)
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
    if v_slip > MJ_MINVAL:
        v_rel = v_slip / transition_velocity
        f_friction = f * (wp.min(v_rel, 1.0) * (ud + 2.0 * (us - ud) / (1.0 + v_rel * v_rel)) + uv * v_slip)

        force += f_friction * v_t / v_slip

    # Apply forces to bodies
    com1, com2 = xipos_in[worldid, body1], xipos_in[worldid, body2]
    wp.atomic_add(xfrc_contact_out[worldid], body1,
                  support.force_at_point(-1.0 * force, location - com1))
    wp.atomic_add(xfrc_contact_out[worldid], body2,
                  support.force_at_point(1.0 * force, location - com2))

    wp.atomic_add(geom_cforce_out[worldid], geom[0], wp.length(force))
    wp.atomic_add(geom_cforce_out[worldid], geom[1], wp.length(force))

    if body1 == 0:
        wp.atomic_add(grf_out, worldid, force)
    elif body2 == 0:
        wp.atomic_add(grf_out, worldid, -force)


@event_scope
def apply_contact_forces(m: Model, d: Data):
    d.grf.zero_()
    wp.launch(
        _process_contacts_hc,
        dim=(d.naconmax),
        inputs=[
            m.body_rootid,
            m.geom_bodyid,
            d.integration_done,
            d.xipos,
            d.cvel,
            d.subtree_com,
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
