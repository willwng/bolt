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
        xipos_in: wp.array2d(dtype=wp.vec3),
        cvel_in: wp.array2d(dtype=wp.spatial_vector),
        subtree_com_in: wp.array2d(dtype=wp.vec3),
        nacon_in: wp.array(dtype=int),
        # In:
        dist_in: wp.array(dtype=float),
        curvature_in: wp.array(dtype=float),
        worldid_in: wp.array(dtype=int),
        geom_in: wp.array(dtype=wp.vec2i),
        pos_in: wp.array(dtype=wp.vec3),
        frame_in: wp.array(dtype=wp.mat33),
        friction_in: wp.array(dtype=vec5),
        # Data out:
        xfrc_applied_out: wp.array2d(dtype=wp.spatial_vector),
        grf_out: wp.array(dtype=wp.vec3)
):
    conid = wp.tid()
    if conid >= nacon_in[0]:
        return

    depth = -dist_in[conid]
    if depth < 0.0:
        return

    worldid = worldid_in[conid]
    geom = geom_in[conid]
    body1 = geom_bodyid[geom[0]]
    body2 = geom_bodyid[geom[1]]
    cpos = pos_in[conid]
    frame = frame_in[conid]
    radius = curvature_in[conid]
    normal = frame[0]

    # TODO: get these from material properties
    stiffness1 = wp.pow(1.6e6, 2.0 / 3.0)
    stiffness2 = wp.pow(1.6e6, 2.0 / 3.0)
    dissipation1 = 0.072
    dissipation2 = 0.072
    us1, us2 = 0.95, 0.95
    ud1, ud2 = 0.3, 0.3
    uv1, uv2 = 0.3, 0.3
    transition_velocity = 0.001

    # Adjust the contact location based on the relative stiffness
    s1 = stiffness2 / (stiffness1 + stiffness2)
    s2 = 1.0 - s1
    location = cpos + depth * (0.5 - s1) * frame[0]

    # Calculate the Hertz force.
    k = stiffness1 * s1
    c = dissipation1 * s1 + dissipation2 * s2
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
        has_static = (us1 != 0.0 or us2 != 0.0)
        has_dynamic = (ud1 != 0.0 or ud2 != 0.0)
        has_viscous = (uv1 != 0.0 or uv2 != 0.0)
        us = (2.0 * us1 * us2) / (us1 + us2) if has_static else 0.0
        ud = (2.0 * ud1 * ud2) / (ud1 + ud2) if has_dynamic else 0.0
        uv = (2.0 * uv1 * uv2) / (uv1 + uv2) if has_viscous else 0.0

        v_rel = v_slip / transition_velocity
        f_friction = f * (wp.min(v_rel, 1.0) * (ud + 2.0 * (us - ud) / (1.0 + v_rel * v_rel)) + uv * v_slip)

        force += f_friction * v_t / v_slip

    # Apply forces to bodies
    com1, com2 = xipos_in[worldid, body1], xipos_in[worldid, body2]
    wp.atomic_add(xfrc_applied_out[worldid], body1,
                  support.force_at_point(-1.0 * force, location - com1))
    wp.atomic_add(xfrc_applied_out[worldid], body2,
                  support.force_at_point(1.0 * force, location - com2))

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
            d.xipos,
            d.cvel,
            d.subtree_com,
            d.nacon,
            d.contact.dist,
            d.contact.curvature,
            d.contact.worldid,
            d.contact.geom,
            d.contact.pos,
            d.contact.frame,
            d.contact.friction,
        ],
        outputs=[
            d.xfrc_applied,
            d.grf
        ],
    )
    return
