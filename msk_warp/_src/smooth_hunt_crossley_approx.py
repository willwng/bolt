import warp as wp

from . import support
from .types import Data
from .types import Model
from .types import vec5
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

    # Calculate the relative velocity of the two bodies at the contact point
    location = cpos
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
    normal = frame[0]
    v_n = wp.dot(v, normal)
    v_t = v - (v_n * normal)

    stiffness = 1.6e6
    dissipation = 0.072

    # Calculate the hertz force
    k = 0.5 * wp.pow(stiffness, 2.0 / 3.0)
    indentation = depth
    cf = 1e-5
    bd = 300.0
    fh_pos = (4.0 / 3.0) * k * wp.sqrt(radius * k) * wp.pow(
        wp.sqrt(indentation * indentation + cf), 3. / 2.)
    fh_smooth = fh_pos * (1.0 / 2.0 + (1.0 / 2.0) * wp.tanh(bd * indentation))

    # Calculate the Hunt-Crossley force
    c = dissipation
    bv = 50.0
    fhc_pos = fh_smooth * (1.0 + (3.0 / 2.0) * c * v_n)
    fhc_smooth = fhc_pos * (
            1. / 2. + (1. / 2.) * wp.tanh(bv * (v_n + (2.0 / (3.0 * c)))))
    force = fhc_smooth * normal

    # Calculate the friction force.
    us = 0.95
    ud = 0.3
    uv = 0.3
    vt = 0.001

    aux = wp.length_sq(v_t) + cf
    vslip = wp.pow(aux, 1.0 / 2.0)
    vrel = vslip / vt
    ff = fhc_smooth * (wp.min(vrel, 1.0) * (
                ud + 2.0 * (us - ud) / (1.0 + vrel * vrel)) + uv * vslip)
    force += ff * v_t / vslip

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
def contact_forces(m: Model, d: Data):
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
