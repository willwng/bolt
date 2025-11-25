# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


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
        xpos_in: wp.array2d(dtype=wp.vec3),
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

    # TODO: get these from material properties
    stiffness1 = wp.pow(5e6, 2.0 / 3.0)
    stiffness2 = wp.pow(5e6, 2.0 / 3.0)
    dissipation1 = 1.0
    dissipation2 = 1.0

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
    normal = frame[0]
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
        # TODO: hardcoded
        mu_s = 0.9
        mu_d = 0.6
        mu_v = 0.
        transition_velocity = 0.1

        us = (2.0 * mu_s * mu_s) / (mu_s + mu_s) if mu_s != 0 else 0.0
        ud = (2.0 * mu_d * mu_d) / (mu_d + mu_d) if mu_d != 0 else 0.0
        uv = (2.0 * mu_v * mu_v) / (mu_v + mu_v) if mu_v != 0 else 0.0

        v_rel = v_slip / transition_velocity
        f_friction = (f * (wp.min(v_rel, 1.0) * (ud + 2.0 * (us - ud) / (
                1.0 + v_rel * v_rel)) + uv * v_slip))
        force += f_friction * v_t / v_slip

    # Apply forces to bodies
    com1, com2 = xpos_in[worldid, body1], xpos_in[worldid, body2]
    wp.atomic_add(xfrc_applied_out[worldid], body1,
                  support.force_at_point(-1.0 * force, location - com1))
    wp.atomic_add(xfrc_applied_out[worldid], body2,
                  support.force_at_point(1.0 * force, location - com2))

    # todo check for which body is ground
    if body1 == 0:
        wp.atomic_add(grf_out, worldid, -force)
    elif body2 == 0:
        wp.atomic_add(grf_out, worldid, force)



@event_scope
def contact_forces(m: Model, d: Data):
    d.grf.zero_()
    wp.launch(
        _process_contacts_hc,
        dim=(d.naconmax),
        inputs=[
            m.body_rootid,
            m.geom_bodyid,
            d.xpos,
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
