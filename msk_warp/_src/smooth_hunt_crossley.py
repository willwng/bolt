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
        worldid_in: wp.array(dtype=int),
        geom_in: wp.array(dtype=wp.vec2i),
        pos_in: wp.array(dtype=wp.vec3),
        frame_in: wp.array(dtype=wp.mat33),
        friction_in: wp.array(dtype=vec5),
        # Data out:
        xfrc_applied_out: wp.array2d(dtype=wp.spatial_vector),
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

    # Compute velocities of bodies with contact as origin
    cvel1 = cvel_in[worldid, body1]
    cvel2 = cvel_in[worldid, body2]
    subtree_com1 = subtree_com_in[worldid, body_rootid[body1]]
    subtree_com2 = subtree_com_in[worldid, body_rootid[body2]]
    dif1 = cpos - subtree_com1
    dif2 = cpos - subtree_com2
    vel1 = support.transform_velocity(cvel1, dif1)
    vel2 = support.transform_velocity(cvel2, dif2)

    # Compute relative velocities of the bodies
    v = wp.spatial_bottom(vel1 - vel2)

    # Project into contact frame
    normal = frame[0]
    v_n = wp.dot(v, normal)
    v_t = v - (v_n * normal)

    # Calculate the Hertz force. These are hard-coded for now
    stiffness = 5e5
    dissipation = 1.0

    k = stiffness
    c = dissipation
    radius = 0.02
    fH = (4.0 / 3.0) * k * depth * wp.sqrt(radius * k * depth)

    # Hunt-Crossley correction forces
    f = fH * (1.0 + 1.5 * c * v_n)
    if f <= 0:
        return
    force = f * normal

    # Friction cone
    v_slip = wp.length(v_t)
    if v_slip > MJ_MINVAL:
        mu_s = 0.9
        mu_d = 0.6
        mu_v = 0.
        # mu_s = friction_in[conid][0]
        # mu_d = friction_in[conid][1]
        # mu_v = friction_in[conid][2]

        transition_velocity = 0.1  # TODO: hardcoded
        us = (2.0 * mu_s * mu_s) / (mu_s + mu_s) if mu_s != 0 else 0.0
        ud = (2.0 * mu_d * mu_d) / (mu_d + mu_d) if mu_d != 0 else 0.0
        uv = (2.0 * mu_v * mu_v) / (mu_v + mu_v) if mu_v != 0 else 0.0

        v_rel = v_slip / transition_velocity
        f_friction = f * (wp.min(v_rel, 1.0) *
                          (ud + 2.0 * (us - ud) /
                           (1.0 + v_rel * v_rel)) + uv * v_slip)
        force += f_friction * (v_t / v_slip)

    # Apply forces to bodies
    frc_vec = wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), force)
    com1, com2 = xpos_in[worldid, body1], xpos_in[worldid, body2]
    wp.atomic_add(xfrc_applied_out[worldid], body1,
                  support.transform_force(-frc_vec, com1 - cpos))
    wp.atomic_sub(xfrc_applied_out[worldid], body2,
                  support.transform_force(frc_vec, com2 - cpos))


@event_scope
def contact_forces(m: Model, d: Data):
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
            d.contact.worldid,
            d.contact.geom,
            d.contact.pos,
            d.contact.frame,
            d.contact.friction,
        ],
        outputs=[
            d.xfrc_applied,
        ],
    )
    return
