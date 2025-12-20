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
from .types import ContactType
from .types import LimitType
from .types import Data
from .types import Model
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.func
def contact_force_fn(
        # Data in:
        contact_frame_in: wp.array(dtype=wp.mat33),
        contact_dim_in: wp.array(dtype=int),
        contact_efc_address_in: wp.array2d(dtype=int),
        efc_force_in: wp.array2d(dtype=float),
        njmax_in: int,
        nacon_in: wp.array(dtype=int),
        # In:
        worldid: int,
        contact_id: int,
        to_world_frame: bool,
) -> wp.spatial_vector:
    """Extract 6D force:torque for one contact, in contact frame by default."""
    force = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    condim = contact_dim_in[contact_id]
    efc_address = contact_efc_address_in[contact_id, 0]

    if contact_id >= 0 and contact_id <= nacon_in[0] and efc_address >= 0:
        for i in range(condim):
            if contact_efc_address_in[contact_id, i] < njmax_in:
                force[i] = efc_force_in[
                    worldid, contact_efc_address_in[contact_id, i]]

    if to_world_frame:
        # Transform both top and bottom parts of spatial vector by
        # the full contact frame matrix
        t = wp.spatial_top(force) @ contact_frame_in[contact_id]
        b = wp.spatial_bottom(force) @ contact_frame_in[contact_id]
        force = wp.spatial_vector(t, b)

    return force


@wp.func
def joint_limit_torque_fn(
        # Data in:
        dof_lim_efc_address_in: wp.array2d(dtype=int),
        efc_force_in: wp.array2d(dtype=float),
        # In:
        worldid: int,
        dof_limit_id: int,
) -> float:
    efc_address = dof_lim_efc_address_in[worldid, dof_limit_id]
    if efc_address >= 0:
        torque = efc_force_in[worldid, efc_address]
    else:
        torque = 0.0

    return torque


@wp.kernel
def joint_limit_torque_kernel(
        # Data in:
        dof_lim_efc_address_in: wp.array2d(dtype=int),
        efc_force_in: wp.array2d(dtype=float),
        # Data out:
        dof_lim_torque: wp.array2d(dtype=float)
):
    worldid, limitdofid = wp.tid()
    torque = joint_limit_torque_fn(
        dof_lim_efc_address_in,
        efc_force_in,
        worldid,
        limitdofid,
    )
    dof_lim_torque[worldid, limitdofid] = torque
    return


@wp.kernel
def compute_grf_kernel(
        # Model:
        geom_bodyid: wp.array(dtype=int),
        # Data in:
        contact_frame_in: wp.array(dtype=wp.mat33),
        contact_dim_in: wp.array(dtype=int),
        contact_efc_address_in: wp.array2d(dtype=int),
        contact_worldid_in: wp.array(dtype=int),
        contact_geom_in: wp.array(dtype=wp.vec2i),
        efc_force_in: wp.array2d(dtype=float),
        njmax_in: int,
        nacon_in: wp.array(dtype=int),
        # In:
        to_world_frame: bool,
        # Data out:
        grf_out: wp.array(dtype=wp.vec3)
):
    conid = wp.tid()
    if conid >= nacon_in[0]:
        return

    worldid = contact_worldid_in[conid]
    geom = contact_geom_in[conid]
    body1 = geom_bodyid[geom[0]]
    body2 = geom_bodyid[geom[1]]
    contact_force = contact_force_fn(
        contact_frame_in,
        contact_dim_in,
        contact_efc_address_in,
        efc_force_in,
        njmax_in,
        nacon_in,
        worldid,
        conid,
        to_world_frame,
    )

    # Linear GRF
    force = wp.spatial_top(contact_force)
    if body1 == 0:
        wp.atomic_add(grf_out, worldid, force)
    elif body2 == 0:
        wp.atomic_add(grf_out, worldid, -force)


@event_scope
def compute_grf(m: Model, d: Data):
    if wp.static(m.opt.contact_type == ContactType.HUNT_CROSSLEY):
        return  # already handled

    d.grf.zero_()
    wp.launch(
        compute_grf_kernel,
        dim=(d.naconmax),
        inputs=[
            m.geom_bodyid,
            d.contact.frame, d.contact.dim, d.contact.efc_address,
            d.contact.worldid, d.contact.geom,
            d.efc.force, d.njmax, d.nacon,
            True,  # to_world_frame
        ],
        outputs=[d.grf],
    )
    return


@event_scope
def compute_limit_torques(m: Model, d: Data):
    if wp.static(m.opt.limit_type == LimitType.EXPONENTIAL):
        return  # already handled
    d.dof_lim_torque.zero_()

    wp.launch(
        joint_limit_torque_kernel,
        dim=(d.nworld, m.ndoflimit),
        inputs=[
            d.dof_lim_efc_address,
            d.efc.force,
        ],
        outputs=[d.dof_lim_torque],
    )
    return


@event_scope
def compute_joint_moments(m: Model, d: Data):
    d.joint_moments.zero_()
    support.mul_m(m, d, d.joint_moments, d.qacc)
    return
