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

from typing import Optional, Tuple

import warp as wp

from .math import motion_cross
from .types import Data
from .types import JointType
from .types import Model
from .types import TileSet
from .types import vec5
from .warp_util import cache_kernel
from .warp_util import event_scope
from .warp_util import kernel as nested_kernel

wp.set_module_options({"enable_backward": False})


@cache_kernel
def mul_m_dense(tile: TileSet, check_skip: bool):
    """Returns a matmul kernel for some tile size."""

    @nested_kernel(module="unique", enable_backward=False)
    def _mul_m_dense(
            # Data In:
            qM_in: wp.array3d(dtype=float),
            # In:
            adr: wp.array(dtype=int),
            vec: wp.array3d(dtype=float),
            skip: wp.array(dtype=bool),
            # Out:
            res: wp.array3d(dtype=float),
    ):
        worldid, nodeid = wp.tid()
        TILE_SIZE = wp.static(tile.size)

        if wp.static(check_skip):
            if skip[worldid]:
                return

        dofid = adr[nodeid]
        qM_tile = wp.tile_load(qM_in[worldid], shape=(TILE_SIZE, TILE_SIZE),
                               offset=(dofid, dofid))
        vec_tile = wp.tile_load(vec[worldid], shape=(TILE_SIZE, 1),
                                offset=(dofid, 0))
        res_tile = wp.tile_matmul(qM_tile, vec_tile)
        wp.tile_store(res[worldid], res_tile, offset=(dofid, 0))

    return _mul_m_dense


@event_scope
def mul_m(
        m: Model,
        d: Data,
        res: wp.array2d(dtype=float),
        vec: wp.array2d(dtype=float),
        skip: Optional[wp.array] = None,
        M: Optional[wp.array] = None,
):
    """Multiply vectors by inertia matrix; optionally skip per world.

    Args:
      m: The model containing kinematic and dynamic information (device).
      d: The data object containing the current state and output arrays (device).
      res: Result: qM @ vec.
      vec: Input vector to multiply by qM.
      skip: Per-world bitmask to skip computing output.
      M: Input matrix: M @ vec.
    """
    check_skip = skip is not None
    skip = skip or wp.empty(0, dtype=bool)

    if M is None:
        M = d.qM

    for tile in m.qM_tiles:
        wp.launch_tiled(
            mul_m_dense(tile, check_skip),
            dim=(d.nworld, tile.adr.size),
            inputs=[
                M,
                tile.adr,
                # note reshape: tile_matmul expects 2d input
                vec.reshape(vec.shape + (1,)),
                skip,
            ],
            outputs=[res.reshape(res.shape + (1,))],
            block_dim=m.block_dim.mul_m_dense,
        )


@wp.kernel
def _apply_ft(
        # Model:
        nbody: int,
        body_parentid: wp.array(dtype=int),
        body_rootid: wp.array(dtype=int),
        dof_bodyid: wp.array(dtype=int),
        # Data in:
        xipos_in: wp.array2d(dtype=wp.vec3),
        subtree_com_in: wp.array2d(dtype=wp.vec3),
        cdof_in: wp.array2d(dtype=wp.spatial_vector),
        # In:
        ft_in: wp.array2d(dtype=wp.spatial_vector),
        flg_add: bool,
        # Out:
        qfrc_out: wp.array2d(dtype=float),
):
    worldid, dofid = wp.tid()
    cdof = cdof_in[worldid, dofid]
    rotational_cdof = wp.vec3(cdof[0], cdof[1], cdof[2])
    jac = wp.spatial_vector(cdof[3], cdof[4], cdof[5], cdof[0], cdof[1],
                            cdof[2])

    dofbodyid = dof_bodyid[dofid]
    accumul = float(0.0)

    for bodyid in range(dofbodyid, nbody):
        ft_body = ft_in[worldid, bodyid]
        if ft_body == wp.spatial_vector():
            continue
        # any body that is in the subtree of dofbodyid is part of the jacobian
        parentid = bodyid
        while parentid != 0 and parentid != dofbodyid:
            parentid = body_parentid[parentid]
        if parentid == 0:
            continue  # body is not part of the subtree
        offset = xipos_in[worldid, bodyid] - subtree_com_in[
            worldid, body_rootid[bodyid]]
        cross_term = wp.cross(rotational_cdof, offset)
        accumul += wp.dot(jac, ft_body) + wp.dot(cross_term,
                                                 wp.spatial_top(ft_body))

    if flg_add:
        qfrc_out[worldid, dofid] += accumul
    else:
        qfrc_out[worldid, dofid] = accumul


def apply_ft(
        m: Model,
        d: Data,
        ft: wp.array2d(dtype=wp.spatial_vector),
        qfrc: wp.array2d(dtype=float),
        flg_add: bool
):
    wp.launch(
        kernel=_apply_ft,
        dim=(d.nworld, m.nv),
        inputs=[m.nbody, m.body_parentid, m.body_rootid, m.dof_bodyid, d.xipos,
                d.subtree_com, d.cdof, ft, flg_add],
        outputs=[qfrc],
    )


@event_scope
def xfrc_accumulate(m: Model, d: Data, qfrc: wp.array2d(dtype=float)):
    """Map applied forces at each body via Jacobians to dof space and accumulate.

    Args:
      m: The model containing kinematic and dynamic information (device).
      d: The data object containing the current state and output arrays (device).
      qfrc: Total applied force mapped to dof space.
    """
    apply_ft(m, d, d.xfrc_applied, qfrc, True)


@wp.func
def contact_force_fn(
        # Model:
        opt_cone: int,
        # Data in:
        contact_frame_in: wp.array(dtype=wp.mat33),
        contact_friction_in: wp.array(dtype=vec5),
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
        # Transform both top and bottom parts of spatial vector by the full contact frame matrix
        t = wp.spatial_top(force) @ contact_frame_in[contact_id]
        b = wp.spatial_bottom(force) @ contact_frame_in[contact_id]
        force = wp.spatial_vector(t, b)

    return force


@wp.kernel
def contact_force_kernel(
        # Model:
        opt_cone: int,
        # Data in:
        contact_frame_in: wp.array(dtype=wp.mat33),
        contact_friction_in: wp.array(dtype=vec5),
        contact_dim_in: wp.array(dtype=int),
        contact_efc_address_in: wp.array2d(dtype=int),
        contact_worldid_in: wp.array(dtype=int),
        efc_force_in: wp.array2d(dtype=float),
        njmax_in: int,
        nacon_in: wp.array(dtype=int),
        # In:
        contact_ids: wp.array(dtype=int),
        to_world_frame: bool,
        # Out:
        out: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    contactid = contact_ids[tid]

    if contactid >= nacon_in[0]:
        return

    worldid = contact_worldid_in[contactid]

    out[tid] = contact_force_fn(
        opt_cone,
        contact_frame_in,
        contact_friction_in,
        contact_dim_in,
        contact_efc_address_in,
        efc_force_in,
        njmax_in,
        nacon_in,
        worldid,
        contactid,
        to_world_frame,
    )


def contact_force(
        m: Model, d: Data, contact_ids: wp.array(dtype=int),
        to_world_frame: bool,
        force: wp.array(dtype=wp.spatial_vector)
):
    """Compute forces for contacts in Data.

    Args:
      m: The model containing kinematic and dynamic information (device).
      d: The data object containing the current state and output arrays (device).
      contact_ids: IDs for each contact.
      to_world_frame: If True, map force from contact to world frame.
      force: Contact forces.
    """
    wp.launch(
        contact_force_kernel,
        dim=contact_ids.size,
        inputs=[
            m.opt.cone,
            d.contact.frame,
            d.contact.friction,
            d.contact.dim,
            d.contact.efc_address,
            d.contact.worldid,
            d.efc.force,
            d.njmax,
            d.nacon,
            contact_ids,
            to_world_frame,
        ],
        outputs=[force],
    )


@wp.func
def force_at_point(frc: wp.vec3, offset: wp.vec3) -> wp.spatial_vector:
    torque = wp.cross(offset, frc)
    return wp.spatial_vector(frc, torque)


@wp.func
def transform_velocity(cvel: wp.spatial_vector,
                       offset: wp.vec3) -> wp.spatial_vector:
    ang = wp.spatial_top(cvel)
    lin = wp.spatial_bottom(cvel)
    pvel_lin = lin + wp.cross(ang, offset)
    return wp.spatial_vector(ang, pvel_lin)


@wp.func
def jac(
        # Model:
        body_parentid: wp.array(dtype=int),
        body_rootid: wp.array(dtype=int),
        dof_bodyid: wp.array(dtype=int),
        # Data in:
        subtree_com_in: wp.array2d(dtype=wp.vec3),
        cdof_in: wp.array2d(dtype=wp.spatial_vector),
        # In:
        point: wp.vec3,
        bodyid: int,
        dofid: int,
        worldid: int,
) -> Tuple[wp.vec3, wp.vec3]:
    dof_bodyid_ = dof_bodyid[dofid]
    in_tree = int(dof_bodyid_ == 0)
    parentid = bodyid
    while parentid != 0:
        if parentid == dof_bodyid_:
            in_tree = 1
            break
        parentid = body_parentid[parentid]

    if not in_tree:
        return wp.vec3(0.0), wp.vec3(0.0)

    offset = point - wp.vec3(subtree_com_in[worldid, body_rootid[bodyid]])

    cdof = cdof_in[worldid, dofid]
    cdof_ang = wp.spatial_top(cdof)
    cdof_lin = wp.spatial_bottom(cdof)

    jacp = cdof_lin + wp.cross(cdof_ang, offset)
    jacr = cdof_ang

    return jacp, jacr
