# Copyright 2025 The Newton Developers
# Modified for MSKWarp by Will Wang
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

from . import math
from .types import Data
from .types import Model
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _acc_world(
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        # In:
        gravity: float,
        # Data out:
        body_A_GB_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid = wp.tid()
    if integration_done_in[worldid]:
        return
    body_A_GB_out[worldid, 0] = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0, -gravity, 0.0))
    return


@wp.kernel
def _calc_udot_pass_inward(
        # Model:
        body_children: wp.array(dtype=int),
        body_children_num: wp.array(dtype=int),
        body_children_adr: wp.array(dtype=int),
        jnt_dofnum: wp.array(dtype=int),
        jnt_dofadr: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        mob_phi_in: wp.array2d(dtype=wp.vec3),
        mob_H_in: wp.array2d(dtype=wp.spatial_vector),
        mob_G_in: wp.array2d(dtype=wp.spatial_vector),
        body_articulated_centrifugal_force_in: wp.array2d(dtype=wp.spatial_vector),
        # In:
        body_tree_: wp.array(dtype=int),
        # Data out:
        body_eps_out: wp.array2d(dtype=wp.spatial_vector),
        body_zPlus_out: wp.array2d(dtype=wp.spatial_vector)
):
    worldid, nodeid = wp.tid()
    if integration_done_in[worldid]:
        return
    bodyid = body_tree_[nodeid]
    dofnum = jnt_dofnum[bodyid]
    dofadr = jnt_dofadr[bodyid]

    # Load in H, G as matrices for convenience
    H = math.load_mat66(mob_H_in[worldid], dofadr, dofnum)
    G = math.load_mat66(mob_G_in[worldid], dofadr, dofnum)

    # z = Pa + b - F
    z = body_articulated_centrifugal_force_in[worldid, bodyid]  # todo ext forces

    # z += sum(Phi(child) * zPlus(child)) for all children
    body_children_adr_ = body_children_adr[bodyid]
    body_children_num_ = body_children_num[bodyid]
    for i in range(body_children_num_):
        childid = body_children[body_children_adr_ + i]
        phi_child = mob_phi_in[worldid, childid]
        zPlus_child = body_zPlus_out[worldid, childid]
        z += math.multiply_phi(phi_child, zPlus_child)

    # eps = f - ~H * z
    eps = -wp.transpose(H) @ z
    # zPlus = z + G * eps
    zPlus = z + G @ eps

    body_eps_out[worldid, bodyid] = eps
    body_zPlus_out[worldid, bodyid] = zPlus
    return


@wp.kernel
def _calc_udot_pass_outward(
        # Model:
        body_parentid: wp.array(dtype=int),
        jnt_dofnum: wp.array(dtype=int),
        jnt_dofadr: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        mob_phi_in: wp.array2d(dtype=wp.vec3),
        mob_H_in: wp.array2d(dtype=wp.spatial_vector),
        mob_G_in: wp.array2d(dtype=wp.spatial_vector),
        mob_DI_in: wp.array2d(dtype=wp.spatial_vector),
        body_A_GB_in: wp.array2d(dtype=wp.spatial_vector),
        body_eps_in: wp.array2d(dtype=wp.spatial_vector),
        # In:
        body_tree_: wp.array(dtype=int),
        # Data out:
        qacc_out: wp.array2d(dtype=float),
        body_A_GB_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, nodeid = wp.tid()
    if integration_done_in[worldid]:
        return
    bodyid = body_tree_[nodeid]
    pid = body_parentid[bodyid]
    dofnum = jnt_dofnum[bodyid]
    dofadr = jnt_dofadr[bodyid]

    # Load in H, G as matrices for convenience
    H = math.load_mat66(mob_H_in[worldid], dofadr, dofnum)
    G = math.load_mat66(mob_G_in[worldid], dofadr, dofnum)
    DI = math.load_mat66(mob_DI_in[worldid], dofadr, dofnum)

    A_GB = body_A_GB_in[worldid, bodyid]

    # Shift parent's acceleration outward
    phi = mob_phi_in[worldid, bodyid]
    A_GP = body_A_GB_in[worldid, pid]
    APlus = math.multiply_phi_transpose(phi, A_GP)

    eps = body_eps_in[worldid, bodyid]
    udot = DI @ eps - wp.transpose(G) @ APlus

    # Store joint accelerations
    for i in range(dofnum):
        qacc_out[worldid, dofadr + i] = udot[i]

    A_GB = APlus + H @ udot + A_GB
    body_A_GB_out[worldid, bodyid] = A_GB
    return


@event_scope
def calc_udot(m: Model, d: Data):
    # Initialize world acceleration
    wp.launch(
        _acc_world,
        dim=[d.nworld],
        inputs=[ d.integration_done, m.opt.gravity ],
        outputs=[d.body_A_GB]
    )

    # tip to base, first inward pass
    for i in reversed(range(len(m.body_tree))):
        body_tree = m.body_tree[i]
        wp.launch(
            _calc_udot_pass_inward,
            dim=(d.nworld, body_tree.size),
            inputs=[
                m.body_children, m.body_children_num, m.body_children_adr, m.jnt_dofnum, m.jnt_dofadr,
                d.integration_done, d.mob_phi, d.mob_H, d.mob_G, d.body_articulated_centrifugal_force,
                body_tree,
            ],
            outputs=[d.body_eps, d.body_zPlus]
        )

    # base to tip: acceleration in internal coordinates
    for i in range(1, len(m.body_tree)):
        body_tree = m.body_tree[i]
        wp.launch(
            _calc_udot_pass_outward,
            dim=(d.nworld, body_tree.size),
            inputs=[
                m.body_parentid, m.jnt_dofnum, m.jnt_dofadr,
                d.integration_done, d.mob_phi, d.mob_H, d.mob_G, d.mob_DI, d.body_A_GB, d.body_eps,
                body_tree,
            ],
            outputs=[d.qacc, d.body_A_GB]
        )
