import warp as wp

from . import math
from .types import Data
from .types import Model
from .types import ArticulatedInertia
from .types import SpatialInertia
from .types import mat66
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _accumulate_articulated_inertia(
        # Model:
        body_children: wp.array(dtype=int),
        body_children_num: wp.array(dtype=int),
        body_children_adr: wp.array(dtype=int),
        jnt_dofnum: wp.array(dtype=int),
        jnt_dofadr: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        mob_H_in: wp.array2d(dtype=wp.spatial_vector),
        mob_phi_in: wp.array2d(dtype=wp.vec3),
        body_Mk_G_in: wp.array2d(dtype=SpatialInertia),
        body_PPlus_in: wp.array2d(dtype=ArticulatedInertia),
        # In:
        body_tree_: wp.array(dtype=int),
        # Data out:
        body_PPlus_out: wp.array2d(dtype=ArticulatedInertia),
):
    worldid, nodeid = wp.tid()
    if integration_done_in[worldid]:
        return
    bodyid = body_tree_[nodeid]

    body_children_adr_ = body_children_adr[bodyid]
    body_children_num_ = body_children_num[bodyid]
    dofnum = jnt_dofnum[bodyid]
    dofadr = jnt_dofadr[bodyid]

    Mk_G = body_Mk_G_in[worldid, bodyid]
    P = math.spatial_inertia_to_articulated_inertia(Mk_G)

    # For each child, we already computed its body inertia P and removed the portion
    # that can't be felt from the parent
    for i in range(body_children_num_):
        childid = body_children[body_children_adr_ + i]
        phi_child = mob_phi_in[worldid, childid]
        PPlus_child = body_PPlus_in[worldid, childid]
        P = math.articulated_inertia_add(P, math.articulated_inertia_shift(PPlus_child, phi_child))

    # Now compute P+.
    # We're going to shove H, PH into matrices to make our life easier.
    # todo: optimize this
    H, PH = mat66(0.0), mat66(0.0)
    for i in range(dofnum):
        H[i] = mob_H_in[worldid, dofadr + i]
        PH[i] = math.articulated_inertia_mul(P, H[i])
    # We stored H and PH as columns, but warp matrices are row major
    H, PH = wp.transpose(H), wp.transpose(PH)

    # First compute D, DI, G, then P+
    # D = ~H @ P @ H
    D = wp.transpose(H) @ PH
    DI = math.invert_upper_left(D, dofnum)
    G = PH * DI

    # Want P+ = P - G * ~PH
    G_PH_T = G @ wp.transpose(PH)
    inertia, mass_moment, _, mass = math.extract_33_blocks(G_PH_T)

    PPlus = math.articulated_inertia_sub(P, ArticulatedInertia(mass, inertia, mass_moment))
    body_PPlus_out[worldid, bodyid] = PPlus
    return


@event_scope
def articulated_body_inertia(m: Model, d: Data):
    # Backward pass: compute P, P+ and propagate to parent
    for i in reversed(range(len(m.body_tree))):
        body_tree = m.body_tree[i]
        wp.launch(
            _accumulate_articulated_inertia,
            dim=(d.nworld, body_tree.size),
            inputs=[
                m.body_children, m.body_children_num, m.body_children_adr, m.jnt_dofnum, m.jnt_dofadr,
                d.integration_done, d.mob_H, d.mob_phi, d.body_Mk_G, d.body_PPlus,
                body_tree,
            ],
            outputs=[d.body_PPlus],
        )
