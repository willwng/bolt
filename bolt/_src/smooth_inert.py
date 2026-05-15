import warp as wp

from . import math
from .types import Data
from .types import Model
from .types import ArticulatedInertia
from .types import SpatialInertia
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _initialize_articulated_inertia(
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        body_Mk_G_in: wp.array2d(dtype=SpatialInertia),
        # Data out:
        body_P_out: wp.array2d(dtype=ArticulatedInertia),
):
    worldid, bodyid = wp.tid()
    if integration_done_in[worldid]:
        return

    Mk_G = body_Mk_G_in[worldid, bodyid]
    P = math.spatial_inertia_to_articulated_inertia(Mk_G)
    body_P_out[worldid, bodyid] = P
    return


@wp.kernel
def _accumulate_child_articulated_inertia(
        # Model:
        body_children: wp.array(dtype=int),
        body_children_num: wp.array(dtype=int),
        body_children_adr: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        mob_phi_in: wp.array2d(dtype=wp.vec3),
        body_P_in: wp.array2d(dtype=ArticulatedInertia),
        body_PPlus_in: wp.array2d(dtype=ArticulatedInertia),
        # In:
        body_tree_: wp.array(dtype=int),
        # Data out:
        body_P_out: wp.array2d(dtype=ArticulatedInertia),
):
    worldid, nodeid = wp.tid()
    if integration_done_in[worldid]:
        return
    bodyid = body_tree_[nodeid]
    body_children_adr_ = body_children_adr[bodyid]
    body_children_num_ = body_children_num[bodyid]

    # Start with the spatial inertia of the current body (in Ground frame)
    P = body_P_in[worldid, bodyid]
    # For each child, we already computed its body inertia P and removed the portion
    # that can't be felt from the parent
    for i in range(body_children_num_):
        childid = body_children[body_children_adr_ + i]
        phi_child = mob_phi_in[worldid, childid]
        PPlus_child = body_PPlus_in[worldid, childid]
        P = math.articulated_inertia_add(P, math.articulated_inertia_shift(PPlus_child, phi_child))
    body_P_out[worldid, bodyid] = P
    return


@wp.kernel
def _compute_articulated_inertia(
        # Model:
        mob_dofnum: wp.array(dtype=int),
        mob_dofadr: wp.array(dtype=int),
        dof_damping: wp.array(dtype=float),
        dof_armature: wp.array(dtype=float),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        mob_H_in: wp.array2d(dtype=wp.spatial_vector),
        body_P_in: wp.array2d(dtype=ArticulatedInertia),
        actual_step_size_in: wp.array(dtype=float),
        # In:
        body_tree_: wp.array(dtype=int),
        implicit_damping: bool,
        # Data out:
        mob_G_out: wp.array2d(dtype=wp.spatial_vector),
        mob_DI_out: wp.array2d(dtype=wp.spatial_vector),
        body_PPlus_out: wp.array2d(dtype=ArticulatedInertia),
):
    worldid, nodeid = wp.tid()
    if integration_done_in[worldid]:
        return
    bodyid = body_tree_[nodeid]
    dofnum = mob_dofnum[bodyid]
    dofadr = mob_dofadr[bodyid]

    # Compute P+, starting with P
    P = body_P_in[worldid, bodyid]

    # We're going to shove H, PH into matrices to make our life easier.
    H, PH = wp.spatial_matrix(0.0), wp.spatial_matrix(0.0)
    for i in range(dofnum):
        H[i] = mob_H_in[worldid, dofadr + i]
        PH[i] = math.articulated_inertia_mul(P, H[i])
    # We stored H and PH as columns, but filled up the matrix row by row
    H, PH = wp.transpose(H), wp.transpose(PH)

    # First compute D, DI, G, then P+
    # D = ~H @ P @ H
    D = wp.transpose(H) @ PH

    # Add armature here, so that M = M + armature TODO(checkme does this make sense?)
    for i in range(dofnum):
        D[i, i] += dof_armature[dofadr + i]

    # DI = D^{-1}, G = P @ H @ D^{-1}
    DI = math.invert_upper_left(D, dofnum)
    G = PH @ DI
    # P+ = P - G * ~PH
    G_PH_T = G @ wp.transpose(PH)
    inertia, mass_moment, _, mass = math.extract_33_blocks(G_PH_T)
    PPlus = math.articulated_inertia_sub(P, ArticulatedInertia(mass, inertia, mass_moment))
    PPlus = math.symmetrize_articulated_inertia(PPlus)
    body_PPlus_out[worldid, bodyid] = PPlus

    # Implicit damping: modify "M^{-1}" to be (M + h * D)^{-1}, but do not modify M itself
    if implicit_damping:
        h = actual_step_size_in[worldid]
        for i in range(dofnum):
            D[i, i] += h * dof_damping[dofadr + i]
        DI = math.invert_upper_left(D, dofnum)
        G = PH @ DI

    # Need G and DI for computing accelerations. here we store col by col
    math.store_mat66(mob_G_out[worldid], G, dofadr, dofnum)
    math.store_mat66(mob_DI_out[worldid], DI, dofadr, dofnum)
    return


@wp.kernel
def _articulated_body_velocity(
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        mob_coriolis_acc_in: wp.array2d(dtype=wp.spatial_vector),
        body_gyro_force_in: wp.array2d(dtype=wp.spatial_vector),
        body_P_in: wp.array2d(dtype=ArticulatedInertia),
        # Data out:
        body_articulated_centrifugal_force_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, bodyid = wp.tid()
    if integration_done_in[worldid] or bodyid == 0:
        return

    P = body_P_in[worldid, bodyid]
    mob_coriolis_acc = mob_coriolis_acc_in[worldid, bodyid]
    gyro_force = body_gyro_force_in[worldid, bodyid]
    body_articulated_centrifugal_force_out[worldid, bodyid] = (
            math.articulated_inertia_mul(P, mob_coriolis_acc) + gyro_force)
    return


@event_scope
def initialize_articulated_body_inertia(m: Model, d: Data):
    """ Initialize articulated inertia based on body's own spatial inertia """
    wp.launch(
        _initialize_articulated_inertia,
        dim=(d.nworld, m.nbody),
        inputs=[d.integration_done, d.body_Mk_G, ],
        outputs=[d.body_P],
    )


@event_scope
def accumulate_articulated_body_inertia(m: Model, d: Data):
    """ Accumulate articulated inertia from children to parent """
    # Backward pass: compute P+. Also store G and DI here
    for i in reversed(range(len(m.body_tree))):
        body_tree = m.body_tree[i]
        # should we fuse these kernels? performance doesn't seem to take a hit keeping them un-fused
        wp.launch(
            _accumulate_child_articulated_inertia,
            dim=(d.nworld, body_tree.size),
            inputs=[
                m.body_children, m.body_children_num, m.body_children_adr,
                d.integration_done, d.mob_phi, d.body_P, d.body_PPlus,
                body_tree,
            ],
            outputs=[d.body_P],
        )
        wp.launch(
            _compute_articulated_inertia,
            dim=(d.nworld, body_tree.size),
            inputs=[
                m.mob_dofnum, m.mob_dofadr, m.dof_damping, m.dof_armature,
                d.integration_done, d.mob_H, d.body_P, d.actual_step_size,
                body_tree, m.opt.implicit_damping
            ],
            outputs=[d.mob_G, d.mob_DI, d.body_PPlus],
        )


@event_scope
def articulated_body_velocity(m: Model, d: Data):
    """ Calculate velocity-related quantities that also depend on articulated body inertias """
    wp.launch(
        _articulated_body_velocity,
        dim=(d.nworld, m.nbody),
        inputs=[
            d.integration_done, d.mob_coriolis_acc, d.body_gyro_force, d.body_P,
        ],
        outputs=[d.body_articulated_centrifugal_force]
    )
