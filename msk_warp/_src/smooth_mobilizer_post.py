import warp as wp

from . import math
from .types import Data
from .types import Model
from .types import SpatialInertia
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _multiply_by_jacobian_transpose_kernel(
        # Model:
        body_children: wp.array(dtype=int),
        body_children_num: wp.array(dtype=int),
        body_children_adr: wp.array(dtype=int),
        mob_dofadr: wp.array(dtype=int),
        mob_dofnum: wp.array(dtype=int),
        mob_phi_in: wp.array2d(dtype=wp.vec3),
        mob_H_in: wp.array2d(dtype=wp.spatial_vector),
        # Data in:
        body_zTmp_in: wp.array2d(dtype=wp.spatial_vector),
        # In:
        body_tree_: wp.array(dtype=int),
        X_in: wp.array2d(dtype=wp.spatial_vector),
        # Data out:
        body_zTmp_out: wp.array2d(dtype=wp.spatial_vector),
        JtX_out: wp.array2d(dtype=float),
):
    worldid, nodeid = wp.tid()

    bodyid = body_tree_[nodeid]
    dofnum = mob_dofnum[bodyid]
    dofadr = mob_dofadr[bodyid]
    H = math.load_mat66(mob_H_in[worldid], dofadr, dofnum)

    # z = X[body]
    z = X_in[worldid, bodyid]

    # z += sum(Phi(child) * z(child)) for all children
    body_children_adr_ = body_children_adr[bodyid]
    body_children_num_ = body_children_num[bodyid]
    for i in range(body_children_num_):
        childid = body_children[body_children_adr_ + i]
        phi_child = mob_phi_in[worldid, childid]
        zPlus_child = body_zTmp_in[worldid, childid]
        z += math.multiply_phi(phi_child, zPlus_child)
    body_zTmp_out[worldid, bodyid] = z

    # Store H^T z
    out = wp.transpose(H) @ z
    for i in range(dofnum):
        JtX_out[worldid, dofadr + i] = out[i]
    return


@wp.kernel
def _multiply_by_M_pass1(
        # Model:
        body_parentid: wp.array(dtype=int),
        mob_dofnum: wp.array(dtype=int),
        mob_dofadr: wp.array(dtype=int),
        # In:
        body_tree_: wp.array(dtype=int),
        qacc: wp.array2d(dtype=float),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        mob_phi_in: wp.array2d(dtype=wp.vec3),
        mob_H_in: wp.array2d(dtype=wp.spatial_vector),
        body_A_GB_in: wp.array2d(dtype=wp.spatial_vector),
        # Data out:
        body_A_GB_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, nodeid = wp.tid()
    if integration_done_in[worldid]:
        return
    bodyid = body_tree_[nodeid]

    # Shift parent's A_GB outward
    pid = body_parentid[bodyid]
    phi = mob_phi_in[worldid, bodyid]
    A_GP = math.multiply_phi_transpose(phi, body_A_GB_in[worldid, pid])

    # A_GB = A_GP + H @ udot
    dofadr, dofnum = mob_dofadr[bodyid], mob_dofnum[bodyid]
    udot = wp.spatial_vector()
    for i in range(dofnum):
        udot[i] = qacc[worldid, dofadr + i]
    H = math.load_mat66(mob_H_in[worldid], dofadr, dofnum)

    A_GB = A_GP + H @ udot
    body_A_GB_out[worldid, bodyid] = A_GB
    return


@wp.kernel
def _multiply_by_M_pass2(
        # Model:
        body_children: wp.array(dtype=int),
        body_children_num: wp.array(dtype=int),
        body_children_adr: wp.array(dtype=int),
        mob_dofnum: wp.array(dtype=int),
        mob_dofadr: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        mob_phi_in: wp.array2d(dtype=wp.vec3),
        mob_H_in: wp.array2d(dtype=wp.spatial_vector),
        body_A_GB_in: wp.array2d(dtype=wp.spatial_vector),
        body_Mk_G_in: wp.array2d(dtype=SpatialInertia),
        all_F_in: wp.array2d(dtype=wp.spatial_vector),
        # In:
        body_tree_: wp.array(dtype=int),
        # Data out:
        all_F_out: wp.array2d(dtype=wp.spatial_vector),
        tau_out: wp.array2d(dtype=float)
):
    worldid, nodeid = wp.tid()
    if integration_done_in[worldid]:
        return
    bodyid = body_tree_[nodeid]

    A_GB = body_A_GB_in[worldid, bodyid]
    Mk_G = body_Mk_G_in[worldid, bodyid]
    F = math.multiply_spatial_inertia(Mk_G, A_GB)

    body_children_adr_ = body_children_adr[bodyid]
    body_children_num_ = body_children_num[bodyid]

    for i in range(body_children_num_):
        childid = body_children[body_children_adr_ + i]
        phi_child = mob_phi_in[worldid, childid]
        F_child = all_F_in[worldid, childid]
        F += math.multiply_phi(phi_child, F_child)

    all_F_out[worldid, bodyid] = F

    # tau = ~H * F
    dofadr, dofnum = mob_dofadr[bodyid], mob_dofnum[bodyid]
    H = math.load_mat66(mob_H_in[worldid], dofadr, dofnum)
    tau = wp.transpose(H) @ F
    for i in range(dofnum):
        tau_out[worldid, dofadr + i] = tau[i]
    return


@event_scope
def multiply_by_jacobian_transpose(m: Model, d: Data, X_in: wp.array2d, JtX_out: wp.array2d):
    # Forward pass, parallelize over bodies within a tree level
    for i in reversed(range(len(m.body_tree))):
        body_tree = m.body_tree[i]
        wp.launch(
            _multiply_by_jacobian_transpose_kernel,
            dim=(d.nworld, body_tree.size),
            inputs=[
                m.body_children, m.body_children_num, m.body_children_adr, m.mob_dofadr, m.mob_dofnum,
                d.mob_phi, d.mob_H, d.body_zTmp,
                body_tree, X_in,
            ],
            outputs=[d.body_zTmp, JtX_out],
        )


@event_scope
def multiply_by_mass(m: Model, d: Data, qacc: wp.array, tau_out: wp.array):
    """ Multiply qacc by M to recover net joint moments """
    body_F_scratch = wp.zeros_like(d.body_F)

    for i in range(1, len(m.body_tree)):
        body_tree = m.body_tree[i]
        wp.launch(
            _multiply_by_M_pass1,
            dim=(d.nworld, body_tree.size),
            inputs=[
                m.body_parentid, m.mob_dofnum, m.mob_dofadr,
                body_tree, qacc,
                d.integration_done, d.mob_phi, d.mob_H, d.body_A_GB,
            ],
            outputs=[d.body_A_GB],
        )

    for i in reversed(range(len(m.body_tree))):
        body_tree = m.body_tree[i]
        wp.launch(
            _multiply_by_M_pass2,
            dim=(d.nworld, body_tree.size),
            inputs=[
                m.body_children, m.body_children_num, m.body_children_adr,
                m.mob_dofnum, m.mob_dofadr,
                d.integration_done, d.mob_phi, d.mob_H, d.body_A_GB, d.body_Mk_G,
                body_F_scratch, body_tree,
            ],
            outputs=[body_F_scratch, tau_out],
        )
