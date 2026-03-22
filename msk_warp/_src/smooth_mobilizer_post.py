import warp as wp

from . import math
from .types import Data
from .types import Model
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
