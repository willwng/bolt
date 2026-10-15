import warp as wp

from bolt.types import Data
from bolt.types import Model

from .. import math
from ..warp_util import event_scope
from ..warp_util import launch
from ..warp_util import tree_pass_per_world

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _acc_world(
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        # In:
        gravity: wp.vec3,
        # Data out:
        body_A_GB_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid = wp.tid()
    if integration_done_in[worldid]:
        return
    body_A_GB_out[worldid, 0] = wp.spatial_vector(wp.vec3(0.0), -gravity)
    return


@wp.func
def _udot_inward_body(
        worldid: int,
        bodyid: int,
        # Model:
        body_children: wp.array(dtype=int),
        body_children_num: wp.array(dtype=int),
        body_children_adr: wp.array(dtype=int),
        mob_dofnum: wp.array(dtype=int),
        mob_dofadr: wp.array(dtype=int),
        # Data in:
        ufrc_total_in: wp.array2d(dtype=float),
        body_F_in: wp.array2d(dtype=wp.spatial_vector),
        mob_phi_in: wp.array2d(dtype=wp.vec3),
        mob_H_in: wp.array2d(dtype=wp.spatial_vector),
        mob_G_in: wp.array2d(dtype=wp.spatial_vector),
        body_articulated_centrifugal_force_in: wp.array2d(dtype=wp.spatial_vector),
        # Data out:
        body_eps_out: wp.array2d(dtype=wp.spatial_vector),
        body_zPlus_out: wp.array2d(dtype=wp.spatial_vector),
):
    """ Inward (tip to base) pass for one body: eps and zPlus """
    dofnum = mob_dofnum[bodyid]
    dofadr = mob_dofadr[bodyid]

    # Load in H, G as matrices for convenience
    H = math.load_mat66(mob_H_in[worldid], dofadr, dofnum)
    G = math.load_mat66(mob_G_in[worldid], dofadr, dofnum)

    # z = Pa + b - F
    F = body_F_in[worldid, bodyid]
    z = body_articulated_centrifugal_force_in[worldid, bodyid] - F

    # z += sum(Phi(child) * zPlus(child)) for all children
    body_children_adr_ = body_children_adr[bodyid]
    body_children_num_ = body_children_num[bodyid]
    for i in range(body_children_num_):
        childid = body_children[body_children_adr_ + i]
        phi_child = mob_phi_in[worldid, childid]
        zPlus_child = body_zPlus_out[worldid, childid]
        z += math.multiply_phi(phi_child, zPlus_child)

    # eps = f - ~H * z.
    f = math.load_spatial_vec(ufrc_total_in[worldid], dofadr, dofnum)
    eps = f - wp.transpose(H) @ z
    # zPlus = z + G * eps
    zPlus = z + G @ eps

    body_eps_out[worldid, bodyid] = eps
    body_zPlus_out[worldid, bodyid] = zPlus


@wp.func
def _udot_outward_body(
        worldid: int,
        bodyid: int,
        # Model:
        body_parentid: wp.array(dtype=int),
        mob_dofnum: wp.array(dtype=int),
        mob_dofadr: wp.array(dtype=int),
        # Data in:
        mob_phi_in: wp.array2d(dtype=wp.vec3),
        mob_coriolis_in: wp.array2d(dtype=wp.spatial_vector),
        mob_H_in: wp.array2d(dtype=wp.spatial_vector),
        mob_G_in: wp.array2d(dtype=wp.spatial_vector),
        mob_DI_in: wp.array2d(dtype=wp.spatial_vector),
        body_eps_in: wp.array2d(dtype=wp.spatial_vector),
        # Data out:
        qacc_out: wp.array2d(dtype=float),
        body_A_GB_out: wp.array2d(dtype=wp.spatial_vector),
):
    """ Outward (base to tip) pass for one body: joint accelerations and body acceleration """
    pid = body_parentid[bodyid]
    dofnum = mob_dofnum[bodyid]
    dofadr = mob_dofadr[bodyid]

    # Load in H, G as matrices for convenience
    H = math.load_mat66(mob_H_in[worldid], dofadr, dofnum)
    G = math.load_mat66(mob_G_in[worldid], dofadr, dofnum)
    DI = math.load_mat66(mob_DI_in[worldid], dofadr, dofnum)

    # Shift parent's acceleration outward
    phi = mob_phi_in[worldid, bodyid]
    A_GP = body_A_GB_out[worldid, pid]
    APlus = math.multiply_phi_transpose(phi, A_GP)

    eps = body_eps_in[worldid, bodyid]
    udot = DI @ eps - wp.transpose(G) @ APlus

    # Store joint accelerations
    for i in range(dofnum):
        qacc_out[worldid, dofadr + i] = udot[i]

    a = mob_coriolis_in[worldid, bodyid]
    A_GB = APlus + H @ udot + a
    body_A_GB_out[worldid, bodyid] = A_GB


@wp.kernel
def _calc_udot_tree(
        # Model:
        body_parentid: wp.array(dtype=int),
        body_children: wp.array(dtype=int),
        body_children_num: wp.array(dtype=int),
        body_children_adr: wp.array(dtype=int),
        mob_dofnum: wp.array(dtype=int),
        mob_dofadr: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        ufrc_total_in: wp.array2d(dtype=float),
        body_F_in: wp.array2d(dtype=wp.spatial_vector),
        mob_phi_in: wp.array2d(dtype=wp.vec3),
        mob_coriolis_in: wp.array2d(dtype=wp.spatial_vector),
        mob_H_in: wp.array2d(dtype=wp.spatial_vector),
        mob_G_in: wp.array2d(dtype=wp.spatial_vector),
        mob_DI_in: wp.array2d(dtype=wp.spatial_vector),
        body_articulated_centrifugal_force_in: wp.array2d(dtype=wp.spatial_vector),
        # In:
        nbody: int,
        # Data out:
        body_eps_out: wp.array2d(dtype=wp.spatial_vector),
        body_zPlus_out: wp.array2d(dtype=wp.spatial_vector),
        qacc_out: wp.array2d(dtype=float),
        body_A_GB_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid = wp.tid()
    if integration_done_in[worldid]:
        return

    # Tip to base, first inward pass (children have larger ids than their parents)
    for bodyid in range(nbody - 1, -1, -1):
        _udot_inward_body(
            worldid, bodyid,
            body_children, body_children_num, body_children_adr, mob_dofnum, mob_dofadr,
            ufrc_total_in, body_F_in, mob_phi_in, mob_H_in, mob_G_in, body_articulated_centrifugal_force_in,
            body_eps_out, body_zPlus_out,
        )

    # Base to tip: acceleration in internal coordinates
    for bodyid in range(1, nbody):
        _udot_outward_body(
            worldid, bodyid,
            body_parentid, mob_dofnum, mob_dofadr,
            mob_phi_in, mob_coriolis_in, mob_H_in, mob_G_in, mob_DI_in, body_eps_out,
            qacc_out, body_A_GB_out,
        )


@wp.kernel
def _calc_udot_pass_inward(
        # Model:
        body_children: wp.array(dtype=int),
        body_children_num: wp.array(dtype=int),
        body_children_adr: wp.array(dtype=int),
        mob_dofnum: wp.array(dtype=int),
        mob_dofadr: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        ufrc_total_in: wp.array2d(dtype=float),
        body_F_in: wp.array2d(dtype=wp.spatial_vector),
        mob_phi_in: wp.array2d(dtype=wp.vec3),
        mob_H_in: wp.array2d(dtype=wp.spatial_vector),
        mob_G_in: wp.array2d(dtype=wp.spatial_vector),
        body_articulated_centrifugal_force_in: wp.array2d(dtype=wp.spatial_vector),
        # In:
        body_tree_: wp.array(dtype=int),
        # Data out:
        body_eps_out: wp.array2d(dtype=wp.spatial_vector),
        body_zPlus_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, nodeid = wp.tid()
    if integration_done_in[worldid]:
        return
    _udot_inward_body(
        worldid, body_tree_[nodeid],
        body_children, body_children_num, body_children_adr, mob_dofnum, mob_dofadr,
        ufrc_total_in, body_F_in, mob_phi_in, mob_H_in, mob_G_in, body_articulated_centrifugal_force_in,
        body_eps_out, body_zPlus_out,
    )


@wp.kernel
def _calc_udot_pass_outward(
        # Model:
        body_parentid: wp.array(dtype=int),
        mob_dofnum: wp.array(dtype=int),
        mob_dofadr: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        mob_phi_in: wp.array2d(dtype=wp.vec3),
        mob_coriolis_in: wp.array2d(dtype=wp.spatial_vector),
        mob_H_in: wp.array2d(dtype=wp.spatial_vector),
        mob_G_in: wp.array2d(dtype=wp.spatial_vector),
        mob_DI_in: wp.array2d(dtype=wp.spatial_vector),
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
    _udot_outward_body(
        worldid, body_tree_[nodeid],
        body_parentid, mob_dofnum, mob_dofadr,
        mob_phi_in, mob_coriolis_in, mob_H_in, mob_G_in, mob_DI_in, body_eps_in,
        qacc_out, body_A_GB_out,
    )


@event_scope
def calc_udot(m: Model, d: Data):
    # Initialize world acceleration if not explicitly using gravity force
    if not m.opt.explicit_gravity:
        launch(
            _acc_world,
            dim=[d.nworld],
            inputs=[d.integration_done, m.opt.gravity],
            outputs=[d.body_A_GB]
        )

    if tree_pass_per_world(m.body_tree):  # one thread per world
        launch(
            _calc_udot_tree,
            dim=d.nworld,
            inputs=[
                m.body_parentid, m.body_children, m.body_children_num, m.body_children_adr, m.mob_dofnum,
                m.mob_dofadr,
                d.integration_done, d.ufrc_total, d.body_F, d.mob_phi, d.mob_coriolis_acc, d.mob_H, d.mob_G,
                d.mob_DI, d.body_articulated_centrifugal_force,
                m.nbody,
            ],
            outputs=[d.body_eps, d.body_zPlus, d.qacc, d.body_A_GB],
        )
        return

    # tip to base, first inward pass
    for i in reversed(range(len(m.body_tree))):
        body_tree = m.body_tree[i]
        launch(
            _calc_udot_pass_inward,
            dim=(d.nworld, body_tree.size),
            inputs=[
                m.body_children, m.body_children_num, m.body_children_adr, m.mob_dofnum, m.mob_dofadr,
                d.integration_done, d.ufrc_total, d.body_F, d.mob_phi, d.mob_H, d.mob_G,
                d.body_articulated_centrifugal_force,
                body_tree,
            ],
            outputs=[d.body_eps, d.body_zPlus]
        )

    # base to tip: acceleration in internal coordinates
    for i in range(1, len(m.body_tree)):
        body_tree = m.body_tree[i]
        launch(
            _calc_udot_pass_outward,
            dim=(d.nworld, body_tree.size),
            inputs=[
                m.body_parentid, m.mob_dofnum, m.mob_dofadr,
                d.integration_done, d.mob_phi, d.mob_coriolis_acc, d.mob_H, d.mob_G, d.mob_DI, d.body_eps,
                body_tree,
            ],
            outputs=[d.qacc, d.body_A_GB]
        )
