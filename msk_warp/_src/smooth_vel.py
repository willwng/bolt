import warp as wp

from . import math
from . import mobilizers
from .types import Data
from .types import Model
from .types import SpatialInertia
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _across_joint_velocity_jacobian_dot(
        # Model:
        mob_type: wp.array(dtype=int),
        mob_dofadr: wp.array(dtype=int),
        mob_extra_info: wp.array(dtype=wp.vec3),
        mob_dofnum: wp.array(dtype=int),
        mob_to_cst_id: wp.array(dtype=int),
        cst_txfm_dof: wp.array2d(dtype=int),
        cst_txfm_axes: wp.array2d(dtype=wp.vec3),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qvel_in: wp.array2d(dtype=float),
        mob_scratch_in: wp.array3d(dtype=wp.vec3),
        mob_V_FM_in: wp.array2d(dtype=wp.spatial_vector),
        mob_H_FM_in: wp.array2d(dtype=wp.spatial_vector),
        # Data out:
        HDot_FM_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, bodyid = wp.tid()
    if integration_done_in[worldid]:
        return

    mob_type_ = mob_type[bodyid]
    dofadr = mob_dofadr[bodyid]
    extra_info = mob_extra_info[bodyid]
    HDot_FM = HDot_FM_out[worldid]
    mob_scratch = mob_scratch_in[worldid, bodyid]
    qvel = qvel_in[worldid]
    V_FM = mob_V_FM_in[worldid, bodyid]
    H_FM = mob_H_FM_in[worldid]
    dofnum = mob_dofnum[bodyid]
    cst_id = mob_to_cst_id[bodyid]
    # Stores Jacobian in H_FM
    mobilizers.calc_across_joint_velocity_jacobian_dot(mob_type_, dofadr, extra_info, mob_scratch, qvel, V_FM, H_FM,
                                                       dofnum, cst_id, cst_txfm_dof, cst_txfm_axes, HDot_FM)
    return


@wp.kernel
def _compute_body_velocities(
        # Model:
        mob_dofnum: wp.array(dtype=int),
        mob_dofadr: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qvel_in: wp.array2d(dtype=float),
        mob_H_FM_in: wp.array2d(dtype=wp.spatial_vector),
        mob_H_in: wp.array2d(dtype=wp.spatial_vector),
        # Data out:
        v_FM_out: wp.array2d(dtype=wp.spatial_vector),
        v_PB_G_out: wp.array2d(dtype=wp.spatial_vector)
):
    worldid, bodyid = wp.tid()
    if integration_done_in[worldid]:
        return

    dofnum = mob_dofnum[bodyid]
    dofadr = mob_dofadr[bodyid]

    H_FM = mob_H_FM_in[worldid]
    H = mob_H_in[worldid]
    qv = qvel_in[worldid]

    v_FM = wp.spatial_vector()
    v_PB_G = wp.spatial_vector()
    for i in range(dofnum):
        v_FM += H_FM[dofadr + i] * qv[dofadr + i]
        v_PB_G += H[dofadr + i] * qv[dofadr + i]

    v_FM_out[worldid, bodyid] = v_FM
    v_PB_G_out[worldid, bodyid] = v_PB_G
    return


@wp.kernel
def _compute_body_velocities_in_ground(
        # Model:
        body_parentid: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        mob_phi_in: wp.array2d(dtype=wp.vec3),
        v_GB_in: wp.array2d(dtype=wp.spatial_vector),
        V_PB_G_in: wp.array2d(dtype=wp.spatial_vector),
        # In:
        body_tree_: wp.array(dtype=int),
        # Data out:
        V_GB_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, nodeid = wp.tid()
    if integration_done_in[worldid]:
        return

    bodyid = body_tree_[nodeid]
    pid = body_parentid[bodyid]

    V_GP = v_GB_in[worldid, pid]  # parent P's velocity
    V_PB_G = V_PB_G_in[worldid, bodyid]  # child B's vel in P, expressed in G
    phi = mob_phi_in[worldid, bodyid]  # need the "transpose" to shift outward

    V_GB = math.multiply_phi_transpose(phi, V_GP) + V_PB_G
    V_GB_out[worldid, bodyid] = V_GB
    return


@wp.kernel
def _compute_parent_to_child_accelerations(
        # Model:
        mob_dofnum: wp.array(dtype=int),
        mob_dofadr: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qvel_in: wp.array2d(dtype=float),
        mob_HDot_in: wp.array2d(dtype=wp.spatial_vector),
        # Data out:
        VD_PB_G_out: wp.array2d(dtype=wp.spatial_vector)
):
    worldid, bodyid = wp.tid()
    if integration_done_in[worldid]:
        return

    dofnum = mob_dofnum[bodyid]
    dofadr = mob_dofadr[bodyid]

    HDot = mob_HDot_in[worldid]
    qv = qvel_in[worldid]

    VD_PB_G = wp.spatial_vector()
    for i in range(dofnum):
        VD_PB_G += HDot[dofadr + i] * qv[dofadr + i]

    VD_PB_G_out[worldid, bodyid] = VD_PB_G
    return


@wp.kernel
def _parent_to_child_joint_velocity_jacobian_in_ground_dot(
        # Model:
        body_parentid: wp.array(dtype=int),
        mob_X_PF: wp.array(dtype=wp.transform),
        mob_X_MB: wp.array(dtype=wp.transform),
        mob_dofnum: wp.array(dtype=int),
        mob_dofadr: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        mob_X_GB_in: wp.array2d(dtype=wp.transform),
        mob_X_FM_in: wp.array2d(dtype=wp.transform),
        mob_H_FM_in: wp.array2d(dtype=wp.spatial_vector),
        mob_H_PB_G_in: wp.array2d(dtype=wp.spatial_vector),
        mob_HDot_FM_in: wp.array2d(dtype=wp.spatial_vector),
        body_V_GB_in: wp.array2d(dtype=wp.spatial_vector),
        body_V_FM_in: wp.array2d(dtype=wp.spatial_vector),
        # Data out:
        mob_HDot_PB_G_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, bodyid = wp.tid()
    if integration_done_in[worldid]:
        return

    # Collect joint information
    pid = body_parentid[bodyid]
    X_PF = mob_X_PF[bodyid]
    X_MB = mob_X_MB[bodyid]
    dofnum = mob_dofnum[bodyid]
    dofadr = mob_dofadr[bodyid]

    # Pre-computed transform and cross-joint Jacobian
    X_FM = mob_X_FM_in[worldid, bodyid]
    H_FM = mob_H_FM_in[worldid]
    HDot_FM = mob_HDot_FM_in[worldid]

    # We want R_GF so we can re-express the cross-joint velocity V_FB (==V_PB)
    #   in the ground frame, to get V_PB_G.
    R_PF = wp.transform_get_rotation(X_PF)

    # Orientation of the parent joint frame in ground
    X_GP = mob_X_GB_in[worldid, pid]
    R_GP = wp.transform_get_rotation(X_GP)
    R_GF = R_GP * R_PF

    # F and P have the same angular velocity
    w_GF = wp.spatial_top(body_V_GB_in[worldid, pid])

    # Note: time derivative of R_GF is crossMat(w_GF)*R_GF.
    #      H = H_PB_G = R_GF * (H_FM + H_MB_F)
    r_MB = wp.transform_get_translation(X_MB)
    R_FM = wp.transform_get_rotation(X_FM)
    r_MB_F = wp.quat_rotate(R_FM, r_MB)

    # local angular velocity
    w_FM = wp.spatial_top(body_V_FM_in[worldid, bodyid])

    H_PB_G = mob_H_PB_G_in[worldid]
    HDot_PB_G = mob_HDot_PB_G_out[worldid]
    w_FM_x_r_MB_f = wp.cross(w_FM, r_MB_F)
    for i in range(dofnum):
        H_FM_i, HDot_FM_i = H_FM[dofadr + i], HDot_FM[dofadr + i]
        H_FM_i_0, HDot_FM_i_0 = wp.spatial_top(H_FM_i), wp.spatial_top(HDot_FM_i)
        HDot_MB_F_i = wp.spatial_vector(
            wp.vec3(),
            -wp.cross(r_MB_F, HDot_FM_i_0) - wp.cross(w_FM_x_r_MB_f, H_FM_i_0)
        )

        H_PB_G_i = H_PB_G[dofadr + i]
        H_PB_G_i_0, H_PB_G_i_1 = wp.spatial_top(H_PB_G_i), wp.spatial_bottom(H_PB_G_i)
        HDot_PB_G[dofadr + i] = (math.rotate_spatial_vec(R_GF, (HDot_FM_i + HDot_MB_F_i)) +
                                 wp.spatial_vector(wp.cross(w_GF, H_PB_G_i_0), wp.cross(w_GF, H_PB_G_i_1)))
    return


@wp.kernel
def _gyroscopic_forces(
        # Model:
        body_mass: wp.array(dtype=float),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        body_V_GB_in: wp.array2d(dtype=wp.spatial_vector),
        body_Mk_G_in: wp.array2d(dtype=SpatialInertia),
        # Data out:
        body_gyro_force_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, bodyid = wp.tid()
    if integration_done_in[worldid] or bodyid == 0:
        return

    V_GB = body_V_GB_in[worldid, bodyid]
    w_GB = wp.spatial_top(V_GB)

    # Calculate gyroscopic moment and force
    unit_inertia_OB_G = body_Mk_G_in[worldid, bodyid].G
    CB_G = body_Mk_G_in[worldid, bodyid].p
    b = (body_mass[bodyid] *
         wp.spatial_vector(wp.cross(w_GB, unit_inertia_OB_G * w_GB),
                           wp.cross(w_GB, wp.cross(w_GB, CB_G))))
    body_gyro_force_out[worldid, bodyid] = b
    return


@wp.kernel
def _coriolis_acceleration(
        # Model:
        body_parentid: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        mob_phi_in: wp.array2d(dtype=wp.vec3),
        body_V_GB_in: wp.array2d(dtype=wp.spatial_vector),
        body_VD_PB_G_in: wp.array2d(dtype=wp.spatial_vector),
        body_total_coriolis_acc_in: wp.array2d(dtype=wp.spatial_vector),
        # In:
        body_tree_: wp.array(dtype=int),
        # Data out:
        mob_coriolis_acc_out: wp.array2d(dtype=wp.spatial_vector),
        body_total_coriolis_acc_out: wp.array2d(dtype=wp.spatial_vector)
):
    worldid, nodeid = wp.tid()
    if integration_done_in[worldid]:
        return

    bodyid = body_tree_[nodeid]

    V_GB = body_V_GB_in[worldid, bodyid]
    w_GB = wp.spatial_top(V_GB)
    v_GB = wp.spatial_bottom(V_GB)

    # Parent velocity
    pid = body_parentid[bodyid]
    V_GP = body_V_GB_in[worldid, pid]
    w_GP = wp.spatial_top(V_GP)
    v_GP = wp.spatial_bottom(V_GP)

    # Calculate this mobilizer's incremental contribution to the coriolis acceleration
    VD_PB_G = body_VD_PB_G_in[worldid, bodyid]
    A = wp.spatial_vector(
        wp.spatial_top(VD_PB_G),
        wp.spatial_bottom(VD_PB_G) + wp.cross(w_GP, v_GB - v_GP)
    )
    mob_coriolis_acc_out[worldid, bodyid] = A

    # Next, the total coriolis acceleration a of body B is the total coriolis acceleration
    # of parent shifted outward, plus B's local contribution A
    parentA = body_total_coriolis_acc_in[worldid, pid]
    phi = mob_phi_in[worldid, bodyid]
    a = math.multiply_phi_transpose(phi, parentA) + A
    body_total_coriolis_acc_out[worldid, bodyid] = a
    return


@wp.kernel
def _centrifugal_forces(
        # Model:
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        body_Mk_G_in: wp.array2d(dtype=SpatialInertia),
        body_total_coriolis_acc_in: wp.array2d(dtype=wp.spatial_vector),
        body_gyro_force_in: wp.array2d(dtype=wp.spatial_vector),
        # Data out:
        body_total_centrifugal_force_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, bodyid = wp.tid()
    if integration_done_in[worldid] or bodyid == 0:
        return

    Mk_G = body_Mk_G_in[worldid, bodyid]
    a = body_total_coriolis_acc_in[worldid, bodyid]
    b = body_gyro_force_in[worldid, bodyid]

    body_total_centrifugal_force_out[worldid, bodyid] = math.multiply_spatial_inertia(Mk_G, a) + b
    return


@event_scope
def joint_velocity_jacobian_dot(m: Model, d: Data):
    """ Computes the derivative of the Jacobian """
    wp.launch(
        _across_joint_velocity_jacobian_dot,
        dim=(d.nworld, m.nbody),
        inputs=[
            m.mob_type, m.mob_dofadr, m.mob_extra_info, m.mob_dofnum, m.mob_to_cst_id, m.cst_txfm_dof, m.cst_txfm_axes,
            d.integration_done, d.qvel, d.mob_scratch, d.body_V_FM, d.mob_H_FM
        ],
        outputs=[d.mob_HDot_FM],
    )

    wp.launch(
        _parent_to_child_joint_velocity_jacobian_in_ground_dot,
        dim=(d.nworld, m.nbody),
        inputs=[
            m.body_parentid, m.mob_X_PF, m.mob_X_MB, m.mob_dofnum, m.mob_dofadr,
            d.integration_done, d.mob_X_GB, d.mob_X_FM, d.mob_H_FM, d.mob_H, d.mob_HDot_FM,
            d.body_V_GB, d.body_V_FM
        ],
        outputs=[d.mob_HDot],
    )


@event_scope
def compute_body_velocities(m: Model, d: Data):
    """
    Compute mobilizer relative velocities and body velocities in parent frame,
     then body velocities in the ground frame.
    """
    # Compute mobilizer spatial velocity and body velocity in parent (measured in ground)
    wp.launch(
        _compute_body_velocities,
        dim=(d.nworld, m.nbody),
        inputs=[
            m.mob_dofnum, m.mob_dofadr,
            d.integration_done, d.qvel, d.mob_H_FM, d.mob_H
        ],
        outputs=[d.body_V_FM, d.body_V_PB_G],
    )

    # Body velocity in ground frame: requires forward pass
    for i in range(1, len(m.body_tree)):
        body_tree = m.body_tree[i]
        wp.launch(
            _compute_body_velocities_in_ground,
            dim=(d.nworld, body_tree.size),
            inputs=[
                m.body_parentid,
                d.integration_done, d.mob_phi, d.body_V_GB, d.body_V_PB_G,
                body_tree,
            ],
            outputs=[d.body_V_GB],
        )


@event_scope
def compute_parent_to_child_accelerations(m: Model, d: Data):
    wp.launch(
        _compute_parent_to_child_accelerations,
        dim=(d.nworld, m.nbody),
        inputs=[
            m.mob_dofnum, m.mob_dofadr,
            d.integration_done, d.qvel, d.mob_HDot
        ],
        outputs=[d.body_VD_PB_G],
    )
    return


@event_scope
def joint_independent_kinematics_vel(m: Model, d: Data):
    """ Computes remaining kinematic-dependent quantities """
    wp.launch(
        _gyroscopic_forces,
        dim=(d.nworld, m.nbody),
        inputs=[
            m.body_mass,
            d.integration_done, d.body_V_GB, d.body_Mk_G
        ],
        outputs=[d.body_gyro_force],
    )

    # Coriolis acceleration, requires forward pass
    for i in range(1, len(m.body_tree)):
        body_tree = m.body_tree[i]
        wp.launch(
            _coriolis_acceleration,
            dim=(d.nworld, body_tree.size),
            inputs=[
                m.body_parentid,
                d.integration_done, d.mob_phi, d.body_V_GB, d.body_VD_PB_G, d.body_total_coriolis_acc,
                body_tree
            ],
            outputs=[d.mob_coriolis_acc, d.body_total_coriolis_acc],
        )

    # Total of the rotational velocity-dependent forces acting on this body
    wp.launch(
        _centrifugal_forces,
        dim=(d.nworld, m.nbody),
        inputs=[
            d.integration_done, d.body_Mk_G, d.body_total_coriolis_acc, d.body_gyro_force
        ],
        outputs=[d.body_total_centrifugal_force],
    )
