import warp as wp

from . import math
from . import mobilizers
from .types import Data
from .types import Model
from .types import SpatialInertia
from .types import MobilizerType
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _calc_mobilizer_X_FM(
        # Model:
        mob_type: wp.array(dtype=int),
        mob_qposadr: wp.array(dtype=int),
        mob_extra_info: wp.array(dtype=wp.vec3),
        mob_to_cst_id: wp.array(dtype=int),
        cst_txfm_axes: wp.array2d(dtype=wp.vec3),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qpos_in: wp.array2d(dtype=float),
        # Data out:
        mob_X_FM_out: wp.array2d(dtype=wp.transform),
        mob_scratch_out: wp.array3d(dtype=wp.vec3),
):
    worldid, bodyid = wp.tid()
    if integration_done_in[worldid]:
        return

    # Collect joint information
    mob_type_ = mob_type[bodyid]
    qpos_start = mob_qposadr[bodyid]
    extra_info = mob_extra_info[bodyid]
    mob_scratch = mob_scratch_out[worldid, bodyid]
    cst_id = mob_to_cst_id[bodyid]

    # Joint transform: parent mobilizer to child mobilizer
    X_FM = mobilizers.calcX_FM(mob_type_, qpos_start, qpos_in[worldid], extra_info,
                               cst_id, cst_txfm_axes, mob_scratch)
    mob_X_FM_out[worldid, bodyid] = X_FM
    return


@wp.kernel
def _body_transforms_ground(
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        # Data out:
        mob_X_PB_out: wp.array2d(dtype=wp.transform),
        mob_X_GB_out: wp.array2d(dtype=wp.transform),
):
    worldid = wp.tid()
    if integration_done_in[worldid]:
        return
    mob_X_PB_out[worldid, 0] = wp.transform_identity()
    mob_X_GB_out[worldid, 0] = wp.transform_identity()


@wp.kernel
def _body_transforms_level(
        # Model:
        body_parentid: wp.array(dtype=int),
        mob_type: wp.array(dtype=int),
        mob_X_PF: wp.array(dtype=wp.transform),
        mob_X_MB: wp.array(dtype=wp.transform),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        mob_X_FM_in: wp.array2d(dtype=wp.transform),
        mob_X_GB_in: wp.array2d(dtype=wp.transform),
        # In:
        body_tree_: wp.array(dtype=int),
        # Data out:
        mob_X_PB_out: wp.array2d(dtype=wp.transform),
        mob_X_GB_out: wp.array2d(dtype=wp.transform),
):
    worldid, nodeid = wp.tid()
    if integration_done_in[worldid]:
        return

    bodyid = body_tree_[nodeid]
    pid = body_parentid[bodyid]

    X_MB = mob_X_MB[bodyid]  # Transform from mobilizer frame M to body frame B
    X_PF = mob_X_PF[bodyid]  # Transform from parent frame P to mobilizer fixed frame F
    X_FM = mob_X_FM_in[worldid, bodyid]  # just calculated
    X_GP = mob_X_GB_in[worldid, pid]  # already calculated
    if pid == 0 and mob_type[bodyid] == MobilizerType.FREE:
        X_PF = wp.transform_identity()

    X_PB = X_PF * X_FM * X_MB
    X_GB = X_GP * X_PB

    mob_X_PB_out[worldid, bodyid] = X_PB
    mob_X_GB_out[worldid, bodyid] = X_GB
    return


@wp.kernel
def _geom_local_to_global(
        # Model:
        geom_bodyid: wp.array(dtype=int),
        geom_X_loc: wp.array(dtype=wp.transform),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        mob_X_GB_in: wp.array2d(dtype=wp.transform),
        # Data out:
        geom_X_out: wp.array2d(dtype=wp.transform),
):
    worldid, geomid = wp.tid()
    if integration_done_in[worldid]:
        return

    bodyid = geom_bodyid[geomid]
    X_GB = mob_X_GB_in[worldid, bodyid]
    geom_X_out[worldid, geomid] = X_GB * geom_X_loc[geomid]


@wp.kernel
def _vis_local_to_global(
        # Model:
        vis_bodyid: wp.array(dtype=int),
        vis_X_loc: wp.array(dtype=wp.transform),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        body_X_in: wp.array2d(dtype=wp.transform),
        # Data out:
        vis_X_out: wp.array2d(dtype=wp.transform),
):
    worldid, visid = wp.tid()
    if integration_done_in[worldid]:
        return

    bodyid = vis_bodyid[visid]
    body_X = body_X_in[worldid, bodyid]
    vis_X_out[worldid, visid] = body_X * vis_X_loc[visid]


@wp.kernel
def _site_local_to_global(
        # Model:
        site_bodyid: wp.array(dtype=int),
        site_pos: wp.array(dtype=wp.vec3),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        body_X_in: wp.array2d(dtype=wp.transform),
        # Data out:
        site_rel_pos_B_out: wp.array2d(dtype=wp.vec3),
        site_pos_G_out: wp.array2d(dtype=wp.vec3),
):
    worldid, siteid = wp.tid()
    if integration_done_in[worldid]:
        return

    bodyid = site_bodyid[siteid]
    body_X = body_X_in[worldid, bodyid]
    body_quat = wp.transform_get_rotation(body_X)
    body_pos = wp.transform_get_translation(body_X)
    # Relative to body
    rpos = wp.quat_rotate(body_quat, site_pos[siteid])
    site_rel_pos_B_out[worldid, siteid] = rpos
    # Position measured in ground
    site_pos_G_out[worldid, siteid] = body_pos + rpos
    return


@wp.kernel
def _across_joint_velocity_jacobian(
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
        mob_scratch_in: wp.array3d(dtype=wp.vec3),
        # Data out:
        H_FM_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, bodyid = wp.tid()
    if integration_done_in[worldid]:
        return

    mob_type_ = mob_type[bodyid]
    dofadr = mob_dofadr[bodyid]
    extra_info = mob_extra_info[bodyid]
    H_FM = H_FM_out[worldid]
    mob_scratch = mob_scratch_in[worldid, bodyid]
    dofnum = mob_dofnum[bodyid]
    cst_id = mob_to_cst_id[bodyid]

    # Stores Jacobian in H_FM
    mobilizers.calc_across_joint_velocity_jacobian(mob_type_, dofadr, extra_info, mob_scratch,
                                                   dofnum, cst_id, cst_txfm_dof, cst_txfm_axes, H_FM)
    return


@wp.kernel
def _parent_to_child_joint_velocity_jacobian_in_ground(
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
        H_FM_in: wp.array2d(dtype=wp.spatial_vector),
        # Data out:
        H_PB_G_out: wp.array2d(dtype=wp.spatial_vector),
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
    H_FM = H_FM_in[worldid]

    # We want R_GF so we can re-express the cross-joint velocity V_FB (==V_PB)
    #   in the ground frame, to get V_PB_G.
    R_PF = wp.transform_get_rotation(X_PF)

    # Compute orientation of the parent joint frame in ground
    X_GP = mob_X_GB_in[worldid, pid]
    R_GP = wp.transform_get_rotation(X_GP)
    R_GF = R_GP * R_PF

    # We want r_MB_F, that is, the vector from Mo to Bo, expressed in F
    r_MB = wp.transform_get_translation(X_MB)
    R_FM = wp.transform_get_rotation(X_FM)
    r_MB_F = wp.quat_rotate(R_FM, r_MB)

    H_PB_G = H_PB_G_out[worldid]
    for i in range(dofnum):
        H_FM_i = H_FM[dofadr + i]
        H_MB_F_i = wp.spatial_vector(wp.vec3(), -wp.cross(r_MB_F, wp.spatial_top(H_FM_i)))
        H_PB_G[dofadr + i] = math.rotate_spatial_vec(R_GF, (H_FM_i + H_MB_F_i))
    return


@wp.kernel
def _joint_independent_kinematics(
        # Model:
        body_parentid: wp.array(dtype=int),
        body_mass_center: wp.array(dtype=wp.vec3),
        body_mass: wp.array(dtype=float),
        body_unit_inertia: wp.array(dtype=wp.mat33),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        mob_X_GB_in: wp.array2d(dtype=wp.transform),
        mob_X_PB_in: wp.array2d(dtype=wp.transform),
        # Data out:
        mob_phi_out: wp.array2d(dtype=wp.vec3),
        body_COM_G_out: wp.array2d(dtype=wp.vec3),
        body_Mk_G_out: wp.array2d(dtype=SpatialInertia),
):
    worldid, bodyid = wp.tid()
    if integration_done_in[worldid] or bodyid == 0:
        return

    pid = body_parentid[bodyid]
    X_GB = mob_X_GB_in[worldid, bodyid]
    X_GP = mob_X_GB_in[worldid, pid]
    X_PB = mob_X_PB_in[worldid, bodyid]

    # Re-express parent-to-child shift vector (Bo-Po) into the ground frame.
    p_PB_G = wp.quat_rotate(wp.transform_get_rotation(X_GP), wp.transform_get_translation(X_PB))

    # Phi matrix
    mob_phi_out[worldid, bodyid] = p_PB_G

    # Calculate spatial mass properties
    R_GB = wp.transform_get_rotation(X_GB)
    p_GB = wp.transform_get_translation(X_GB)

    # re-express inertia in ground:
    G_Bo_G = math.reexpress_inertia(body_unit_inertia[bodyid], wp.quat_inverse(R_GB))
    p_BBc_G = wp.quat_rotate(R_GB, body_mass_center[bodyid])

    body_COM_G_out[worldid, bodyid] = p_GB + p_BBc_G

    # Mk: the spatial inertia matrix about the body origin
    body_Mk_G_out[worldid, bodyid] = SpatialInertia(body_mass[bodyid], p_BBc_G, G_Bo_G)
    return


@wp.kernel
def _beam_visuals(
        # Model:
        body_parentid: wp.array(dtype=int),
        beam_to_mob_id: wp.array(dtype=int),
        mob_X_PF: wp.array(dtype=wp.transform),
        mob_qposadr: wp.array(dtype=int),
        mob_extra_info: wp.array(dtype=wp.vec3),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qpos_in: wp.array2d(dtype=float),
        mob_X_GB_in: wp.array2d(dtype=wp.transform),
        # In:
        nbeam_visuals: int,
        # Data out:
        vis_beam_pos_out: wp.array3d(dtype=wp.vec3),
):
    worldid, beamid = wp.tid()
    if integration_done_in[worldid]:
        return

    bodyid = beam_to_mob_id[beamid]
    pid = body_parentid[bodyid]
    X_PF = mob_X_PF[bodyid]
    X_GP = mob_X_GB_in[worldid, pid]
    extra_info = mob_extra_info[bodyid]
    qpos = qpos_in[worldid]
    qpos_adr = mob_qposadr[bodyid]

    step = 1.0 / (float(nbeam_visuals) - 1.0)
    for i in range(nbeam_visuals):
        fraction = step * float(i)
        pt_local = mobilizers.compute_beam_transform(fraction, qpos, qpos_adr, extra_info)
        pt_G = wp.transform_point(X_GP * X_PF, pt_local)
        vis_beam_pos_out[worldid, beamid, i] = pt_G
    return


@event_scope
def calc_mobilizer_X_FM(m: Model, d: Data):
    wp.launch(
        _calc_mobilizer_X_FM,
        dim=(d.nworld, m.nbody),
        inputs=[
            m.mob_type, m.mob_qposadr, m.mob_extra_info, m.mob_to_cst_id, m.cst_txfm_axes,
            d.integration_done, d.qpos,
        ],
        outputs=[d.mob_X_FM, d.mob_scratch],
    )


@event_scope
def calc_body_transforms(m: Model, d: Data):
    """ Computes world-space transformations for all bodies """
    # World body
    wp.launch(
        _body_transforms_ground,
        dim=(d.nworld),
        inputs=[d.integration_done],
        outputs=[d.mob_X_PB, d.mob_X_GB]
    )

    # Forward pass, parallelize over bodies within a tree level
    for i in range(1, len(m.body_tree)):
        body_tree = m.body_tree[i]
        wp.launch(
            _body_transforms_level,
            dim=(d.nworld, body_tree.size),
            inputs=[
                m.body_parentid, m.mob_type, m.mob_X_PF, m.mob_X_MB,
                d.integration_done, d.mob_X_FM, d.mob_X_GB,
                body_tree,
            ],
            outputs=[d.mob_X_PB, d.mob_X_GB],
        )


@event_scope
def joint_velocity_jacobian(m: Model, d: Data):
    """ Computes the Jacobian mapping joint velocities to body velocities """
    wp.launch(
        _across_joint_velocity_jacobian,
        dim=(d.nworld, m.nbody),
        inputs=[
            m.mob_type, m.mob_dofadr, m.mob_extra_info, m.mob_dofnum, m.mob_to_cst_id, m.cst_txfm_dof, m.cst_txfm_axes,
            d.integration_done, d.mob_scratch,
        ],
        outputs=[d.mob_H_FM],
    )
    wp.launch(
        _parent_to_child_joint_velocity_jacobian_in_ground,
        dim=(d.nworld, m.nbody),
        inputs=[
            m.body_parentid, m.mob_X_PF, m.mob_X_MB, m.mob_dofnum, m.mob_dofadr,
            d.integration_done, d.mob_X_GB, d.mob_X_FM, d.mob_H_FM,
        ],
        outputs=[d.mob_H],
    )


@event_scope
def joint_independent_kinematics(m: Model, d: Data):
    """ Computes remaining kinematic-dependent quantities """
    wp.launch(
        _joint_independent_kinematics,
        dim=(d.nworld, m.nbody),
        inputs=[
            m.body_parentid,
            m.body_mass_center,
            m.body_mass,
            m.body_unit_inertia_OB_B,
            d.integration_done,
            d.mob_X_GB,
            d.mob_X_PB,
        ],
        outputs=[d.mob_phi, d.body_COM_G, d.body_Mk_G],
    )


@event_scope
def attachment_kinematics(m: Model, d: Data):
    # Collision geometry: only position is needed
    wp.launch(
        _geom_local_to_global,
        dim=(d.nworld, m.ngeom),
        inputs=[m.geom_bodyid, m.geom_X_loc, d.integration_done, d.mob_X_GB],
        outputs=[d.geom_X],
    )

    # Sites
    wp.launch(
        _site_local_to_global,
        dim=(d.nworld, m.nsite),
        inputs=[m.site_bodyid, m.site_offset, d.integration_done, d.mob_X_GB],
        outputs=[d.site_rel_pos_B, d.site_pos_G],
    )

    # Visuals: only position is needed
    if wp.static(m.opt.visuals):
        wp.launch(
            _vis_local_to_global,
            dim=(d.nworld, m.nvis),
            inputs=[m.vis_bodyid, m.vis_X_loc, d.integration_done, d.mob_X_GB],
            outputs=[d.vis_X],
        )

        wp.launch(
            _beam_visuals,
            dim=(d.nworld, m.nbeams),
            inputs=[
                m.body_parentid, m.beam_to_mob_id, m.mob_X_PF, m.mob_qposadr, m.mob_extra_info,
                d.integration_done, d.qpos, d.mob_X_GB,
                m.opt.nbeam_visuals,
            ],
            outputs=[d.vis_beam_pos],
        )
