import warp as wp

from . import math
from .types import MobilizerType
from .types import Model
from .types import Data
from .types import mat36
from .consts import MSK_MINVAL
from .consts import (IDX_SCRATCH_ROT_F, IDX_SCRATCH_ROT_DF, IDX_SCRATCH_ROT_D2F,
                     IDX_SCRATCH_TRANS_F, IDX_SCRATCH_TRANS_DF, IDX_SCRATCH_TRANS_D2F)
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.func
def ensure_valid_qpos(
        mobtype: int,
        qadr: int,
        qpos: wp.array(dtype=float),
):
    """ Ensures that coordinates are valid (i.e., quaternions are normalized) """
    if mobtype == MobilizerType.FREE or mobtype == MobilizerType.BALL:
        q = wp.quat(qpos[qadr + 0], qpos[qadr + 1], qpos[qadr + 2], qpos[qadr + 3])
        if wp.length(q) < MSK_MINVAL:
            q = wp.quat_identity()
        q = wp.normalize(q)
        for i in range(4):
            qpos[qadr + i] = q[i]
    return


@wp.func
def calcX_FM(
        mobtype: int,
        qadr: int,
        qpos: wp.array(dtype=float),
        extra_info: wp.vec3,
        # custom joints
        cst_id: int,
        cst_txfm_axes: wp.array2d(dtype=wp.vec3),
        # out:
        mob_scratch_out: wp.array(dtype=wp.vec3),
) -> wp.transform:
    """ Computes the mobilizer transformation from the parent joint frame F to the mobilizer frame M """
    if mobtype == MobilizerType.FREE:
        r = wp.quat(qpos[qadr + 0], qpos[qadr + 1], qpos[qadr + 2], qpos[qadr + 3])
        p = wp.vec3(qpos[qadr + 4], qpos[qadr + 5], qpos[qadr + 6])
        r = wp.normalize(r)
        return wp.transform(p, r)

    elif mobtype == MobilizerType.PIN:
        pin_axis = wp.vec3(0.0, 0.0, 1.0)
        r = wp.quat_from_axis_angle(pin_axis, qpos[qadr])
        return wp.transform(wp.vec3(), r)

    elif mobtype == MobilizerType.SLIDER:
        slide_axis = wp.vec3(1.0, 0.0, 0.0)
        p = qpos[qadr] * slide_axis
        return wp.transform(p, wp.quat_identity())

    elif mobtype == MobilizerType.UNIVERSAL:
        axis0 = wp.vec3(1.0, 0.0, 0.0)
        axis1 = wp.vec3(0.0, 1.0, 0.0)

        qloc0 = wp.quat_from_axis_angle(axis0, qpos[qadr + 0])
        qloc1 = wp.quat_from_axis_angle(axis1, qpos[qadr + 1])
        r = qloc0 * qloc1

        mob_scratch_out[0] = wp.quat_rotate(r, axis1)
        return wp.transform(wp.vec3(), r)

    elif mobtype == MobilizerType.GIMBAL:
        # Euler/Body-Fixed XYZ order
        r = math.quat_from_xyz(qpos[qadr + 0], qpos[qadr + 1], qpos[qadr + 2])
        # store q0, q1, q2 for later
        mob_scratch_out[0] = wp.vec3(qpos[qadr + 0], qpos[qadr + 1], qpos[qadr + 2])
        return wp.transform(wp.vec3(), r)

    elif mobtype == MobilizerType.BEAM:
        r = math.quat_from_xyz(qpos[qadr + 0], qpos[qadr + 1], qpos[qadr + 2])
        # Beam deflection
        length, deflection_coeff, displacement_coeff = extra_info[0], extra_info[1], extra_info[2]
        theta_sq = qpos[qadr + 0] * qpos[qadr + 0] + qpos[qadr + 1] * qpos[qadr + 1]
        p = wp.vec3(
            qpos[qadr + 1] * deflection_coeff,
            -qpos[qadr + 0] * deflection_coeff,
            length - displacement_coeff * theta_sq
        )
        # store q0, q1, q2 for later
        mob_scratch_out[0] = wp.vec3(qpos[qadr + 0], qpos[qadr + 1], qpos[qadr + 2])
        return wp.transform(p, r)

    elif mobtype == MobilizerType.ELLIPSOID:
        r = math.quat_from_xyz(qpos[qadr + 0], qpos[qadr + 1], qpos[qadr + 2])
        # Ellipsoid translation: the z axis of body is assumed to be normal to the joint ellipsoid
        semi = extra_info
        n = wp.quat_rotate(r, wp.vec(0.0, 0.0, 1.0))
        p = wp.vec3(semi.x * n.x, semi.y * n.y, semi.z * n.z)

        mob_scratch_out[0] = n
        return wp.transform(p, r)

    elif mobtype == MobilizerType.BALL:
        r = wp.quat(qpos[qadr + 0], qpos[qadr + 1], qpos[qadr + 2], qpos[qadr + 3])
        r = wp.normalize(r)
        return wp.transform(wp.vec3(), r)

    elif mobtype == MobilizerType.CUSTOM:
        # For custom joints, we already put function evaluation into scratch
        f_rot, f_trans = mob_scratch_out[IDX_SCRATCH_ROT_F], mob_scratch_out[IDX_SCRATCH_TRANS_F]
        # Fetch the axes to perform the transformation
        rot_ax1, rot_ax2, rot_ax3 = cst_txfm_axes[cst_id, 0], cst_txfm_axes[cst_id, 1], cst_txfm_axes[cst_id, 2]
        trans_ax1, trans_ax2, trans_ax3 = cst_txfm_axes[cst_id, 3], cst_txfm_axes[cst_id, 4], cst_txfm_axes[cst_id, 5]

        r = math.quat_from_three_angle_axes(f_rot[0], f_rot[1], f_rot[2], rot_ax1, rot_ax2, rot_ax3)
        p = math.trans_from_three_shift_axes(f_trans[0], f_trans[1], f_trans[2], trans_ax1, trans_ax2, trans_ax3)
        return wp.transform(p, r)

    elif mobtype == MobilizerType.WELD:
        return wp.transform(wp.vec3(), wp.quat_identity())

    elif mobtype == MobilizerType.WORLD:
        return wp.transform(wp.vec3(), wp.quat_identity())
    else:
        assert False, f"Unknown joint type {mobtype}"

    return wp.transform_identity()


@wp.func
def compute_beam_transform(
        fraction: float,
        qpos: wp.array(dtype=float),
        qpos_adr: int,
        extra_info: wp.vec3
) -> wp.vec3:
    """ For a given fraction along the beam, compute the mobilizer transform for that point """
    q0, q1, q2 = qpos[qpos_adr], qpos[qpos_adr + 1], qpos[qpos_adr + 2]
    L, deflection_coeff, displacement_coeff = extra_info[0], extra_info[1], extra_info[2]
    z = fraction * L

    theta_sq = q0 * q0 + q1 * q1

    C_deflection = (z * z * (3.0 * L - z)) / (3.0 * L ** 2.0)
    C_displacement = -(z ** 3.0 * (20.0 * L ** 2.0 - 15.0 * L * z + 3.0 * z ** 2.0)) / (30.0 * L ** 4.0)
    d_x = q1 * C_deflection
    d_y = -q0 * C_deflection
    d_z = C_displacement * theta_sq
    pt_local = wp.vec3(d_x, d_y, z + d_z)
    return pt_local


@wp.func
def calc_across_joint_velocity_jacobian(
        mobtype: int,
        dofadr: int,
        extra_info: wp.vec3,
        mob_scratch: wp.array(dtype=wp.vec3),
        dofnum: int,
        # custom joints
        cst_id: int,
        cst_txfm_dof: wp.array2d(dtype=int),
        cst_txfm_axes: wp.array2d(dtype=wp.vec3),
        # Out
        H_FM: wp.array(dtype=wp.spatial_vector),
):
    """
    Computes the motion subspace and joint velocity contribution
    """
    if mobtype == MobilizerType.FREE:
        # Rotations
        H_FM[dofadr + 0] = wp.spatial_vector(wp.vec3(1.0, 0.0, 0.0), wp.vec3())
        H_FM[dofadr + 1] = wp.spatial_vector(wp.vec3(0.0, 1.0, 0.0), wp.vec3())
        H_FM[dofadr + 2] = wp.spatial_vector(wp.vec3(0.0, 0.0, 1.0), wp.vec3())
        # Translations
        H_FM[dofadr + 3] = wp.spatial_vector(wp.vec3(), wp.vec3(1.0, 0.0, 0.0))
        H_FM[dofadr + 4] = wp.spatial_vector(wp.vec3(), wp.vec3(0.0, 1.0, 0.0))
        H_FM[dofadr + 5] = wp.spatial_vector(wp.vec3(), wp.vec3(0.0, 0.0, 1.0))

    elif mobtype == MobilizerType.PIN:
        pin_axis = wp.vec3(0.0, 0.0, 1.0)
        H_FM[dofadr] = wp.spatial_vector(pin_axis, wp.vec3())

    elif mobtype == MobilizerType.SLIDER:
        slide_axis = wp.vec3(1.0, 0.0, 0.0)
        H_FM[dofadr] = wp.spatial_vector(wp.vec3(), slide_axis)

    elif mobtype == MobilizerType.UNIVERSAL:
        R_FM_y = mob_scratch[0]
        H_FM[dofadr + 0] = wp.spatial_vector(wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0))
        H_FM[dofadr + 1] = wp.spatial_vector(R_FM_y, wp.vec3(0.0))

    elif mobtype == MobilizerType.GIMBAL:
        gimbal_q0, gimbal_q1, gimbal_q2 = mob_scratch[0][0], mob_scratch[0][1], mob_scratch[0][2]
        c0, c1 = wp.cos(gimbal_q0), wp.cos(gimbal_q1)
        s0, s1 = wp.sin(gimbal_q0), wp.sin(gimbal_q1)
        H_FM[dofadr + 0] = wp.spatial_vector(wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0))
        H_FM[dofadr + 1] = wp.spatial_vector(wp.vec3(0.0, c0, s0), wp.vec3(0.0))
        H_FM[dofadr + 2] = wp.spatial_vector(wp.vec3(s1, -s0 * c1, c0 * c1), wp.vec3(0.0))

    elif mobtype == MobilizerType.BEAM:
        beam_q0, beam_q1, beam_q2 = mob_scratch[0][0], mob_scratch[0][1], mob_scratch[0][2]
        length, deflection_coeff, displacement_coeff = extra_info[0], extra_info[1], extra_info[2]

        c0, c1 = wp.cos(beam_q0), wp.cos(beam_q1)
        s0, s1 = wp.sin(beam_q0), wp.sin(beam_q1)
        H_FM[dofadr + 0] = wp.spatial_vector(wp.vec3(1.0, 0.0, 0.0),
                                             wp.vec3(0.0, -deflection_coeff, -2.0 * displacement_coeff * beam_q0))
        H_FM[dofadr + 1] = wp.spatial_vector(wp.vec3(0.0, c0, s0),
                                             wp.vec3(deflection_coeff, 0.0, -2.0 * displacement_coeff * beam_q1))
        H_FM[dofadr + 2] = wp.spatial_vector(wp.vec3(s1, -s0 * c1, c0 * c1), wp.vec3(0.0))

    elif mobtype == MobilizerType.ELLIPSOID:
        semi = extra_info
        n = mob_scratch[0]
        H_FM[dofadr + 0] = wp.spatial_vector(wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, -n[2] * semi[1], n[1] * semi[2]))
        H_FM[dofadr + 1] = wp.spatial_vector(wp.vec3(0.0, 1.0, 0.0), wp.vec3(n[2] * semi[0], 0.0, -n[0] * semi[2]))
        H_FM[dofadr + 2] = wp.spatial_vector(wp.vec3(0.0, 0.0, 1.0), wp.vec3(-n[1] * semi[0], n[0] * semi[1], 0.0))

    elif mobtype == MobilizerType.BALL:
        H_FM[dofadr + 0] = wp.spatial_vector(wp.vec3(1.0, 0.0, 0.0), wp.vec3())
        H_FM[dofadr + 1] = wp.spatial_vector(wp.vec3(0.0, 1.0, 0.0), wp.vec3())
        H_FM[dofadr + 2] = wp.spatial_vector(wp.vec3(0.0, 0.0, 1.0), wp.vec3())

    elif mobtype == MobilizerType.CUSTOM:
        f_rot, f_trans = mob_scratch[IDX_SCRATCH_ROT_F], mob_scratch[IDX_SCRATCH_TRANS_F]
        df_rot, df_trans = mob_scratch[IDX_SCRATCH_ROT_DF], mob_scratch[IDX_SCRATCH_TRANS_DF]
        rot_ax1, rot_ax2, rot_ax3 = cst_txfm_axes[cst_id, 0], cst_txfm_axes[cst_id, 1], cst_txfm_axes[cst_id, 2]
        trans_ax1, trans_ax2, trans_ax3 = cst_txfm_axes[cst_id, 3], cst_txfm_axes[cst_id, 4], cst_txfm_axes[cst_id, 5]

        # Build dF/dq
        F_rot, F_trans = mat36(0.0), mat36(0.0)
        for i in range(3):
            df_rot_i, df_trans_i = df_rot[i], df_trans[i]
            # Find which dof index this function operates on (relative to joint dofadr)
            rot_dof_idx_i = cst_txfm_dof[cst_id, i]
            trans_dof_idx_i = cst_txfm_dof[cst_id, i + 3]

            # -1 means there are no dependent dofs (i.e., constant function)
            if rot_dof_idx_i != -1:
                F_rot[i, rot_dof_idx_i] = df_rot_i

            if trans_dof_idx_i != -1:
                F_trans[i, trans_dof_idx_i] = df_trans_i

        # A = [ a_trans1, a_trans2, a_trans3 ]
        A = wp.matrix_from_cols(trans_ax1, trans_ax2, trans_ax3)
        # W = [a_rot1, rotated a_rot2, rotated a_rot3] - note these are body-fixed axes rotations
        R_F1 = wp.quat_from_axis_angle(rot_ax1, f_rot[0])
        R_F2 = R_F1 * wp.quat_from_axis_angle(rot_ax2, f_rot[1])
        w1, w2, w3 = rot_ax1, wp.quat_rotate(R_F1, rot_ax2), wp.quat_rotate(R_F2, rot_ax3)
        W = wp.matrix_from_cols(w1, w2, w3)
        # H = [ W * dF_rot/dq
        #       A * dF_trans/dt ]
        H_rot = W * F_rot
        H_trans = A * F_trans
        for i in range(dofnum):
            H_rot_col = wp.vec3(H_rot[0, i], H_rot[1, i], H_rot[2, i])
            H_trans_col = wp.vec3(H_trans[0, i], H_trans[1, i], H_trans[2, i])
            H_FM[dofadr + i] = wp.spatial_vector(H_rot_col, H_trans_col)

    elif mobtype == MobilizerType.WELD:
        pass
    elif mobtype == MobilizerType.WORLD:
        pass
    else:
        assert False, f"Unknown joint type {mobtype}"
    return


@wp.func
def calc_across_joint_velocity_jacobian_dot(
        mobtype: int,
        dofadr: int,
        extra_info: wp.vec3,
        mob_scratch: wp.array(dtype=wp.vec3),
        qvel: wp.array(dtype=float),
        V_FM: wp.spatial_vector,
        H_FM: wp.array(dtype=wp.spatial_vector),
        dofnum: int,
        # custom joints
        cst_id: int,
        cst_txfm_dof: wp.array2d(dtype=int),
        cst_txfm_axes: wp.array2d(dtype=wp.vec3),
        # Out
        HDot_FM: wp.array(dtype=wp.spatial_vector),
):
    """
    Computes the additional joint acceleration contribution S_dot * q_vel
    """
    if mobtype == MobilizerType.FREE:
        HDot_FM[dofadr + 0] = wp.spatial_vector()
        HDot_FM[dofadr + 1] = wp.spatial_vector()
        HDot_FM[dofadr + 2] = wp.spatial_vector()
        HDot_FM[dofadr + 3] = wp.spatial_vector()
        HDot_FM[dofadr + 4] = wp.spatial_vector()
        HDot_FM[dofadr + 5] = wp.spatial_vector()

    elif mobtype == MobilizerType.SLIDER:
        HDot_FM[dofadr + 0] = wp.spatial_vector()

    elif mobtype == MobilizerType.PIN:
        HDot_FM[dofadr + 0] = wp.spatial_vector()

    elif mobtype == MobilizerType.UNIVERSAL:
        R_FM_y = mob_scratch[0]
        w_FM = wp.spatial_top(V_FM)
        HDot_FM[dofadr + 0] = wp.spatial_vector()
        HDot_FM[dofadr + 1] = wp.spatial_vector(wp.cross(w_FM, R_FM_y), wp.vec3(0.0))

    elif mobtype == MobilizerType.GIMBAL:
        gimbal_q0, gimbal_q1, gimbal_q2 = mob_scratch[0][0], mob_scratch[0][1], mob_scratch[0][2]
        c0, c1 = wp.cos(gimbal_q0), wp.cos(gimbal_q1)
        s0, s1 = wp.sin(gimbal_q0), wp.sin(gimbal_q1)

        gimbal_qd0, gimbal_qd1 = qvel[dofadr + 0], qvel[dofadr + 1]
        dc0, dc1 = -s0 * gimbal_qd0, -s1 * gimbal_qd1  # derivatives of c0,c1,s0,s1
        ds0, ds1 = c0 * gimbal_qd0, c1 * gimbal_qd1

        HDot_FM[dofadr + 0] = wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0))
        HDot_FM[dofadr + 1] = wp.spatial_vector(wp.vec3(0.0, dc0, ds0), wp.vec3(0.0))
        HDot_FM[dofadr + 2] = wp.spatial_vector(wp.vec3(ds1, -ds0 * c1 - s0 * dc1, dc0 * c1 + c0 * dc1), wp.vec3(0.0))

    elif mobtype == MobilizerType.BEAM:
        beam_q0, beam_q1, beam_q2 = mob_scratch[0][0], mob_scratch[0][1], mob_scratch[0][2]
        length, deflection_coeff, displacement_coeff = extra_info[0], extra_info[1], extra_info[2]

        c0, c1 = wp.cos(beam_q0), wp.cos(beam_q1)
        s0, s1 = wp.sin(beam_q0), wp.sin(beam_q1)

        beam_qd0, beam_qd1 = qvel[dofadr + 0], qvel[dofadr + 1]
        dc0, dc1 = -s0 * beam_qd0, -s1 * beam_qd1
        ds0, ds1 = c0 * beam_qd0, c1 * beam_qd1

        HDot_FM[dofadr + 0] = wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0),
                                                wp.vec3(0.0, 0.0, -2.0 * displacement_coeff * beam_qd0))
        HDot_FM[dofadr + 1] = wp.spatial_vector(wp.vec3(0.0, dc0, ds0),
                                                wp.vec3(0.0, 0.0, -2.0 * displacement_coeff * beam_qd1))
        HDot_FM[dofadr + 2] = wp.spatial_vector(wp.vec3(ds1, -ds0 * c1 - s0 * dc1, dc0 * c1 + c0 * dc1), wp.vec3(0.0))

    elif mobtype == MobilizerType.ELLIPSOID:
        semi = extra_info
        n = mob_scratch[0]
        w_FM = wp.spatial_top(V_FM)
        ndot = wp.cross(w_FM, n)

        HDot_FM[dofadr + 0] = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0, -ndot[2] * semi[1], ndot[1] * semi[2]))
        HDot_FM[dofadr + 1] = wp.spatial_vector(wp.vec3(0.0), wp.vec3(ndot[2] * semi[0], 0.0, -ndot[0] * semi[2]))
        HDot_FM[dofadr + 2] = wp.spatial_vector(wp.vec3(0.0), wp.vec3(-ndot[1] * semi[0], ndot[0] * semi[1], 0.0))

    elif mobtype == MobilizerType.BALL:
        HDot_FM[dofadr + 0] = wp.spatial_vector()
        HDot_FM[dofadr + 1] = wp.spatial_vector()
        HDot_FM[dofadr + 2] = wp.spatial_vector()

    elif mobtype == MobilizerType.CUSTOM:
        f_rot, f_trans = mob_scratch[IDX_SCRATCH_ROT_F], mob_scratch[IDX_SCRATCH_TRANS_F]
        df_rot, df_trans = mob_scratch[IDX_SCRATCH_ROT_DF], mob_scratch[IDX_SCRATCH_TRANS_DF]
        d2f_rot, d2f_trans = mob_scratch[IDX_SCRATCH_ROT_D2F], mob_scratch[IDX_SCRATCH_TRANS_D2F]
        rot_ax1, rot_ax2, rot_ax3 = cst_txfm_axes[cst_id, 0], cst_txfm_axes[cst_id, 1], cst_txfm_axes[cst_id, 2]
        trans_ax1, trans_ax2, trans_ax3 = cst_txfm_axes[cst_id, 3], cst_txfm_axes[cst_id, 4], cst_txfm_axes[cst_id, 5]

        # pre-fetch qv
        qv = wp.spatial_vector()
        for i in range(dofnum):
            qv[i] = qvel[dofadr + i]

        # Build dF/dq, d2F/dq * qdot
        F_rot = mat36(0.0)
        Fqdot_rot, Fqdot_trans = mat36(0.0), mat36(0.0)
        for i in range(3):
            df_rot_i, df_trans_i = df_rot[i], df_trans[i]
            d2f_rot_i, d2f_trans_i = d2f_rot[i], d2f_trans[i]
            # relative to joint dofadr
            rot_dof_idx_i = cst_txfm_dof[cst_id, i]
            trans_dof_idx_i = cst_txfm_dof[cst_id, i + 3]

            if rot_dof_idx_i != -1:
                F_rot[i, rot_dof_idx_i] = df_rot_i
                Fqdot_rot[i, rot_dof_idx_i] = d2f_rot_i * qv[rot_dof_idx_i]
            if trans_dof_idx_i != -1:
                Fqdot_trans[i, trans_dof_idx_i] = d2f_trans_i * qv[trans_dof_idx_i]

        # A = [ a_trans1, a_trans2, a_trans3 ]
        A = wp.matrix_from_cols(trans_ax1, trans_ax2, trans_ax3)
        # W = [a_rot1, rotated a_rot2, rotated a_rot3] - note these are body-fixed axes rotations
        R_F1 = wp.quat_from_axis_angle(rot_ax1, f_rot[0])
        R_F2 = R_F1 * wp.quat_from_axis_angle(rot_ax2, f_rot[1])
        w1, w2, w3 = rot_ax1, wp.quat_rotate(R_F1, rot_ax2), wp.quat_rotate(R_F2, rot_ax3)
        W = wp.matrix_from_cols(w1, w2, w3)

        # Compute WDot: requires angular velocity contributions
        # note: this assumes that the rotations use the first three dofs only
        v1 = H_FM[dofadr + 0] * qv[0]
        v2 = v1
        if dofnum > 1:
            v2 += H_FM[dofadr + 1] * qv[1]
        omega1 = wp.spatial_top(v1)
        omega2 = wp.spatial_top(v2)
        WDot = wp.matrix_from_cols(wp.vec3(), wp.cross(omega1, w2), wp.cross(omega2, w3))

        # HDot = [ W * dF_rot/dq + Wdot * dF_rot/dt
        #          A * d2F_trans/dt ]
        HDot_rot = W * Fqdot_rot + WDot * F_rot
        HDot_trans = A * Fqdot_trans
        for i in range(dofnum):
            H_rot_col = wp.vec3(HDot_rot[0, i], HDot_rot[1, i], HDot_rot[2, i])
            H_trans_col = wp.vec3(HDot_trans[0, i], HDot_trans[1, i], HDot_trans[2, i])
            HDot_FM[dofadr + i] = wp.spatial_vector(H_rot_col, H_trans_col)

    elif mobtype == MobilizerType.WELD:
        pass
    elif mobtype == MobilizerType.WORLD:
        pass
    else:
        assert False, f"Unknown joint type {mobtype}"
    return wp.spatial_vector()


@wp.func
def integrate(
        mobtype: int,
        qpos: wp.array(dtype=float),
        qvel: wp.array(dtype=float),
        qpos_adr: int,
        dof_adr: int,
        timestep: float,
        dof_num: int,
        # Out:
        qpos_next: wp.array(dtype=float),
):
    if mobtype == MobilizerType.FREE:
        qpos_quat = wp.quat(qpos[qpos_adr + 0], qpos[qpos_adr + 1], qpos[qpos_adr + 2], qpos[qpos_adr + 3])
        qvel_ang = wp.vec3(qvel[dof_adr + 0], qvel[dof_adr + 1], qvel[dof_adr + 2])
        dq_ang = math.calc_unnormalized_quaternion_N(qpos_quat) @ qvel_ang

        qpos_pos = wp.vec3(qpos[qpos_adr + 4], qpos[qpos_adr + 5], qpos[qpos_adr + 6])
        qvel_lin = wp.vec3(qvel[dof_adr + 3], qvel[dof_adr + 4], qvel[dof_adr + 5])
        qpos_new = qpos_pos + timestep * qvel_lin

        qpos_next[qpos_adr + 0] = qpos_quat[0] + timestep * dq_ang[0]
        qpos_next[qpos_adr + 1] = qpos_quat[1] + timestep * dq_ang[1]
        qpos_next[qpos_adr + 2] = qpos_quat[2] + timestep * dq_ang[2]
        qpos_next[qpos_adr + 3] = qpos_quat[3] + timestep * dq_ang[3]
        math.quat_normalize_in_place(qpos_next, qpos_adr)

        qpos_next[qpos_adr + 4] = qpos_new[0]
        qpos_next[qpos_adr + 5] = qpos_new[1]
        qpos_next[qpos_adr + 6] = qpos_new[2]

    elif mobtype == MobilizerType.BALL:
        qpos_quat = wp.quat(qpos[qpos_adr + 0], qpos[qpos_adr + 1], qpos[qpos_adr + 2], qpos[qpos_adr + 3])
        qvel_ang = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2])
        dq_ang = math.calc_unnormalized_quaternion_N(qpos_quat) @ qvel_ang
        qpos_next[qpos_adr + 0] = qpos_quat[0] + timestep * dq_ang[0]
        qpos_next[qpos_adr + 1] = qpos_quat[1] + timestep * dq_ang[1]
        qpos_next[qpos_adr + 2] = qpos_quat[2] + timestep * dq_ang[2]
        qpos_next[qpos_adr + 3] = qpos_quat[3] + timestep * dq_ang[3]
        math.quat_normalize_in_place(qpos_next, qpos_adr)

    elif mobtype == MobilizerType.ELLIPSOID:
        cosxy = wp.vec2(wp.cos(qpos[qpos_adr + 0]), wp.cos(qpos[qpos_adr + 1]))
        sinxy = wp.vec2(wp.sin(qpos[qpos_adr + 0]), wp.sin(qpos[qpos_adr + 1]))
        oocosy = 1.0 / wp.cos(qpos[qpos_adr + 1])
        w = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2])
        dq = math.mul_body_xyz_N(cosxy, sinxy, oocosy, w)
        qpos_next[qpos_adr + 0] = qpos[qpos_adr + 0] + timestep * dq[0]
        qpos_next[qpos_adr + 1] = qpos[qpos_adr + 1] + timestep * dq[1]
        qpos_next[qpos_adr + 2] = qpos[qpos_adr + 2] + timestep * dq[2]

    # The remaining use u = q_dot
    else:
        for i in range(dof_num):
            qpos_next[qpos_adr + i] = (qpos[qpos_adr + i] + timestep * qvel[dof_adr + i])
    return


@wp.func
def multiply_by_N(
        mobtype: int,
        qpos: wp.array(dtype=float),
        qvel: wp.array(dtype=float),
        qpos_adr: int,
        dof_adr: int,
        dof_num: int,
        # Out
        dq: wp.array(dtype=float),
):
    """ Maps from u to q_dot (Nu = q_dot) """
    if mobtype == MobilizerType.FREE:
        # rotation
        rot = wp.quat(qpos[qpos_adr + 0], qpos[qpos_adr + 1], qpos[qpos_adr + 2], qpos[qpos_adr + 3])
        ang_v = wp.vec3(qvel[dof_adr + 0], qvel[dof_adr + 1], qvel[dof_adr + 2])
        dq_rot = math.calc_unnormalized_quaternion_N(rot) @ ang_v
        dq[qpos_adr + 0] = dq_rot[0]
        dq[qpos_adr + 1] = dq_rot[1]
        dq[qpos_adr + 2] = dq_rot[2]
        dq[qpos_adr + 3] = dq_rot[3]
        # translation
        dq[qpos_adr + 4] = qvel[dof_adr + 3]
        dq[qpos_adr + 5] = qvel[dof_adr + 4]
        dq[qpos_adr + 6] = qvel[dof_adr + 5]
    elif mobtype == MobilizerType.BALL:
        rot = wp.quat(qpos[qpos_adr + 0], qpos[qpos_adr + 1], qpos[qpos_adr + 2], qpos[qpos_adr + 3])
        ang_v = wp.vec3(qvel[dof_adr + 0], qvel[dof_adr + 1], qvel[dof_adr + 2])
        dq_rot = math.calc_unnormalized_quaternion_N(rot) @ ang_v
        dq[qpos_adr + 0] = dq_rot[0]
        dq[qpos_adr + 1] = dq_rot[1]
        dq[qpos_adr + 2] = dq_rot[2]
        dq[qpos_adr + 3] = dq_rot[3]
    elif mobtype == MobilizerType.ELLIPSOID:
        cosxy = wp.vec2(wp.cos(qpos[qpos_adr + 0]), wp.cos(qpos[qpos_adr + 1]))
        sinxy = wp.vec2(wp.sin(qpos[qpos_adr + 0]), wp.sin(qpos[qpos_adr + 1]))
        oocosy = 1.0 / wp.cos(qpos[qpos_adr + 1])
        w = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2])
        dq_eul = math.mul_body_xyz_N(cosxy, sinxy, oocosy, w)
        dq[qpos_adr + 0] = dq_eul[0]
        dq[qpos_adr + 1] = dq_eul[1]
        dq[qpos_adr + 2] = dq_eul[2]
    else:  # standard, everything else u = q_dot
        for i in range(dof_num):
            dq[qpos_adr + i] = qvel[dof_adr + i]
    return


@wp.func
def multiply_by_N_transpose(
        mobtype: int,
        qpos: wp.array(dtype=float),
        qfrc: wp.array(dtype=float),
        qpos_adr: int,
        dof_adr: int,
        dof_num: int,
        # Out
        ufrc: wp.array(dtype=float),
):
    """ Maps from q_force to u_force (N^T q_force = u_force) """
    if mobtype == MobilizerType.FREE:
        rot = wp.quat(qpos[qpos_adr + 0], qpos[qpos_adr + 1], qpos[qpos_adr + 2], qpos[qpos_adr + 3])
        f_q_rot = wp.vec4(qfrc[qpos_adr + 0], qfrc[qpos_adr + 1], qfrc[qpos_adr + 2], qfrc[qpos_adr + 3])
        f_u_rot = wp.transpose(math.calc_unnormalized_quaternion_N(rot)) @ f_q_rot
        ufrc[dof_adr + 0] = f_u_rot[0]
        ufrc[dof_adr + 1] = f_u_rot[1]
        ufrc[dof_adr + 2] = f_u_rot[2]
        ufrc[dof_adr + 3] = qfrc[qpos_adr + 4]
        ufrc[dof_adr + 4] = qfrc[qpos_adr + 5]
        ufrc[dof_adr + 5] = qfrc[qpos_adr + 6]
    elif mobtype == MobilizerType.BALL:
        rot = wp.quat(qpos[qpos_adr + 0], qpos[qpos_adr + 1], qpos[qpos_adr + 2], qpos[qpos_adr + 3])
        f_q_rot = wp.vec4(qfrc[qpos_adr + 0], qfrc[qpos_adr + 1], qfrc[qpos_adr + 2], qfrc[qpos_adr + 3])
        f_u_rot = wp.transpose(math.calc_unnormalized_quaternion_N(rot)) @ f_q_rot
        ufrc[dof_adr + 0] = f_u_rot[0]
        ufrc[dof_adr + 1] = f_u_rot[1]
        ufrc[dof_adr + 2] = f_u_rot[2]
    elif mobtype == MobilizerType.ELLIPSOID:
        cosxy = wp.vec2(wp.cos(qpos[qpos_adr + 0]), wp.cos(qpos[qpos_adr + 1]))
        sinxy = wp.vec2(wp.sin(qpos[qpos_adr + 0]), wp.sin(qpos[qpos_adr + 1]))
        oocosy = 1.0 / wp.cos(qpos[qpos_adr + 1])
        f_q_eul = wp.vec3(qfrc[qpos_adr + 0], qfrc[qpos_adr + 1], qfrc[qpos_adr + 2])
        f_u_eul = math.mul_body_xyz_NT(cosxy, sinxy, oocosy, f_q_eul)
        ufrc[dof_adr + 0] = f_u_eul[0]
        ufrc[dof_adr + 1] = f_u_eul[1]
        ufrc[dof_adr + 2] = f_u_eul[2]
    else:  # standard, N is the identity matrix
        for i in range(dof_num):
            ufrc[dof_adr + i] = qfrc[qpos_adr + i]
    return


@wp.func
def multiply_by_N_inv(
        qpos: wp.array(dtype=float),
        dq: wp.array(dtype=float),
        mobtype: int,
        qpos_adr: int,
        dof_adr: int,
        dof_num: int,
        # Out
        qvel_out: wp.array(dtype=float),
):
    """ Maps from q_dot to u (N^-1 q_dot = u) """
    if mobtype == MobilizerType.FREE:
        # rotation first
        rot = wp.quat(qpos[qpos_adr + 0], qpos[qpos_adr + 1], qpos[qpos_adr + 2], qpos[qpos_adr + 3])
        dq_rot = wp.vec4(dq[qpos_adr + 0], dq[qpos_adr + 1], dq[qpos_adr + 2], dq[qpos_adr + 3])
        qvel_rot = math.calc_unnormalized_quaternion_N_inv(rot) @ dq_rot
        qvel_out[dof_adr + 0] = qvel_rot[0]
        qvel_out[dof_adr + 1] = qvel_rot[1]
        qvel_out[dof_adr + 2] = qvel_rot[2]
        # translation second
        qvel_out[dof_adr + 3] = dq[qpos_adr + 4]
        qvel_out[dof_adr + 4] = dq[qpos_adr + 5]
        qvel_out[dof_adr + 5] = dq[qpos_adr + 6]
    elif mobtype == MobilizerType.BALL:  # ball
        rot = wp.quat(qpos[qpos_adr + 0], qpos[qpos_adr + 1], qpos[qpos_adr + 2], qpos[qpos_adr + 3])
        dq_rot = wp.vec4(dq[qpos_adr + 0], dq[qpos_adr + 1], dq[qpos_adr + 2], dq[qpos_adr + 3])
        qvel_rot = math.calc_unnormalized_quaternion_N_inv(rot) @ dq_rot
        qvel_out[dof_adr + 0] = qvel_rot[0]
        qvel_out[dof_adr + 1] = qvel_rot[1]
        qvel_out[dof_adr + 2] = qvel_rot[2]
    elif mobtype == MobilizerType.ELLIPSOID:
        cosxy = wp.vec2(wp.cos(qpos[qpos_adr + 0]), wp.cos(qpos[qpos_adr + 1]))
        sinxy = wp.vec2(wp.sin(qpos[qpos_adr + 0]), wp.sin(qpos[qpos_adr + 1]))
        dq_eul = wp.vec3(dq[qpos_adr + 0], dq[qpos_adr + 1], dq[qpos_adr + 2])
        w = math.mul_body_xyz_N_inv(cosxy, sinxy, dq_eul)
        qvel_out[dof_adr + 0] = w[0]
        qvel_out[dof_adr + 1] = w[1]
        qvel_out[dof_adr + 2] = w[2]
    else:  # standard, nothing else uses quaternions
        for i in range(dof_num):
            qvel_out[dof_adr + i] = dq[qpos_adr + i]
    return


@wp.kernel
def multiply_N_inv_kernel(
        # Model
        mob_type: wp.array(dtype=int),
        mob_qposadr: wp.array(dtype=int),
        mob_dofadr: wp.array(dtype=int),
        mob_dofnum: wp.array(dtype=int),
        # Data in
        qpos_in: wp.array2d(dtype=float),
        # In
        dq: wp.array2d(dtype=float),
        # Data out
        ninv_dq_tmp_out: wp.array2d(dtype=float),
):
    worldid, bodyid = wp.tid()
    qpos = qpos_in[worldid]
    mob_type_ = mob_type[bodyid]
    qpos_adr = mob_qposadr[bodyid]
    dof_adr = mob_dofadr[bodyid]
    dof_num = mob_dofnum[bodyid]

    multiply_by_N_inv(
        qpos,
        dq[worldid],
        mob_type_,
        qpos_adr,
        dof_adr,
        dof_num,
        ninv_dq_tmp_out[worldid],
    )
    return


@event_scope
def multiply_W(m: Model, d: Data):
    @wp.kernel
    def multiply_W_kernel(
            # Model in:
            qvel_weights: wp.array(dtype=float),
            # Data in:
            qvel_diff_in: wp.array2d(dtype=float),
            # Out:
            ninv_dq_tmp_out: wp.array2d(dtype=float),
    ):
        worldid = wp.tid()
        nv = wp.static(m.nv)

        qvel_diff_tile = wp.tile_load(qvel_diff_in[worldid], nv)
        qvel_scales_tile = wp.tile_load(qvel_weights, nv)
        qvel_scaled_diff_tile = wp.tile_map(wp.mul, qvel_diff_tile, qvel_scales_tile)

        wp.tile_store(ninv_dq_tmp_out[worldid], qvel_scaled_diff_tile)
        return

    wp.launch_tiled(
        multiply_W_kernel,
        dim=d.nworld,
        inputs=[m.opt.qvel_weights, d.ninv_dq_tmp, ],
        outputs=[d.ninv_dq_tmp, ],
        block_dim=m.block_dim.error_step,
    )


@wp.kernel
def multiply_N_kernel(
        # Model
        mob_type: wp.array(dtype=int),
        mob_qposadr: wp.array(dtype=int),
        mob_dofadr: wp.array(dtype=int),
        mob_dofnum: wp.array(dtype=int),
        # Data in
        qpos_in: wp.array2d(dtype=float),
        # In
        qvel_scaled: wp.array2d(dtype=float),
        # Data out
        scaled_qdiff_out: wp.array2d(dtype=float),
):
    worldid, bodyid = wp.tid()
    qpos = qpos_in[worldid]
    mob_type_ = mob_type[bodyid]
    qpos_adr = mob_qposadr[bodyid]
    dof_adr = mob_dofadr[bodyid]
    dof_num = mob_dofnum[bodyid]

    multiply_by_N(
        mob_type_,
        qpos,
        qvel_scaled[worldid],
        qpos_adr,
        dof_adr,
        dof_num,
        scaled_qdiff_out[worldid],
    )
    return


def scale_dq(
        m: Model,
        d: Data,
        dq: wp.array2d(dtype=float),
        dq_scaled: wp.array2d(dtype=float)
):
    # The weights of u correspond to weights of q_dot
    # q_dot = N u (this is how we integrate qpos from qvel)
    # so u = N_inv q_dot (N_inv is pseudo-inverse of N)
    # Therefore they way we weigh q_dot is:
    # u_scaled = W_u * u
    # u_scaled = W_u * N_inv * q_dot
    # N(u_scaled) = N * W_u * N_inv * q_dot
    # q_dot_scaled = (N * W_u * N_inv) * q_dot

    # N_inv * q_dot
    wp.launch(
        kernel=multiply_N_inv_kernel,
        dim=(d.nworld, m.nbody),
        inputs=[m.mob_type, m.mob_qposadr, m.mob_dofadr, m.mob_dofnum, d.qpos, dq, ],
        outputs=[d.ninv_dq_tmp, ],
    )

    # W * N_inv * q_dot
    multiply_W(m, d)

    # N * W * N_inv * q_dot
    wp.launch(
        kernel=multiply_N_kernel,
        dim=(d.nworld, m.nbody),
        inputs=[m.mob_type, m.mob_qposadr, m.mob_dofadr, m.mob_dofnum, d.qpos, d.ninv_dq_tmp, ],
        outputs=[dq_scaled, ],
    )
    return
