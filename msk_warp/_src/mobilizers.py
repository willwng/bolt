import warp as wp

from . import math
from .types import JointType
from .types import CustomFnType
from .types import Model
from .types import Data
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


# returns f(x) and df(x)
@wp.func
def evaluate_txfm(
        # data in
        qpos_in: wp.array(dtype=float),
        # in
        txfm_fn_type: int,
        txfm_fn_adr: int,
        txfm_qadr: int,
        # model in
        const_fns: wp.array(dtype=float),
        linear_fns: wp.array(dtype=wp.vec2),
) -> wp.vec2:
    if txfm_fn_type == CustomFnType.CONSTANT:
        return wp.vec2(const_fns[txfm_fn_adr], 0.0)
    elif txfm_fn_type == CustomFnType.LINEAR:
        if txfm_qadr == -1:
            return wp.vec2(0.0, 0.0)  # this should be impossible
        lin_fn = linear_fns[txfm_fn_adr]
        return wp.vec2(lin_fn[0] * qpos_in[txfm_qadr] + lin_fn[1], lin_fn[0])
    return wp.vec2(0.0, 0.0)


@wp.func
def fk_joint(
        jnttype: int,
        qadr: int,
        qpos: wp.array(dtype=float),
        # Parent information
        pid: int,
        xpos_in: wp.array(dtype=wp.vec3),
        xquat_in: wp.array(dtype=wp.quat),
        jnt_rel_parent: wp.vec3,
        jnt_rel_parent_rot: wp.quat,
        jnt_rel_child: wp.vec3,
        jnt_rel_child_rot: wp.quat,
        jnt_extra_info: wp.vec3,
        # Custom joints
        txfm_axes: wp.array(dtype=wp.vec3),
        txfm_fn: wp.array(dtype=int),
        cst_txfm_fn_adr: wp.array(dtype=int),
        cst_txfm_qadr: wp.array(dtype=int),
        const_fns: wp.array(dtype=float),
        linear_fns: wp.array(dtype=wp.vec2),
        # Output
        xaxis_out: wp.array(dtype=wp.vec3),
) -> tuple[wp.vec3, wp.quat, wp.vec3, wp.quat]:
    if jnttype == JointType.FREE:
        xpos = wp.vec3(qpos[qadr], qpos[qadr + 1], qpos[qadr + 2])
        xquat = wp.quat(qpos[qadr + 3], qpos[qadr + 4], qpos[qadr + 5], qpos[qadr + 6])
        xquat = wp.normalize(xquat)
        xanchor = xpos
        return xpos, xquat, xanchor, xquat

    # Grab parent frame information
    p_pos = xpos_in[pid]
    p_rot = xquat_in[pid]

    # compute the joint frame in world space
    p_to_jnt = wp.quat_rotate(p_rot, jnt_rel_parent)
    jnt_pos = p_pos + p_to_jnt
    jnt_rot = p_rot * jnt_rel_parent_rot

    # compute the local joint transformation
    qloc_ = wp.quat_identity(dtype=wp.float32)
    xloc_ = wp.vec3()
    if jnttype == JointType.PIN:
        hinge_axis = wp.vec3(0.0, 0.0, 1.0)
        qloc_ = wp.quat_from_axis_angle(hinge_axis, qpos[qadr])
        xaxis_out[0] = wp.normalize(wp.quat_rotate(jnt_rot, hinge_axis))

    if jnttype == JointType.BALL:
        qloc_ = wp.quat(qpos[qadr + 0], qpos[qadr + 1], qpos[qadr + 2], qpos[qadr + 3])
        qloc_ = wp.normalize(qloc_)

    elif jnttype == JointType.SLIDE:
        slide_axis = wp.vec3(1.0, 0.0, 0.0)
        xloc_ = qpos[qadr] * slide_axis
        xaxis_out[0] = wp.normalize(wp.quat_rotate(jnt_rot, slide_axis))

    elif jnttype == JointType.UNIVERSAL:
        axis0 = wp.vec3(1.0, 0.0, 0.0)
        axis1 = wp.vec3(0.0, 1.0, 0.0)

        qloc0 = wp.quat_from_axis_angle(axis0, qpos[qadr + 0])
        qloc1 = wp.quat_from_axis_angle(axis1, qpos[qadr + 1])
        qloc_ = qloc0 * qloc1

        # Keep track of first rotation
        xaxis_out[0] = wp.normalize(wp.quat_rotate(jnt_rot, axis0))
        xaxis_out[1] = wp.normalize(wp.quat_rotate(jnt_rot * qloc0, axis1))
    elif jnttype == JointType.GIMBAL or jnttype == JointType.BEAM or jnttype == JointType.ELLIPSOID:
        # Euler/Body-Fixed XYZ order
        q0, q1, q2 = qpos[qadr + 0], qpos[qadr + 1], qpos[qadr + 2]
        qloc0 = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), q0)
        qloc1 = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), q1)
        qloc2 = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), q2)
        qloc_ = qloc0 * qloc1 * qloc2
        xaxis_out[0] = wp.vec3(q0, q1, q2)

        # Deflection contribution for beam
        if jnttype == JointType.BEAM:
            length = jnt_extra_info[0]
            deflection_coeff = (2.0 / 3.0) * length
            displacement_coeff = (4.0 / 15.0) * length
            xloc_ = wp.vec3(
                q1 * deflection_coeff,
                -q0 * deflection_coeff,
                length - displacement_coeff * (q0 * q0 + q1 * q1)
            )
            # Save these for later
            xaxis_out[1] = wp.vec3(length, deflection_coeff, displacement_coeff)

        # Ellipsoid translation: the z axis of body is assumed to be normal to the joint ellipsoid
        if jnttype == JointType.ELLIPSOID:
            n = wp.quat_rotate(qloc_, wp.vec(0.0, 0.0, 1.0))
            semi = jnt_extra_info
            xloc_ = wp.vec3(semi.x * n.x, semi.y * n.y, semi.z * n.z)
            xaxis_out[1] = semi
            xaxis_out[2] = n

    elif jnttype == JointType.CUSTOM:
        # First 3 are rotation
        for i in range(wp.static(3)):
            fn_eval = evaluate_txfm(
                qpos,
                txfm_fn[i],
                cst_txfm_fn_adr[i],
                cst_txfm_qadr[i],
                const_fns,
                linear_fns,
            )
            # store intermediate rotated axes
            xaxis_out[i] = fn_eval[1] * wp.normalize(wp.quat_rotate(jnt_rot * qloc_, txfm_axes[i]))
            qloc_ = qloc_ * wp.quat_from_axis_angle(txfm_axes[i], fn_eval[0])
        # Next 3 are translation
        for i in range(wp.static(3), wp.static(6)):
            fn_eval = evaluate_txfm(
                qpos,
                txfm_fn[i],
                cst_txfm_fn_adr[i],
                cst_txfm_qadr[i],
                const_fns,
                linear_fns,
            )
            xloc_ += fn_eval[0] * txfm_axes[i]
            # store the rotated linear axes, with derivative
            xaxis_out[i] = fn_eval[1] * wp.normalize(wp.quat_rotate(jnt_rot, txfm_axes[i]))
    elif jnttype == JointType.WELD:
        pass
    elif jnttype == JointType.DUMMY:
        pass

    # To world coordinates
    # R_joint * R_local * R_body_to_body_joint^-1
    xquat = jnt_rot * qloc_ * wp.quat_inverse(jnt_rel_child_rot)
    # p_joint + R_joint * p_local is the position of the child's "joint frame"
    # add R_child * -p_body_to_body_joint to get the position of the child's body frame
    xpos = jnt_pos + wp.quat_rotate(jnt_rot, xloc_) + wp.quat_rotate(xquat, -jnt_rel_child)

    xanchor = jnt_pos
    return xpos, xquat, xanchor, jnt_rot


@wp.func
def cdof_joint(
        jnttype: int,
        dofid: int,
        offset: wp.vec3,
        xaxis_in: wp.array(dtype=wp.vec3),
        jnt_rot: wp.quat,
        # Custom joints
        dof_num: int,
        cst_txfm_dofadr: wp.array(dtype=int),
        # out
        cdof_out: wp.array(dtype=wp.spatial_vector),
        cdof_tmp_out: wp.array(dtype=wp.spatial_vector)
):
    if jnttype == JointType.FREE:
        # [0, I]
        cdof_out[dofid + 0] = wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        cdof_out[dofid + 1] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        cdof_out[dofid + 2] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        # [R, -[r]_x R]
        ux, uy, uz = wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 1.0)
        rx, ry, rz = wp.quat_rotate(jnt_rot, ux), wp.quat_rotate(jnt_rot, uy), wp.quat_rotate(jnt_rot, uz)
        cdof_out[dofid + 3] = wp.spatial_vector(rx, wp.cross(rx, offset))
        cdof_out[dofid + 4] = wp.spatial_vector(ry, wp.cross(ry, offset))
        cdof_out[dofid + 5] = wp.spatial_vector(rz, wp.cross(rz, offset))

        d1, d2, d3 = cdof_out[dofid + 3], cdof_out[dofid + 4], cdof_out[dofid + 5]
    elif jnttype == JointType.BALL or jnttype == JointType.ELLIPSOID:
        # u is angular velocity defined in the *joint frame*
        ux, uy, uz = wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 1.0)
        rx, ry, rz = wp.quat_rotate(jnt_rot, ux), wp.quat_rotate(jnt_rot, uy), wp.quat_rotate(jnt_rot, uz)
        # [R, -[r]_x R]
        cdof_out[dofid + 0] = wp.spatial_vector(rx, wp.cross(rx, offset))
        cdof_out[dofid + 1] = wp.spatial_vector(ry, wp.cross(ry, offset))
        cdof_out[dofid + 2] = wp.spatial_vector(rz, wp.cross(rz, offset))

        # Translational contribution for ellipsoid: [0, -R diag(semi)[n]_x ]
        #  Note the R on the outside since n is defined in the joint frame
        if jnttype == JointType.ELLIPSOID:
            semi, n = xaxis_in[1], xaxis_in[2]
            diag_semi = wp.diag(semi)
            cdof_out[dofid + 0] += wp.spatial_vector(
                wp.vec3(0.0), wp.quat_rotate(jnt_rot, diag_semi @ wp.cross(ux, n)))
            cdof_out[dofid + 1] += wp.spatial_vector(
                wp.vec3(0.0), wp.quat_rotate(jnt_rot, diag_semi @ wp.cross(uy, n)))
            cdof_out[dofid + 2] += wp.spatial_vector(
                wp.vec3(0.0), wp.quat_rotate(jnt_rot, diag_semi @ wp.cross(uz, n)))
    elif jnttype == JointType.SLIDE:
        xaxis = xaxis_in[0]
        cdof_out[dofid] = wp.spatial_vector(wp.vec3(0.0), xaxis)
    elif jnttype == JointType.PIN:  # hinge
        xaxis = xaxis_in[0]
        cdof_out[dofid] = wp.spatial_vector(xaxis, wp.cross(xaxis, offset))
    elif jnttype == JointType.UNIVERSAL:
        xaxis0 = xaxis_in[0]
        xaxis1 = xaxis_in[1]
        cdof_out[dofid + 0] = wp.spatial_vector(xaxis0, wp.cross(xaxis0, offset))
        cdof_out[dofid + 1] = wp.spatial_vector(xaxis1, wp.cross(xaxis1, offset))
    elif jnttype == JointType.GIMBAL or jnttype == JointType.BEAM:
        q0, q1, q2 = xaxis_in[0][0], xaxis_in[0][1], xaxis_in[0][2]
        c0, s0, c1, s1 = wp.cos(q0), wp.sin(q0), wp.cos(q1), wp.sin(q1)
        xaxis0 = wp.normalize(wp.quat_rotate(jnt_rot, wp.vec3(1.0, 0.0, 0.0)))
        xaxis1 = wp.normalize(wp.quat_rotate(jnt_rot, wp.vec3(0.0, c0, s0)))
        xaxis2 = wp.normalize(wp.quat_rotate(jnt_rot, wp.vec3(s1, -s0 * c1, c0 * c1)))
        cdof_out[dofid + 0] = wp.spatial_vector(xaxis0, wp.cross(xaxis0, offset))
        cdof_out[dofid + 1] = wp.spatial_vector(xaxis1, wp.cross(xaxis1, offset))
        cdof_out[dofid + 2] = wp.spatial_vector(xaxis2, wp.cross(xaxis2, offset))
        # Deflection contribution for beam
        if jnttype == JointType.BEAM:
            length, deflection_coeff, displacement_coeff = xaxis_in[1][0], xaxis_in[1][1], xaxis_in[1][2]
            trans_from_def0 = wp.vec3(0.0, -deflection_coeff, -2.0 * displacement_coeff * q0)
            trans_from_def1 = wp.vec3(deflection_coeff, 0.0, -2.0 * displacement_coeff * q1)
            cdof_out[dofid + 0] += wp.spatial_vector(wp.vec3(0.0), wp.quat_rotate(jnt_rot, trans_from_def0))
            cdof_out[dofid + 1] += wp.spatial_vector(wp.vec3(0.0), wp.quat_rotate(jnt_rot, trans_from_def1))
    elif jnttype == JointType.CUSTOM:
        # Initialize to zero
        for i in range(dof_num):
            cdof_out[dofid + i] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        # Accumulate over all spatial txfm
        for i in range(wp.static(6)):
            dof_adr = cst_txfm_dofadr[i]
            if dof_adr == -1:  # not attached to a dof
                continue
            xaxis = xaxis_in[i]
            if i < 3:  # rotation
                c = wp.spatial_vector(xaxis, wp.cross(xaxis, offset))
            else:  # translation
                c = wp.spatial_vector(wp.vec3(0.0), xaxis)
            cdof_out[dof_adr] += c
            cdof_tmp_out[i] = c
    elif jnttype == JointType.WELD:
        pass
    elif jnttype == JointType.DUMMY:
        pass


@wp.func
def cvel_joint(
        jnttype: int,
        dofid: int,
        cvel: wp.spatial_vector,
        cdof: wp.array(dtype=wp.spatial_vector),
        qvel: wp.array(dtype=float),
        offset: wp.vec3,
        xaxis_in: wp.array(dtype=wp.vec3),
        jnt_rot: wp.quat,
        # Custom joints
        dof_num: int,
        cst_txfm_dofadr: wp.array(dtype=int),
        cdof_tmp: wp.array(dtype=wp.spatial_vector),
        # Out
        cdof_dot_out: wp.array(dtype=wp.spatial_vector),
) -> wp.spatial_vector:
    if jnttype == JointType.FREE:
        cvel += cdof[dofid + 0] * qvel[dofid + 0]
        cvel += cdof[dofid + 1] * qvel[dofid + 1]
        cvel += cdof[dofid + 2] * qvel[dofid + 2]

        cdof_dot_out[dofid + 3] = math.motion_cross(cvel, cdof[dofid + 3])
        cdof_dot_out[dofid + 4] = math.motion_cross(cvel, cdof[dofid + 4])
        cdof_dot_out[dofid + 5] = math.motion_cross(cvel, cdof[dofid + 5])

        cvel += cdof[dofid + 3] * qvel[dofid + 3]
        cvel += cdof[dofid + 4] * qvel[dofid + 4]
        cvel += cdof[dofid + 5] * qvel[dofid + 5]
    elif jnttype == JointType.BALL or jnttype == JointType.ELLIPSOID:
        cdof_dot_out[dofid + 0] = math.motion_cross(cvel, cdof[dofid + 0])
        cdof_dot_out[dofid + 1] = math.motion_cross(cvel, cdof[dofid + 1])
        cdof_dot_out[dofid + 2] = math.motion_cross(cvel, cdof[dofid + 2])

        cvel += cdof[dofid + 0] * qvel[dofid + 0]
        cvel += cdof[dofid + 1] * qvel[dofid + 1]
        cvel += cdof[dofid + 2] * qvel[dofid + 2]
        if jnttype == JointType.ELLIPSOID:
            semi, n = xaxis_in[1], xaxis_in[2]
            w = wp.vec3(qvel[dofid + 0], qvel[dofid + 1], qvel[dofid + 2])
            n_dot = wp.cross(w, n)
            cdof_dot_out[dofid + 0] += wp.spatial_vector(
                wp.vec3(0.0), wp.quat_rotate(jnt_rot, wp.vec3(0.0, -n_dot[2] * semi[1], n_dot[1] * semi[2])))
            cdof_dot_out[dofid + 1] += wp.spatial_vector(
                wp.vec3(0.0), wp.quat_rotate(jnt_rot, wp.vec3(n_dot[2] * semi[0], 0.0, -n_dot[0] * semi[2])))
            cdof_dot_out[dofid + 2] += wp.spatial_vector(
                wp.vec3(0.0), wp.quat_rotate(jnt_rot, wp.vec3(-n_dot[1] * semi[0], n_dot[0] * semi[1], 0.0)))

    elif jnttype == JointType.PIN or jnttype == JointType.SLIDE:
        cdof_dot_out[dofid] = math.motion_cross(cvel, cdof[dofid])
        cvel += cdof[dofid] * qvel[dofid]
    elif jnttype == JointType.UNIVERSAL:
        # The second transformation is dependent on the first
        for i in range(2):
            cdof_dot_out[dofid + i] = math.motion_cross(cvel, cdof[dofid + i])
            cvel += cdof[dofid + i] * qvel[dofid + i]
    elif jnttype == JointType.GIMBAL or jnttype == JointType.BEAM:
        q0, q1, q2 = xaxis_in[0][0], xaxis_in[0][1], xaxis_in[0][2]
        dq0, dq1, dq2 = qvel[dofid + 0], qvel[dofid + 1], qvel[dofid + 2]
        # cos, sin, derivatives of cos and sin
        c0, s0, c1, s1 = wp.cos(q0), wp.sin(q0), wp.cos(q1), wp.sin(q1)
        dc0, dc1, ds0, ds1 = -s0 * dq0, -s1 * dq1, c0 * dq0, c1 * dq1
        # Derivative of S_gimbal wrst time
        dx0 = wp.quat_rotate(jnt_rot, wp.vec3(0.0, 0.0, 0.0))
        dx1 = wp.quat_rotate(jnt_rot, wp.vec3(0.0, dc0, ds0))
        dx2 = wp.quat_rotate(jnt_rot, wp.vec3(ds1, -ds0 * c1 - s0 * dc1, dc0 * c1 + c0 * dc1))
        cdof_dot_out[dofid + 0] = (math.motion_cross(cvel, cdof[dofid + 0]) +
                                   wp.spatial_vector(dx0, wp.cross(dx0, offset)))
        cdof_dot_out[dofid + 1] = (math.motion_cross(cvel, cdof[dofid + 1]) +
                                   wp.spatial_vector(dx1, wp.cross(dx1, offset)))
        cdof_dot_out[dofid + 2] = (math.motion_cross(cvel, cdof[dofid + 2]) +
                                   wp.spatial_vector(dx2, wp.cross(dx2, offset)))
        # Add deflection contribution
        if jnttype == JointType.BEAM:
            length, deflection_coeff, displacement_coeff = xaxis_in[1][0], xaxis_in[1][1], xaxis_in[1][2]
            cdof_dot_out[dofid + 0] += wp.spatial_vector(
                wp.vec3(0.0), wp.quat_rotate(jnt_rot, wp.vec3(0.0, 0.0, -2.0 * displacement_coeff * dq0)))
            cdof_dot_out[dofid + 1] += wp.spatial_vector(
                wp.vec3(0.0), wp.quat_rotate(jnt_rot, wp.vec3(0.0, 0.0, -2.0 * displacement_coeff * dq1)))
        # Now update cvel
        cvel += cdof[dofid + 0] * dq0
        cvel += cdof[dofid + 1] * dq1
        cvel += cdof[dofid + 2] * dq2
    elif jnttype == JointType.CUSTOM:
        # Initialize to zero
        for i in range(dof_num):
            cdof_dot_out[dofid + i] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        # For custom joints, we need to process them in order of transformation
        #  translations don't depend on previous transforms, so do them first
        #  but each rotation depends on previous rotation
        # That's why we stored the intermediate cdof in cdof_tmp
        for i in range(wp.static(6)):
            j = i + 3 if i < 3 else i - 3  # translations first
            dof_adr = cst_txfm_dofadr[j]
            if dof_adr == -1:
                continue
            cdof_dot_out[dof_adr] += math.motion_cross(cvel, cdof_tmp[j])
            cvel += cdof_tmp[j] * qvel[dof_adr]
    elif jnttype == JointType.WELD:
        pass
    elif jnttype == JointType.DUMMY:
        pass
    return cvel


@wp.func
def integrate(
        jnttype: int,
        qpos: wp.array(dtype=float),
        qvel: wp.array(dtype=float),
        qpos_adr: int,
        dof_adr: int,
        timestep: float,
        # Custom joints
        dof_num: int,
        # Out:
        qpos_next: wp.array(dtype=float),
):
    if jnttype == JointType.FREE:
        qpos_pos = wp.vec3(qpos[qpos_adr], qpos[qpos_adr + 1], qpos[qpos_adr + 2])
        qvel_lin = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2])
        qpos_new = qpos_pos + timestep * qvel_lin

        qpos_quat = wp.quat(qpos[qpos_adr + 3], qpos[qpos_adr + 4], qpos[qpos_adr + 5], qpos[qpos_adr + 6])
        qvel_ang = wp.vec3(qvel[dof_adr + 3], qvel[dof_adr + 4], qvel[dof_adr + 5])
        dq_ang = math.calc_unnormalized_quaternion_N(qpos_quat) @ qvel_ang

        qpos_next[qpos_adr + 0] = qpos_new[0]
        qpos_next[qpos_adr + 1] = qpos_new[1]
        qpos_next[qpos_adr + 2] = qpos_new[2]
        qpos_next[qpos_adr + 3] = qpos_quat[0] + timestep * dq_ang[0]
        qpos_next[qpos_adr + 4] = qpos_quat[1] + timestep * dq_ang[1]
        qpos_next[qpos_adr + 5] = qpos_quat[2] + timestep * dq_ang[2]
        qpos_next[qpos_adr + 6] = qpos_quat[3] + timestep * dq_ang[3]
        math.quat_normalize_in_place(qpos_next, qpos_adr + 3)

    elif jnttype == JointType.BALL:
        qpos_quat = wp.quat(qpos[qpos_adr + 0], qpos[qpos_adr + 1], qpos[qpos_adr + 2], qpos[qpos_adr + 3])
        qvel_ang = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2])
        dq_ang = math.calc_unnormalized_quaternion_N(qpos_quat) @ qvel_ang
        qpos_next[qpos_adr + 0] = qpos_quat[0] + timestep * dq_ang[0]
        qpos_next[qpos_adr + 1] = qpos_quat[1] + timestep * dq_ang[1]
        qpos_next[qpos_adr + 2] = qpos_quat[2] + timestep * dq_ang[2]
        qpos_next[qpos_adr + 3] = qpos_quat[3] + timestep * dq_ang[3]
        math.quat_normalize_in_place(qpos_next, qpos_adr)

    elif jnttype == JointType.ELLIPSOID:
        cosxy = wp.vec2(wp.cos(qpos[qpos_adr + 0]), wp.cos(qpos[qpos_adr + 1]))
        sinxy = wp.vec2(wp.sin(qpos[qpos_adr + 0]), wp.sin(qpos[qpos_adr + 1]))
        oocosy = 1.0 / wp.cos(qpos[qpos_adr + 1])
        w = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2])
        dq = math.mul_body_xyz_N(cosxy, sinxy, oocosy, w)
        qpos_next[qpos_adr + 0] = qpos[qpos_adr + 0] + timestep * dq[0]
        qpos_next[qpos_adr + 1] = qpos[qpos_adr + 1] + timestep * dq[1]
        qpos_next[qpos_adr + 2] = qpos[qpos_adr + 2] + timestep * dq[2]

    elif jnttype == JointType.SLIDE or jnttype == JointType.PIN:
        qpos_next[qpos_adr] = qpos[qpos_adr] + timestep * qvel[dof_adr]

    elif jnttype == JointType.UNIVERSAL:
        qpos_next[qpos_adr] = qpos[qpos_adr] + timestep * qvel[dof_adr]
        qpos_next[qpos_adr + 1] = qpos[qpos_adr + 1] + timestep * qvel[dof_adr + 1]

    elif jnttype == JointType.GIMBAL or jnttype == JointType.BEAM:
        for i in range(3):
            qpos_next[qpos_adr + i] = (qpos[qpos_adr + i] + timestep * qvel[dof_adr + i])

    elif jnttype == JointType.CUSTOM:
        for i in range(dof_num):
            qpos_next[qpos_adr + i] = (qpos[qpos_adr + i] + timestep * qvel[dof_adr + i])

    elif jnttype == JointType.WELD:
        return
    elif jnttype == JointType.DUMMY:
        return
    else:
        assert False


@wp.func
def multiply_by_N(
        qpos: wp.array(dtype=float),
        qvel: wp.array(dtype=float),
        jnttype: int,
        qpos_adr: int,
        dof_adr: int,
        dof_num: int,
        # Out
        dq: wp.array(dtype=float),
):
    """ Maps from u to q_dot (Nu = q_dot) """
    if jnttype == JointType.FREE:
        # translation, nothing to do
        dq[qpos_adr + 0] = qvel[dof_adr + 0]
        dq[qpos_adr + 1] = qvel[dof_adr + 1]
        dq[qpos_adr + 2] = qvel[dof_adr + 2]
        # rotation
        rot = wp.quat(qpos[qpos_adr + 3], qpos[qpos_adr + 4], qpos[qpos_adr + 5], qpos[qpos_adr + 6])
        ang_v = wp.vec3(qvel[dof_adr + 3], qvel[dof_adr + 4], qvel[dof_adr + 5])
        dq_rot = math.calc_unnormalized_quaternion_N(rot) @ ang_v
        dq[qpos_adr + 3] = dq_rot[0]
        dq[qpos_adr + 4] = dq_rot[1]
        dq[qpos_adr + 5] = dq_rot[2]
        dq[qpos_adr + 6] = dq_rot[3]
    elif jnttype == JointType.BALL:  # ball
        rot = wp.quat(qpos[qpos_adr + 0], qpos[qpos_adr + 1], qpos[qpos_adr + 2], qpos[qpos_adr + 3])
        rot_N = math.calc_unnormalized_quaternion_N(rot)
        ang_v = wp.vec3(qvel[dof_adr + 0], qvel[dof_adr + 1], qvel[dof_adr + 2])
        dq_rot = rot_N @ ang_v
        dq[qpos_adr + 0] = dq_rot[0]
        dq[qpos_adr + 1] = dq_rot[1]
        dq[qpos_adr + 2] = dq_rot[2]
        dq[qpos_adr + 3] = dq_rot[3]
    elif jnttype == JointType.ELLIPSOID:
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
def multiply_by_N_inv(
        qpos: wp.array(dtype=float),
        dq: wp.array(dtype=float),
        jnttype: int,
        qpos_adr: int,
        dof_adr: int,
        dof_num: int,
        # Out
        qvel_out: wp.array(dtype=float),
):
    if jnttype == JointType.FREE:
        # translation
        qvel_out[dof_adr + 0] = dq[qpos_adr + 0]
        qvel_out[dof_adr + 1] = dq[qpos_adr + 1]
        qvel_out[dof_adr + 2] = dq[qpos_adr + 2]
        # rotation
        rot = wp.quat(qpos[qpos_adr + 3], qpos[qpos_adr + 4], qpos[qpos_adr + 5], qpos[qpos_adr + 6])
        dq_rot = wp.vec4(dq[qpos_adr + 3], dq[qpos_adr + 4], dq[qpos_adr + 5], dq[qpos_adr + 6])
        qvel_rot = math.calc_unnormalized_quaternion_N_inv(rot) @ dq_rot
        qvel_out[dof_adr + 3] = qvel_rot[0]
        qvel_out[dof_adr + 4] = qvel_rot[1]
        qvel_out[dof_adr + 5] = qvel_rot[2]
    elif jnttype == JointType.BALL:  # ball
        rot = wp.quat(qpos[qpos_adr + 0], qpos[qpos_adr + 1], qpos[qpos_adr + 2], qpos[qpos_adr + 3])
        dq_rot = wp.vec4(dq[qpos_adr + 0], dq[qpos_adr + 1], dq[qpos_adr + 2], dq[qpos_adr + 3])
        qvel_rot = math.calc_unnormalized_quaternion_N_inv(rot) @ dq_rot
        qvel_out[dof_adr + 0] = qvel_rot[0]
        qvel_out[dof_adr + 1] = qvel_rot[1]
        qvel_out[dof_adr + 2] = qvel_rot[2]
    elif jnttype == JointType.ELLIPSOID:
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
        jnt_type: wp.array(dtype=int),
        jnt_qposadr: wp.array(dtype=int),
        jnt_dofadr: wp.array(dtype=int),
        jnt_dofnum: wp.array(dtype=int),
        # Data in
        qpos_in: wp.array2d(dtype=float),
        # In
        dq: wp.array2d(dtype=float),
        # Data out
        ninv_dq_tmp_out: wp.array2d(dtype=float),
):
    worldid, bodyid = wp.tid()
    qpos = qpos_in[worldid]
    jnt_type_ = jnt_type[bodyid]
    qpos_adr = jnt_qposadr[bodyid]
    dof_adr = jnt_dofadr[bodyid]
    dof_num = jnt_dofnum[bodyid]

    multiply_by_N_inv(
        qpos,
        dq[worldid],
        jnt_type_,
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
        jnt_type: wp.array(dtype=int),
        jnt_qposadr: wp.array(dtype=int),
        jnt_dofadr: wp.array(dtype=int),
        jnt_dofnum: wp.array(dtype=int),
        # Data in
        qpos_in: wp.array2d(dtype=float),
        # In
        qvel_scaled: wp.array2d(dtype=float),
        # Data out
        scaled_qdiff_out: wp.array2d(dtype=float),
):
    worldid, bodyid = wp.tid()
    qpos = qpos_in[worldid]
    jnt_type_ = jnt_type[bodyid]
    qpos_adr = jnt_qposadr[bodyid]
    dof_adr = jnt_dofadr[bodyid]
    dof_num = jnt_dofnum[bodyid]

    multiply_by_N(
        qpos,
        qvel_scaled[worldid],
        jnt_type_,
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
        inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, m.jnt_dofnum, d.qpos, dq, ],
        outputs=[d.ninv_dq_tmp, ],
    )

    # W * N_inv * q_dot
    multiply_W(m, d)

    # N * W * N_inv * q_dot
    wp.launch(
        kernel=multiply_N_kernel,
        dim=(d.nworld, m.nbody),
        inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, m.jnt_dofnum, d.qpos, d.ninv_dq_tmp, ],
        outputs=[dq_scaled, ],
    )
    return
