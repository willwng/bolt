import warp as wp

from . import math
from .types import JointType
from .types import CustomFnType

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
            return wp.vec2(0.0, 0.0)
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
        # Custom joints
        txfm_axes: wp.array(dtype=wp.vec3),
        txfm_fn: wp.array(dtype=int),
        cst_txfm_fn_adr: wp.array(dtype=int),
        cst_txfm_qadr: wp.array(dtype=int),
        const_fns: wp.array(dtype=float),
        linear_fns: wp.array(dtype=wp.vec2),
        # Output
        xaxis_out: wp.array(dtype=wp.vec3),
) -> tuple[wp.vec3, wp.quat, wp.vec3]:
    if jnttype == JointType.FREE:
        xpos = wp.vec3(qpos[qadr], qpos[qadr + 1], qpos[qadr + 2])
        xquat = wp.quat(qpos[qadr + 3], qpos[qadr + 4],
                        qpos[qadr + 5], qpos[qadr + 6])
        xquat = wp.normalize(xquat)
        xanchor = xpos
        return xpos, xquat, xanchor

    # Grab parent frame information
    p_pos = xpos_in[pid]
    p_rot = xquat_in[pid]

    # compute the joint frame
    p_to_jnt = math.rot_vec_quat(jnt_rel_parent, p_rot)
    p_to_jnt_rot = jnt_rel_parent_rot
    jnt_pos = p_pos + p_to_jnt
    jnt_rot = math.mul_quat(p_rot, p_to_jnt_rot)

    # local joint transformation
    qloc_ = wp.quat(1.0, 0.0, 0.0, 0.0)
    xloc_ = wp.vec3(0.0, 0.0, 0.0)
    if jnttype == JointType.PIN:
        hinge_axis = wp.vec3(0.0, 0.0, 1.0)
        qloc_ = math.axis_angle_to_quat(hinge_axis, qpos[qadr])
        xaxis_out[0] = math.rot_vec_quat(hinge_axis, jnt_rot)

    if jnttype == JointType.BALL:
        qloc_ = wp.quat(qpos[qadr + 0], qpos[qadr + 1],
                        qpos[qadr + 2], qpos[qadr + 3])
        qloc_ = wp.normalize(qloc_)

    elif jnttype == JointType.SLIDE:
        slide_axis = wp.vec3(1.0, 0.0, 0.0)
        xloc_ = qpos[qadr] * slide_axis
        xaxis_out[0] = math.rot_vec_quat(slide_axis, jnt_rot)

    elif jnttype == JointType.UNIVERSAL:
        axis1 = wp.vec3(1.0, 0.0, 0.0)
        axis2 = wp.vec3(0.0, 1.0, 0.0)

        qloc1 = math.axis_angle_to_quat(axis1, qpos[qadr + 0])
        qloc2 = math.axis_angle_to_quat(axis2, qpos[qadr + 1])
        qloc_ = math.mul_quat(qloc1, qloc2)

        # Keep track of first rotation
        xaxis_out[0] = (math.rot_vec_quat(axis1, jnt_rot))
        xaxis_out[1] = (math.rot_vec_quat(axis2, math.mul_quat(jnt_rot, qloc1)))
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
            xaxis_out[i] = fn_eval[1] * math.rot_vec_quat(
                txfm_axes[i], math.mul_quat(jnt_rot, qloc_))

            qloc_ = math.mul_quat(
                qloc_, math.axis_angle_to_quat(txfm_axes[i], fn_eval[0]))

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
            # store intermediate rotated axes, with derivative
            xaxis_out[i] = fn_eval[1] * math.rot_vec_quat(txfm_axes[i], jnt_rot)
    elif jnttype == JointType.DUMMY:
        pass

    # world coordinates
    xquat = math.mul_quat(jnt_rot, math.mul_quat(qloc_, jnt_rel_child_rot))
    xpos = (jnt_pos + math.rot_vec_quat(xloc_, jnt_rot) +
            math.rot_vec_quat(jnt_rel_child, xquat))
    xanchor = jnt_pos
    return xpos, xquat, xanchor


@wp.func
def cdof_joint(
        jnttype: int,
        dofid: int,
        xmat: wp.mat33,
        offset: wp.vec3,
        xaxis_in: wp.array(dtype=wp.vec3),
        # Custom joints
        dof_num: int,
        cst_txfm_dofadr: wp.array(dtype=int),
        # out
        res: wp.array(dtype=wp.spatial_vector),
):
    if jnttype == JointType.FREE:
        res[dofid + 0] = wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        res[dofid + 1] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        res[dofid + 2] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        # I_3 rotation in child frame (assume no subsequent rotations)
        res[dofid + 3] = wp.spatial_vector(xmat[0], wp.cross(xmat[0], offset))
        res[dofid + 4] = wp.spatial_vector(xmat[1], wp.cross(xmat[1], offset))
        res[dofid + 5] = wp.spatial_vector(xmat[2], wp.cross(xmat[2], offset))
    elif jnttype == JointType.BALL:  # ball
        # I_3 rotation in child frame (assume no subsequent rotations)
        res[dofid + 0] = wp.spatial_vector(xmat[0], wp.cross(xmat[0], offset))
        res[dofid + 1] = wp.spatial_vector(xmat[1], wp.cross(xmat[1], offset))
        res[dofid + 2] = wp.spatial_vector(xmat[2], wp.cross(xmat[2], offset))
    elif jnttype == JointType.SLIDE:
        xaxis = xaxis_in[0]
        res[dofid] = wp.spatial_vector(wp.vec3(0.0), xaxis)
    elif jnttype == JointType.PIN:  # hinge
        xaxis = xaxis_in[0]
        res[dofid] = wp.spatial_vector(xaxis, wp.cross(xaxis, offset))
    elif jnttype == JointType.UNIVERSAL:
        xaxis1 = xaxis_in[0]
        xaxis2 = xaxis_in[1]

        res[dofid + 0] = wp.spatial_vector(xaxis1, wp.cross(xaxis1, offset))
        res[dofid + 1] = wp.spatial_vector(xaxis2, wp.cross(xaxis2, offset))
    elif jnttype == JointType.CUSTOM:
        # initialize to zero
        for i in range(dof_num):
            res[dofid + i] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # accumulate over all spatial txfm
        for i in range(wp.static(6)):
            dof_adr = cst_txfm_dofadr[i]
            if dof_adr == -1:  # not attached to a dof
                continue

            xaxis = xaxis_in[i]
            if i < 3:  # rotation
                res[dof_adr] += wp.spatial_vector(xaxis,
                                                  wp.cross(xaxis, offset))
            else:  # translation
                res[dof_adr] += wp.spatial_vector(wp.vec3(0.0), xaxis)
    elif jnttype == JointType.DUMMY:
        pass


@wp.func
def cvel_joint(
        cvel: wp.spatial_vector,
        cdof: wp.array(dtype=wp.spatial_vector),
        qvel: wp.array(dtype=float),
        jnttype: int,
        dofid: int,
        dof_num: int,
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
    elif jnttype == JointType.BALL:
        cdof_dot_out[dofid + 0] = math.motion_cross(cvel, cdof[dofid + 0])
        cdof_dot_out[dofid + 1] = math.motion_cross(cvel, cdof[dofid + 1])
        cdof_dot_out[dofid + 2] = math.motion_cross(cvel, cdof[dofid + 2])

        cvel += cdof[dofid + 0] * qvel[dofid + 0]
        cvel += cdof[dofid + 1] * qvel[dofid + 1]
        cvel += cdof[dofid + 2] * qvel[dofid + 2]
    elif jnttype == JointType.PIN or jnttype == JointType.SLIDE:
        cdof_dot_out[dofid] = math.motion_cross(cvel, cdof[dofid])
        cvel += cdof[dofid] * qvel[dofid]
    elif jnttype == JointType.UNIVERSAL:
        # The second transformation is dependent on the first
        for i in range(2):
            cdof_dot_out[dofid] = math.motion_cross(cvel, cdof[dofid])
            cvel += cdof[dofid] * qvel[dofid]
            dofid += 1
    elif jnttype == JointType.CUSTOM:
        # TODO: This doesn't seem right, DOF ordering may be arbitrary
        for i in range(dof_num):
            cdof_dot_out[dofid] = math.motion_cross(cvel, cdof[dofid])
            cvel += cdof[dofid] * qvel[dofid]
            dofid += 1
    elif jnttype == JointType.DUMMY:
        pass
    return cvel
