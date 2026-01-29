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
        xaxis_out[0] = wp.normalize(math.rot_vec_quat(hinge_axis, jnt_rot))

    if jnttype == JointType.BALL:
        qloc_ = wp.quat(qpos[qadr + 0], qpos[qadr + 1], qpos[qadr + 2], qpos[qadr + 3])
        qloc_ = wp.normalize(qloc_)

    elif jnttype == JointType.SLIDE:
        slide_axis = wp.vec3(1.0, 0.0, 0.0)
        xloc_ = qpos[qadr] * slide_axis
        xaxis_out[0] = wp.normalize(math.rot_vec_quat(slide_axis, jnt_rot))

    elif jnttype == JointType.UNIVERSAL:
        axis1 = wp.vec3(1.0, 0.0, 0.0)
        axis2 = wp.vec3(0.0, 1.0, 0.0)

        qloc1 = math.axis_angle_to_quat(axis1, qpos[qadr + 0])
        qloc2 = math.axis_angle_to_quat(axis2, qpos[qadr + 1])
        qloc_ = math.mul_quat(qloc1, qloc2)

        # Keep track of first rotation
        xaxis_out[0] = wp.normalize(math.rot_vec_quat(axis1, jnt_rot))
        xaxis_out[1] = wp.normalize(math.rot_vec_quat(axis2, math.mul_quat(jnt_rot, qloc1)))
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
            xaxis_out[i] = fn_eval[1] * wp.normalize(math.rot_vec_quat(
                txfm_axes[i], math.mul_quat(jnt_rot, qloc_)))

            qloc_ = math.mul_quat(qloc_, math.axis_angle_to_quat(txfm_axes[i], fn_eval[0]))

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
            xaxis_out[i] = fn_eval[1] * wp.normalize(math.rot_vec_quat(txfm_axes[i], jnt_rot))
    elif jnttype == JointType.WELD:
        pass
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
        cst_res_tmp: wp.array(dtype=wp.spatial_vector)
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
        # Initialize to zero
        for i in range(dof_num):
            res[dofid + i] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
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
            res[dof_adr] += c
            cst_res_tmp[i] = c
    elif jnttype == JointType.WELD:
        pass
    elif jnttype == JointType.DUMMY:
        pass


@wp.func
def cvel_joint(
        cvel: wp.spatial_vector,
        cdof: wp.array(dtype=wp.spatial_vector),
        qvel: wp.array(dtype=float),
        jnttype: int,
        dofid: int,
        # Custom joints
        dof_num: int,
        cst_txfm_dofadr: wp.array(dtype=int),
        cdof_tmp: wp.array(dtype=wp.spatial_vector),
        # Out
        res: wp.array(dtype=wp.spatial_vector),
) -> wp.spatial_vector:
    if jnttype == JointType.FREE:
        cvel += cdof[dofid + 0] * qvel[dofid + 0]
        cvel += cdof[dofid + 1] * qvel[dofid + 1]
        cvel += cdof[dofid + 2] * qvel[dofid + 2]

        res[dofid + 3] = math.motion_cross(cvel, cdof[dofid + 3])
        res[dofid + 4] = math.motion_cross(cvel, cdof[dofid + 4])
        res[dofid + 5] = math.motion_cross(cvel, cdof[dofid + 5])

        cvel += cdof[dofid + 3] * qvel[dofid + 3]
        cvel += cdof[dofid + 4] * qvel[dofid + 4]
        cvel += cdof[dofid + 5] * qvel[dofid + 5]
    elif jnttype == JointType.BALL:
        res[dofid + 0] = math.motion_cross(cvel, cdof[dofid + 0])
        res[dofid + 1] = math.motion_cross(cvel, cdof[dofid + 1])
        res[dofid + 2] = math.motion_cross(cvel, cdof[dofid + 2])

        cvel += cdof[dofid + 0] * qvel[dofid + 0]
        cvel += cdof[dofid + 1] * qvel[dofid + 1]
        cvel += cdof[dofid + 2] * qvel[dofid + 2]
    elif jnttype == JointType.PIN or jnttype == JointType.SLIDE:
        res[dofid] = math.motion_cross(cvel, cdof[dofid])
        cvel += cdof[dofid] * qvel[dofid]
    elif jnttype == JointType.UNIVERSAL:
        # The second transformation is dependent on the first
        for i in range(2):
            res[dofid + i] = math.motion_cross(cvel, cdof[dofid + i])
            cvel += cdof[dofid + i] * qvel[dofid + i]
    elif jnttype == JointType.CUSTOM:
        # Initialize to zero
        for i in range(dof_num):
            res[dofid + i] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        # For custom joints, we need to process them in order of transformation
        #  translations don't depend on previous transforms, so do them first
        #  but each rotation depends on previous rotation
        # That's why we stored the intermediate cdof in cdof_tmp
        for i in range(wp.static(6)):
            j = i + 3 if i < 3 else i - 3  # translations first
            dof_adr = cst_txfm_dofadr[j]
            if dof_adr == -1:
                continue
            res[dof_adr] += math.motion_cross(cvel, cdof_tmp[j])
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
        qpos_pos = wp.vec3(qpos[qpos_adr], qpos[qpos_adr + 1],
                           qpos[qpos_adr + 2])
        qvel_lin = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2])
        qpos_new = qpos_pos + timestep * qvel_lin

        qpos_quat = wp.quat(
            qpos[qpos_adr + 3], qpos[qpos_adr + 4],
            qpos[qpos_adr + 5], qpos[qpos_adr + 6],
        )
        qvel_ang = wp.vec3(qvel[dof_adr + 3], qvel[dof_adr + 4],
                           qvel[dof_adr + 5])
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
        qpos_quat = wp.quat(qpos[qpos_adr + 0], qpos[qpos_adr + 1],
                            qpos[qpos_adr + 2], qpos[qpos_adr + 3])
        qvel_ang = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2])
        dq_ang = math.calc_unnormalized_quaternion_N(qpos_quat) @ qvel_ang
        qpos_next[qpos_adr + 0] = qpos_quat[0] + timestep * dq_ang[0]
        qpos_next[qpos_adr + 1] = qpos_quat[1] + timestep * dq_ang[1]
        qpos_next[qpos_adr + 2] = qpos_quat[2] + timestep * dq_ang[2]
        qpos_next[qpos_adr + 3] = qpos_quat[3] + timestep * dq_ang[3]
        math.quat_normalize_in_place(qpos_next, qpos_adr)

    elif jnttype == JointType.SLIDE or jnttype == JointType.PIN:
        qpos_next[qpos_adr] = qpos[qpos_adr] + timestep * qvel[dof_adr]

    elif jnttype == JointType.UNIVERSAL:
        qpos_next[qpos_adr] = qpos[qpos_adr] + timestep * qvel[dof_adr]
        qpos_next[qpos_adr + 1] = qpos[qpos_adr + 1] + timestep * qvel[
            dof_adr + 1]

    elif jnttype == JointType.CUSTOM:
        for i in range(dof_num):
            qpos_next[qpos_adr + i] = (
                    qpos[qpos_adr + i] + timestep * qvel[dof_adr + i])

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
        rot = wp.quat(qpos[qpos_adr + 3], qpos[qpos_adr + 4],
                      qpos[qpos_adr + 5], qpos[qpos_adr + 6], )
        ang_v = wp.vec3(qvel[dof_adr + 3], qvel[dof_adr + 4], qvel[dof_adr + 5])
        dq_rot = math.calc_unnormalized_quaternion_N(rot) @ ang_v
        dq[qpos_adr + 3] = dq_rot[0]
        dq[qpos_adr + 4] = dq_rot[1]
        dq[qpos_adr + 5] = dq_rot[2]
        dq[qpos_adr + 6] = dq_rot[3]
    elif jnttype == JointType.BALL:  # ball
        rot = wp.quat(qpos[qpos_adr + 0], qpos[qpos_adr + 1],
                      qpos[qpos_adr + 2], qpos[qpos_adr + 3], )
        rot_N = math.calc_unnormalized_quaternion_N(rot)
        ang_v = wp.vec3(qvel[dof_adr + 0], qvel[dof_adr + 1], qvel[dof_adr + 2])
        dq_rot = rot_N @ ang_v
        dq[qpos_adr + 0] = dq_rot[0]
        dq[qpos_adr + 1] = dq_rot[1]
        dq[qpos_adr + 2] = dq_rot[2]
        dq[qpos_adr + 3] = dq_rot[3]
    else:  # standard, nothing else uses quaternions
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
        rot = wp.quat(qpos[qpos_adr + 3], qpos[qpos_adr + 4],
                      qpos[qpos_adr + 5], qpos[qpos_adr + 6], )
        dq_rot = wp.vec4(dq[qpos_adr + 3], dq[qpos_adr + 4],
                         dq[qpos_adr + 5], dq[qpos_adr + 6])
        qvel_rot = math.calc_unnormalized_quaternion_N_inv(rot) @ dq_rot
        qvel_out[dof_adr + 3] = qvel_rot[0]
        qvel_out[dof_adr + 4] = qvel_rot[1]
        qvel_out[dof_adr + 5] = qvel_rot[2]
    elif jnttype == JointType.BALL:  # ball
        rot = wp.quat(qpos[qpos_adr + 0], qpos[qpos_adr + 1],
                      qpos[qpos_adr + 2], qpos[qpos_adr + 3], )
        dq_rot = wp.vec4(dq[qpos_adr + 0], dq[qpos_adr + 1],
                         dq[qpos_adr + 2], dq[qpos_adr + 3])
        qvel_rot = math.calc_unnormalized_quaternion_N_inv(rot) @ dq_rot
        qvel_out[dof_adr + 0] = qvel_rot[0]
        qvel_out[dof_adr + 1] = qvel_rot[1]
        qvel_out[dof_adr + 2] = qvel_rot[2]
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
        qvel_scaled_diff_tile = wp.tile_map(
            wp.mul, qvel_diff_tile, qvel_scales_tile)

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
        inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, m.jnt_dofnum,
                d.qpos, dq, ],
        outputs=[d.ninv_dq_tmp, ],
    )

    # W * N_inv * q_dot
    multiply_W(m, d)

    # N * W * N_inv * q_dot
    wp.launch(
        kernel=multiply_N_kernel,
        dim=(d.nworld, m.nbody),
        inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, m.jnt_dofnum,
                d.qpos, d.ninv_dq_tmp, ],
        outputs=[dq_scaled, ],
    )
    return
