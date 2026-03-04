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
def jcalc_transform(
        jnttype: int,
        qadr: int,
        qpos: wp.array(dtype=float),
        extra_info: wp.vec3,
) -> wp.transform:
    if jnttype == JointType.FREE:
        r = wp.quat(qpos[qadr + 0], qpos[qadr + 1], qpos[qadr + 2], qpos[qadr + 3])
        p = wp.vec3(qpos[qadr + 4], qpos[qadr + 5], qpos[qadr + 6])
        r = wp.normalize(r)
        return wp.transform(p, r)

    if jnttype == JointType.PIN:
        pin_axis = wp.vec3(0.0, 0.0, 1.0)
        r = wp.quat_from_axis_angle(pin_axis, qpos[qadr])
        return wp.transform(wp.vec3(), r)

    elif jnttype == JointType.SLIDE:
        slide_axis = wp.vec3(1.0, 0.0, 0.0)
        p = qpos[qadr] * slide_axis
        return wp.transform(p, wp.quat_identity())

    elif jnttype == JointType.UNIVERSAL:
        axis0 = wp.vec3(1.0, 0.0, 0.0)
        axis1 = wp.vec3(0.0, 1.0, 0.0)

        qloc0 = wp.quat_from_axis_angle(axis0, qpos[qadr + 0])
        qloc1 = wp.quat_from_axis_angle(axis1, qpos[qadr + 1])
        r = qloc0 * qloc1
        return wp.transform(wp.vec3(), r)

    elif jnttype == JointType.BALL:
        r = wp.quat(qpos[qadr + 0], qpos[qadr + 1], qpos[qadr + 2], qpos[qadr + 3])
        r = wp.normalize(r)
        return wp.transform(wp.vec3(), r)

    elif jnttype == JointType.GIMBAL:
        # Euler/Body-Fixed XYZ order
        r = math.quat_from_xyz(qpos[qadr + 0], qpos[qadr + 1], qpos[qadr + 2])
        return wp.transform(wp.vec3(), r)

    elif jnttype == JointType.BEAM:
        r = math.quat_from_xyz(qpos[qadr + 0], qpos[qadr + 1], qpos[qadr + 2])
        # Beam deflection
        length = extra_info[0]
        deflection_coeff, displacement_coeff = (2.0 / 3.0) * length, (4.0 / 15.0) * length
        theta_sq = qpos[qadr + 0] * qpos[qadr + 0] + qpos[qadr + 1] * qpos[qadr + 1]
        p = wp.vec3(
            qpos[qadr + 1] * deflection_coeff,
            -qpos[qadr + 0] * deflection_coeff,
            length - displacement_coeff * theta_sq
        )
        return wp.transform(p, r)

    elif jnttype == JointType.ELLIPSOID:
        r = math.quat_from_xyz(qpos[qadr + 0], qpos[qadr + 1], qpos[qadr + 2])
        # Ellipsoid translation: the z axis of body is assumed to be normal to the joint ellipsoid
        n = wp.quat_rotate(r, wp.vec(0.0, 0.0, 1.0))
        semi = extra_info
        p = wp.vec3(semi.x * n.x, semi.y * n.y, semi.z * n.z)
        return wp.transform(p, r)

    elif jnttype == JointType.WELD:
        return wp.transform(wp.vec3(), wp.quat_identity())

    elif jnttype == JointType.DUMMY:
        return wp.transform(wp.vec3(), wp.quat_identity())
    else:
        assert False, f"Unknown joint type {jnttype}"

    return wp.transform_identity()


@wp.func
def joint_motion(
        jnttype: int,
        dofadr: int,
        qvel: wp.array(dtype=float),
        extra_info: wp.vec3,
        # Out
        S_out: wp.array(dtype=wp.spatial_vector),
):
    """
    Computes the motion subspace and joint velocity contribution
    """
    if jnttype == JointType.FREE:
        # Rotations
        S_out[dofadr + 0] = wp.spatial_vector(wp.vec3(1.0, 0.0, 0.0), wp.vec3())
        S_out[dofadr + 1] = wp.spatial_vector(wp.vec3(0.0, 1.0, 0.0), wp.vec3())
        S_out[dofadr + 2] = wp.spatial_vector(wp.vec3(0.0, 0.0, 1.0), wp.vec3())
        # Translations
        S_out[dofadr + 3] = wp.spatial_vector(wp.vec3(), wp.vec3(1.0, 0.0, 0.0))
        S_out[dofadr + 4] = wp.spatial_vector(wp.vec3(), wp.vec3(0.0, 1.0, 0.0))
        S_out[dofadr + 5] = wp.spatial_vector(wp.vec3(), wp.vec3(0.0, 0.0, 1.0))

        return wp.spatial_vector(qvel[dofadr + 0], qvel[dofadr + 1], qvel[dofadr + 2],
                                 qvel[dofadr + 3], qvel[dofadr + 4], qvel[dofadr + 5])

    elif jnttype == JointType.SLIDE:
        slide_axis = wp.vec3(1.0, 0.0, 0.0)
        S_j = wp.spatial_vector(wp.vec3(), slide_axis)
        S_out[dofadr] = wp.spatial_vector(wp.vec3(), slide_axis)
        return S_j * qvel[dofadr]

    elif jnttype == JointType.PIN:
        pin_axis = wp.vec3(0.0, 0.0, 1.0)
        S_j = wp.spatial_vector(pin_axis, wp.vec3())
        S_out[dofadr] = wp.spatial_vector(pin_axis, wp.vec3())
        return S_j * qvel[dofadr]

    elif jnttype == JointType.BALL:
        S_out[dofadr + 0] = wp.spatial_vector(wp.vec3(1.0, 0.0, 0.0), wp.vec3())
        S_out[dofadr + 1] = wp.spatial_vector(wp.vec3(0.0, 1.0, 0.0), wp.vec3())
        S_out[dofadr + 2] = wp.spatial_vector(wp.vec3(0.0, 0.0, 1.0), wp.vec3())
        return wp.spatial_vector(wp.vec3(qvel[dofadr + 0], qvel[dofadr + 1], qvel[dofadr + 2]), wp.vec3())

    elif jnttype == JointType.ELLIPSOID:
        pass
    elif jnttype == JointType.UNIVERSAL:
        pass
    elif jnttype == JointType.GIMBAL:
        pass
    elif jnttype == JointType.BEAM:
        pass
    elif jnttype == JointType.CUSTOM:
        pass

    elif jnttype == JointType.WELD:
        return wp.spatial_vector()

    elif jnttype == JointType.DUMMY:
        return wp.spatial_vector()

    else:
        assert False, f"Unknown joint type {jnttype}"
    return wp.spatial_vector()


@wp.func
def joint_acc(
        jnttype: int,
        dofadr: int,
        qvel: wp.array(dtype=float),
        extra_info: wp.vec3,
):
    """
    Computes the additional joint acceleration contribution S_dot * q_vel
    """
    if jnttype == JointType.FREE:
        return wp.spatial_vector()

    elif jnttype == JointType.SLIDE:
        return wp.spatial_vector()

    elif jnttype == JointType.PIN:
        return wp.spatial_vector()

    elif jnttype == JointType.BALL:
        return wp.spatial_vector()

    elif jnttype == JointType.ELLIPSOID:
        pass
    elif jnttype == JointType.UNIVERSAL:
        pass
    elif jnttype == JointType.GIMBAL:
        pass
    elif jnttype == JointType.BEAM:
        pass
    elif jnttype == JointType.CUSTOM:
        pass

    elif jnttype == JointType.WELD:
        return wp.spatial_vector()

    elif jnttype == JointType.DUMMY:
        return wp.spatial_vector()

    else:
        assert False, f"Unknown joint type {jnttype}"
    return wp.spatial_vector()


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
