import warp as wp

from . import math
from .consts import MSK_MINVAL
from .consts import MSK_SIG_REAL
from .types import Data
from .types import Model
from .types import CoordinateLimitForce
from .types import SwingTwistLimit
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _process_limit_forces(
        # Model in:
        coordinate_limit_force: wp.array(dtype=CoordinateLimitForce),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qpos_in: wp.array2d(dtype=float),
        qvel_in: wp.array2d(dtype=float),
        # Data out:
        ufrc_limit_out: wp.array2d(dtype=float),
):
    worldid, limitid = wp.tid()
    if integration_done_in[worldid]:
        return

    lf = coordinate_limit_force[limitid]
    qpos_range = lf.qpos_range
    qpos_adr = lf.qpos_adr
    dof_adr = lf.dof_adr
    stiffness = lf.stiffness
    damp = lf.damping
    transition = lf.transition

    q_low, q_up = qpos_range[0], qpos_range[1]
    lower_stiffness, upper_stiffness = stiffness[0], stiffness[1]

    q = qpos_in[worldid, qpos_adr]
    qdot = qvel_in[worldid, dof_adr]

    weight_up = math.step_function(q, q_up, q_up + transition, 0.0, 1.0)
    weight_low = math.step_function(q, q_low - transition, q_low, 1.0, 0.0)

    # Scale to get the stiffness values
    K_up = weight_up * upper_stiffness
    K_low = weight_low * lower_stiffness

    f_up = -K_up * (q - q_up)
    f_low = K_low * (q_low - q)

    f_damp = -damp * (weight_up + weight_low) * qdot
    force = f_up + f_low + f_damp
    wp.atomic_add(ufrc_limit_out, worldid, dof_adr, force)
    return


@wp.kernel
def _process_swing_twist_limits(
        # Model in:
        swing_twist_limits: wp.array(dtype=SwingTwistLimit),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qpos_in: wp.array2d(dtype=float),
        qvel_in: wp.array2d(dtype=float),
        # Data out:
        ufrc_limit_out: wp.array2d(dtype=float),
):
    worldid, limitid = wp.tid()
    if integration_done_in[worldid]:
        return
    qpos = qpos_in[worldid]
    qvel = qvel_in[worldid]

    stl = swing_twist_limits[limitid]
    stiffness = stl.stiffness
    damping = stl.damping
    qpos_adr = stl.qpos_adr
    dof_adr = stl.dof_adr
    twist_range = stl.twist_range
    swing1_range = stl.swing1_range
    swing2_range = stl.swing2_range

    # Obtain rotation, angular velocity
    rot = wp.quat(qpos[qpos_adr + 0], qpos[qpos_adr + 1], qpos[qpos_adr + 2], qpos[qpos_adr + 3])
    omega = wp.vec3(qvel[dof_adr + 0], qvel[dof_adr + 1], qvel[dof_adr + 2])
    if rot.w < 0.0:
        rot = -rot  # ensure shortest path

    # Decompose into swing, twist
    twist_axis = wp.vec3(0.0, 1.0, 0.0)
    q_swing, q_twist = math.quat_swing_twist(rot, twist_axis)

    twist_angle = 2.0 * wp.atan2(q_twist.y, q_twist.w)
    swing_angle = 2.0 * wp.acos(wp.clamp(q_swing.w, -1.0, 1.0))

    # Compute twist violation
    violation_twist = 0.0
    if twist_angle > twist_range[1]:
        violation_twist = twist_angle - twist_range[1]
    elif twist_angle < twist_range[0]:
        violation_twist = twist_angle - twist_range[0]

    # Swing violation: based on elliptical limit in the plane
    swing_axis_len = wp.sqrt(q_swing.x * q_swing.x + q_swing.z * q_swing.z)
    swing_axis = wp.vec3(1.0, 0.0, 0.0)
    if swing_axis_len > MSK_SIG_REAL:
        swing_axis.x = q_swing.x / swing_axis_len
        swing_axis.z = q_swing.z / swing_axis_len
    nx = swing_axis.x
    nz = swing_axis.z
    swing_max_x = wp.where(nx < 0.0, swing1_range[0], swing1_range[1])
    swing_max_z = wp.where(nz < 0.0, swing2_range[0], swing2_range[1])
    # Elliptical boundary
    dynamic_swing_max = 0.0
    denominator = wp.sqrt((swing_max_z * nx) * (swing_max_z * nx) + (swing_max_x * nz) * (swing_max_x * nz))
    if denominator > MSK_SIG_REAL:
        dynamic_swing_max = (swing_max_x * swing_max_z) / denominator
    violation_swing = wp.where(swing_angle > dynamic_swing_max, swing_angle - dynamic_swing_max, 0.0)

    # Twist correction force
    twist_corrective = wp.vec3(0.0, 0.0, 0.0)
    if wp.abs(violation_twist) > MSK_SIG_REAL:
        actual_twist_axis = wp.quat_rotate(q_swing, wp.vec3(0.0, 1.0, 0.0))
        omega_twist = wp.dot(omega, actual_twist_axis)
        if violation_twist > 0.0:
            tau_mag_twist = wp.min(-stiffness * violation_twist * (1.0 + damping * omega_twist), 0.0)
        else:
            tau_mag_twist = wp.max(-stiffness * violation_twist * (1.0 - damping * omega_twist), 0.0)
        twist_corrective = tau_mag_twist * actual_twist_axis

    # Swing correction force
    swing_corrective = wp.vec3(0.0, 0.0, 0.0)
    if violation_swing > MSK_SIG_REAL:
        omega_swing = wp.dot(omega, swing_axis)
        # Swing violation is an absolute magnitude (x > 0), so it acts as an upper limit
        tau_mag_swing = wp.min(0.0, -stiffness * violation_swing * (1.0 + damping * omega_swing))
        swing_corrective = swing_axis * tau_mag_swing

    tau_corrective = twist_corrective + swing_corrective
    if wp.length(tau_corrective) > MSK_MINVAL:
        wp.atomic_add(ufrc_limit_out[worldid], dof_adr + 0, tau_corrective.x)
        wp.atomic_add(ufrc_limit_out[worldid], dof_adr + 1, tau_corrective.y)
        wp.atomic_add(ufrc_limit_out[worldid], dof_adr + 2, tau_corrective.z)
    return


@event_scope
def coordinate_limit_force(m: Model, d: Data):
    wp.launch(
        _process_limit_forces,
        dim=(d.nworld, m.nlimitforce),
        inputs=[
            m.coordinate_limit_force,
            d.integration_done, d.qpos, d.qvel
        ],
        outputs=[d.ufrc_limit, ],
    )
    return


@event_scope
def swing_twist_limit_force(m: Model, d: Data):
    wp.launch(
        _process_swing_twist_limits,
        dim=(d.nworld, m.nswingtwist),
        inputs=[
            m.swing_twist_limit,
            d.integration_done, d.qpos, d.qvel
        ],
        outputs=[d.ufrc_limit, ],
    )
    return
