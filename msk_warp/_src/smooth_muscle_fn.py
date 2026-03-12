import warp as wp

from . import math
from .types import mat411
from .types import Data
from .types import Model
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _compute_path_kernel(
        # Model:
        muscle_poly_coeffs: wp.array(dtype=float),
        muscle_poly_adr: wp.array(dtype=int),
        muscle_poly_order: wp.array(dtype=int),
        muscle_poly_qpos_adr: wp.array(dtype=int),
        muscle_poly_dof_adr: wp.array(dtype=int),
        muscle_poly_dep_dof_num: wp.array(dtype=int),
        muscle_poly_dep_dof_adr: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qpos_in: wp.array2d(dtype=float),
        qvel_in: wp.array2d(dtype=float),
        # Data out:
        muscle_length_out: wp.array2d(dtype=float),
        muscle_velocity_out: wp.array2d(dtype=float),
        muscle_moment_arm_out: wp.array3d(dtype=float),
):
    worldid, muscle_id = wp.tid()
    if integration_done_in[worldid]:
        return
    # Fetch polynomial data: address into coeffs, order
    poly_adr = muscle_poly_adr[muscle_id]
    order = muscle_poly_order[muscle_id]
    # Number of dependent DOFs, and address into dep DOF array
    n_dof = muscle_poly_dep_dof_num[muscle_id]
    dep_adr = muscle_poly_dep_dof_adr[muscle_id]
    # Pre-fetch the dependent qpos and qvel values
    poly_tmp_q = wp.vec4f(0.0)
    poly_tmp_qv = wp.vec4f(0.0)
    for i in range(n_dof):
        qpos_adr_i = muscle_poly_qpos_adr[dep_adr + i]
        dof_adr_i = muscle_poly_dof_adr[dep_adr + i]
        poly_tmp_q[i] = qpos_in[worldid, qpos_adr_i]
        poly_tmp_qv[i] = qvel_in[worldid, dof_adr_i]

    # Pre-calculate powers up to order 10
    # mat411 provides 4 rows (DOFs) and 11 columns (powers 0 to 10)
    q_pows = mat411(1.0)
    for d in range(4):
        val = poly_tmp_q[d]
        # q^0 is already 1.0 from initialization
        for p in range(1, order + 1):
            q_pows[d, p] = q_pows[d, p - 1] * val

    # Evaluate polynomial
    length = float(0.0)
    df_dq = wp.vec4f(0.0)
    coeff_idx = int(0)
    for in1 in range(order + 1):
        # If n_dof < 2, in2 can only be 0 (the power of a non-existent DOF)
        max_in2 = (order - in1) if n_dof >= 2 else 0
        for in2 in range(max_in2 + 1):
            max_in3 = (order - in1 - in2) if n_dof >= 3 else 0
            for in3 in range(max_in3 + 1):
                max_in4 = (order - in1 - in2 - in3) if n_dof >= 4 else 0
                for in4 in range(max_in4 + 1):
                    c = muscle_poly_coeffs[poly_adr + coeff_idx]
                    t1, t2, t3, t4 = q_pows[0, in1], q_pows[1, in2], q_pows[2, in3], q_pows[3, in4]

                    # Function eval (L)
                    term_all = t1 * t2 * t3 * t4
                    length += c * term_all

                    # Partial Derivatives (dL/dq)
                    if in1 > 0:
                        df_dq[0] += c * float(in1) * q_pows[0, in1 - 1] * t2 * t3 * t4
                    if in2 > 0:
                        df_dq[1] += c * t1 * float(in2) * q_pows[1, in2 - 1] * t3 * t4
                    if in3 > 0:
                        df_dq[2] += c * t1 * t2 * float(in3) * q_pows[2, in3 - 1] * t4
                    if in4 > 0:
                        df_dq[3] += c * t1 * t2 * t3 * float(in4) * q_pows[3, in4 - 1]

                    coeff_idx += 1

    # l = f(q), v = dL/dq * dq/dt, moment_arm = -dL/dq
    muscle_length_out[worldid, muscle_id] = length
    muscle_velocity_out[worldid, muscle_id] = wp.dot(df_dq, poly_tmp_qv)
    for i in range(n_dof):
        dof_adr_i = muscle_poly_dof_adr[dep_adr + i]
        muscle_moment_arm_out[worldid, muscle_id, dof_adr_i] = -df_dq[i]
    return


@wp.kernel
def _apply_muscle_frc(
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        muscle_actuation_in: wp.array2d(dtype=float),
        muscle_moment_arm_in: wp.array3d(dtype=float),
        # Data out:
        qfrc_applied_out: wp.array2d(dtype=float),
):
    worldid, muscle_id, dofid = wp.tid()
    if integration_done_in[worldid]:
        return
    actuation = muscle_actuation_in[worldid, muscle_id]
    moment_arm = muscle_moment_arm_in[worldid, muscle_id]

    # most moment arms are zero, hopefully not much contention on the atomic adds
    q_applied = actuation * moment_arm[dofid]
    if q_applied != 0.0:
        wp.atomic_add(qfrc_applied_out[worldid], dofid, q_applied)
    return


@event_scope
def muscle_path(m: Model, d: Data):
    """
    Computes the muscle path length and velocity.
    Length calculations can be done after fwd_position,
        but it's easier to fuse with path velocity calculation
     """
    if not m.nmuscle:
        return

    # Now we can compute the path
    d.muscle_moment_arm.zero_()
    wp.launch(
        _compute_path_kernel,
        dim=(d.nworld, m.nmuscle),
        inputs=[
            m.muscle_poly_coeffs, m.muscle_poly_adr, m.muscle_poly_order,
            m.muscle_poly_qpos_adr, m.muscle_poly_dof_adr,
            m.muscle_dep_dof_num, m.muscle_dep_dof_adr,
            d.integration_done, d.qpos, d.qvel,
        ],
        outputs=[d.muscle_length, d.muscle_velocity, d.muscle_moment_arm],
    )


@event_scope
def muscle_force(m: Model, d: Data):
    if m.nmuscle:
        wp.launch(
            _apply_muscle_frc,
            dim=(d.nworld, m.nmuscle, m.nv),
            inputs=[d.integration_done, d.muscle_actuation, d.muscle_moment_arm, ],
            outputs=[d.qfrc_muscle],
        )
