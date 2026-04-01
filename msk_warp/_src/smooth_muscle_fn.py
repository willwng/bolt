import warp as wp

from . import math
from .consts import MAX_POLY_NUM_DOFS
from .consts import MAX_POLY_ORDER
from .consts import POLY_TILE_SIZE
from .types import Data
from .types import Model
from .types import MuscleMetadata
from .types import PolyInts
from .types import PolyVec
from .types import PolyPowCache
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _compute_path_kernel(
        # Model:
        muscle_metadata: wp.array(dtype=MuscleMetadata),
        fn_path_dimension: wp.array(dtype=int),
        fn_path_order: wp.array(dtype=int),
        fn_path_term_coeff: wp.array(dtype=float),
        fn_path_term_start: wp.array(dtype=int),
        fn_path_qpos_adr: wp.array(dtype=PolyInts),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qpos_in: wp.array2d(dtype=float),
        qdot_in: wp.array2d(dtype=float),
        # Data out:
        muscle_length_out: wp.array2d(dtype=float),
        muscle_moment_arm_out: wp.array3d(dtype=float),
        muscle_velocity_out: wp.array2d(dtype=float),
):
    worldid, muscle_id = wp.tid()
    if integration_done_in[worldid]:
        return
    if not muscle_metadata[muscle_id].fn_based_path:
        return

    # Fetch polynomial data: dimension, order, address into coeffs, and dependent dof addresses
    n_dof = fn_path_dimension[muscle_id]
    order = fn_path_order[muscle_id]
    start_idx = fn_path_term_start[muscle_id]
    qpos_adr = fn_path_qpos_adr[muscle_id]

    # Fetch q values into registers
    q = PolyVec(0.0)
    for i in range(wp.static(MAX_POLY_NUM_DOFS)):
        q[i] = qpos_in[worldid, qpos_adr[i]]

    # Pre-calculate powers
    q_pows = PolyPowCache(1.0)
    for d in range(MAX_POLY_NUM_DOFS):
        for p in range(1, MAX_POLY_ORDER + 1):
            q_pows[d, p] = q_pows[d, p - 1] * q[d]

    # Evaluate polynomial and derivative
    length, df_dq = math.evaluate_polynomial(fn_path_term_coeff, q_pows, start_idx, order, n_dof)

    # Write out length and moment arms, note the negative sign since moment arm is -dL/dq
    muscle_length_out[worldid, muscle_id] = length
    for i in range(wp.static(MAX_POLY_NUM_DOFS)):
        if i >= n_dof:
            break
        muscle_moment_arm_out[worldid, muscle_id, qpos_adr[i]] = -df_dq[i]

    # Compute velocity
    velocity = float(0.0)
    for i in range(wp.static(MAX_POLY_NUM_DOFS)):
        if i >= n_dof:
            break
        velocity += df_dq[i] * qdot_in[worldid, qpos_adr[i]]
    muscle_velocity_out[worldid, muscle_id] = velocity
    return


@wp.kernel
def _compute_path_kernel_tiled(
        # Model:
        muscle_metadata: wp.array(dtype=MuscleMetadata),
        fn_path_term_coeff: wp.array(dtype=float),
        fn_path_term_exps: wp.array(dtype=PolyInts),
        fn_path_term_start: wp.array(dtype=int),
        fn_path_term_count: wp.array(dtype=int),
        fn_path_qpos_adr: wp.array(dtype=PolyInts),
        fn_path_dof_adr: wp.array(dtype=PolyInts),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qpos_in: wp.array2d(dtype=float),
        qvel_in: wp.array2d(dtype=float),
        # Data out:
        muscle_length_out: wp.array2d(dtype=float),
        muscle_moment_arm_out: wp.array3d(dtype=float),
        muscle_velocity_out: wp.array2d(dtype=float),
):
    worldid, muscle_id = wp.tid()
    if integration_done_in[worldid]:
        return
    if not muscle_metadata[muscle_id].fn_based_path:
        return

    # Fetch polynomial data: address into coeffs, order, and dependent dof addresses
    start_idx = fn_path_term_start[muscle_id]
    num_terms = fn_path_term_count[muscle_id]
    qpos_adr = fn_path_qpos_adr[muscle_id]
    dof_adr = fn_path_dof_adr[muscle_id]

    # Fetch q values into registers
    q = PolyVec(0.0)
    qv = PolyVec(0.0)
    for i in range(wp.static(MAX_POLY_NUM_DOFS)):
        q[i] = qpos_in[worldid, qpos_adr[i]]
        qv[i] = qvel_in[worldid, dof_adr[i]]

    f_accum = float(0.0)
    ma_accum = PolyVec(0.0)
    TILE_SIZE = wp.static(POLY_TILE_SIZE)
    for i in range(0, num_terms, TILE_SIZE):
        # Fetch coefficients and exponents
        coeffs_tile = wp.tile_load(fn_path_term_coeff, shape=(TILE_SIZE,), offset=(start_idx + i,))
        exps_tile = wp.tile_load(fn_path_term_exps, shape=(TILE_SIZE,), offset=(start_idx + i,))

        # Compute term value and derivative, accumulate
        term_deriv = wp.tile_map(math.evaluate_term_and_deriv, coeffs_tile, exps_tile, q)
        term_deriv_sum = wp.tile_sum(term_deriv)[0]
        f_accum += term_deriv_sum[0]
        ma_accum += math.poly_vec_from_eval(term_deriv_sum)

    # Write out length and moment arms, note the negative sign since moment arm is -dL/dq
    muscle_length_out[worldid, muscle_id] = f_accum
    for i in range(wp.static(MAX_POLY_NUM_DOFS)):
        muscle_moment_arm_out[worldid, muscle_id, dof_adr[i]] = -ma_accum[i]
    muscle_velocity_out[worldid, muscle_id] = -wp.dot(ma_accum, qv)
    return


@wp.kernel
def _apply_muscle_frc_kernel(
        # Model:
        muscle_metadata: wp.array(dtype=MuscleMetadata),
        fn_dimension: wp.array(dtype=int),
        fn_qpos_adr: wp.array(dtype=PolyInts),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        muscle_actuation_in: wp.array2d(dtype=float),
        muscle_moment_arm_in: wp.array3d(dtype=float),
        # Data out:
        qfrc_applied_out: wp.array2d(dtype=float),
):
    worldid, muscle_id = wp.tid()
    if integration_done_in[worldid]:
        return
    if not muscle_metadata[muscle_id].fn_based_path:
        return
    actuation = muscle_actuation_in[worldid, muscle_id]
    moment_arm = muscle_moment_arm_in[worldid, muscle_id]
    dimension = fn_dimension[muscle_id]

    # Iterate over the dependent dofs
    for varid in range(wp.static(MAX_POLY_NUM_DOFS)):
        if varid >= dimension:
            break
        # q_forces[i] = r[i] * F_muscle
        qposadr = fn_qpos_adr[muscle_id][varid]
        q_applied = actuation * moment_arm[qposadr]
        if q_applied != 0.0:
            wp.atomic_add(qfrc_applied_out[worldid], qposadr, q_applied)
    return


@event_scope
def muscle_fn_path(m: Model, d: Data):
    """ Computes the muscle path length and moment arms using a polynomial function approximation """
    if m.nmuscle:
        # wp.launch_tiled(
        #     _compute_path_kernel_tiled,
        #     dim=(d.nworld, m.nmuscle),
        #     inputs=[
        #         m.muscle_metadata, m.fn_path_term_coeffs, m.fn_path_term_exps, m.fn_path_term_start, m.fn_path_term_count,
        #         m.fn_path_qpos_adr, m.fn_path_dof_adr,
        #         d.integration_done, d.qpos, d.qvel
        #     ],
        #     outputs=[d.muscle_length, d.muscle_moment_arm, d.muscle_velocity],
        #     block_dim=m.block_dim.muscle_path,
        # )

        # non-tiled seems faster, probably because of how many threads are launched in the tiled version
        wp.launch(
            _compute_path_kernel,
            dim=(d.nworld, m.nmuscle),
            inputs=[
                m.muscle_metadata, m.fn_path_dimension, m.fn_path_order, m.fn_path_term_coeffs, m.fn_path_term_start,
                m.fn_path_qpos_adr,
                d.integration_done, d.qpos, d.qdot
            ],
            outputs=[d.muscle_length, d.muscle_moment_arm, d.muscle_velocity],
        )
    return


@event_scope
def apply_muscle_force(m: Model, d: Data):
    if m.nmuscle:
        wp.launch(
            _apply_muscle_frc_kernel,
            dim=(d.nworld, m.nmuscle),
            inputs=[m.muscle_metadata, m.fn_path_dimension, m.fn_path_qpos_adr,
                    d.integration_done, d.muscle_actuation, d.muscle_moment_arm, ],
            outputs=[d.qfrc_muscle],
        )
