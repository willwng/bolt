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
        muscle_fn_to_mid: wp.array(dtype=int),
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
    worldid, nodeid = wp.tid()
    if integration_done_in[worldid]:
        return
    muscle_id = muscle_fn_to_mid[nodeid]

    # Fetch polynomial data: dimension, order, address into coeffs, and dependent dof addresses
    n_dof = fn_path_dimension[muscle_id]
    order = fn_path_order[muscle_id]
    start_idx = fn_path_term_start[muscle_id]
    qpos_adr = fn_path_qpos_adr[muscle_id]

    # Fetch q values into registers
    q = PolyVec(0.0)
    for i in range(wp.static(MAX_POLY_NUM_DOFS)):
        if i >= n_dof:
            break
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
def _prepare_tiled_path(
        # Model:
        muscle_fn_tiled_to_mid: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        # Data out:
        muscle_length_out: wp.array2d(dtype=float),
        muscle_fn_tile_ma_tmp_out: wp.array3d(dtype=float),
):
    worldid, nodeid = wp.tid()
    if integration_done_in[worldid]:
        return
    muscle_id = muscle_fn_tiled_to_mid[nodeid]
    muscle_length_out[worldid, muscle_id] = 0.0
    for i in range(wp.static(MAX_POLY_NUM_DOFS)):
        muscle_fn_tile_ma_tmp_out[worldid, muscle_id, i] = 0.0
    return


@wp.kernel
def _compute_path_kernel_tiled(
        # Model:
        fn_tile_muscle_id: wp.array(dtype=int),
        fn_tile_offset: wp.array(dtype=int),
        fn_path_dimension: wp.array(dtype=int),
        fn_path_term_coeff: wp.array(dtype=float),
        fn_path_term_exps: wp.array(dtype=PolyInts),
        fn_path_term_start: wp.array(dtype=int),
        fn_path_qpos_adr: wp.array(dtype=PolyInts),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qpos_in: wp.array2d(dtype=float),
        # Data out:
        muscle_length_out: wp.array2d(dtype=float),
        muscle_fn_tile_ma_tmp_out: wp.array3d(dtype=float),
):
    worldid, tile_id, tid = wp.tid()
    if integration_done_in[worldid]:
        return

    muscle_id = fn_tile_muscle_id[tile_id]

    n_dof = fn_path_dimension[muscle_id]

    q = PolyVec(0.0)
    qpos_adr = fn_path_qpos_adr[muscle_id]
    for i in range(wp.static(MAX_POLY_NUM_DOFS)):
        if i < n_dof:
            q[i] = qpos_in[worldid, qpos_adr[i]]

    TILE_SIZE = wp.static(POLY_TILE_SIZE)
    offset_idx = fn_path_term_start[muscle_id] + fn_tile_offset[tile_id] * TILE_SIZE
    coeffs_tile = wp.tile_load(fn_path_term_coeff, shape=(TILE_SIZE,), offset=(offset_idx,))
    exps_tile = wp.tile_load(fn_path_term_exps, shape=(TILE_SIZE,), offset=(offset_idx,))

    # Compute term value and derivative, accumulate
    term_deriv = wp.tile_map(math.evaluate_term_and_deriv, coeffs_tile, exps_tile, q)
    term_deriv_sum = wp.tile_sum(term_deriv)[0]

    # Write out length and moment arms
    if tid == 0:
        f_accum = term_deriv_sum[0]
        df_dq = math.poly_vec_from_eval(term_deriv_sum)
        wp.atomic_add(muscle_length_out[worldid], muscle_id, f_accum)
        for i in range(wp.static(MAX_POLY_NUM_DOFS)):
            if i < n_dof:
                wp.atomic_add(muscle_fn_tile_ma_tmp_out[worldid], muscle_id, i, df_dq[i])
    return


@wp.kernel
def _post_tile_muscle(
        # Model:
        muscle_fn_tiled_to_mid: wp.array(dtype=int),
        fn_path_dimension: wp.array(dtype=int),
        fn_path_qpos_adr: wp.array(dtype=PolyInts),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        muscle_fn_tile_ma_tmp_in: wp.array3d(dtype=float),
        qdot_in: wp.array2d(dtype=float),
        # Data out:
        muscle_moment_arm_out: wp.array3d(dtype=float),
        muscle_velocity_out: wp.array2d(dtype=float),
):
    worldid, nodeid = wp.tid()
    if integration_done_in[worldid]:
        return
    muscle_id = muscle_fn_tiled_to_mid[nodeid]

    # Fetch polynomial data: dimension, order, address into coeffs, and dependent dof addresses
    n_dof = fn_path_dimension[muscle_id]
    qpos_adr = fn_path_qpos_adr[muscle_id]
    df_dq = muscle_fn_tile_ma_tmp_in[worldid, muscle_id]

    # Compute velocity
    velocity = float(0.0)
    for i in range(wp.static(MAX_POLY_NUM_DOFS)):
        if i >= n_dof:
            break
        velocity += df_dq[i] * qdot_in[worldid, qpos_adr[i]]
        muscle_moment_arm_out[worldid, muscle_id, qpos_adr[i]] = -df_dq[i]
    muscle_velocity_out[worldid, muscle_id] = velocity
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
def muscle_fn_path_tiled(m: Model, d: Data):
    wp.launch(
        _prepare_tiled_path,
        dim=(d.nworld, m.nm_fntilepaths),
        inputs=[m.muscle_fn_tiled_to_mid, d.integration_done,],
        outputs=[d.muscle_length, d.muscle_fn_tile_ma_tmp]
    )
    wp.launch_tiled(
        _compute_path_kernel_tiled,
        dim=(d.nworld, m.n_fn_path_tiles),
        inputs=[
            m.fn_tile_muscle_id, m.fn_tile_offset, m.fn_path_dimension, m.fn_path_term_coeffs,
            m.fn_path_term_exps, m.fn_path_term_start, m.fn_path_qpos_adr,
            d.integration_done, d.qpos,
        ],
        outputs=[d.muscle_length, d.muscle_fn_tile_ma_tmp],
        block_dim=m.block_dim.muscle_path,
    )
    wp.launch(
        _post_tile_muscle,
        dim=(d.nworld, m.nm_fntilepaths),
        inputs=[
            m.muscle_fn_tiled_to_mid, m.fn_path_dimension, m.fn_path_qpos_adr,
            d.integration_done, d.muscle_fn_tile_ma_tmp, d.qdot
        ],
        outputs=[d.muscle_moment_arm, d.muscle_velocity],
    )


@event_scope
def muscle_fn_path_standard(m: Model, d: Data):
    """ Computes the muscle path length and moment arms using a polynomial function approximation """
    wp.launch(
        _compute_path_kernel,
        dim=(d.nworld, m.nm_fnpaths),
        inputs=[
            m.muscle_fn_to_mid, m.fn_path_dimension, m.fn_path_order, m.fn_path_term_coeffs, m.fn_path_term_start,
            m.fn_path_qpos_adr,
            d.integration_done, d.qpos, d.qdot
        ],
        outputs=[d.muscle_length, d.muscle_moment_arm, d.muscle_velocity],
    )
    return


@event_scope
def muscle_fn_path(m: Model, d: Data):
    """ Computes the muscle path length and moment arms using a polynomial function approximation """
    if m.nm_fnpaths:
        muscle_fn_path_standard(m, d)

    if m.nm_fntilepaths:
        muscle_fn_path_tiled(m, d)


@event_scope
def apply_muscle_force_fn(m: Model, d: Data, passive_only: bool = False):
    if m.nmuscle:
        actuation_in = d.muscle_actuation_passive if passive_only else d.muscle_actuation
        qfrc_out = d.qfrc_muscle_passive if passive_only else d.qfrc_muscle
        wp.launch(
            _apply_muscle_frc_kernel,
            dim=(d.nworld, m.nmuscle),
            inputs=[m.muscle_metadata, m.fn_path_dimension, m.fn_path_qpos_adr,
                    d.integration_done, actuation_in, d.muscle_moment_arm, ],
            outputs=[qfrc_out],
        )
    return
