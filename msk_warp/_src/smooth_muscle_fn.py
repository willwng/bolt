import warp as wp

from . import math
from .consts import MAX_POLY_NUM_DOFS
from .types import Data
from .types import Model
from .types import PolyInts
from .types import PolyVec
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _compute_path_kernel_tiled(
        # Model:
        term_coeff: wp.array(dtype=float),
        term_exponents: wp.array(dtype=PolyInts),
        term_start: wp.array(dtype=int),
        term_count: wp.array(dtype=int),
        fn_qpos_adr: wp.array(dtype=PolyInts),
        fn_dof_adr: wp.array(dtype=PolyInts),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qpos_in: wp.array2d(dtype=float),
        # Data out:
        muscle_length_out: wp.array2d(dtype=float),
        muscle_moment_arm_out: wp.array3d(dtype=float),
):
    worldid, muscle_id = wp.tid()
    if integration_done_in[worldid]:
        return

    # Fetch polynomial data: address into coeffs, order, and dependent dof addresses
    start_idx = term_start[muscle_id]
    num_terms = term_count[muscle_id]
    TILE_SIZE = wp.static(MAX_POLY_NUM_DOFS)

    qpos_adr = fn_qpos_adr[muscle_id]
    qvel_adr = fn_dof_adr[muscle_id]

    q = PolyVec(0.0)
    for i in range(MAX_POLY_NUM_DOFS):
        q[i] = qpos_in[worldid, qpos_adr[i]]

    f_accum = float(0.0)
    ma_accum = PolyVec(0.0)
    for i in range(0, num_terms, TILE_SIZE):
        coeffs_tile = wp.tile_load(term_coeff, shape=(TILE_SIZE,), offset=(start_idx + i,))
        exps_tile = wp.tile_load(term_exponents, shape=(TILE_SIZE,), offset=(start_idx + i,))

        term_deriv = wp.tile_map(math.evaluate_term_and_deriv, coeffs_tile, exps_tile, q)
        term_deriv_sum = wp.tile_sum(term_deriv)[0]
        f_accum += term_deriv_sum[0]
        ma_accum += math.poly_vec_from_eval(term_deriv_sum)

    muscle_length_out[worldid, muscle_id] = f_accum
    for i in range(MAX_POLY_NUM_DOFS):
        muscle_moment_arm_out[worldid, muscle_id, qvel_adr[i]] = -ma_accum[i]
    return


@event_scope
def compute_path_and_moment_arm(m: Model, d: Data):
    wp.launch_tiled(
        _compute_path_kernel_tiled,
        dim=(d.nworld, m.nmuscle),
        inputs=[
            m.term_coeff, m.term_exponents, m.term_start, m.term_count, m.fn_qpos_idx, m.fn_dof_idx,
            d.integration_done, d.qpos,
        ],
        outputs=[d.muscle_length, d.muscle_moment_arm],
        block_dim=MAX_POLY_NUM_DOFS
    )


@event_scope
def compute_muscle_velocity(m: Model, d: Data):
    @wp.kernel
    def _compute_muscle_velocity_tiled(
            # Data in:
            integration_done_in: wp.array(dtype=bool),
            qvel_in: wp.array2d(dtype=float),
            muscle_moment_arm_in: wp.array3d(dtype=float),
            # Data out:
            muscle_velocity_out: wp.array2d(dtype=float),
    ):
        worldid = wp.tid()
        if integration_done_in[worldid]:
            return
        nv, nmuscle = wp.static(m.nv), wp.static(m.nmuscle)

        qvel = wp.tile_reshape(wp.tile_load(qvel_in[worldid], nv), (nv, 1))
        moment_arms = wp.tile_load(muscle_moment_arm_in[worldid], (nmuscle, nv))
        muscle_velocities = wp.tile_squeeze(wp.tile_matmul(moment_arms, qvel), axis=(1,))
        wp.tile_store(muscle_velocity_out[worldid], -muscle_velocities)  # need to negate for since moment arm is -dL/dq
        return

    if m.nmuscle:
        wp.launch_tiled(
            _compute_muscle_velocity_tiled,
            dim=(d.nworld),
            inputs=[d.integration_done, d.qvel, d.muscle_moment_arm],
            outputs=[d.muscle_velocity],
            block_dim=32
        )


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

    # most moment arms are zero, but hopefully not much contention on the adds
    q_applied = actuation * moment_arm[dofid]
    if q_applied != 0.0:
        wp.atomic_add(qfrc_applied_out[worldid], dofid, q_applied)
    return


@event_scope
def muscle_fn_path_moment_arms(m: Model, d: Data):
    """ Computes the muscle path length and moment arms using a polynomial function approximation """
    if not m.nmuscle:
        return

    compute_path_and_moment_arm(m, d)
    return


@event_scope
def muscle_velocity(m: Model, d: Data):
    """ Computes the muscle path velocity from the moment arms and joint velocities. """
    compute_muscle_velocity(m, d)
    return


@event_scope
def apply_muscle_force(m: Model, d: Data):
    if m.nmuscle:
        wp.launch(
            _apply_muscle_frc,
            dim=(d.nworld, m.nmuscle, m.nv),
            inputs=[d.integration_done, d.muscle_actuation, d.muscle_moment_arm, ],
            outputs=[d.qfrc_muscle],
        )
