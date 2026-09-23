import functools

import warp as wp

from bolt.consts import MAX_POLY_NUM_DOFS
from bolt.types import Data
from bolt.types import Model
from bolt.types import PolyInts
from bolt.types import PolyPowCache
from bolt.types import PolyVec

from ..warp_util import event_scope
from . import polynomial_evaluator

wp.set_module_options({"enable_backward": False})


@functools.cache
def _compute_path_kernel(dimension: int, order: int):
    """
    Computes length, moment arms and velocity for function-path muscles whose polynomial has this (dimension, order)
    """
    if (dimension, order) not in polynomial_evaluator.EVALUATORS:
        raise ValueError(f"Function-based paths with dimension {dimension} and order {order} are not supported. "
                         f"Add it to generate_polynomial_evaluator.py and regenerate")
    evaluate_polynomial = polynomial_evaluator.EVALUATORS[(dimension, order)]

    @wp.kernel(module="unique")
    def kernel(
            # Model:
            fn_path_term_coeff: wp.array(dtype=float),
            fn_path_term_start: wp.array(dtype=int),
            fn_path_qpos_adr: wp.array(dtype=PolyInts),
            # Data in:
            integration_done_in: wp.array(dtype=bool),
            qpos_in: wp.array2d(dtype=float),
            qdot_in: wp.array2d(dtype=float),
            # In:
            fn_group: wp.array(dtype=int),
            # Data out:
            muscle_length_out: wp.array2d(dtype=float),
            muscle_moment_arm_out: wp.array3d(dtype=float),
            muscle_velocity_out: wp.array2d(dtype=float),
    ):
        worldid, nodeid = wp.tid()
        if integration_done_in[worldid]:
            return
        muscle_id = fn_group[nodeid]

        # Fetch polynomial data: address into coeffs, and dependent dof addresses
        start_idx = fn_path_term_start[muscle_id]
        qpos_adr = fn_path_qpos_adr[muscle_id]

        # Fetch q values into registers
        q = PolyVec(0.0)
        for i in range(wp.static(dimension)):
            q[i] = qpos_in[worldid, qpos_adr[i]]

        # Pre-calculate powers
        q_pows = PolyPowCache(1.0)
        for dof in range(wp.static(dimension)):
            for p in range(1, wp.static(order + 1)):
                q_pows[dof, p] = q_pows[dof, p - 1] * q[dof]

        # Evaluate polynomial and derivative
        length, df_dq = evaluate_polynomial(fn_path_term_coeff, q_pows, start_idx)

        # Write out length
        muscle_length_out[worldid, muscle_id] = length
        # Write moment arm and compute velocity
        velocity = float(0.0)
        for i in range(wp.static(dimension)):
            muscle_moment_arm_out[worldid, muscle_id, qpos_adr[i]] = -df_dq[i]
            velocity += df_dq[i] * qdot_in[worldid, qpos_adr[i]]
        muscle_velocity_out[worldid, muscle_id] = velocity

    return kernel


@wp.kernel
def _apply_muscle_frc_kernel(
        # Model:
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


@wp.kernel
def _apply_muscle_frc_breakdown_kernel(
        # Model:
        fn_dimension: wp.array(dtype=int),
        fn_qpos_adr: wp.array(dtype=PolyInts),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        muscle_actuation_in: wp.array2d(dtype=float),
        muscle_moment_arm_in: wp.array3d(dtype=float),
        # Data out:
        qfrc_applied_out: wp.array3d(dtype=float),
):
    # Same as above but for breakdowns only
    worldid, muscle_id = wp.tid()
    if integration_done_in[worldid]:
        return
    actuation = muscle_actuation_in[worldid, muscle_id]
    moment_arm = muscle_moment_arm_in[worldid, muscle_id]
    dimension = fn_dimension[muscle_id]

    for varid in range(wp.static(MAX_POLY_NUM_DOFS)):
        if varid >= dimension:
            break
        qposadr = fn_qpos_adr[muscle_id][varid]
        q_applied = actuation * moment_arm[qposadr]
        qfrc_applied_out[worldid, qposadr, muscle_id] = q_applied
    return


@event_scope
def muscle_fn_path(m: Model, d: Data):
    """ Computes the muscle path length and moment arms using a polynomial function approximation """
    for (dimension, order), fn_group in zip(m.muscle_fn_group_dim_order, m.muscle_fn_groups):
        wp.launch(
            _compute_path_kernel(dimension, order),
            dim=(d.nworld, fn_group.size),
            inputs=[
                m.fn_path_term_coeffs, m.fn_path_term_start, m.fn_path_qpos_adr,
                d.integration_done, d.qpos, d.qdot,
                fn_group
            ],
            outputs=[d.muscle_length, d.muscle_moment_arm, d.muscle_velocity],
        )
    return


@event_scope
def apply_muscle_force_fn(m: Model, d: Data, passive_only: bool = False):
    if m.nmuscle:
        actuation_in = d.muscle_actuation_passive if passive_only else d.muscle_actuation
        qfrc_out = d.qfrc_muscle_passive if passive_only else d.qfrc_muscle
        wp.launch(
            _apply_muscle_frc_kernel,
            dim=(d.nworld, m.nmuscle),
            inputs=[m.fn_path_dimension, m.fn_path_qpos_adr,
                    d.integration_done, actuation_in, d.muscle_moment_arm, ],
            outputs=[qfrc_out],
        )
    return


@event_scope
def apply_muscle_force_fn_breakdown(m: Model, d: Data, actuation: wp.array, qfrc_breakdown_out: wp.array):
    if m.nmuscle:
        wp.launch(
            _apply_muscle_frc_breakdown_kernel,
            dim=(d.nworld, m.nmuscle),
            inputs=[m.fn_path_dimension, m.fn_path_qpos_adr,
                    d.integration_done, actuation, d.muscle_moment_arm, ],
            outputs=[qfrc_breakdown_out],
        )
    return
