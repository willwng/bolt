import warp as wp

from . import dgf
from .types import Data
from .types import Model
from .types import ResidualResult
from .types import MuscleMetadata
from .types import MuscleLengthInfo
from .types import FiberVelocityInfo
from .types import MuscleDynamicsInfo
from .consts import M_MIN_NORM_TENDON_FORCE
from .consts import M_MAX_NORM_TENDON_FORCE
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _compute_activation_dot(
        # Data in:
        mexcitation_in: wp.array2d(dtype=float),
        act_in: wp.array2d(dtype=float),
        # Data out:
        act_dot_out: wp.array2d(dtype=float),
):
    worldid, muscle_id = wp.tid()
    excitation = mexcitation_in[worldid, muscle_id]
    activation = act_in[worldid, muscle_id]
    act_dot = dgf.calc_activation_derivative(activation, excitation)
    act_dot_out[worldid, muscle_id] = act_dot
    return


@event_scope
def compute_act_dot(m: Model, d: Data):
    wp.launch(
        _compute_activation_dot,
        dim=(d.nworld, m.nmuscle),
        inputs=[d.mexcitations, d.act],
        outputs=[d.act_dot],
    )


@wp.kernel
def _reset_prev_path(
        # Data in:
        world_reset_in: wp.array(dtype=bool),
        muscle_length_in: wp.array2d(dtype=float),
        muscle_velocity_in: wp.array2d(dtype=float),
        # Data out:
        muscle_length_prev_out: wp.array2d(dtype=float),
        muscle_velocity_prev_out: wp.array2d(dtype=float),
):
    worldid, muscle_id = wp.tid()
    if world_reset_in[worldid]:
        muscle_length_prev_out[worldid, muscle_id] = muscle_length_in[
            worldid, muscle_id]
        muscle_velocity_prev_out[worldid, muscle_id] = muscle_velocity_in[
            worldid, muscle_id]
    return


@wp.func
def _calc_eq_residual(
        norm_tendon_force: float,
        path_length: float,
        path_velocity: float,
        activation: float,
        mm: MuscleMetadata,
) -> ResidualResult:
    # Retrieve tendon length from force
    norm_tendon_length = dgf.calc_tendon_force_length_inverse_curve(
        norm_tendon_force)
    tendon_length = norm_tendon_length * mm.tendon_slack_length
    # From path and tendon length compute fiber length
    fiber_width = dgf.get_fiber_width(mm.optimal_fiber_length,
                                      mm.optimal_pennation_angle)
    fiber_length_along_tendon = path_length - tendon_length
    fiber_length = wp.sqrt(
        fiber_length_along_tendon ** 2.0 + fiber_width ** 2.0)
    norm_fiber_length = fiber_length / mm.optimal_fiber_length
    # Pennation angle
    cos_pennation_angle = fiber_length_along_tendon / fiber_length
    sin_pennation_angle = fiber_width / fiber_length
    pennation_angle = wp.asin(sin_pennation_angle)
    if pennation_angle > wp.acos(0.1):
        pennation_angle = wp.acos(0.1)
        cos_pennation_angle = wp.cos(pennation_angle)
        sin_pennation_angle = wp.sin(pennation_angle)
    # Tendon velocity
    norm_tendon_velocity = (
        dgf.calc_tendon_force_length_inverse_curve_derivative(
            0.0, norm_tendon_length))
    tendon_velocity = mm.tendon_slack_length * norm_tendon_velocity
    # Fiber velocity
    fiber_velocity_along_tendon = path_velocity - tendon_velocity
    fiber_velocity = fiber_velocity_along_tendon * cos_pennation_angle
    norm_fiber_velocity = (fiber_velocity /
                           dgf.get_max_contraction_velocity_in_meters_per_second(
                               mm.v_max, mm.optimal_fiber_length))
    # Residual
    active_fiber_force = dgf.calc_active_fiber_force(
        mm.max_isometric_force, activation, norm_fiber_length,
        norm_fiber_velocity)
    passive_fiber_force = dgf.calc_passive_fiber_force(
        mm.max_isometric_force, norm_fiber_length, norm_fiber_velocity,
        mm.fiber_damping, mm.min_norm_fiber_length)
    fiber_force = (active_fiber_force + passive_fiber_force)
    fiber_force_along_tendon = fiber_force * cos_pennation_angle
    residual = (norm_tendon_force -
                fiber_force_along_tendon / mm.max_isometric_force)

    return ResidualResult(
        norm_tendon_force=norm_tendon_force,
        residual=residual,
        pennation_angle=pennation_angle,
        fiber_length=fiber_length,
        norm_fiber_length=norm_fiber_length,
        tendon_length=tendon_length,
        norm_tendon_length=norm_tendon_length,
        norm_tendon_velocity=norm_tendon_velocity,
        active_fiber_force=active_fiber_force,
        fiber_velocity=fiber_velocity,
        fiber_force_along_tendon=fiber_force_along_tendon)


@wp.kernel
def _equilibrate(
        # Model:
        muscle_metadata: wp.array(dtype=MuscleMetadata),
        # Data in:
        world_reset_in: wp.array(dtype=bool),
        muscle_length_in: wp.array2d(dtype=float),
        muscle_velocity_in: wp.array2d(dtype=float),
        act_in: wp.array2d(dtype=float),
        # Data out:
        mstate_out: wp.array2d(dtype=float)
):
    worldid, muscle_id = wp.tid()

    # Only equilibrate if the world was reset
    if not world_reset_in[worldid]:
        return

    # Bisection to solve for equilibrium
    lower = M_MIN_NORM_TENDON_FORCE
    upper = M_MAX_NORM_TENDON_FORCE
    mid = 0.5 * (lower + upper)
    tol = 1e-8
    max_iters = 30

    path_length = muscle_length_in[worldid, muscle_id]
    path_velocity = muscle_velocity_in[worldid, muscle_id]
    activation = act_in[worldid, muscle_id]
    metadata = muscle_metadata[muscle_id]

    res_lower = _calc_eq_residual(lower, path_length, path_velocity,
                                  activation, metadata)
    res_upper = _calc_eq_residual(upper, path_length, path_velocity,
                                  activation, metadata)
    res_mid = _calc_eq_residual(mid, path_length, path_velocity,
                                activation, metadata)
    res_best = res_lower if wp.abs(res_lower.residual) < wp.abs(
        res_upper.residual) else res_upper

    for i in range(max_iters):
        # Converted or interval is sufficiently small
        if wp.abs(res_best.residual) < tol or 0.5 * (upper - lower) < tol:
            break
        # Update bounds
        if res_lower.residual * res_mid.residual > 0.0:
            lower = mid
            res_lower = res_mid
        else:
            upper = mid
        # New midpoint
        mid = 0.5 * (lower + upper)
        res_mid = _calc_eq_residual(mid, path_length, path_velocity,
                                    activation, metadata)
        # Update best
        if abs(res_mid.residual) < abs(res_best.residual):
            res_best = res_mid

    # Set state
    fiber_length = res_best.fiber_length
    norm_fiber_length = fiber_length / metadata.optimal_fiber_length

    mstate_out[worldid, muscle_id] = dgf.clamp_fiber_length(
        norm_fiber_length, metadata.min_norm_fiber_length,
        metadata.max_norm_fiber_length)
    return


@wp.kernel
def _update_length_info(
        # Model:
        muscle_metadata: wp.array(dtype=MuscleMetadata),
        # Data in:
        mstate_in: wp.array2d(dtype=float),
        muscle_length_in: wp.array2d(dtype=float),
        # Data out:
        muscle_length_info_out: wp.array2d(dtype=MuscleLengthInfo),
):
    worldid, muscle_id = wp.tid()

    mm = muscle_metadata[muscle_id]
    norm_fiber_length = mstate_in[worldid, muscle_id]
    path_length = muscle_length_in[worldid, muscle_id]

    # Fiber
    fiber_length = norm_fiber_length * mm.optimal_fiber_length
    min_norm_fiber_length = mm.min_norm_fiber_length
    # Pennation angle
    pennation_angle = dgf.calc_pennation_angle(mm.optimal_pennation_angle,
                                               mm.optimal_fiber_length,
                                               norm_fiber_length,
                                               min_norm_fiber_length)
    cos_pennation_angle = wp.cos(pennation_angle)
    sin_pennation_angle = wp.sin(pennation_angle)
    fiber_length_along_tendon = fiber_length * cos_pennation_angle
    # Tendon
    tendon_length = path_length - fiber_length_along_tendon
    norm_tendon_length = tendon_length / mm.tendon_slack_length
    tendon_strain = norm_tendon_length - 1.0
    # Force multipliers
    fiber_passive_force_length_multiplier = (
        dgf.calc_passive_force_multiplier(norm_fiber_length,
                                          min_norm_fiber_length))
    fiber_active_force_length_multiplier = (
        dgf.calc_active_force_length_multiplier(norm_fiber_length))
    force_multiplier = (
        dgf.calc_tendon_force_multiplier(norm_tendon_length, True))

    # Set info
    mli = muscle_length_info_out[worldid]
    mli[muscle_id].fiber_length = fiber_length
    mli[muscle_id].pennation_angle = pennation_angle
    mli[muscle_id].cos_pennation_angle = cos_pennation_angle
    mli[muscle_id].sin_pennation_angle = sin_pennation_angle
    mli[muscle_id].norm_fiber_length = norm_fiber_length
    mli[muscle_id].fiber_length_along_tendon = fiber_length_along_tendon
    mli[muscle_id].tendon_length = tendon_length
    mli[muscle_id].norm_tendon_length = norm_tendon_length
    mli[muscle_id].tendon_strain = tendon_strain
    mli[muscle_id].fiber_passive_force_length_multiplier = (
        fiber_passive_force_length_multiplier)
    mli[muscle_id].fiber_active_force_length_multiplier = (
        fiber_active_force_length_multiplier)
    mli[muscle_id].tendon_force_multiplier = force_multiplier

    return


@wp.kernel
def _update_velocity_info(
        # Model:
        muscle_metadata: wp.array(dtype=MuscleMetadata),
        # Data in:
        act_in: wp.array2d(dtype=float),
        muscle_length_info_in: wp.array2d(dtype=MuscleLengthInfo),
        muscle_velocity_in: wp.array2d(dtype=float),
        # Data out:
        muscle_velocity_info_out: wp.array2d(dtype=FiberVelocityInfo)
):
    worldid, muscle_id = wp.tid()

    mm = muscle_metadata[muscle_id]
    mli = muscle_length_info_in[worldid, muscle_id]
    path_velocity = muscle_velocity_in[worldid, muscle_id]
    activation = act_in[worldid, muscle_id]

    # Compute fiber velocity multiplier
    if mm.fiber_damping > 0.0:
        dlceN_dt, fv = dgf.calc_damped_norm_fiber_velocity(
            mm.max_isometric_force,
            activation,
            mli.fiber_active_force_length_multiplier,
            mli.fiber_passive_force_length_multiplier,
            mli.tendon_force_multiplier,
            mm.fiber_damping,
            mli.cos_pennation_angle)
        norm_fiber_velocity = dlceN_dt
        fiber_force_velocity_multiplier = fv
    else:
        fv = dgf.calc_undamped_fiber_force_velocity_multiplier(
            activation,
            mli.fiber_active_force_length_multiplier,
            mli.fiber_passive_force_length_multiplier,
            mli.tendon_force_multiplier,
            mli.cos_pennation_angle
        )
        norm_fiber_velocity = dgf.calc_force_velocity_inverse_curve(fv)
        fiber_force_velocity_multiplier = fv

    fiber_velocity = (norm_fiber_velocity *
                      dgf.get_max_contraction_velocity_in_meters_per_second(
                          mm.v_max, mm.optimal_fiber_length))
    pennation_angular_velocity = dgf.calc_pennation_angular_velocity(
        mm.optimal_pennation_angle, mli.fiber_length, fiber_velocity,
        wp.tan(mli.pennation_angle))
    fiber_velocity_along_tendon = dgf.calc_fiber_velocity_along_tendon(
        mli.fiber_length, fiber_velocity, mli.sin_pennation_angle,
        mli.cos_pennation_angle, pennation_angular_velocity)

    tendon_velocity = dgf.calc_tendon_velocity(
        mli.cos_pennation_angle, mli.sin_pennation_angle,
        pennation_angular_velocity, mli.fiber_length,
        fiber_velocity, path_velocity)
    norm_tendon_velocity = tendon_velocity / mm.tendon_slack_length

    # Check to see whether the fiber length was clamped
    min_norm_fiber_length = mm.min_norm_fiber_length
    mli.fiber_state_clamped = dgf.is_fiber_state_clamped(
        mli.norm_fiber_length, norm_fiber_velocity, min_norm_fiber_length)
    if mli.fiber_state_clamped:
        norm_fiber_velocity = 0.0
        fiber_velocity = 0.0
        fiber_velocity_along_tendon = 0.0
        pennation_angular_velocity = 0.0
        tendon_velocity = path_velocity
        norm_tendon_velocity = tendon_velocity / mm.tendon_slack_length
        fiber_force_velocity_multiplier = 1.0  # consistent w fiber vel 0

    fvi = muscle_velocity_info_out[worldid]
    fvi[muscle_id].fiber_velocity = fiber_velocity
    fvi[muscle_id].fiber_velocity_along_tendon = fiber_velocity_along_tendon
    fvi[muscle_id].norm_fiber_velocity = norm_fiber_velocity
    fvi[muscle_id].pennation_angular_velocity = pennation_angular_velocity
    fvi[muscle_id].tendon_velocity = tendon_velocity
    fvi[muscle_id].norm_tendon_velocity = norm_tendon_velocity
    fvi[muscle_id].fiber_force_velocity_multiplier = (
        fiber_force_velocity_multiplier)
    return


@wp.kernel
def _update_dynamics_info(
        # Model:
        muscle_metadata: wp.array(dtype=MuscleMetadata),
        # Data in:
        act_in: wp.array2d(dtype=float),
        muscle_length_info_in: wp.array2d(dtype=MuscleLengthInfo),
        muscle_velocity_info_in: wp.array2d(dtype=FiberVelocityInfo),
        # Data out:
        muscle_dynamics_info_out: wp.array2d(dtype=MuscleDynamicsInfo),
):
    worldid, muscle_id = wp.tid()

    mm = muscle_metadata[muscle_id]
    mli = muscle_length_info_in[worldid, muscle_id]
    fvi = muscle_velocity_info_in[worldid, muscle_id]
    activation = act_in[worldid, muscle_id]

    fm, aFm, p1Fm, p2Fm, pFm, fmAT = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    fse = mli.tendon_force_multiplier

    if not mli.fiber_state_clamped:
        aFm = (mm.max_isometric_force * activation *
               mli.fiber_active_force_length_multiplier *
               fvi.fiber_force_velocity_multiplier)
        p1Fm = (mm.max_isometric_force *
                mli.fiber_passive_force_length_multiplier)
        p2Fm = (mm.max_isometric_force *
                mm.fiber_damping * fvi.norm_fiber_velocity)
        pFm = p1Fm + p2Fm

        fm = aFm + pFm
        fmAT = fm * mli.cos_pennation_angle

    mdi = muscle_dynamics_info_out[worldid]
    mdi[muscle_id].fiber_force = fm
    mdi[muscle_id].fiber_force_along_tendon = fmAT
    mdi[muscle_id].norm_fiber_force = fm / mm.max_isometric_force
    mdi[muscle_id].active_fiber_force = aFm
    mdi[muscle_id].passive_fiber_force = pFm
    mdi[muscle_id].tendon_force = fse * mm.max_isometric_force
    mdi[muscle_id].norm_tendon_force = fse
    return


@wp.func
def compute_state_derivative(
        mm: MuscleMetadata,
        norm_fiber_length: float,
        path_length: float,
        activation: float,
) -> float:
    # Fiber
    fiber_length = norm_fiber_length * mm.optimal_fiber_length
    min_norm_fiber_length = mm.min_norm_fiber_length
    # Pennation angle
    pennation_angle = dgf.calc_pennation_angle(mm.optimal_pennation_angle,
                                               mm.optimal_fiber_length,
                                               norm_fiber_length,
                                               min_norm_fiber_length)
    cos_pennation_angle = wp.cos(pennation_angle)
    fiber_length_along_tendon = fiber_length * cos_pennation_angle
    # Tendon
    tendon_length = path_length - fiber_length_along_tendon
    norm_tendon_length = tendon_length / mm.tendon_slack_length
    # Force multipliers
    fiber_passive_force_length_multiplier = (
        dgf.calc_passive_force_multiplier(norm_fiber_length,
                                          min_norm_fiber_length))
    fiber_active_force_length_multiplier = (
        dgf.calc_active_force_length_multiplier(norm_fiber_length))
    tendon_force_multiplier = (
        dgf.calc_tendon_force_multiplier(norm_tendon_length, True))

    # Compute fiber velocity multiplier
    if mm.fiber_damping > 0.0:
        dlceN_dt, fv = dgf.calc_damped_norm_fiber_velocity(
            mm.max_isometric_force,
            activation,
            fiber_active_force_length_multiplier,
            fiber_passive_force_length_multiplier,
            tendon_force_multiplier,
            mm.fiber_damping,
            cos_pennation_angle)
        norm_fiber_velocity = dlceN_dt
    else:
        fv = dgf.calc_undamped_fiber_force_velocity_multiplier(
            activation,
            fiber_active_force_length_multiplier,
            fiber_passive_force_length_multiplier,
            tendon_force_multiplier,
            cos_pennation_angle
        )
        norm_fiber_velocity = dgf.calc_force_velocity_inverse_curve(fv)

    fiber_velocity = (norm_fiber_velocity *
                      dgf.get_max_contraction_velocity_in_meters_per_second(
                          mm.v_max, mm.optimal_fiber_length))

    # Check to see whether the fiber length was clamped
    min_norm_fiber_length = mm.min_norm_fiber_length
    fiber_state_clamped = dgf.is_fiber_state_clamped(
        norm_fiber_length, norm_fiber_velocity, min_norm_fiber_length)
    fiber_velocity = wp.where(fiber_state_clamped, 0.0, fiber_velocity)

    mstate_dot = fiber_velocity / mm.optimal_fiber_length
    return mstate_dot


@wp.func
def lerp(
        a: float,
        b: float,
        f: float,
) -> float:
    return a + f * (b - a)


@event_scope
def substep_fused(m: Model, d: Data, scale: float):
    @wp.kernel
    def substep_fused_kernel(
            # Model:
            muscle_metadata: wp.array(dtype=MuscleMetadata),
            # Data in:
            act_in: wp.array2d(dtype=float),
            mstate_in: wp.array2d(dtype=float),
            muscle_length_in: wp.array2d(dtype=float),
            muscle_length_prev_in: wp.array2d(dtype=float),
            actual_step_size_in: wp.array(dtype=float),
            # Data out:
            mstate_out: wp.array2d(dtype=float),
    ):
        worldid, muscle_id = wp.tid()

        mm = muscle_metadata[muscle_id]
        prev_path_length = muscle_length_prev_in[worldid, muscle_id]
        path_length = muscle_length_in[worldid, muscle_id]
        activation = act_in[worldid, muscle_id]
        norm_fiber_length = mstate_in[worldid, muscle_id]

        # Substep integration
        num_substeps = wp.static(m.opt.muscle_dyn_substeps)
        h = (scale * actual_step_size_in[0]) / float(num_substeps)
        for i in range(wp.static(num_substeps)):
            # Interpolate path length and velocity
            frac = (float(i) / float(num_substeps))
            c_path_length = lerp(prev_path_length, path_length, frac)
            # Compute state derivative
            mstate_dot = compute_state_derivative(
                mm, norm_fiber_length, c_path_length, activation)
            # Integrate
            norm_fiber_length += h * mstate_dot
            norm_fiber_length = dgf.clamp_fiber_length(
                norm_fiber_length, mm.min_norm_fiber_length, mm.max_norm_fiber_length)

        # Store updated state
        mstate_out[worldid, muscle_id] = norm_fiber_length

    wp.launch(
        substep_fused_kernel,
        dim=(d.nworld, m.nmuscle),
        inputs=[m.muscle_metadata, d.act, d.mstate,
                d.muscle_length, d.muscle_length_prev,
                d.actual_step_size],
        outputs=[d.mstate],
    )


@wp.kernel
def _update_info_fused(
        # Model:
        muscle_metadata: wp.array(dtype=MuscleMetadata),
        # Data in:
        act_in: wp.array2d(dtype=float),
        mstate_in: wp.array2d(dtype=float),
        muscle_length_in: wp.array2d(dtype=float),
        muscle_velocity_in: wp.array2d(dtype=float),
        # Data out:
        muscle_length_info_out: wp.array2d(dtype=MuscleLengthInfo),
        muscle_velocity_info_out: wp.array2d(dtype=FiberVelocityInfo),
        muscle_dynamics_info_out: wp.array2d(dtype=MuscleDynamicsInfo),
):
    worldid, muscle_id = wp.tid()

    mm = muscle_metadata[muscle_id]
    norm_fiber_length = mstate_in[worldid, muscle_id]
    path_length = muscle_length_in[worldid, muscle_id]
    path_velocity = muscle_velocity_in[worldid, muscle_id]
    activation = act_in[worldid, muscle_id]

    # Fiber
    fiber_length = norm_fiber_length * mm.optimal_fiber_length
    min_norm_fiber_length = mm.min_norm_fiber_length
    # Pennation angle
    pennation_angle = dgf.calc_pennation_angle(mm.optimal_pennation_angle,
                                               mm.optimal_fiber_length,
                                               norm_fiber_length,
                                               min_norm_fiber_length)
    cos_pennation_angle = wp.cos(pennation_angle)
    sin_pennation_angle = wp.sin(pennation_angle)
    fiber_length_along_tendon = fiber_length * cos_pennation_angle
    # Tendon
    tendon_length = path_length - fiber_length_along_tendon
    norm_tendon_length = tendon_length / mm.tendon_slack_length
    tendon_strain = norm_tendon_length - 1.0
    # Force multipliers
    fiber_passive_force_length_multiplier = (
        dgf.calc_passive_force_multiplier(norm_fiber_length,
                                          min_norm_fiber_length))
    fiber_active_force_length_multiplier = (
        dgf.calc_active_force_length_multiplier(norm_fiber_length))
    tendon_force_multiplier = (
        dgf.calc_tendon_force_multiplier(norm_tendon_length, True))

    # Compute fiber velocity multiplier
    if mm.fiber_damping > 0.0:
        dlceN_dt, fv = dgf.calc_damped_norm_fiber_velocity(
            mm.max_isometric_force,
            activation,
            fiber_active_force_length_multiplier,
            fiber_passive_force_length_multiplier,
            tendon_force_multiplier,
            mm.fiber_damping,
            cos_pennation_angle)
        norm_fiber_velocity = dlceN_dt
        fiber_force_velocity_multiplier = fv
    else:
        fv = dgf.calc_undamped_fiber_force_velocity_multiplier(
            activation,
            fiber_active_force_length_multiplier,
            fiber_passive_force_length_multiplier,
            tendon_force_multiplier,
            cos_pennation_angle
        )
        norm_fiber_velocity = dgf.calc_force_velocity_inverse_curve(fv)
        fiber_force_velocity_multiplier = fv

    fiber_velocity = (norm_fiber_velocity *
                      dgf.get_max_contraction_velocity_in_meters_per_second(
                          mm.v_max, mm.optimal_fiber_length))
    pennation_angular_velocity = dgf.calc_pennation_angular_velocity(
        mm.optimal_pennation_angle, fiber_length, fiber_velocity,
        wp.tan(pennation_angle))
    fiber_velocity_along_tendon = dgf.calc_fiber_velocity_along_tendon(
        fiber_length, fiber_velocity, sin_pennation_angle,
        cos_pennation_angle, pennation_angular_velocity)

    tendon_velocity = dgf.calc_tendon_velocity(
        cos_pennation_angle, sin_pennation_angle,
        pennation_angular_velocity, fiber_length,
        fiber_velocity, path_velocity)
    norm_tendon_velocity = tendon_velocity / mm.tendon_slack_length

    # Check to see whether the fiber length was clamped
    min_norm_fiber_length = mm.min_norm_fiber_length
    fiber_state_clamped = dgf.is_fiber_state_clamped(
        norm_fiber_length, norm_fiber_velocity, min_norm_fiber_length)
    if fiber_state_clamped:
        norm_fiber_velocity = 0.0
        fiber_velocity = 0.0
        fiber_velocity_along_tendon = 0.0
        pennation_angular_velocity = 0.0
        tendon_velocity = path_velocity
        norm_tendon_velocity = tendon_velocity / mm.tendon_slack_length
        fiber_force_velocity_multiplier = 1.0  # consistent w fiber vel 0

    fm, aFm, p1Fm, p2Fm, pFm, fmAT = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    fse = tendon_force_multiplier

    if not fiber_state_clamped:
        aFm = (mm.max_isometric_force * activation *
               fiber_active_force_length_multiplier *
               fiber_force_velocity_multiplier)
        p1Fm = (mm.max_isometric_force *
                fiber_passive_force_length_multiplier)
        p2Fm = (mm.max_isometric_force *
                mm.fiber_damping * norm_fiber_velocity)
        pFm = p1Fm + p2Fm

        fm = aFm + pFm
        fmAT = fm * cos_pennation_angle

    # Final write
    mli = muscle_length_info_out[worldid]
    fvi = muscle_velocity_info_out[worldid]
    mdi = muscle_dynamics_info_out[worldid]

    mli[muscle_id].fiber_length = fiber_length
    mli[muscle_id].pennation_angle = pennation_angle
    mli[muscle_id].cos_pennation_angle = cos_pennation_angle
    mli[muscle_id].sin_pennation_angle = sin_pennation_angle
    mli[muscle_id].norm_fiber_length = norm_fiber_length
    mli[muscle_id].fiber_length_along_tendon = fiber_length_along_tendon
    mli[muscle_id].tendon_length = tendon_length
    mli[muscle_id].norm_tendon_length = norm_tendon_length
    mli[muscle_id].tendon_strain = tendon_strain
    mli[muscle_id].fiber_passive_force_length_multiplier = (
        fiber_passive_force_length_multiplier)
    mli[muscle_id].fiber_active_force_length_multiplier = (
        fiber_active_force_length_multiplier)
    mli[muscle_id].tendon_force_multiplier = tendon_force_multiplier

    fvi[muscle_id].fiber_velocity = fiber_velocity
    fvi[muscle_id].fiber_velocity_along_tendon = fiber_velocity_along_tendon
    fvi[muscle_id].norm_fiber_velocity = norm_fiber_velocity
    fvi[muscle_id].pennation_angular_velocity = pennation_angular_velocity
    fvi[muscle_id].tendon_velocity = tendon_velocity
    fvi[muscle_id].norm_tendon_velocity = norm_tendon_velocity
    fvi[muscle_id].fiber_force_velocity_multiplier = (
        fiber_force_velocity_multiplier)

    mdi[muscle_id].fiber_force = fm
    mdi[muscle_id].fiber_force_along_tendon = fmAT
    mdi[muscle_id].norm_fiber_force = fm / mm.max_isometric_force
    mdi[muscle_id].active_fiber_force = aFm
    mdi[muscle_id].passive_fiber_force = pFm
    mdi[muscle_id].tendon_force = fse * mm.max_isometric_force
    mdi[muscle_id].norm_tendon_force = fse
    return


@wp.kernel
def _set_state(
        # Model:
        muscle_metadata: wp.array(dtype=MuscleMetadata),
        # Data in:
        muscle_velocity_info_in: wp.array2d(dtype=FiberVelocityInfo),
        muscle_dynamics_info_in: wp.array2d(dtype=MuscleDynamicsInfo),
        # Data out:
        mstate_dot_out: wp.array2d(dtype=float),
        muscle_actuation_out: wp.array2d(dtype=float),
):
    worldid, muscle_id = wp.tid()
    mm = muscle_metadata[muscle_id]
    fvi = muscle_velocity_info_in[worldid, muscle_id]
    mdi = muscle_dynamics_info_in[worldid, muscle_id]

    muscle_actuation_out[worldid, muscle_id] = mdi.tendon_force
    mstate_dot_out[worldid, muscle_id] = (
            fvi.fiber_velocity / mm.optimal_fiber_length)
    return


@event_scope
def update_length_info(m: Model, d: Data):
    wp.launch(
        _update_length_info,
        dim=(d.nworld, m.nmuscle),
        inputs=[m.muscle_metadata, d.mstate, d.muscle_length, ],
        outputs=[d.muscle_length_info],
    )


@event_scope
def update_velocity_info(m: Model, d: Data):
    wp.launch(
        _update_velocity_info,
        dim=(d.nworld, m.nmuscle),
        inputs=[m.muscle_metadata, d.act, d.muscle_length_info,
                d.muscle_velocity, ],
        outputs=[d.muscle_velocity_info],
    )


@event_scope
def update_dynamics_info(m: Model, d: Data):
    wp.launch(
        _update_dynamics_info,
        dim=(d.nworld, m.nmuscle),
        inputs=[m.muscle_metadata, d.act,
                d.muscle_length_info, d.muscle_velocity_info, ],
        outputs=[d.muscle_dynamics_info],
    )


@event_scope
def update_info_fused(m: Model, d: Data):
    wp.launch(
        _update_info_fused,
        dim=(d.nworld, m.nmuscle),
        inputs=[m.muscle_metadata, d.act, d.mstate,
                d.muscle_length, d.muscle_velocity, ],
        outputs=[d.muscle_length_info, d.muscle_velocity_info,
                 d.muscle_dynamics_info],
    )


@event_scope
def muscle_equilibrate(m: Model, d: Data):
    """ Equilibrate muscles """
    if not m.nmuscle:
        return
    # Set the previous length/velocity to current
    wp.launch(
        _reset_prev_path,
        dim=(d.nworld, m.nmuscle),
        inputs=[d.world_reset, d.muscle_length, d.muscle_velocity],
        outputs=[d.muscle_length_prev, d.muscle_velocity_prev],
    )

    # Equilibrate (bisection)
    wp.launch(
        _equilibrate,
        dim=(d.nworld, m.nmuscle),
        inputs=[m.muscle_metadata, d.world_reset, d.muscle_length,
                d.muscle_velocity, d.act],
        outputs=[d.mstate],
    )


@event_scope
def realize_muscle_state(m: Model, d: Data):
    """ Muscle dynamics """
    if not m.nmuscle:
        return

    update_info_fused(m, d)

    # Set actuation and muscle state derivatives
    wp.launch(
        _set_state,
        dim=(d.nworld, m.nmuscle),
        inputs=[m.muscle_metadata, d.muscle_velocity_info,
                d.muscle_dynamics_info],
        outputs=[d.mstate_dot, d.muscle_actuation],
    )
