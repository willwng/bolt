import warp as wp

from . import dgf
from . import millard
from . import consts
from .types import ContractionType
from .types import MuscleMetadata

wp.set_module_options({"enable_backward": False})



# --- FIBER FORCE MULTIPLIERS ---
@wp.func
def calc_active_fiber_force_length(
        norm_fiber_length: float,
        contraction_type: int,
) -> float:
    if contraction_type == ContractionType.DGF:
        return dgf.calc_active_force_length_multiplier(
            norm_fiber_length=norm_fiber_length,
        )
    elif contraction_type == ContractionType.MILLARD:
        return millard.calc_active_force_length_multiplier(
            norm_fiber_length=norm_fiber_length,
        )
    assert False


@wp.func
def calc_active_fiber_force_velocity(
        norm_fiber_velocity: float,
        contraction_type: int,
) -> tuple[float, float]:
    if contraction_type == ContractionType.DGF:
        fv = dgf.calc_force_velocity_multiplier(
            norm_fiber_velocity=norm_fiber_velocity
        )
        fvDer = dgf.calc_force_velocity_multiplier_derivative(
            norm_fiber_velocity=norm_fiber_velocity
        )
        return fv, fvDer
    elif contraction_type == ContractionType.MILLARD:
        fv, fvDer = millard.calc_force_velocity_multiplier(
            norm_fiber_velocity=norm_fiber_velocity
        )
        return fv, fvDer
    assert False


@wp.func
def calc_active_fiber_force_velocity_inverse(
        force_velocity_mult: float,
        contraction_type: int,
) -> float:
    if contraction_type == ContractionType.DGF:
        return dgf.calc_force_velocity_inverse_curve(
            force_velocity_mult=force_velocity_mult
        )
    elif contraction_type == ContractionType.MILLARD:
        return millard.calc_force_velocity_multiplier_inverse(
            force_velocity_mult=force_velocity_mult,
        )
    assert False


@wp.func
def calc_passive_fiber_force_length(
        norm_fiber_length: float,
        mm: MuscleMetadata,
        contraction_type: int,
) -> float:
    if contraction_type == ContractionType.DGF:
        return dgf.calc_passive_force_multiplier(
            norm_fiber_length=norm_fiber_length,
            min_norm_fiber_length=mm.min_norm_fiber_length,
            passive_fiber_strain_at_one_norm_force=mm.strain_at_one_norm_force
        )
    elif contraction_type == ContractionType.MILLARD:
        return millard.calc_passive_force_multiplier(
            norm_fiber_length=norm_fiber_length,
            strain_at_zero_force=mm.strain_at_zero_force,
            strain_at_one_norm_force=mm.strain_at_one_norm_force,
            stiffness_at_low_force=mm.stiffness_at_low_force,
            stiffness_at_one_norm_force=mm.stiffness_at_one_norm_force,
            curviness=mm.curviness,
        )
    assert False


# --- TENDON ---
@wp.func
def calc_tendon_force_length(
        norm_tendon_length: float,
        contraction_type: int,
) -> float:
    if contraction_type == ContractionType.DGF:
        return dgf.calc_tendon_force_multiplier(
            norm_tendon_length=norm_tendon_length,
            clamped=True
        )
    elif contraction_type == ContractionType.MILLARD:
        return millard.calc_tendon_force_multiplier(
            norm_tendon_length=norm_tendon_length,
        )
    assert False


@wp.func
def calc_tendon_force_inverse(
        norm_tendon_force: float,
        contraction_type: int,
):
    if contraction_type == ContractionType.DGF:
        return dgf.calc_tendon_force_length_inverse(
            norm_tendon_force=norm_tendon_force
        )
    elif contraction_type == ContractionType.MILLARD:
        return millard.calc_tendon_force_multiplier_inverse(
            norm_tendon_force=norm_tendon_force
        )

    assert False


@wp.func
def calc_tendon_force_inverse_derivative(
        norm_tendon_length: float,
        contraction_type: int,
):
    if contraction_type == ContractionType.DGF:
        return dgf.calc_tendon_force_length_inverse_curve_derivative(0.0, norm_tendon_length)
    elif contraction_type == ContractionType.MILLARD:
        return 0.0

    assert False


# --- TOTAL FIBER FORCES ---
@wp.func
def calc_active_fiber_force(
        max_isometric_force: float,
        activation: float,
        norm_fiber_length: float,
        norm_fiber_velocity: float,
        contraction_type: int,
):
    fl = calc_active_fiber_force_length(norm_fiber_length, contraction_type)
    fv, _ = calc_active_fiber_force_velocity(norm_fiber_velocity, contraction_type)
    fiber_force = max_isometric_force * (activation * fl * fv)
    return fiber_force


@wp.func
def calc_passive_fiber_force(
        norm_fiber_length: float,
        norm_fiber_velocity: float,
        mm: MuscleMetadata,
        contraction_type: int,
) -> float:
    fp = calc_passive_fiber_force_length(
        norm_fiber_length=norm_fiber_length,
        mm=mm,
        contraction_type=contraction_type
    )
    fd = mm.fiber_damping * norm_fiber_velocity
    passive_force = mm.max_isometric_force * (fp + fd)
    return passive_force


# --- FIBER VELOCITY ESTIMATION ---
@wp.func
def calc_undamped_fiber_force_velocity_multiplier(
        a: float,
        fal: float,
        fp: float,
        fse: float,
        cos_phi: float
) -> float:
    return (fse / cos_phi - fp) / (a * fal)


@wp.func
def calc_damped_norm_fiber_velocity(
        f_iso: float,
        a: float,
        fal: float,
        fpe: float,
        fse: float,
        beta: float,
        cos_phi: float,
        contraction_type: int,
        state: wp.array(dtype=wp.uint32)
) -> tuple[float, float]:
    max_iter = wp.static(20)
    tol = wp.max(1e-10 * f_iso, consts.MSK_SIG_REAL * 100.0)
    err = float(1e10)
    i = int(0)

    # use undamped estimate as initial guess
    fv = calc_undamped_fiber_force_velocity_multiplier(
        wp.max(a, 0.01),
        wp.max(fal, 0.01),
        fpe,
        fse,
        wp.max(cos_phi, 0.01)
    )
    dlceN_dt = calc_active_fiber_force_velocity_inverse(
        force_velocity_mult=fv,
        contraction_type=contraction_type,
    )

    # approximation is poor beyond maximum velocities
    dlceN_dt = wp.clamp(dlceN_dt, -1.0, 1.0)

    while wp.abs(err) > tol and i < max_iter:
        fv, fvDer = calc_active_fiber_force_velocity(
            norm_fiber_velocity=dlceN_dt,
            contraction_type=contraction_type,
        )
        fiber_force = f_iso * (a * fal * fv + fpe + beta * dlceN_dt)

        err = fiber_force * cos_phi - fse * f_iso
        df_d_dlceNdt = f_iso * (a * fal * fvDer + beta)
        derr_d_dlceNdt = df_d_dlceNdt * cos_phi

        if wp.abs(err) > tol and wp.abs(derr_d_dlceNdt) > consts.MSK_SIG_REAL:
            delta = -err / derr_d_dlceNdt
            dlceN_dt = dlceN_dt + delta
        elif wp.abs(derr_d_dlceNdt) < consts.MSK_SIG_REAL:
            # Perturb the solution if we lost rank: shouldn't happen
            perturbation = 2.0 * wp.randf(state[0]) - 1.0
            wp.atomic_add(state, 0, wp.uint32(1))
            dlceN_dt = dlceN_dt + perturbation * 0.05
        i += 1

    return dlceN_dt, fv
