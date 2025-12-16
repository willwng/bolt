import warp as wp

from . import consts

wp.set_module_options({"enable_backward": False})


@wp.func
def calc_gaussian_like_curve(
        x: float,
        b1: float,
        b2: float,
        b3: float,
        b4: float
) -> float:
    return b1 * wp.exp(-0.5 * (x - b2) ** 2.0 / (b3 + b4 * x) ** 2.0)


@wp.func
def calc_gaussian_like_curve_der(
        x: float,
        b1: float,
        b2: float,
        b3: float,
        b4: float
) -> float:
    return (b1 * wp.exp(-(b2 - x) ** 2.0 / (2 * (b3 + b4 * x) ** 2.0)) *
            (b2 - x) * (b3 + b2 * b4)) / (b3 + b4 * x) ** 3


### --- Begin activation --- ###
@wp.func
def calc_activation_derivative(
        activation: float,
        excitation: float,
) -> float:
    time_const_fact = 0.5 + 1.5 * activation
    tmp_act = 1.0 / (consts.DGF_ACTIVATION_TIME_CONSTANT * time_const_fact)
    tmp_deact = time_const_fact / consts.DGF_DEACTIVATION_TIME_CONSTANT
    f = 0.5 * wp.tanh(
        consts.DGF_ACTIVATION_DYNAMICS_SMOOTHING * (excitation - activation))
    time_const = tmp_act * (f + 0.5) + tmp_deact * (-f + 0.5)
    return time_const * (excitation - activation)


### --- Begin Pennation --- ###
@wp.func
def get_tendon_stiffness_parameter() -> float:
    return wp.log((1.0 + consts.DGF_C3) / consts.DGF_C1) / \
        (1.0 + consts.TENDON_STRAIN_AT_ONE_NORM_FORCE - consts.DGF_C2)


@wp.func
def get_fiber_width(
        optimal_fiber_length: float,
        optimal_pennation_angle: float
) -> float:
    return optimal_fiber_length * wp.sin(optimal_pennation_angle)


@wp.func
def get_max_contraction_velocity_in_meters_per_second(
        v_max: float,
        optimal_fiber_length: float
) -> float:
    return v_max * optimal_fiber_length


@wp.func
def is_fiber_state_clamped(
        norm_fiber_length: float,
        norm_fiber_velocity: float,
        min_norm_fiber_length: float,
) -> bool:
    return (norm_fiber_length <= min_norm_fiber_length and
            norm_fiber_velocity < 0.0) or \
        (norm_fiber_length < min_norm_fiber_length)


@wp.func
def clamp_fiber_length(
        norm_fiber_length: float,
        min_norm_fiber_length: float,
        max_norm_fiber_length: float
) -> float:
    return wp.clamp(norm_fiber_length,
                    min_norm_fiber_length,
                    max_norm_fiber_length)


@wp.func
def calc_pennation_angle(
        optimal_pennation_angle: float,
        optimal_fiber_length: float,
        norm_fiber_length: float,
        min_norm_fiber_length: float,
) -> float:
    phi = 0.0

    if optimal_pennation_angle > 1e-8:
        if norm_fiber_length > min_norm_fiber_length:
            parallelogram_height = get_fiber_width(
                optimal_fiber_length,
                optimal_pennation_angle)
            max_sin_pennation_angle = wp.sin(consts.M_MAX_PENNATION_ANGLE)
            fiber_length = norm_fiber_length * optimal_fiber_length
            sin_phi = parallelogram_height / fiber_length
            phi = wp.where(sin_phi < max_sin_pennation_angle,
                           wp.asin(sin_phi), consts.M_MAX_PENNATION_ANGLE)
        else:
            phi = consts.M_MAX_PENNATION_ANGLE
    return phi


@wp.func
def calc_pennation_angular_velocity(
        optimal_pennation_angle: float,
        fiber_length: float,
        fiber_velocity: float,
        tan_pennation_angle: float
) -> float:
    d_phi = -(fiber_velocity / fiber_length) * tan_pennation_angle
    return wp.where(optimal_pennation_angle > 1e-8, d_phi, 0.0)


@wp.func
def calc_fiber_velocity_along_tendon(
        fiber_length: float,
        fiber_velocity: float,
        sin_pennation_angle: float,
        cos_pennation_angle: float,
        pennation_angular_velocity: float
) -> float:
    return (fiber_velocity * cos_pennation_angle
            - fiber_length * sin_pennation_angle * pennation_angular_velocity)


@wp.func
def calc_tendon_velocity(
        cos_pennation_angle: float,
        sin_pennation_angle: float,
        pennation_angular_velocity: float,
        fiber_length: float,
        fiber_velocity: float,
        muscle_velocity: float
) -> float:
    return muscle_velocity - fiber_velocity * cos_pennation_angle + \
        fiber_length * sin_pennation_angle * pennation_angular_velocity


### --- Begin Fiber --- ###
@wp.func
def calc_active_force_length_multiplier(
        norm_fiber_length: float,
) -> float:
    scale = consts.ACTIVE_FORCE_WIDTH_SCALE
    x = (norm_fiber_length - 1.0) / scale + 1.0
    return calc_gaussian_like_curve(x, consts.DGF_B11, consts.DGF_B21,
                                    consts.DGF_B31, consts.DGF_B41) + \
        calc_gaussian_like_curve(x, consts.DGF_B12, consts.DGF_B22,
                                 consts.DGF_B32, consts.DGF_B42) + \
        calc_gaussian_like_curve(x, consts.DGF_B13, consts.DGF_B23,
                                 consts.DGF_B33, consts.DGF_B43)


@wp.func
def calc_active_force_length_multiplier_derivative(
        norm_fiber_length: float,
) -> float:
    scale = consts.ACTIVE_FORCE_WIDTH_SCALE
    x = (norm_fiber_length - 1.0) / scale + 1.0
    return (1.0 / scale) * (
            calc_gaussian_like_curve_der(x, consts.DGF_B11, consts.DGF_B21,
                                         consts.DGF_B31, consts.DGF_B41) +
            calc_gaussian_like_curve_der(x, consts.DGF_B12, consts.DGF_B22,
                                         consts.DGF_B32, consts.DGF_B42) +
            calc_gaussian_like_curve_der(x, consts.DGF_B13, consts.DGF_B23,
                                         consts.DGF_B33, consts.DGF_B43)
    )


@wp.func
def calc_force_velocity_multiplier(
        norm_fiber_velocity: float,
) -> float:
    temp_v = consts.DGF_D2 * norm_fiber_velocity + consts.DGF_D3

    # Original code (numerically imprecise for negative temp_v)
    #   temp_log_arg = temp_v + wp.sqrt(temp_v ** 2.0 + 1.0)
    sqrt_term = wp.sqrt(temp_v * temp_v + 1.0)
    temp_log_arg = wp.where(
        temp_v >= 0.0,
        temp_v + sqrt_term,
        1.0 / (sqrt_term - temp_v)
    )
    return consts.DGF_D1 * wp.log(temp_log_arg) + consts.DGF_D4


@wp.func
def calc_force_velocity_multiplier_derivative(
        norm_fiber_velocity: float,
) -> float:
    temp_v = consts.DGF_D2 * norm_fiber_velocity + consts.DGF_D3

    # Original code
    # tmp = wp.sqrt(temp_v ** 2.0 + 1.0)
    # return (consts.DGF_D1 * consts.DGF_D2) / (temp_v + tmp) * (1.0 + temp_v / tmp)
    return (consts.DGF_D1 * consts.DGF_D2) / wp.sqrt(temp_v * temp_v + 1.0)


@wp.func
def calc_force_velocity_inverse_curve(
        force_velocity_mult: float,
) -> float:
    return (wp.sinh(1.0 / consts.DGF_D1 * (
            force_velocity_mult - consts.DGF_D4)) - consts.DGF_D3) / consts.DGF_D2


@wp.func
def calc_passive_force_multiplier(
        norm_fiber_length: float,
        min_norm_fiber_length: float,
) -> float:
    kPE = consts.DGF_KPE
    e0 = consts.PASSIVE_FIBER_STRAIN_AT_ONE_NORM_FORCE
    offset = wp.exp(kPE * (min_norm_fiber_length - 1.0) / e0)
    denom = wp.exp(kPE) - offset
    return (wp.exp(kPE * (norm_fiber_length - 1.0) / e0) - offset) / denom


@wp.func
def calc_active_fiber_force(
        max_isometric_force: float,
        activation: float,
        norm_fiber_length: float,
        norm_fiber_velocity: float,
) -> float:
    fl = calc_active_force_length_multiplier(norm_fiber_length)
    fv = calc_force_velocity_multiplier(norm_fiber_velocity)
    fiber_force = max_isometric_force * (activation * fl * fv)
    return fiber_force


@wp.func
def calc_passive_fiber_force(
        max_isometric_force: float,
        norm_fiber_length: float,
        norm_fiber_velocity: float,
        fiber_damping: float,
        min_norm_fiber_length: float,
) -> float:
    fp = calc_passive_force_multiplier(norm_fiber_length, min_norm_fiber_length)
    fd = fiber_damping * norm_fiber_velocity
    passive_force = max_isometric_force * (fp + fd)
    return passive_force


### --- Begin Tendon --- ###
@wp.func
def calc_tendon_force_multiplier(
        norm_tendon_length: float,
        clamped: bool,
) -> float:
    tmp = (consts.DGF_C1 * wp.exp(get_tendon_stiffness_parameter() *
                                  (norm_tendon_length - consts.DGF_C2)) - consts.DGF_C3)
    return wp.where(clamped, wp.clamp(tmp, consts.M_MIN_NORM_TENDON_FORCE,
                                      consts.M_MAX_NORM_TENDON_FORCE), tmp)


@wp.func
def calc_tendon_force_length_inverse_curve(
        norm_tendon_force: float,
) -> float:
    return wp.log((1.0 / consts.DGF_C1) * (norm_tendon_force + consts.DGF_C3)) / \
        get_tendon_stiffness_parameter() + consts.DGF_C2


@wp.func
def calc_tendon_force_length_inverse_curve_derivative(
        deriv_norm_tendon_force: float,
        norm_tendon_length: float,
) -> float:
    return deriv_norm_tendon_force / (
            consts.DGF_C1 * get_tendon_stiffness_parameter()
            * wp.exp(get_tendon_stiffness_parameter() * (
            norm_tendon_length - consts.DGF_C2)))


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
) -> tuple[float, float]:
    k_sig_real = 1e-6
    tol = wp.max(1e-8 * f_iso, k_sig_real * 100.0)
    prev_err = float(1e10)

    # use undamped estimate as initial guess
    fv = calc_undamped_fiber_force_velocity_multiplier(
        wp.max(a, 0.01),
        wp.max(fal, 0.01),
        fpe,
        fse,
        wp.max(cos_phi, 0.01)
    )
    dlceN_dt = calc_force_velocity_inverse_curve(fv)
    # approximation is poor beyond maximum velocities
    dlceN_dt = wp.clamp(dlceN_dt, -1.0, 1.0)

    max_iter = wp.static(20)
    for i in range(max_iter):
        fv = calc_force_velocity_multiplier(dlceN_dt)
        fvDer = calc_force_velocity_multiplier_derivative(dlceN_dt)

        fiber_force = f_iso * (a * fal * fv + fpe + beta * dlceN_dt)
        err = fiber_force * cos_phi - fse * f_iso
        df_d_dlceNdt = f_iso * (a * fal * fvDer + beta)
        derr_d_dlceNdt = df_d_dlceNdt * cos_phi

        if abs(err) < tol:
            break
        if abs(prev_err) - abs(err) < tol:
            break
        if derr_d_dlceNdt < tol:
            break

        delta = -err / derr_d_dlceNdt
        dlceN_dt = dlceN_dt + delta
        prev_err = err

    return dlceN_dt, fv
