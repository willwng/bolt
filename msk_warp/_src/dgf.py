import warp as wp

from .types import MuscleConsts

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
        mc: MuscleConsts
) -> float:
    time_const_fact = 0.5 + 1.5 * activation
    tmp_act = 1.0 / (mc.activation_time_constant * time_const_fact)
    tmp_deact = time_const_fact / mc.deactivation_time_constant
    f = 0.5 * wp.tanh(
        mc.activation_dynamics_smoothing * (excitation - activation))
    time_const = tmp_act * (f + 0.5) + tmp_deact * (-f + 0.5)
    return time_const * (excitation - activation)


### --- Begin Pennation --- ###
@wp.func
def get_tendon_stiffness_parameter(
        mc: MuscleConsts
) -> float:
    return wp.log((1.0 + mc.c3) / mc.c1) / \
        (1.0 + mc.tendon_strain_at_one_norm_force - mc.c2)


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
        mc: MuscleConsts
) -> bool:
    return (norm_fiber_length <= mc.m_minNormFiberLength and
            norm_fiber_velocity < 0.0) or \
        (norm_fiber_length < mc.m_minNormFiberLength)


@wp.func
def clamp_fiber_length(
        norm_fiber_length: float,
        optimal_pennation_angle: float,
        mc: MuscleConsts
) -> float:
    if mc.m_maxPennationAngle > 1e-8:
        minimum_fiber_length = wp.sin(optimal_pennation_angle) / wp.sin(
            mc.m_maxPennationAngle)
    else:
        minimum_fiber_length = 0.01
    minimum_fiber_length = max(minimum_fiber_length,
                               mc.m_minNormFiberLength)

    return wp.clamp(norm_fiber_length,
                    minimum_fiber_length,
                    mc.m_maxNormFiberLength)


@wp.func
def calc_pennation_angle(
        optimal_pennation_angle: float,
        optimal_fiber_length: float,
        norm_fiber_length: float,
        mc: MuscleConsts
) -> float:
    phi = 0.0

    if optimal_pennation_angle > 1e-8:
        if norm_fiber_length > mc.m_minNormFiberLength:
            parallelogram_height = get_fiber_width(
                optimal_fiber_length,
                optimal_pennation_angle)
            max_sin_pennation_angle = wp.sin(mc.m_maxPennationAngle)
            fiber_length = norm_fiber_length * optimal_fiber_length
            sin_phi = parallelogram_height / fiber_length
            if sin_phi < max_sin_pennation_angle:
                phi = wp.asin(sin_phi)
            else:
                phi = mc.m_maxPennationAngle
        else:
            phi = mc.m_maxPennationAngle
    return phi


@wp.func
def calc_pennation_angular_velocity(
        optimal_pennation_angle: float,
        fiber_length: float,
        fiber_velocity: float,
        tan_pennation_angle: float
) -> float:
    d_phi = 0.0
    if optimal_pennation_angle > 1e-8:
        d_phi = -(fiber_velocity / fiber_length) * tan_pennation_angle
    return d_phi


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
        mc: MuscleConsts
) -> float:
    scale = mc.active_force_width_scale
    x = (norm_fiber_length - 1.0) / scale + 1.0
    return calc_gaussian_like_curve(x, mc.b11, mc.b21, mc.b31, mc.b41) + \
        calc_gaussian_like_curve(x, mc.b12, mc.b22, mc.b32, mc.b42) + \
        calc_gaussian_like_curve(x, mc.b13, mc.b23, mc.b33, mc.b43)


@wp.func
def calc_active_force_length_multiplier_derivative(
        norm_fiber_length: float,
        mc: MuscleConsts
) -> float:
    scale = mc.active_force_width_scale
    x = (norm_fiber_length - 1.0) / scale + 1.0
    return (1.0 / scale) * (
            calc_gaussian_like_curve_der(x, mc.b11, mc.b21, mc.b31, mc.b41) +
            calc_gaussian_like_curve_der(x, mc.b12, mc.b22, mc.b32, mc.b42) +
            calc_gaussian_like_curve_der(x, mc.b13, mc.b23, mc.b33, mc.b43)
    )


@wp.func
def calc_force_velocity_multiplier(
        norm_fiber_velocity: float,
        mc: MuscleConsts
) -> float:
    temp_v = mc.d2 * norm_fiber_velocity + mc.d3
    temp_log_arg = temp_v + wp.sqrt(temp_v ** 2.0 + 1.0)
    return mc.d1 * wp.log(temp_log_arg) + mc.d4


@wp.func
def calc_force_velocity_multiplier_derivative(
        norm_fiber_velocity: float,
        mc: MuscleConsts
) -> float:
    temp_v = mc.d2 * norm_fiber_velocity + mc.d3
    tmp = wp.sqrt(temp_v ** 2.0 + 1.0)
    return (mc.d1 * mc.d2) / (temp_v + tmp) * (1.0 + temp_v / tmp)


@wp.func
def calc_force_velocity_multiplier_derivative(
        norm_fiber_velocity: float,
        mc: MuscleConsts
) -> float:
    temp_v = mc.d2 * norm_fiber_velocity + mc.d3
    tmp = wp.sqrt(temp_v ** 2.0 + 1.0)
    return (mc.d1 * mc.d2) / (temp_v + tmp) * (1.0 + temp_v / tmp)


@wp.func
def calc_force_velocity_inverse_curve(
        force_velocity_mult: float,
        mc: MuscleConsts
) -> float:
    return (wp.sinh(1.0 / mc.d1 * (
            force_velocity_mult - mc.d4)) - mc.d3) / mc.d2


@wp.func
def calc_passive_force_multiplier(
        norm_fiber_length: float,
        mc: MuscleConsts
) -> float:
    kPE = mc.kPE
    e0 = mc.passive_fiber_strain_at_one_norm_force
    offset = wp.exp(kPE * (mc.m_minNormFiberLength - 1.0) / e0)
    denom = wp.exp(kPE) - offset
    return (wp.exp(kPE * (norm_fiber_length - 1.0) / e0) - offset) / denom


@wp.func
def calc_active_fiber_force(
        max_isometric_force: float,
        activation: float,
        norm_fiber_length: float,
        norm_fiber_velocity: float,
        mc: MuscleConsts
) -> float:
    fl = calc_active_force_length_multiplier(norm_fiber_length, mc)
    fv = calc_force_velocity_multiplier(norm_fiber_velocity, mc)
    fiber_force = max_isometric_force * (activation * fl * fv)
    return fiber_force


@wp.func
def calc_passive_fiber_force(
        max_isometric_force: float,
        norm_fiber_length: float,
        norm_fiber_velocity: float,
        fiber_damping: float,
        mc: MuscleConsts
) -> float:
    fp = calc_passive_force_multiplier(norm_fiber_length, mc)
    fd = fiber_damping * norm_fiber_velocity
    passive_force = max_isometric_force * (fp + fd)
    return passive_force


### --- Begin Tendon --- ###
@wp.func
def calc_tendon_force_multiplier(
        norm_tendon_length: float,
        clamped: bool,
        mc: MuscleConsts
) -> float:
    tmp = mc.c1 * wp.exp(get_tendon_stiffness_parameter(mc) *
                         (norm_tendon_length - mc.c2)) - mc.c3
    if clamped:
        return wp.clamp(tmp, mc.m_minNormTendonForce, mc.m_maxNormTendonForce)
    else:
        return tmp


@wp.func
def calc_tendon_force_length_inverse_curve(
        norm_tendon_force: float,
        mc: MuscleConsts
) -> float:
    return wp.log((1.0 / mc.c1) * (norm_tendon_force + mc.c3)) / \
        get_tendon_stiffness_parameter(mc) + mc.c2


@wp.func
def calc_tendon_force_length_inverse_curve_derivative(
        deriv_norm_tendon_force: float,
        norm_tendon_length: float,
        mc: MuscleConsts
) -> float:
    return deriv_norm_tendon_force / (
            mc.c1 * get_tendon_stiffness_parameter(mc)
            * wp.exp(get_tendon_stiffness_parameter(mc) * (
            norm_tendon_length - mc.c2)))


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
        mc: MuscleConsts
) -> tuple[float, float]:
    max_iter = 20
    tol = 1e-8 * f_iso
    k_sig_real = 1e-6
    prev_err = 1e10
    if tol < k_sig_real * 100.0:
        tol = k_sig_real * 100.0

    # use undamped estimate as initial guess
    fv = calc_undamped_fiber_force_velocity_multiplier(
        max(a, 0.01),
        max(fal, 0.01),
        fpe,
        fse,
        max(cos_phi, 0.01)
    )
    dlceN_dt = calc_force_velocity_inverse_curve(fv, mc)

    # approximation is poor beyond maximum velocities
    dlceN_dt = wp.clamp(dlceN_dt, -1.0, 1.0)

    for i in range(max_iter):
        fv = calc_force_velocity_multiplier(dlceN_dt, mc)
        fvDer = calc_force_velocity_multiplier_derivative(dlceN_dt, mc)

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

    return dlceN_dt, fv
