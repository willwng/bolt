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


### --- Begin Fiber --- ###
@wp.func
def calc_active_force_length_multiplier(
        norm_fiber_length: float,
        active_force_width_scale: float,
) -> float:
    scale = active_force_width_scale
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
        active_force_width_scale: float,
) -> float:
    scale = active_force_width_scale
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
        passive_fiber_strain_at_one_norm_force: float
) -> float:
    kPE = consts.DGF_KPE
    e0 = passive_fiber_strain_at_one_norm_force
    offset = wp.exp(kPE * (min_norm_fiber_length - 1.0) / e0)
    denom = wp.exp(kPE) - offset
    return (wp.exp(kPE * (norm_fiber_length - 1.0) / e0) - offset) / denom


### --- Begin Tendon --- ###
@wp.func
def get_tendon_stiffness_parameter() -> float:
    return wp.log((1.0 + consts.DGF_C3) / consts.DGF_C1) / \
        (1.0 + consts.TENDON_STRAIN_AT_ONE_NORM_FORCE - consts.DGF_C2)


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
def calc_tendon_force_length_inverse(
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
