import warp as wp

from . import consts

wp.set_module_options({"enable_backward": False})

# DeGroote-Fregly muscle constants
DGF_B11 = 0.8150671134243542
DGF_B21 = 1.055033428970575
DGF_B31 = 0.162384573599574
DGF_B41 = 0.063303448465465
DGF_B12 = 0.433004984392647
DGF_B22 = 0.716775413397760
DGF_B32 = -0.029947116970696
DGF_B42 = 0.200356847296188
DGF_B13 = 0.1
DGF_B23 = 1.0
DGF_B33 = 0.353553390593274
DGF_B43 = 0.0
# Tendon force-length curve
DGF_C1 = 0.200
DGF_C2 = 1.0
DGF_C3 = 0.200
# Muscle force-velocity curve
DGF_D1 = -0.3211346127989808
DGF_D2 = -8.149
DGF_D3 = -0.374
DGF_D4 = 0.8825327733249912
# Muscle passive force-length curve
DGF_KPE = 4.0

TENDON_STRAIN_AT_ONE_NORM_FORCE = 0.049

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
    return calc_gaussian_like_curve(x, DGF_B11, DGF_B21,
                                    DGF_B31, DGF_B41) + \
        calc_gaussian_like_curve(x, DGF_B12, DGF_B22,
                                 DGF_B32, DGF_B42) + \
        calc_gaussian_like_curve(x, DGF_B13, DGF_B23,
                                 DGF_B33, DGF_B43)


@wp.func
def calc_active_force_length_multiplier_derivative(
        norm_fiber_length: float,
        active_force_width_scale: float,
) -> float:
    scale = active_force_width_scale
    x = (norm_fiber_length - 1.0) / scale + 1.0
    return (1.0 / scale) * (
            calc_gaussian_like_curve_der(x, DGF_B11, DGF_B21,
                                         DGF_B31, DGF_B41) +
            calc_gaussian_like_curve_der(x, DGF_B12, DGF_B22,
                                         DGF_B32, DGF_B42) +
            calc_gaussian_like_curve_der(x, DGF_B13, DGF_B23,
                                         DGF_B33, DGF_B43)
    )


@wp.func
def calc_force_velocity_multiplier(
        norm_fiber_velocity: float,
) -> float:
    temp_v = DGF_D2 * norm_fiber_velocity + DGF_D3

    # Original code (numerically imprecise for negative temp_v)
    #   temp_log_arg = temp_v + wp.sqrt(temp_v ** 2.0 + 1.0)
    sqrt_term = wp.sqrt(temp_v * temp_v + 1.0)
    temp_log_arg = wp.where(
        temp_v >= 0.0,
        temp_v + sqrt_term,
        1.0 / (sqrt_term - temp_v)
    )
    return DGF_D1 * wp.log(temp_log_arg) + DGF_D4


@wp.func
def calc_force_velocity_multiplier_derivative(
        norm_fiber_velocity: float,
) -> float:
    temp_v = DGF_D2 * norm_fiber_velocity + DGF_D3

    # Original code
    # tmp = wp.sqrt(temp_v ** 2.0 + 1.0)
    # return (DGF_D1 * DGF_D2) / (temp_v + tmp) * (1.0 + temp_v / tmp)
    return (DGF_D1 * DGF_D2) / wp.sqrt(temp_v * temp_v + 1.0)


@wp.func
def calc_force_velocity_inverse_curve(
        force_velocity_mult: float,
) -> float:
    return (wp.sinh(1.0 / DGF_D1 * (
            force_velocity_mult - DGF_D4)) - DGF_D3) / DGF_D2


@wp.func
def calc_passive_force_multiplier(
        norm_fiber_length: float,
        min_norm_fiber_length: float,
        passive_fiber_strain_at_one_norm_force: float
) -> float:
    kPE = DGF_KPE
    e0 = passive_fiber_strain_at_one_norm_force
    offset = wp.exp(kPE * (min_norm_fiber_length - 1.0) / e0)
    denom = wp.exp(kPE) - offset
    return (wp.exp(kPE * (norm_fiber_length - 1.0) / e0) - offset) / denom


### --- Begin Tendon --- ###
@wp.func
def get_tendon_stiffness_parameter() -> float:
    return wp.log((1.0 + DGF_C3) / DGF_C1) / \
        (1.0 + TENDON_STRAIN_AT_ONE_NORM_FORCE - DGF_C2)


@wp.func
def calc_tendon_force_multiplier(
        norm_tendon_length: float,
        clamped: bool,
) -> float:
    tmp = (DGF_C1 * wp.exp(get_tendon_stiffness_parameter() *
                           (norm_tendon_length - DGF_C2)) - DGF_C3)
    return wp.where(clamped, wp.clamp(tmp, consts.M_MIN_NORM_TENDON_FORCE,
                                      consts.M_MAX_NORM_TENDON_FORCE), tmp)


@wp.func
def calc_tendon_force_length_inverse(
        norm_tendon_force: float,
) -> float:
    return wp.log((1.0 / DGF_C1) * (norm_tendon_force + DGF_C3)) / \
        get_tendon_stiffness_parameter() + DGF_C2


@wp.func
def calc_tendon_force_length_inverse_curve_derivative(
        deriv_norm_tendon_force: float,
        norm_tendon_length: float,
) -> float:
    return deriv_norm_tendon_force / (
            DGF_C1 * get_tendon_stiffness_parameter()
            * wp.exp(get_tendon_stiffness_parameter() * (
            norm_tendon_length - DGF_C2)))
