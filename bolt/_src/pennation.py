import warp as wp

from . import consts

wp.set_module_options({"enable_backward": False})


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
def clamp_norm_fiber_length(
        norm_fiber_length: float,
        min_norm_fiber_length: float,
        max_norm_fiber_length: float
) -> float:
    return wp.clamp(norm_fiber_length,
                    min_norm_fiber_length,
                    max_norm_fiber_length)


@wp.func
def clamp_fiber_length(
        fiber_length: float,
        optimal_fiber_length: float,
        min_norm_fiber_length: float,
        max_norm_fiber_length: float,
) -> float:
    return wp.clamp(fiber_length,
                    min_norm_fiber_length * optimal_fiber_length,
                    max_norm_fiber_length * optimal_fiber_length)


@wp.func
def calc_fiber_length(
        muscle_length: float,
        optimal_fiber_length: float,
        optimal_pennation_angle: float,
        tendon_slack_length: float,
        minimum_fiber_length: float,
) -> float:
    fiber_length_AT = muscle_length - tendon_slack_length
    minimum_fiber_length_along_tendon = minimum_fiber_length * wp.cos(consts.M_MAX_PENNATION_ANGLE)
    if fiber_length_AT >= minimum_fiber_length_along_tendon:
        parallelogram_height = get_fiber_width(optimal_fiber_length, optimal_pennation_angle)
        fiber_length = wp.sqrt(parallelogram_height * parallelogram_height + fiber_length_AT * fiber_length_AT)
    else:
        fiber_length = minimum_fiber_length_along_tendon
    return fiber_length


@wp.func
def calc_fiber_velocity(
        cos_pennation_angle: float,
        muscle_velocity: float,
        tendon_velocity: float,
) -> float:
    return (muscle_velocity - tendon_velocity) * cos_pennation_angle


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
            parallelogram_height = get_fiber_width(optimal_fiber_length, optimal_pennation_angle)
            max_sin_pennation_angle = wp.sin(consts.M_MAX_PENNATION_ANGLE)
            fiber_length = norm_fiber_length * optimal_fiber_length
            sin_phi = parallelogram_height / fiber_length
            phi = wp.where(sin_phi < max_sin_pennation_angle, wp.asin(sin_phi), consts.M_MAX_PENNATION_ANGLE)
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
