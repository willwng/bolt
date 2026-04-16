import warp as wp

from . import consts
from .types import vec6

wp.set_module_options({"enable_backward": False})


# --- BEZIER TOOLS ---
@wp.func
def scale_curviness(curviness: float) -> float:
    c = 0.1 + 0.8 * curviness
    return c


@wp.func
def calc_quintic_bezier_corner_control_points(
        x0: float,
        y0: float,
        dydx0: float,
        x1: float,
        y1: float,
        dydx1: float,
        curviness: float,
) -> tuple[vec6, vec6]:
    root_eps = wp.sqrt(consts.MSK_MINVAL)
    if wp.abs(dydx0 - dydx1) > root_eps:
        xC = (y1 - y0 - x1 * dydx1 + x0 * dydx0) / (dydx0 - dydx1)
    else:
        xC = (x1 + x0) / 2.0
    yC = (xC - x1) * dydx1 + y1

    xCx0 = (xC - x0)
    yCy0 = (yC - y0)
    xCx1 = (xC - x1)
    yCy1 = (yC - y1)

    x0_mid = x0 + curviness * xCx0
    y0_mid = y0 + curviness * yCy0

    x1_mid = x1 + curviness * xCx1
    y1_mid = y1 + curviness * yCy1

    xPts = vec6(x0, x0_mid, x0_mid, x1_mid, x1_mid, x1)
    yPts = vec6(y0, y0_mid, y0_mid, y1_mid, y1_mid, y1)
    return xPts, yPts


@wp.func
def evaluate_quintic_bezier(u: float, pts: vec6) -> float:
    """Evaluates a 1D Quintic Bezier curve at parameter u (0 <= u <= 1)."""
    inv_u = 1.0 - u
    u2, u3, u4, u5 = u ** 2.0, u ** 3.0, u ** 4.0, u ** 5.0
    inv_u2, inv_u3, inv_u4, inv_u5 = inv_u ** 2.0, inv_u ** 3.0, inv_u ** 4.0, inv_u ** 5.0
    return (inv_u5 * pts[0] +
            5.0 * inv_u4 * u * pts[1] +
            10.0 * inv_u3 * u2 * pts[2] +
            10.0 * inv_u2 * u3 * pts[3] +
            5.0 * inv_u * u4 * pts[4] +
            u5 * pts[5])


@wp.func
def evaluate_quintic_bezier_der_u(u: float, pts: vec6) -> float:
    """Evaluates the derivative (wrst u) of a 1D Quintic Bezier curve."""
    inv_u = 1.0 - u
    u2, u3, u4 = u ** 2.0, u ** 3.0, u ** 4.0
    inv_u2, inv_u3, inv_u4 = inv_u ** 2.0, inv_u ** 3.0, inv_u ** 4.0
    d0 = pts[1] - pts[0]
    d1 = pts[2] - pts[1]
    d2 = pts[3] - pts[2]
    d3 = pts[4] - pts[3]
    d4 = pts[5] - pts[4]
    return 5.0 * (inv_u4 * d0 +
                  4.0 * inv_u3 * u * d1 +
                  6.0 * inv_u2 * u2 * d2 +
                  4.0 * inv_u * u3 * d3 +
                  u4 * d4)


@wp.func
def evaluate_quintic_bezier_der_x(u: float, x_pts: vec6, y_pts: vec6) -> float:
    """ Evaluates the derivative (wrst x) """
    dx_du = evaluate_quintic_bezier_der_u(u, x_pts)
    dy_du = evaluate_quintic_bezier_der_u(u, y_pts)
    dy_dx = 0.0
    if wp.abs(dx_du) > consts.MSK_SIG_REAL:
        dy_dx = dy_du / dx_du
    return dy_dx


@wp.func
def compute_u(x: float, x_pts: vec6) -> float:
    u = float(0.5)
    max_iter = 20
    for _ in range(max_iter):
        current_x = evaluate_quintic_bezier(u, x_pts)
        error = current_x - x
        if wp.abs(error) < consts.MSK_SIG_REAL:
            break
        dxdu = evaluate_quintic_bezier_der_u(u, x_pts)
        if wp.abs(dxdu) < consts.MSK_SIG_REAL:
            break
        u -= error / dxdu

        # Clamp u to [0, 1] bounds
        u = wp.max(0.0, wp.min(1.0, u))
    return u


# --- FIBER ACTIVE FORCE LENGTH ---
@wp.func
def calc_active_force_length_multiplier(
        norm_fiber_length: float,
) -> float:
    # TODO: don't hard code this
    min_norm_active_fiber_length = 0.4441
    transition_norm_fiber_length = 0.73
    max_active_norm_fiber_length = 1.8123
    shallow_ascending_slope = 0.8616
    minimum_value = 0.1
    x0 = min_norm_active_fiber_length
    x1 = transition_norm_fiber_length
    x2 = 1.0
    x3 = max_active_norm_fiber_length
    ylow = minimum_value
    dydx = shallow_ascending_slope
    curviness = 1.0

    if norm_fiber_length <= x0:
        return ylow
    elif norm_fiber_length >= x3:
        return ylow

    c = scale_curviness(curviness)

    # Calculate all intermediate structural points
    xDelta = 0.05 * x2
    xs = x2 - xDelta

    y0 = 0.0
    dydx0 = 0.0

    y1 = 1.0 - dydx * (xs - x1)
    dydx01 = 1.25 * (y1 - y0) / (x1 - x0)

    x01 = x0 + 0.5 * (x1 - x0)
    y01 = y0 + 0.5 * (y1 - y0)

    x1s = x1 + 0.5 * (xs - x1)
    y1s = y1 + 0.5 * (1.0 - y1)
    dydx1s = dydx

    y2 = 1.0
    dydx2 = 0.0

    y3 = 0.0
    dydx3 = 0.0

    x23 = (x2 + xDelta) + 0.5 * (x3 - (x2 + xDelta))
    y23 = y2 + 0.5 * (y3 - y2)
    dydx23 = (y3 - y2) / ((x3 - xDelta) - (x2 + xDelta))

    # Determine the correct segment and calculate its control points
    if norm_fiber_length <= x01:
        x_pts, y_pts = calc_quintic_bezier_corner_control_points(
            x0, ylow, dydx0, x01, y01, dydx01, c
        )
    elif norm_fiber_length <= x1s:
        x_pts, y_pts = calc_quintic_bezier_corner_control_points(
            x01, y01, dydx01, x1s, y1s, dydx1s, c
        )
    elif norm_fiber_length <= x2:
        x_pts, y_pts = calc_quintic_bezier_corner_control_points(
            x1s, y1s, dydx1s, x2, y2, dydx2, c
        )
    elif norm_fiber_length <= x23:
        x_pts, y_pts = calc_quintic_bezier_corner_control_points(
            x2, y2, dydx2, x23, y23, dydx23, c
        )
    else:
        x_pts, y_pts = calc_quintic_bezier_corner_control_points(
            x23, y23, dydx23, x3, ylow, dydx3, c
        )
    u = compute_u(norm_fiber_length, x_pts)
    return evaluate_quintic_bezier(u, y_pts)


# --- FIBER ACTIVE FORCE VELOCITY ---
@wp.func
def calc_force_velocity_multiplier(
        norm_fiber_velocity: float,
) -> tuple[float, float]:
    # TODO don't hard code this
    concentric_slope_at_v_max = 0.0
    concentric_slope_near_v_max = 0.25
    isometric_slope = 5.0
    eccentric_slope_at_v_max = 0.0
    eccentric_slope_near_v_max = 0.15
    max_eccentric_velocity_force_multiplier = 1.4
    concentric_curviness = 0.6
    eccentric_curviness = 0.9
    f_max_e = max_eccentric_velocity_force_multiplier
    dy_dx_c = concentric_slope_at_v_max
    dy_dx_near_c = concentric_slope_near_v_max
    dy_dx_iso = isometric_slope
    dy_dx_e = eccentric_slope_at_v_max
    dy_dx_near_e = eccentric_slope_near_v_max
    conc_curviness = concentric_curviness
    ecc_curviness = eccentric_curviness

    xC = -1.0
    yC = 0.0

    xE = 1.0
    yE = f_max_e

    if norm_fiber_velocity <= xC:
        return yC + dy_dx_c * (norm_fiber_velocity - xC), dy_dx_c
    elif norm_fiber_velocity >= xE:
        return yE + dy_dx_e * (norm_fiber_velocity - xE), dy_dx_e

    cC = scale_curviness(conc_curviness)
    cE = scale_curviness(ecc_curviness)

    xNearC = -0.9
    yNearC = yC + 0.5 * dy_dx_near_c * (xNearC - xC) + 0.5 * dy_dx_c * (xNearC - xC)

    xIso = 0.0
    yIso = 1.0

    xNearE = 0.9
    yNearE = yE + 0.5 * dy_dx_near_e * (xNearE - xE) + 0.5 * dy_dx_e * (xNearE - xE)

    if norm_fiber_velocity <= xNearC:
        x_pts, y_pts = calc_quintic_bezier_corner_control_points(
            xC, yC, dy_dx_c, xNearC, yNearC, dy_dx_near_c, cC
        )
    elif norm_fiber_velocity <= xIso:
        x_pts, y_pts = calc_quintic_bezier_corner_control_points(
            xNearC, yNearC, dy_dx_near_c, xIso, yIso, dy_dx_iso, cC
        )
    elif norm_fiber_velocity <= xNearE:
        x_pts, y_pts = calc_quintic_bezier_corner_control_points(
            xIso, yIso, dy_dx_iso, xNearE, yNearE, dy_dx_near_e, cE
        )
    else:
        x_pts, y_pts = calc_quintic_bezier_corner_control_points(
            xNearE, yNearE, dy_dx_near_e, xE, yE, dy_dx_e, cE
        )
    u = compute_u(norm_fiber_velocity, x_pts)
    return evaluate_quintic_bezier(u, y_pts), evaluate_quintic_bezier_der_x(u, x_pts, y_pts)


@wp.func
def calc_force_velocity_multiplier_inverse(
        force_velocity_mult: float,
) -> float:
    # TODO don't hard code
    concentric_slope_at_v_max = 0.1
    concentric_slope_near_v_max = 0.25
    isometric_slope = 5.0
    eccentric_slope_at_v_max = 0.1
    eccentric_slope_near_v_max = 0.15
    max_eccentric_velocity_force_multiplier = 1.4
    concentric_curviness = 0.6
    eccentric_curviness = 0.9
    f_max_e = max_eccentric_velocity_force_multiplier
    dy_dx_c = concentric_slope_at_v_max
    dy_dx_near_c = concentric_slope_near_v_max
    dy_dx_iso = isometric_slope
    dy_dx_e = eccentric_slope_at_v_max
    dy_dx_near_e = eccentric_slope_near_v_max
    conc_curviness = concentric_curviness
    ecc_curviness = eccentric_curviness

    xC = -1.0
    yC = 0.0

    xE = 1.0
    yE = f_max_e

    if force_velocity_mult <= yC:
        return xC + (1.0 / dy_dx_c) * (force_velocity_mult - yC)
    elif force_velocity_mult >= yE:
        return xE + (1.0 / dy_dx_e) * (force_velocity_mult - yE)

    cC = scale_curviness(conc_curviness)
    cE = scale_curviness(ecc_curviness)

    xNearC = -0.9
    yNearC = yC + 0.5 * dy_dx_near_c * (xNearC - xC) + 0.5 * dy_dx_c * (xNearC - xC)

    xIso = 0.0
    yIso = 1.0

    xNearE = 0.9
    yNearE = yE + 0.5 * dy_dx_near_e * (xNearE - xE) + 0.5 * dy_dx_e * (xNearE - xE)

    if force_velocity_mult <= yNearC:
        forward_x_pts, forward_y_pts = calc_quintic_bezier_corner_control_points(
            xC, yC, dy_dx_c, xNearC, yNearC, dy_dx_near_c, cC
        )
    elif force_velocity_mult <= yIso:
        forward_x_pts, forward_y_pts = calc_quintic_bezier_corner_control_points(
            xNearC, yNearC, dy_dx_near_c, xIso, yIso, dy_dx_iso, cC
        )
    elif force_velocity_mult <= yNearE:
        forward_x_pts, forward_y_pts = calc_quintic_bezier_corner_control_points(
            xIso, yIso, dy_dx_iso, xNearE, yNearE, dy_dx_near_e, cE
        )
    else:
        forward_x_pts, forward_y_pts = calc_quintic_bezier_corner_control_points(
            xNearE, yNearE, dy_dx_near_e, xE, yE, dy_dx_e, cE
        )

    inv_x_pts = forward_y_pts
    inv_y_pts = forward_x_pts
    u = compute_u(force_velocity_mult, inv_x_pts)
    return evaluate_quintic_bezier(u, inv_y_pts)


# --- FIBER PASSIVE FORCE ---
@wp.func
def calc_passive_force_multiplier(
        norm_fiber_length: float,
        strain_at_zero_force: float,
        strain_at_one_norm_force: float,
        stiffness_at_low_force: float,
        stiffness_at_one_norm_force: float,
        curviness: float
) -> float:
    x_zero = 1.0 + strain_at_zero_force
    y_zero = 0.0

    x_iso = 1.0 + strain_at_one_norm_force
    y_iso = 1.0

    # Handle linear extrapolation outside the defined curve bounds
    if norm_fiber_length <= x_zero:
        return y_zero
    elif norm_fiber_length >= x_iso:
        return y_iso + stiffness_at_one_norm_force * (norm_fiber_length - x_iso)

    # Calculate intermediate geometry
    c = scale_curviness(curviness)
    delta_x = wp.min(0.1 * (1.0 / stiffness_at_one_norm_force), 0.1 * (x_iso - x_zero))

    x_low = x_zero + delta_x
    x_foot = x_zero + 0.5 * (x_low - x_zero)
    y_foot = 0.0
    y_low = y_foot + stiffness_at_low_force * (x_low - x_foot)

    # determine which Bezier segment 'x' falls into
    if norm_fiber_length <= x_low:
        x_pts, y_pts = calc_quintic_bezier_corner_control_points(
            x_zero, y_zero, 0.0, x_low, y_low, stiffness_at_low_force, c)
    else:
        x_pts, y_pts = calc_quintic_bezier_corner_control_points(
            x_low, y_low, stiffness_at_low_force, x_iso, y_iso, stiffness_at_one_norm_force, c)

    # Root-finding to find parameter u for the selected segment
    u = compute_u(norm_fiber_length, x_pts)

    return evaluate_quintic_bezier(u, y_pts)



# --- TENDON FORCE ----
@wp.func
def calc_tendon_force_multiplier(
        norm_tendon_length: float,
) -> float:
    # TODO don't hard code this
    strain_at_one_norm_force = 0.049
    stiffness_at_one_norm_force = 1.375 / strain_at_one_norm_force
    norm_force_at_toe_end = 2.0 / 3.0
    curviness = 0.5
    e_iso = strain_at_one_norm_force
    k_iso = stiffness_at_one_norm_force
    f_toe = norm_force_at_toe_end
    curviness = curviness

    x0 = 1.0
    y0 = 0.0
    dydx0 = 0.0

    xIso = 1.0 + e_iso
    yIso = 1.0
    dydxIso = k_iso

    # Location where the curved "toe" section ends and becomes fully linear
    yToe = f_toe
    xToe = (yToe - 1.0) / k_iso + xIso

    if norm_tendon_length <= x0:
        return y0  # Tendons do not support compression; force is 0 when slack
    elif norm_tendon_length >= xToe:
        return yToe + dydxIso * (norm_tendon_length - xToe)  # Linear elastic region
    c = scale_curviness(curviness)

    xFoot = 1.0 + (xToe - 1.0) / 10.0
    yFoot = 0.0

    # Intermediate toe parameters
    yToeMid = yToe * 0.5
    xToeMid = (yToeMid - yIso) / k_iso + xIso
    dydxToeMid = (yToeMid - yFoot) / (xToeMid - xFoot)

    # Control point separating the two Bezier segments
    xToeCtrl = xFoot + 0.5 * (xToeMid - xFoot)
    yToeCtrl = yFoot + dydxToeMid * (xToeCtrl - xFoot)

    if norm_tendon_length <= xToeCtrl:
        x_pts, y_pts = calc_quintic_bezier_corner_control_points(
            x0, y0, dydx0, xToeCtrl, yToeCtrl, dydxToeMid, c
        )
    else:
        x_pts, y_pts = calc_quintic_bezier_corner_control_points(
            xToeCtrl, yToeCtrl, dydxToeMid, xToe, yToe, dydxIso, c
        )
    u = compute_u(norm_tendon_length, x_pts)
    return evaluate_quintic_bezier(u, y_pts)


@wp.func
def calc_tendon_force_multiplier_inverse(
        norm_tendon_force: float,
) -> float:
    # TODO don't hard code this
    strain_at_one_norm_force = 0.049
    stiffness_at_one_norm_force = 1.375 / strain_at_one_norm_force
    norm_force_at_toe_end = 2.0 / 3.0
    curviness = 0.5
    e_iso = strain_at_one_norm_force
    k_iso = stiffness_at_one_norm_force
    f_toe = norm_force_at_toe_end
    curviness = curviness

    x0 = 1.0
    y0 = 0.0
    dydx0 = 0.0

    xIso = 1.0 + e_iso
    yIso = 1.0
    dydxIso = k_iso

    # Location where the curved "toe" section ends and becomes fully linear
    yToe = f_toe
    xToe = (yToe - 1.0) / k_iso + xIso

    if norm_tendon_force <= y0:
        return x0
    elif norm_tendon_force >= yToe:
        return xToe + (1.0 / dydxIso) * (norm_tendon_force - yToe)

    c = scale_curviness(curviness)

    # "Foot" point logic to shape the 2nd derivative of the toe region
    xFoot = 1.0 + (xToe - 1.0) / 10.0
    yFoot = 0.0

    # Intermediate toe parameters
    yToeMid = yToe * 0.5
    xToeMid = (yToeMid - yIso) / k_iso + xIso
    dydxToeMid = (yToeMid - yFoot) / (xToeMid - xFoot)

    # Control point separating the two Bezier segments
    xToeCtrl = xFoot + 0.5 * (xToeMid - xFoot)
    yToeCtrl = yFoot + dydxToeMid * (xToeCtrl - xFoot)

    if norm_tendon_force <= yToeCtrl:
        forward_x_pts, forward_y_pts = calc_quintic_bezier_corner_control_points(
            x0, y0, dydx0,
            xToeCtrl, yToeCtrl, dydxToeMid,
            c
        )
    else:
        forward_x_pts, forward_y_pts = calc_quintic_bezier_corner_control_points(
            xToeCtrl, yToeCtrl, dydxToeMid,
            xToe, yToe, dydxIso,
            c
        )

    inv_x_pts = forward_y_pts
    inv_y_pts = forward_x_pts
    u = compute_u(norm_tendon_force, inv_x_pts)
    return evaluate_quintic_bezier(u, inv_y_pts)


@wp.func
def calc_active_fiber_force(
        max_isometric_force: float,
        activation: float,
        norm_fiber_length: float,
        norm_fiber_velocity: float,
) -> float:
    fl = calc_active_force_length_multiplier(norm_fiber_length)
    fv, _ = calc_force_velocity_multiplier(norm_fiber_velocity)
    fiber_force = max_isometric_force * (activation * fl * fv)
    return fiber_force
