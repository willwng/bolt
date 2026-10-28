"""
Obstacle surfaces and geodesics over them for muscle wrapping, ported from Simbody's ContactGeometry,
GeodesicOnImplicitSurface and GeodesicIntegrator.

Everything is in the obstacle's surface frame S. Surfaces are implicit functions that are positive inside and zero on
the surface, so the outward normal is the negative gradient direction. Frenet frames are (tangent, normal, binormal)
with binormal = tangent x normal, as in Scholz et al. (2015).
"""
import warp as wp

from bolt.consts import BOLT_MINVAL
from bolt.consts import WRAP_MAX_GEODESIC_STEPS
from bolt.consts import WRAP_NEAREST_POINT_MAX_ITERATIONS
from bolt.consts import WRAP_NEAREST_POINT_TOL
from bolt.consts import WRAP_PROJECTION_TOL
from bolt.types import WrapGeodesic
from bolt.types import WrapStatus
from bolt.types import WrapType

wp.set_module_options({"enable_backward": False})


class vec10(wp.types.vector(length=10, dtype=float)):
    """ Geodesic integrator state: point (0:3), rotational/translational Jacobi scalars (3, 4), and their derivatives
    along the arc length: tangent (5:8), Jacobi scalar derivatives (8, 9) """
    pass


# --- Implicit surface functions ---
@wp.func
def surface_value(wrap_type: int, size: wp.vec3, x: wp.vec3) -> float:
    if wrap_type == WrapType.SPHERE:
        return size[0] * size[0] - wp.dot(x, x)
    elif wrap_type == WrapType.CYLINDER:
        return size[0] * size[0] - x[0] * x[0] - x[1] * x[1]
    elif wrap_type == WrapType.ELLIPSOID:
        y = wp.cw_div(x, size)
        return 1.0 - wp.dot(y, y)
    rho = wp.sqrt(x[0] * x[0] + x[1] * x[1])
    return 1.0 - ((size[0] - rho) * (size[0] - rho) + x[2] * x[2]) / (size[1] * size[1])


@wp.func
def surface_gradient(wrap_type: int, size: wp.vec3, x: wp.vec3) -> wp.vec3:
    if wrap_type == WrapType.SPHERE:
        return -2.0 * x
    elif wrap_type == WrapType.CYLINDER:
        return wp.vec3(-2.0 * x[0], -2.0 * x[1], 0.0)
    elif wrap_type == WrapType.ELLIPSOID:
        return -2.0 * wp.cw_div(x, wp.cw_mul(size, size))
    rho = wp.sqrt(x[0] * x[0] + x[1] * x[1])
    r2 = size[1] * size[1]
    s = 2.0 * (size[0] - rho) / (r2 * rho)
    return wp.vec3(s * x[0], s * x[1], -2.0 * x[2] / r2)


@wp.func
def surface_hessian(wrap_type: int, size: wp.vec3, x: wp.vec3) -> wp.mat33:
    if wrap_type == WrapType.SPHERE:
        return wp.diag(wp.vec3(-2.0))
    elif wrap_type == WrapType.CYLINDER:
        return wp.diag(wp.vec3(-2.0, -2.0, 0.0))
    elif wrap_type == WrapType.ELLIPSOID:
        return wp.diag(-2.0 * wp.cw_div(wp.vec3(1.0), wp.cw_mul(size, size)))
    rho2 = x[0] * x[0] + x[1] * x[1]
    r2 = size[1] * size[1]
    den = r2 * rho2 * wp.sqrt(rho2)
    fxy = -2.0 * size[0] * x[0] * x[1] / den
    return wp.mat33(
        2.0 * size[0] * x[1] * x[1] / den - 2.0 / r2, fxy, 0.0,
        fxy, 2.0 * size[0] * x[0] * x[0] / den - 2.0 / r2, 0.0,
        0.0, 0.0, -2.0 / r2,
    )


@wp.func
def surface_unit_normal(wrap_type: int, size: wp.vec3, x: wp.vec3) -> wp.vec3:
    g = surface_gradient(wrap_type, size, x)
    g_norm = wp.length(g)
    if g_norm < BOLT_MINVAL:
        # Singular point of the implicit function (e.g. on a cylinder's axis): try a nearby point
        g = surface_gradient(wrap_type, size, x + wp.vec3(1e-4, 1.1e-4, 1.2e-4))
        g_norm = wp.length(g)
    return -g / g_norm


@wp.func
def surface_curvature(wrap_type: int, size: wp.vec3, x: wp.vec3, direction: wp.vec3) -> float:
    """ Normal curvature of the surface in a direction, signed such that d(tangent)/ds = curvature * normal """
    k_num = wp.dot(direction, surface_hessian(wrap_type, size, x) * direction)
    if wp.abs(k_num) < BOLT_MINVAL:
        return 0.0
    return k_num / wp.length(surface_gradient(wrap_type, size, x))


@wp.func
def surface_torsion(wrap_type: int, size: wp.vec3, x: wp.vec3, direction: wp.vec3) -> float:
    """ Geodesic torsion of the surface in a direction """
    if wrap_type == WrapType.SPHERE:
        return 0.0
    g = surface_gradient(wrap_type, size, x)
    h_d = surface_hessian(wrap_type, size, x) * direction
    return -wp.dot(h_d, wp.cross(g, direction)) / wp.dot(g, g)


@wp.func
def gaussian_curvature(g: wp.vec3, H: wp.mat33) -> float:
    """ Gaussian curvature of an implicit surface from its gradient and Hessian """
    A = wp.mat33(
        H[1, 1] * H[2, 2] - H[1, 2] * H[1, 2], H[0, 2] * H[1, 2] - H[0, 1] * H[2, 2],
        H[0, 1] * H[1, 2] - H[0, 2] * H[1, 1],
        H[0, 2] * H[1, 2] - H[0, 1] * H[2, 2], H[0, 0] * H[2, 2] - H[0, 2] * H[0, 2],
        H[0, 1] * H[0, 2] - H[0, 0] * H[1, 2],
        H[0, 1] * H[1, 2] - H[0, 2] * H[1, 1], H[0, 1] * H[0, 2] - H[0, 0] * H[1, 2],
        H[0, 0] * H[1, 1] - H[0, 1] * H[0, 1],
    )
    g2 = wp.dot(g, g)
    return wp.dot(g, A * g) / (g2 * g2)


@wp.func
def is_point_inside(wrap_type: int, size: wp.vec3, x: wp.vec3) -> bool:
    return surface_value(wrap_type, size, x) > 0.0


# --- Frenet frames ---
@wp.func
def binormal(tangent: wp.vec3, normal: wp.vec3) -> wp.vec3:
    return wp.cross(tangent, normal)


@wp.func
def orthogonal_tangent(normal: wp.vec3, tangent: wp.vec3) -> wp.vec3:
    """ The tangent made orthogonal to the (exact) normal, as Simbody's Rotation::setRotationFromTwoAxes """
    b = wp.cross(tangent, normal)
    b_norm = wp.length(b)
    if b_norm < BOLT_MINVAL:
        # tangent is along the normal: any perpendicular direction
        axis = wp.vec3(1.0, 0.0, 0.0)
        if wp.abs(normal[0]) > 0.5:
            axis = wp.vec3(0.0, 1.0, 0.0)
        b = wp.cross(axis, normal)
        b_norm = wp.length(b)
    return wp.cross(normal, b / b_norm)


@wp.func
def set_boundary_frames(
        wrap_type: int, size: wp.vec3, g: WrapGeodesic,
        p_P: wp.vec3, t_P: wp.vec3, p_Q: wp.vec3, t_Q: wp.vec3,
) -> WrapGeodesic:
    """ Sets the Frenet frames, curvatures and torsions at the geodesic's boundary points """
    g.p_P = p_P
    g.n_P = surface_unit_normal(wrap_type, size, p_P)
    g.t_P = orthogonal_tangent(g.n_P, t_P)
    g.p_Q = p_Q
    g.n_Q = surface_unit_normal(wrap_type, size, p_Q)
    g.t_Q = orthogonal_tangent(g.n_Q, t_Q)
    g.curvatures_P = wp.vec2(surface_curvature(wrap_type, size, g.p_P, g.t_P),
                             surface_curvature(wrap_type, size, g.p_P, binormal(g.t_P, g.n_P)))
    g.curvatures_Q = wp.vec2(surface_curvature(wrap_type, size, g.p_Q, g.t_Q),
                             surface_curvature(wrap_type, size, g.p_Q, binormal(g.t_Q, g.n_Q)))
    g.torsion_P = surface_torsion(wrap_type, size, g.p_P, g.t_P)
    g.torsion_Q = surface_torsion(wrap_type, size, g.p_Q, g.t_Q)
    return g


# --- Analytic geodesics ---
@wp.func
def shoot_sphere(size: wp.vec3, g: WrapGeodesic, point: wp.vec3, tangent: wp.vec3, length: float) -> WrapGeodesic:
    """ A great circle arc """
    radius = size[0]
    n = wp.normalize(point)
    t = orthogonal_tangent(n, tangent)
    theta = length / radius
    c, s = wp.cos(theta), wp.sin(theta)
    g = set_boundary_frames(WrapType.SPHERE, size, g, radius * n, t, radius * (c * n + s * t), c * t - s * n)
    g.jacobi_Q = wp.vec2(c, radius * s)
    g.jacobi_dot_Q = wp.vec2(-s / radius, c)
    return g


@wp.func
def shoot_cylinder(size: wp.vec3, g: WrapGeodesic, point: wp.vec3, tangent: wp.vec3, length: float) -> WrapGeodesic:
    """ A helix around the z-axis """
    radius = size[0]
    n = wp.normalize(wp.vec3(point[0], point[1], 0.0))
    t = orthogonal_tangent(n, tangent)
    angle = -length / radius * binormal(t, n)[2]
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)
    n_Q = wp.quat_rotate(rot, n)
    p_P = radius * n + wp.vec3(0.0, 0.0, point[2])
    p_Q = radius * n_Q + wp.vec3(0.0, 0.0, point[2] + length * t[2])
    g = set_boundary_frames(WrapType.CYLINDER, size, g, p_P, t, p_Q, wp.quat_rotate(rot, t))
    # The cylinder unrolls into a plane, where the geodesic is a straight line
    g.jacobi_Q = wp.vec2(1.0, length)
    g.jacobi_dot_Q = wp.vec2(0.0, 1.0)
    return g


# --- Geodesics over implicit surfaces ---
@wp.func
def geodesic_derivs(wrap_type: int, size: wp.vec3, y: vec10) -> vec10:
    """ A unit-speed particle constrained to the surface, plus the Jacobi field equations j'' + K j = 0 """
    p = wp.vec3(y[0], y[1], y[2])
    v = wp.vec3(y[5], y[6], y[7])
    g = surface_gradient(wrap_type, size, p)
    H = surface_hessian(wrap_type, size, p)
    K = gaussian_curvature(g, H)
    a = -(wp.dot(v, H * v) / wp.dot(g, g)) * g
    return vec10(v[0], v[1], v[2], y[8], y[9], a[0], a[1], a[2], -K * y[3], -K * y[4])


@wp.func
def geodesic_project(wrap_type: int, size: wp.vec3, y: vec10) -> vec10:
    """ Projects the point onto the surface along the gradient, and the tangent onto the tangent plane """
    p = wp.vec3(y[0], y[1], y[2])
    for _ in range(10):
        p_err = surface_value(wrap_type, size, p)
        if wp.abs(p_err) <= 0.1 * WRAP_PROJECTION_TOL:
            break
        g = surface_gradient(wrap_type, size, p)
        p = p - (p_err / wp.dot(g, g)) * g
    g = surface_gradient(wrap_type, size, p)
    v = wp.vec3(y[5], y[6], y[7])
    v = wp.normalize(v - (wp.dot(g, v) / wp.dot(g, g)) * g)
    return vec10(p[0], p[1], p[2], y[3], y[4], v[0], v[1], v[2], y[8], y[9])


@wp.func
def geodesic_rkm_step(wrap_type: int, size: wp.vec3, y0: vec10, f0: vec10, h: float):
    """ One Runge-Kutta-Merson step, returns the new state and the inf-norm of its error estimate """
    f1 = geodesic_derivs(wrap_type, size, y0 + (h / 3.0) * f0)
    f2 = geodesic_derivs(wrap_type, size, y0 + (h / 6.0) * (f0 + f1))
    f3 = geodesic_derivs(wrap_type, size, y0 + (h / 8.0) * (f0 + 3.0 * f2))
    y_save = y0 + (h / 2.0) * (f0 - 3.0 * f2 + 4.0 * f3)
    f4 = geodesic_derivs(wrap_type, size, y_save)
    y1 = y0 + (h / 6.0) * (f0 + 4.0 * f3 + f4)
    err = float(0.0)
    for i in range(10):
        err = wp.max(err, wp.abs(y1[i] - y_save[i]) / 5.0)
    return y1, err


@wp.func
def shoot_implicit(
        wrap_type: int, size: wp.vec3, g: WrapGeodesic, point: wp.vec3, tangent: wp.vec3, length: float,
        accuracy: float,
) -> WrapGeodesic:
    """ Integrates the geodesic with an error-controlled RKM integrator, as Simbody's GeodesicIntegrator """
    v0 = wp.normalize(tangent)
    # Jacobi scalars start at jr = 0, jr' = 1, jt = 1, jt' = 0
    y = geodesic_project(wrap_type, size, vec10(point[0], point[1], point[2], 0.0, 1.0,
                                                v0[0], v0[1], v0[2], 1.0, 0.0))
    p_P = wp.vec3(y[0], y[1], y[2])
    t_P = wp.vec3(y[5], y[6], y[7])

    f = geodesic_derivs(wrap_type, size, y)
    s = float(0.0)
    h_init = g.step_size
    h_next = h_init
    h_first = float(0.0)
    h_second = float(0.0)
    num_steps = int(0)
    while s < length and num_steps < WRAP_MAX_GEODESIC_STEPS:
        s1 = s + h_next
        if s1 + 0.1 * h_next > length:
            s1 = length
        # Shrink the step until it is accurate enough
        h = s1 - s
        y1, err = geodesic_rkm_step(wrap_type, size, y, f, h)
        for _ in range(20):
            if err <= accuracy:
                break
            h = wp.clamp(0.9 * h * wp.sqrt(wp.sqrt(accuracy / err)), 0.1 * h, 0.9 * h)
            s1 = s + h
            y1, err = geodesic_rkm_step(wrap_type, size, y, f, h)
        y = geodesic_project(wrap_type, size, y1)
        f = geodesic_derivs(wrap_type, size, y)
        if num_steps == 0:
            h_first = h
        elif num_steps == 1:
            h_second = h
        num_steps += 1
        s = s1
        if s < length:
            h_new = 5.0 * h
            if err > 0.0:
                h_new = 0.9 * h * wp.sqrt(wp.sqrt(accuracy / err))
            if h_new < 1.2 * h:
                h_new = h
            h_next = wp.min(h_new, 5.0 * h)

    g = set_boundary_frames(wrap_type, size, g, p_P, t_P, wp.vec3(y[0], y[1], y[2]), wp.vec3(y[5], y[6], y[7]))
    g.jacobi_Q = wp.vec2(y[4], y[3])
    g.jacobi_dot_Q = wp.vec2(y[9], y[8])

    # Initial step size for the next geodesic: shrink it if it was rejected, grow it if the next step was larger
    if num_steps >= 2:
        if h_first < h_init:
            g.step_size = h_first
        else:
            g.step_size = wp.max(h_init, h_second)
    return g


@wp.func
def shoot_geodesic(
        wrap_type: int, size: wp.vec3, g: WrapGeodesic, point: wp.vec3, tangent: wp.vec3, length: float,
        accuracy: float,
) -> WrapGeodesic:
    """ Shoots a new geodesic of the given length from (near) the given point and tangent, and makes contact """
    if wrap_type == WrapType.SPHERE:
        g = shoot_sphere(size, g, point, tangent, length)
    elif wrap_type == WrapType.CYLINDER:
        g = shoot_cylinder(size, g, point, tangent, length)
    else:
        g = shoot_implicit(wrap_type, size, g, point, tangent, length, accuracy)
    g.length = length
    g.status = WrapStatus.IN_CONTACT
    return g


# --- Touchdown ---
@wp.func
def nearest_point_on_line(wrap_type: int, size: wp.vec3, point_a: wp.vec3, point_b: wp.vec3, guess: wp.vec3):
    """
    Newton iteration for the point on the line segment AB nearest the surface, from the guess.
    Returns the point (the guess if not converged), and whether it falls inside the surface (touchdown)
    """
    d = point_b - point_a
    dd = wp.dot(d, d)
    if dd < BOLT_MINVAL:
        return point_a, is_point_inside(wrap_type, size, point_a)

    alpha = wp.clamp(-wp.dot(d, point_a - guess) / dd, 0.0, 1.0)
    for _ in range(WRAP_NEAREST_POINT_MAX_ITERATIONS):
        point = point_a + alpha * d
        # Distance-like measure to the surface, negative inside
        if -surface_value(wrap_type, size, point) <= 0.0:
            return point, True
        grad = -surface_gradient(wrap_type, size, point)
        hess = -wp.dot(d, surface_hessian(wrap_type, size, point) * d)
        # Damped Newton step (|hessian| in case the surface is not convex), and limit the step size
        step = wp.clamp(wp.dot(grad, d) / (wp.abs(hess) + 0.5), -0.2, 0.2)
        prev_alpha = alpha
        alpha = wp.clamp(alpha - step, 0.0, 1.0)
        if wp.abs(alpha - prev_alpha) < WRAP_NEAREST_POINT_TOL:
            return point_a + alpha * d, False
    return guess, False
