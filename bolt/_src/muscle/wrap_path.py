"""
Muscle paths that wrap over obstacles, ported from Simbody's CableSpan with the Scholz2015 algorithm

The path between two consecutive path points is a sequence of straight lines and geodesics over the obstacles in
contact. Each geodesic has four natural variations: tangential and binormal shifts of its initial point, rotation of
its initial tangent, and lengthening. Each solver iteration computes the corrections that drive the misalignment
between the straight lines and the geodesics' boundary frames to zero, and re-shoots the geodesics. The geodesics are
kept in Data as the warm start of the next evaluation, which also tracks the touchdown and liftoff of each obstacle.
A path that folds back on an obstacle (a straight line running backward along a curve) is never smooth, and the folded
curve is trimmed or restarted, as in Simbody PR #887.
"""
import functools

import warp as wp

from bolt.consts import BOLT_MINVAL
from bolt.consts import WRAP_FOLD_MAX_SAMPLES
from bolt.consts import WRAP_FOLD_MAX_TRIM_TURNING
from bolt.consts import WRAP_FOLD_MIN_RESTART_ALIGNMENT
from bolt.consts import WRAP_FOLD_SAMPLE_ANGLE
from bolt.consts import WRAP_INITIAL_STEP_SIZE
from bolt.consts import WRAP_MAX_STEP_ANGLE
from bolt.types import Data
from bolt.types import Model
from bolt.types import WrapGeodesic
from bolt.types import WrapStatus

from ..warp_util import event_scope
from ..warp_util import launch
from . import wrap_geometry

wp.set_module_options({"enable_backward": False})


@wp.func
def is_in_contact(g: WrapGeodesic) -> bool:
    return g.status == WrapStatus.IN_CONTACT or g.status == WrapStatus.INITIAL_GUESS


@wp.func
def body_frame(X_GB: wp.transform, origin: wp.vec3) -> wp.transform:
    """ A body frame, translated so that the local origin is at zero """
    return wp.transform(wp.transform_get_translation(X_GB) - origin, wp.transform_get_rotation(X_GB))


@wp.func
def surface_frame(wrapid: int, wrap_bodyid: wp.array(dtype=int), wrap_X_BS: wp.array(dtype=wp.transform),
                  X_GB: wp.array(dtype=wp.transform), origin: wp.vec3) -> wp.transform:
    """ An obstacle's surface frame relative to the local origin (see compute_path_kernel) """
    return wp.transform_multiply(body_frame(X_GB[wrap_bodyid[wrapid]], origin), wrap_X_BS[wrapid])


@wp.func
def prev_in_contact(wrapid: int, seg_begin: int, geodesics: wp.array(dtype=WrapGeodesic)) -> int:
    """ The previous obstacle in contact within the segment, -1 if none """
    for i in range(wrapid - seg_begin):
        if is_in_contact(geodesics[wrapid - 1 - i]):
            return wrapid - 1 - i
    return -1


@wp.func
def next_in_contact(wrapid: int, seg_end: int, geodesics: wp.array(dtype=WrapGeodesic)) -> int:
    """ The next obstacle in contact within the segment, -1 if none """
    for j in range(wrapid + 1, seg_end):
        if is_in_contact(geodesics[j]):
            return j
    return -1


@wp.func
def prev_path_point(
        wrapid: int, seg_begin: int, start_G: wp.vec3, geodesics: wp.array(dtype=WrapGeodesic),
        wrap_bodyid: wp.array(dtype=int), wrap_X_BS: wp.array(dtype=wp.transform), X_GB: wp.array(dtype=wp.transform),
        origin: wp.vec3,
) -> wp.vec3:
    """ The final contact point of the previous obstacle in contact, or else the path point starting the segment """
    j = prev_in_contact(wrapid, seg_begin, geodesics)
    if j < 0:
        return start_G
    return wp.transform_point(surface_frame(j, wrap_bodyid, wrap_X_BS, X_GB, origin), geodesics[j].p_Q)


@wp.func
def next_path_point(
        wrapid: int, seg_end: int, end_G: wp.vec3, geodesics: wp.array(dtype=WrapGeodesic),
        wrap_bodyid: wp.array(dtype=int), wrap_X_BS: wp.array(dtype=wp.transform), X_GB: wp.array(dtype=wp.transform),
        origin: wp.vec3,
) -> wp.vec3:
    """ The initial contact point of the next obstacle in contact, or else the path point ending the segment """
    j = next_in_contact(wrapid, seg_end, geodesics)
    if j < 0:
        return end_G
    return wp.transform_point(surface_frame(j, wrap_bodyid, wrap_X_BS, X_GB, origin), geodesics[j].p_P)


@wp.func
def init_geodesic(hint_on_surface: wp.vec3) -> WrapGeodesic:
    """ The geodesic before the first evaluation: start from the contact hint if there is one """
    g = WrapGeodesic()
    g.step_size = WRAP_INITIAL_STEP_SIZE
    if wp.isnan(hint_on_surface[0]):
        g.status = WrapStatus.LIFTED
    else:
        g.status = WrapStatus.INITIAL_GUESS
        g.p_P = hint_on_surface
        g.p_Q = hint_on_surface
    return g


# --- Undoing folds (Simbody PR #887) ---
# A curve folds when the straight line entering it at P, or leaving it at Q, runs opposite to its tangent there. The
# normal and binormal path errors vanish for a fully reversed line as well, so the solver can converge to a fold, or get
# stuck near one, e.g. after a large motion leaves a warm-started curve wrapped too far around the obstacle
@wp.func
def unit_angle(a: wp.vec3, b: wp.vec3) -> float:
    """ Angle between two unit vectors """
    return wp.acos(wp.clamp(wp.dot(a, b), -1.0, 1.0))


@wp.func
def fold_angle(e: wp.vec3, t: wp.vec3) -> float:
    """ Angle between a line direction e and a curve tangent t if they point in opposite directions, else zero """
    cos_angle = wp.dot(e, t)
    if cos_angle < 0.0:
        return wp.acos(wp.max(cos_angle, -1.0))
    return 0.0


@wp.func
def fold_num_samples(g: WrapGeodesic) -> int:
    """ Samples of the curve searched for a trim point: ~WRAP_FOLD_SAMPLE_ANGLE of its largest curvature apart """
    curvature = wp.max(wp.max(wp.abs(g.curvatures_P[0]), wp.abs(g.curvatures_P[1])),
                       wp.max(wp.abs(g.curvatures_Q[0]), wp.abs(g.curvatures_Q[1])))
    intervals = wp.ceil(g.length * curvature / WRAP_FOLD_SAMPLE_ANGLE)
    if not wp.isfinite(intervals):
        return 2
    return wp.max(int(wp.min(intervals, float(WRAP_FOLD_MAX_SAMPLES))) + 1, 2)


@wp.func
def walk_curve(
        wrap_type: int, size: wp.vec3, g: WrapGeodesic, prev_S: wp.vec3, next_S: wp.vec3, num: int, from_P: bool,
        stop: int, total: bool, accuracy: float,
):
    """
    Walks the num samples of the curve (equally spaced in arc length), from P (sample 0) up to sample num - 1, or from Q
    (sample num - 1) down to sample stop, and finds the first sample whose tangent points forward: away from prev_S
    walking from P, toward next_S walking from Q. Then keeps walking while the alignment improves, to its first local
    maximum. That only removes the folded part: the best aligned sample of the whole curve could be whole windings away.
    Gives up if no sample points forward before the curve turns by WRAP_FOLD_MAX_TRIM_TURNING, as an overshooting curve
    folds back by less than half a turn. The turning between samples is the larger of the turns of the tangent and of
    the surface normal: on a steep helix, the tangent turns much less than a full turn per winding, the normal a full
    turn. With total, only sums the turning over the whole walk.
    Returns that sample's index (-1 if none), point and tangent, and the turning walked
    """
    ds = g.length / float(num - 1)
    point = wp.vec3(g.p_P)
    tangent = wp.vec3(g.t_P)
    first = int(0)
    count = num - 1
    if not from_P:
        point = wp.vec3(g.p_Q)
        tangent = wp.vec3(g.t_Q)
        first = num - 1
        count = num - 1 - stop
    normal = wrap_geometry.surface_unit_normal(wrap_type, size, point)
    probe = g
    best = int(-1)
    best_align = float(0.0)
    best_point = point
    best_tangent = tangent
    turning = float(0.0)
    for k in range(count + 1):
        if k > 0:
            # The next sample: walking from Q follows the reversed curve, which is a geodesic too
            walk = tangent
            if not from_P:
                walk = -tangent
            probe = wrap_geometry.shoot_geodesic(wrap_type, size, probe, point, walk, ds, accuracy)
            next_tangent = wp.vec3(probe.t_Q)
            if not from_P:
                next_tangent = -next_tangent
            next_normal = wrap_geometry.surface_unit_normal(wrap_type, size, probe.p_Q)
            turning += wp.max(unit_angle(tangent, next_tangent), unit_angle(normal, next_normal))
            point = wp.vec3(probe.p_Q)
            tangent = next_tangent
            normal = next_normal
        if total:
            if turning > WRAP_FOLD_MAX_TRIM_TURNING:
                break  # only whether the curve turns by more than this is used
            continue
        if best < 0 and turning > WRAP_FOLD_MAX_TRIM_TURNING:
            break  # trimming further would remove a winding
        d = point - prev_S
        if not from_P:
            d = next_S - point
        d_norm = wp.length(d)
        align = float(-1.0)
        if d_norm > 0.0:
            align = wp.dot(tangent, d) / d_norm
        if align > best_align:
            best_align = align
            best = first + k
            if not from_P:
                best = first - k
            best_point = point
            best_tangent = tangent
        elif best >= 0:
            break  # the first local maximum of the alignment
    return best, best_point, best_tangent, turning


@wp.func
def tangential_direction(line: wp.vec3, normal: wp.vec3) -> wp.vec3:
    """ A line projected onto the tangent plane, then normalized (zero if along the normal) """
    tangential = line - wp.dot(line, normal) * normal
    norm = wp.length(tangential)
    if norm > 3.5e-4 * wp.length(line):  # Simbody's SqrtEps, for float32
        return tangential / norm
    return wp.vec3()


@wp.func
def restart_is_forward(tangent: wp.vec3, tangent_norm: float, direction: wp.vec3) -> bool:
    return wp.length(direction) == 0.0 or wp.dot(tangent, direction) > WRAP_FOLD_MIN_RESTART_ALIGNMENT * tangent_norm


@wp.func
def trim_folded_curve(wrap_type: int, size: wp.vec3, g: WrapGeodesic, prev_S: wp.vec3, next_S: wp.vec3,
                      accuracy: float):
    """
    Undoes a fold of a curve in contact. A curve with nonzero length is trimmed at each folded end to the nearest part
    that heads away from prev_S (at P), or toward next_S (at Q) (see walk_curve). This keeps the side of the obstacle
    the curve wraps over, and the useful part of the wrap, including its full windings: a curve that turns by more
    than half a turn is never replaced.
    If nothing useful is left, or the curve has zero length, it restarts as a zero-length curve whose tangent is the
    bisector of the incoming and outgoing lines, projected onto the tangent plane (each before normalizing, so that a
    steep line can't make it point backward). Only if that tangent clearly aligns with both lines: if they point in
    nearly opposite directions, the bisector is dominated by roundoff. Otherwise the curve is left as is, and the solver
    doesn't report the fold as converged (see the fold angle in solve_segment).
    Returns whether to shoot a new geodesic, and its initial point, tangent and length
    """
    no = bool(False)
    p_P = wp.vec3(g.p_P)
    t_P = wp.vec3(g.t_P)
    if not (g.length >= 0.0):
        return no, p_P, t_P, g.length
    # The path point before or after the curve can be the contact point of a neighbouring obstacle, which may lie
    # inside this surface while the solver iterates: leave the curve as is then
    if wrap_geometry.is_point_inside(wrap_type, size, prev_S) or wrap_geometry.is_point_inside(wrap_type, size, next_S):
        return no, p_P, t_P, g.length
    folds_P = wp.dot(g.t_P, g.p_P - prev_S) < 0.0
    folds_Q = wp.dot(g.t_Q, next_S - g.p_Q) < 0.0
    if not (folds_P or folds_Q):
        return no, p_P, t_P, g.length

    restart_point = p_P  # where to place a zero-length curve if nothing is left of this one
    if g.length > 0.0:
        num = fold_num_samples(g)
        ds = g.length / float(num - 1)
        first = int(0)
        first_point = p_P
        first_tangent = t_P
        last = num - 1
        found = bool(True)
        total_turning = float(-1.0)  # unknown
        if folds_P:
            first, first_point, first_tangent, turning_P = walk_curve(wrap_type, size, g, prev_S, next_S, num, True,
                                                                      num - 1, False, accuracy)
            found = first >= 0
            if not found:
                # The walk from P either stopped past the max trim turning, or walked the whole curve: its turning
                # answers the winding test below
                total_turning = turning_P
        if found:
            restart_point = first_point
            if folds_Q:
                last, _point, _tangent, _turning = walk_curve(wrap_type, size, g, prev_S, next_S, num, False, first,
                                                              False, accuracy)
                found = last >= 0
        if found and last > first:
            return True, first_point, first_tangent, float(last - first) * ds
        # Nothing forward-facing is left within reach of the folded end: if the curve winds around the obstacle,
        # leave it as is (the solver may still unfold it without losing the winding)
        if total_turning < 0.0:
            _best, _point, _tangent, total_turning = walk_curve(wrap_type, size, g, prev_S, next_S, num, True,
                                                                num - 1, True, accuracy)
        if total_turning > WRAP_FOLD_MAX_TRIM_TURNING:
            return no, p_P, t_P, g.length

    normal = wrap_geometry.surface_unit_normal(wrap_type, size, restart_point)
    incoming = tangential_direction(restart_point - prev_S, normal)
    outgoing = tangential_direction(next_S - restart_point, normal)
    tangent = incoming + outgoing
    tangent_norm = wp.length(tangent)
    if not (tangent_norm > 0.0 and restart_is_forward(tangent, tangent_norm, incoming)
            and restart_is_forward(tangent, tangent_norm, outgoing)):
        return no, p_P, t_P, g.length
    return True, restart_point, tangent, 0.0


@wp.func
def update_wrap_status(
        wrap_type: int, size: wp.vec3, g: WrapGeodesic, prev_S: wp.vec3, next_S: wp.vec3, accuracy: float,
) -> WrapGeodesic:
    """
    Initializes the geodesic from the contact hint, undoes a fold, or detects liftoff / touchdown, given the
    neighbouring path points
    """
    # The geodesic to shoot, if any: a single call site keeps the kernels small (and quick to compile)
    shoot = bool(False)
    point = wp.vec3()
    tangent = next_S - prev_S
    length = float(0.0)
    in_contact = g.status == WrapStatus.IN_CONTACT
    if g.status == WrapStatus.INITIAL_GUESS:
        # Zero-length geodesic at the hint (already projected onto the surface, see init_geodesic: the nearest point,
        # rather than the geodesic integrator's projection), directed along the straight line
        shoot = True
        point = g.p_P
    elif in_contact:
        # Undo a fold before the solver tests for convergence. This can shrink the curve to zero length, so the liftoff
        # test comes afterwards
        shoot, point, tangent, length = trim_folded_curve(wrap_type, size, g, prev_S, next_S, accuracy)
    elif g.status == WrapStatus.LIFTED:
        # Touch down when the straight line between the neighbouring points dips below the surface
        nearest, touchdown = wrap_geometry.nearest_point_on_line(wrap_type, size, prev_S, next_S, g.tracking_point)
        if touchdown:
            shoot = True
            point = nearest
        else:
            g.tracking_point = nearest
    if shoot:
        g = wrap_geometry.shoot_geodesic(wrap_type, size, g, point, tangent, length, accuracy)
    # Lift off once the geodesic has shrunk to zero length, and the neighbouring points are above the surface
    if in_contact and g.length <= 0.0 and wp.dot(prev_S - g.p_P, g.n_P) > 0.0 and wp.dot(next_S - g.p_P, g.n_P) > 0.0:
        g.status = WrapStatus.LIFTED
        g.tracking_point = g.p_P
    return g


@wp.func
def apply_geodesic_correction(wrap_type: int, size: wp.vec3, g: WrapGeodesic, c: wp.vec4,
                              accuracy: float) -> WrapGeodesic:
    """ Shoots a new geodesic from the initial conditions changed by the natural geodesic correction c """
    # Clamp the length to be non-negative, spreading the clamped part over both ends
    length = g.length + c[3]
    c0 = c[0]
    if length < 0.0:
        c0 += 0.5 * length
        length = 0.0
    t, n = g.t_P, g.n_P
    b = wrap_geometry.binormal(t, n)
    point = g.p_P + c0 * t + c[1] * b
    w = (g.curvatures_P[0] * c0 - g.torsion_P * c[1]) * b - c[2] * n
    return wrap_geometry.shoot_geodesic(wrap_type, size, g, point, t + wp.cross(w, t), length, accuracy)


@wp.func
def variation_P(t: wp.vec3, n: wp.vec3, b: wp.vec3, curvatures: wp.vec2, torsion: float, dc_dv: wp.vec3,
                dc_dw: wp.vec3) -> wp.vec4:
    """
    Gradient of a scalar c to the natural geodesic corrections, given its gradients to the position (dc_dv) and
    rotation (dc_dw) of the initial Frenet frame P. These vary along the columns of v_P = [t, b, 0, 0] and
    w_P = [tau t + kt b, -kb t - tau b, -n, 0], with kt, kb the curvatures in the tangent and binormal directions.
    """
    return wp.vec4(
        wp.dot(t, dc_dv) + wp.dot(torsion * t + curvatures[0] * b, dc_dw),
        wp.dot(b, dc_dv) - wp.dot(curvatures[1] * t + torsion * b, dc_dw),
        -wp.dot(n, dc_dw),
        0.0,
    )


@wp.func
def variation_Q(t: wp.vec3, n: wp.vec3, b: wp.vec3, curvatures: wp.vec2, torsion: float, jacobi: wp.vec2,
                jacobi_dot: wp.vec2, dc_dv: wp.vec3, dc_dw: wp.vec3) -> wp.vec4:
    """
    As variation_P for the final Frenet frame Q, with the translational and rotational Jacobi scalars (a, r):
    v_Q = [t, a b, r b, t] and w_Q = [tau t + kt b, -a kb t - a' n - a tau b, -r kb t - r' n - r tau b, w_Q0]
    """
    v_t = wp.dot(t, dc_dv)
    v_b = wp.dot(b, dc_dv)
    w_0 = wp.dot(torsion * t + curvatures[0] * b, dc_dw)
    w_b = wp.dot(curvatures[1] * t + torsion * b, dc_dw)
    w_n = wp.dot(n, dc_dw)
    return wp.vec4(
        v_t + w_0,
        jacobi[0] * (v_b - w_b) - jacobi_dot[0] * w_n,
        jacobi[1] * (v_b - w_b) - jacobi_dot[1] * w_n,
        v_t + w_0,
    )


@wp.func
def max_angular_displacement(t_P: wp.vec3, n_P: wp.vec3, b_P: wp.vec3, g: WrapGeodesic, t_Q: wp.vec3, n_Q: wp.vec3,
                             b_Q: wp.vec3, c: wp.vec4) -> float:
    """
    Estimated rotation of the boundary frames by a correction: max of |column of w|_inf * |c|. Simbody uses c signed,
    which leaves negative corrections unlimited (and mirror-image paths solving differently: mirroring flips the sign of
    some corrections)
    """
    w_P0 = g.torsion_P * t_P + g.curvatures_P[0] * b_P
    w_P1 = -(g.curvatures_P[1] * t_P + g.torsion_P * b_P)
    w_Q0 = g.torsion_Q * t_Q + g.curvatures_Q[0] * b_Q
    w_Qb = -(g.curvatures_Q[1] * t_Q + g.torsion_Q * b_Q)
    w_Q1 = g.jacobi_Q[0] * w_Qb - g.jacobi_dot_Q[0] * n_Q
    w_Q2 = g.jacobi_Q[1] * w_Qb - g.jacobi_dot_Q[1] * n_Q
    disp = float(0.0)
    disp = wp.max(disp, wp.max(wp.abs(w_P0)) * wp.abs(c[0]))
    disp = wp.max(disp, wp.max(wp.abs(w_P1)) * wp.abs(c[1]))
    disp = wp.max(disp, wp.max(wp.abs(n_P)) * wp.abs(c[2]))
    disp = wp.max(disp, wp.max(wp.abs(w_Q0)) * wp.abs(c[0]))
    disp = wp.max(disp, wp.max(wp.abs(w_Q1)) * wp.abs(c[1]))
    disp = wp.max(disp, wp.max(wp.abs(w_Q2)) * wp.abs(c[2]))
    disp = wp.max(disp, wp.max(wp.abs(w_Q0)) * wp.abs(c[3]))
    return disp


@wp.func
def contact_point_velocity(x_G: wp.vec3, X_GB: wp.transform, V_GB: wp.spatial_vector) -> wp.vec3:
    """ Velocity of a point fixed to a body, at x_G in ground """
    return wp.spatial_bottom(V_GB) + wp.cross(wp.spatial_top(V_GB), x_G - wp.transform_get_translation(X_GB))


@wp.func
def segment_obstacle_end(wrap_begin: int, wrap_end: int, wrap_segment: wp.array(dtype=int), segment: int) -> int:
    """ Obstacles are ordered along the path: returns the end of the run of obstacles in this segment """
    o = wrap_begin
    while o < wrap_end and wrap_segment[o] == segment:
        o += 1
    return o


@functools.cache
def _wrap_kernels(max_per_segment: int):
    """ Kernels for paths with up to max_per_segment obstacles between two consecutive path points """
    NW = max_per_segment
    NQ = 4 * NW  # natural geodesic corrections
    NC = 5 * NW  # 4 path errors per obstacle, and a penalty on changing its length

    class MatA(wp.types.matrix(shape=(NC, NQ), dtype=float)):
        pass

    class VecC(wp.types.vector(length=NC, dtype=float)):
        pass

    class VecQ(wp.types.vector(length=NQ, dtype=float)):
        pass

    @wp.func
    def solve_least_squares(A: MatA, b: VecC) -> VecQ:
        """ Least squares solution of A x = b by Householder QR, dropping columns with a negligible pivot """
        R_diag = VecQ()
        R_max = float(0.0)
        for j in range(NQ):
            norm2 = float(0.0)
            for i in range(j, NC):
                norm2 += A[i, j] * A[i, j]
            norm = wp.sqrt(norm2)
            if norm < BOLT_MINVAL:
                continue
            alpha = -norm
            if A[j, j] < 0.0:
                alpha = norm
            v_norm2 = norm2 - A[j, j] * A[j, j] + (A[j, j] - alpha) * (A[j, j] - alpha)
            A[j, j] = A[j, j] - alpha
            R_diag[j] = alpha
            R_max = wp.max(R_max, norm)
            for k in range(j + 1, NQ):
                dot = float(0.0)
                for i in range(j, NC):
                    dot += A[i, j] * A[i, k]
                f = 2.0 * dot / v_norm2
                for i in range(j, NC):
                    A[i, k] = A[i, k] - f * A[i, j]
            dot = float(0.0)
            for i in range(j, NC):
                dot += A[i, j] * b[i]
            f = 2.0 * dot / v_norm2
            for i in range(j, NC):
                b[i] = b[i] - f * A[i, j]
        x = VecQ()
        for jj in range(NQ):
            j = NQ - 1 - jj
            if wp.abs(R_diag[j]) <= 1e-6 * R_max:
                continue
            s = b[j]
            for k in range(j + 1, NQ):
                s -= A[j, k] * x[k]
            x[j] = s / R_diag[j]
        return x

    @wp.func
    def solve_segment(
            # Model:
            wrap_type: wp.array(dtype=int),
            wrap_bodyid: wp.array(dtype=int),
            wrap_X_BS: wp.array(dtype=wp.transform),
            wrap_size: wp.array(dtype=wp.vec3),
            smoothness_tol: float,
            max_iterations: int,
            accuracy: float,
            # Data in:
            X_GB: wp.array(dtype=wp.transform),
            # In:
            origin: wp.vec3,
            seg_begin: int,
            seg_end: int,
            start_G: wp.vec3,
            end_G: wp.vec3,
            # Data out:
            geodesics: wp.array(dtype=WrapGeodesic),
    ) -> float:
        """
        Solves the path between two consecutive path points, over the obstacles seg_begin..seg_end-1. Returns its
        smoothness: the max of the path errors and of the fold angles (see fold_angle), at most smoothness_tol if it
        converged within max_iterations
        """
        smoothness = float(0.0)
        for loop_iter in range(max_iterations + 1):
            # Update each obstacle's wrapping status, in path order
            for o in range(seg_begin, seg_end):
                X_SG = wp.transform_inverse(surface_frame(o, wrap_bodyid, wrap_X_BS, X_GB, origin))
                prev_S = wp.transform_point(X_SG, prev_path_point(
                    o, seg_begin, start_G, geodesics, wrap_bodyid, wrap_X_BS, X_GB, origin))
                next_S = wp.transform_point(X_SG, next_path_point(
                    o, seg_end, end_G, geodesics, wrap_bodyid, wrap_X_BS, X_GB, origin))
                geodesics[o] = update_wrap_status(wrap_type[o], wrap_size[o], geodesics[o], prev_S, next_S, accuracy)

            # Path errors: misalignment of the straight lines with the normals and binormals of the boundary frames,
            # (P normal, P binormal, Q normal, Q binormal) per obstacle in contact, and their Jacobian
            A = MatA()
            b = VecC()
            max_error = float(0.0)
            max_fold = float(0.0)
            slot = int(0)
            for o in range(seg_begin, seg_end):
                g = geodesics[o]
                if not is_in_contact(g):
                    continue
                X_GS = surface_frame(o, wrap_bodyid, wrap_X_BS, X_GB, origin)
                R = wp.transform_get_rotation(X_GS)
                t_P, n_P = wp.quat_rotate(R, g.t_P), wp.quat_rotate(R, g.n_P)
                t_Q, n_Q = wp.quat_rotate(R, g.t_Q), wp.quat_rotate(R, g.n_Q)
                b_P, b_Q = wrap_geometry.binormal(t_P, n_P), wrap_geometry.binormal(t_Q, n_Q)
                line_P = wp.transform_point(X_GS, g.p_P) - prev_path_point(
                    o, seg_begin, start_G, geodesics, wrap_bodyid, wrap_X_BS, X_GB, origin)
                line_Q = next_path_point(o, seg_end, end_G, geodesics, wrap_bodyid, wrap_X_BS,
                                         X_GB, origin) - wp.transform_point(X_GS, g.p_Q)
                l_P, l_Q = wp.length(line_P), wp.length(line_Q)
                # A path point on the contact point leaves the line's direction undefined: no path error from it
                e_P, e_Q = t_P, t_Q
                if l_P >= BOLT_MINVAL:
                    e_P = line_P / l_P
                else:
                    l_P = 1.0
                if l_Q >= BOLT_MINVAL:
                    e_Q = line_Q / l_Q
                else:
                    l_Q = 1.0
                # A fold has no normal or binormal path error, but is never smooth
                max_fold = wp.max(max_fold, wp.max(fold_angle(e_P, t_P), fold_angle(e_Q, t_Q)))
                o_prev = prev_in_contact(o, seg_begin, geodesics)
                o_next = next_in_contact(o, seg_end, geodesics)

                for k in range(4):
                    row = 4 * slot + k
                    err = float(0.0)
                    if k < 2:
                        axis = n_P
                        if k == 1:
                            axis = b_P
                        err = wp.dot(e_P, axis)
                        dc_dv = (axis - e_P * err) / l_P
                        block = variation_P(t_P, n_P, b_P, g.curvatures_P, g.torsion_P, dc_dv, wp.cross(axis, e_P))
                        for j in range(4):
                            A[row, 4 * slot + j] = block[j]
                        # The line at P also moves with the final point of the previous obstacle in contact
                        if o_prev >= 0:
                            g_prev = geodesics[o_prev]
                            R_prev = wp.transform_get_rotation(surface_frame(o_prev, wrap_bodyid, wrap_X_BS, X_GB, origin))
                            t_prev = wp.quat_rotate(R_prev, g_prev.t_Q)
                            n_prev = wp.quat_rotate(R_prev, g_prev.n_Q)
                            block_prev = variation_Q(t_prev, n_prev, wrap_geometry.binormal(t_prev, n_prev),
                                                     g_prev.curvatures_Q, g_prev.torsion_Q, g_prev.jacobi_Q,
                                                     g_prev.jacobi_dot_Q, -dc_dv, wp.vec3())
                            for j in range(4):
                                A[row, 4 * slot - 4 + j] = block_prev[j]
                    else:
                        axis = n_Q
                        if k == 3:
                            axis = b_Q
                        err = wp.dot(e_Q, axis)
                        dc_dv = -(axis - e_Q * err) / l_Q
                        block = variation_Q(t_Q, n_Q, b_Q, g.curvatures_Q, g.torsion_Q, g.jacobi_Q, g.jacobi_dot_Q,
                                            dc_dv, wp.cross(axis, e_Q))
                        for j in range(4):
                            A[row, 4 * slot + j] = block[j]
                        # The line at Q also moves with the initial point of the next obstacle in contact
                        if o_next >= 0:
                            g_next = geodesics[o_next]
                            R_next = wp.transform_get_rotation(surface_frame(o_next, wrap_bodyid, wrap_X_BS, X_GB, origin))
                            t_next = wp.quat_rotate(R_next, g_next.t_P)
                            n_next = wp.quat_rotate(R_next, g_next.n_P)
                            block_next = variation_P(t_next, n_next, wrap_geometry.binormal(t_next, n_next),
                                                     g_next.curvatures_P, g_next.torsion_P, -dc_dv, wp.vec3())
                            for j in range(4):
                                A[row, 4 * slot + 4 + j] = block_next[j]
                    b[row] = err
                    max_error = wp.max(max_error, wp.abs(err))
                slot += 1

            # The fold angle only decides convergence (when the path errors are small, i.e. a line is nearly reversed):
            # the corrections below still use the path errors alone
            smoothness = wp.max(max_error, max_fold)
            if smoothness <= smoothness_tol or loop_iter >= max_iterations:
                break

            # Penalize changing the length of each curve, weighted by the path error: heavily when far from the
            # solution, less so when closer. Unused slots get identity rows, so their corrections are zero
            for k in range(NW):
                if k < slot:
                    A[4 * NW + k, 4 * k + 3] = max_error
                else:
                    for j in range(4):
                        A[4 * k + j, 4 * k + j] = 1.0
            q = -solve_least_squares(A, b)

            # Limit the step so that no boundary frame rotates by more than the max step angle
            step = float(1.0)
            slot = int(0)
            for o in range(seg_begin, seg_end):
                g = geodesics[o]
                if not is_in_contact(g):
                    continue
                R = wp.transform_get_rotation(surface_frame(o, wrap_bodyid, wrap_X_BS, X_GB, origin))
                t_P, n_P = wp.quat_rotate(R, g.t_P), wp.quat_rotate(R, g.n_P)
                t_Q, n_Q = wp.quat_rotate(R, g.t_Q), wp.quat_rotate(R, g.n_Q)
                c = wp.vec4(q[4 * slot], q[4 * slot + 1], q[4 * slot + 2], q[4 * slot + 3])
                disp = max_angular_displacement(t_P, n_P, wrap_geometry.binormal(t_P, n_P), g, t_Q, n_Q,
                                                wrap_geometry.binormal(t_Q, n_Q), c)
                if disp > WRAP_MAX_STEP_ANGLE:
                    step = wp.min(step, wp.abs(WRAP_MAX_STEP_ANGLE / disp))
                slot += 1

            # Apply the corrections
            slot = int(0)
            for o in range(seg_begin, seg_end):
                g = geodesics[o]
                if not is_in_contact(g):
                    continue
                c = step * wp.vec4(q[4 * slot], q[4 * slot + 1], q[4 * slot + 2], q[4 * slot + 3])
                assert wp.isfinite(c[0]) and wp.isfinite(c[1]) and wp.isfinite(c[2]) and wp.isfinite(c[3])  # debug mode
                geodesics[o] = apply_geodesic_correction(wrap_type[o], wrap_size[o], g, c, accuracy)
                slot += 1
        return smoothness

    @wp.kernel(module="unique")
    def compute_path_kernel(
            # Model:
            muscle_pts_adr: wp.array(dtype=int),
            muscle_pts_num: wp.array(dtype=int),
            muscle_wrap_adr: wp.array(dtype=int),
            muscle_wrap_num: wp.array(dtype=int),
            wrap_type: wp.array(dtype=int),
            wrap_bodyid: wp.array(dtype=int),
            wrap_X_BS: wp.array(dtype=wp.transform),
            wrap_size: wp.array(dtype=wp.vec3),
            wrap_segment: wp.array(dtype=int),
            wrap_hint_on_surface_S: wp.array(dtype=wp.vec3),
            smoothness_tol: float,
            max_iterations: int,
            accuracy: float,
            # Data in:
            integration_done_in: wp.array(dtype=bool),
            mob_X_GB_in: wp.array2d(dtype=wp.transform),
            body_V_GB_in: wp.array2d(dtype=wp.spatial_vector),
            site_pos_G_in: wp.array2d(dtype=wp.vec3),
            site_vel_G_in: wp.array2d(dtype=wp.vec3),
            # In:
            muscle_wrap_group: wp.array(dtype=int),
            # Data out:
            wrap_geodesic_out: wp.array2d(dtype=WrapGeodesic),
            muscle_length_out: wp.array2d(dtype=float),
            muscle_velocity_out: wp.array2d(dtype=float),
            muscle_wrap_smoothness_out: wp.array2d(dtype=float),
    ):
        worldid, nodeid = wp.tid()
        if integration_done_in[worldid]:
            return
        muscle_id = muscle_wrap_group[nodeid]
        pts_adr = muscle_pts_adr[muscle_id]
        wrap_begin = muscle_wrap_adr[muscle_id]
        wrap_end = wrap_begin + muscle_wrap_num[muscle_id]
        X_GB = mob_X_GB_in[worldid]
        V_GB = body_V_GB_in[worldid]
        geodesics = wrap_geodesic_out[worldid]

        for o in range(wrap_begin, wrap_end):
            if geodesics[o].status == WrapStatus.UNINITIALIZED:
                geodesics[o] = init_geodesic(wrap_hint_on_surface_S[o])

        # Solve relative to the muscle's first path point: in float32, ground coordinates far from the origin (e.g. a
        # runner 15 m down the track) carry ~1e-6 m roundoff, which over a short straight line between an obstacle
        # and a path point exceeds the smoothness tolerance in angle, so the solver never converges
        origin = site_pos_G_in[worldid, pts_adr]
        length = float(0.0)
        velocity = float(0.0)
        smoothness = float(0.0)
        seg_begin = wrap_begin
        for s in range(muscle_pts_num[muscle_id] - 1):
            seg_end = segment_obstacle_end(seg_begin, wrap_end, wrap_segment, s)
            x0 = site_pos_G_in[worldid, pts_adr + s] - origin
            x1 = site_pos_G_in[worldid, pts_adr + s + 1] - origin
            if seg_end > seg_begin:
                seg_smoothness = solve_segment(wrap_type, wrap_bodyid, wrap_X_BS, wrap_size, smoothness_tol,
                                               max_iterations, accuracy, X_GB, origin, seg_begin, seg_end, x0, x1, geodesics)
                smoothness = wp.max(smoothness, seg_smoothness)

            # Sum the straight lines and the curves. The lengthening speed is that of the straight lines, whose
            # end points move with their bodies (the curves slide over the surfaces)
            x_prev = x0
            v_prev = site_vel_G_in[worldid, pts_adr + s]
            for o in range(seg_begin, seg_end):
                g = geodesics[o]
                if not is_in_contact(g):
                    continue
                X_GS = surface_frame(o, wrap_bodyid, wrap_X_BS, X_GB, origin)
                bodyid = wrap_bodyid[o]
                X_OB = body_frame(X_GB[bodyid], origin)
                x_P, x_Q = wp.transform_point(X_GS, g.p_P), wp.transform_point(X_GS, g.p_Q)
                v_P = contact_point_velocity(x_P, X_OB, V_GB[bodyid])
                line = x_P - x_prev
                dist = wp.length(line)
                length += dist + g.length
                if dist >= BOLT_MINVAL:
                    velocity += wp.dot(line / dist, v_P - v_prev)
                x_prev = x_Q
                v_prev = contact_point_velocity(x_Q, X_OB, V_GB[bodyid])
            line = x1 - x_prev
            dist = wp.length(line)
            length += dist
            if dist >= BOLT_MINVAL:
                velocity += wp.dot(line / dist, site_vel_G_in[worldid, pts_adr + s + 1] - v_prev)
            seg_begin = seg_end

        muscle_length_out[worldid, muscle_id] = length
        muscle_velocity_out[worldid, muscle_id] = velocity
        muscle_wrap_smoothness_out[worldid, muscle_id] = smoothness

    return compute_path_kernel


@wp.func
def apply_point_force(bodyid: int, x: wp.vec3, frc: wp.vec3, X_GB: wp.array(dtype=wp.transform), origin: wp.vec3,
                      body_F_out: wp.array(dtype=wp.spatial_vector)):
    """ A force at x (relative to the local origin) on a body, as a spatial force about the body origin """
    arm = x - (wp.transform_get_translation(X_GB[bodyid]) - origin)
    wp.atomic_add(body_F_out, bodyid, wp.spatial_vector(wp.cross(arm, frc), frc))


@wp.func
def apply_wrap_force_to_bodies(
        actuation: float,
        pts_adr: int,
        pts_num: int,
        wrap_begin: int,
        wrap_end: int,
        site_bodyid: wp.array(dtype=int),
        wrap_bodyid: wp.array(dtype=int),
        wrap_X_BS: wp.array(dtype=wp.transform),
        wrap_segment: wp.array(dtype=int),
        X_GB: wp.array(dtype=wp.transform),
        site_pos_G: wp.array(dtype=wp.vec3),
        geodesics: wp.array(dtype=WrapGeodesic),
        body_F_out: wp.array(dtype=wp.spatial_vector),
):
    """
    The tension pulls each path point along the straight lines leaving it, and each obstacle along the curve's
    tangents at its contact points
    """
    # Relative to the first path point, as the path solve (see compute_path_kernel)
    origin = site_pos_G[pts_adr]
    seg_begin = wrap_begin
    for s in range(pts_num - 1):
        seg_end = segment_obstacle_end(seg_begin, wrap_end, wrap_segment, s)
        site0, site1 = pts_adr + s, pts_adr + s + 1
        x0, x1 = site_pos_G[site0] - origin, site_pos_G[site1] - origin
        x_prev = x0
        first_line = bool(True)
        for o in range(seg_begin, seg_end):
            g = geodesics[o]
            if not is_in_contact(g):
                continue
            X_GS = surface_frame(o, wrap_bodyid, wrap_X_BS, X_GB, origin)
            R = wp.transform_get_rotation(X_GS)
            x_P, x_Q = wp.transform_point(X_GS, g.p_P), wp.transform_point(X_GS, g.p_Q)
            if first_line:
                apply_point_force(site_bodyid[site0], x0, actuation * wp.normalize(x_P - x_prev), X_GB, origin,
                                  body_F_out)
                first_line = False
            apply_point_force(wrap_bodyid[o], x_P, -actuation * wp.quat_rotate(R, g.t_P), X_GB, origin, body_F_out)
            apply_point_force(wrap_bodyid[o], x_Q, actuation * wp.quat_rotate(R, g.t_Q), X_GB, origin, body_F_out)
            x_prev = x_Q
        line = x1 - x_prev
        if wp.length(line) >= BOLT_MINVAL:
            direction = wp.normalize(line)
            if first_line:
                apply_point_force(site_bodyid[site0], x0, actuation * direction, X_GB, origin, body_F_out)
            apply_point_force(site_bodyid[site1], x1, -actuation * direction, X_GB, origin, body_F_out)
        seg_begin = seg_end


@wp.kernel
def _apply_wrap_force_kernel(
        # Model:
        muscle_pts_adr: wp.array(dtype=int),
        muscle_pts_num: wp.array(dtype=int),
        muscle_wrap_adr: wp.array(dtype=int),
        muscle_wrap_num: wp.array(dtype=int),
        site_bodyid: wp.array(dtype=int),
        wrap_bodyid: wp.array(dtype=int),
        wrap_X_BS: wp.array(dtype=wp.transform),
        wrap_segment: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        muscle_actuation_in: wp.array2d(dtype=float),
        mob_X_GB_in: wp.array2d(dtype=wp.transform),
        site_pos_G_in: wp.array2d(dtype=wp.vec3),
        wrap_geodesic_in: wp.array2d(dtype=WrapGeodesic),
        # In:
        muscle_wrap_group: wp.array(dtype=int),
        # Data out:
        body_F_muscle_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, nodeid = wp.tid()
    if integration_done_in[worldid]:
        return
    muscle_id = muscle_wrap_group[nodeid]
    wrap_begin = muscle_wrap_adr[muscle_id]
    apply_wrap_force_to_bodies(
        muscle_actuation_in[worldid, muscle_id], muscle_pts_adr[muscle_id], muscle_pts_num[muscle_id], wrap_begin,
        wrap_begin + muscle_wrap_num[muscle_id], site_bodyid, wrap_bodyid, wrap_X_BS, wrap_segment,
        mob_X_GB_in[worldid], site_pos_G_in[worldid], wrap_geodesic_in[worldid], body_F_muscle_out[worldid]
    )


@wp.kernel
def _apply_unit_wrap_force_one_muscle_kernel(
        # Model:
        muscle_pts_adr: wp.array(dtype=int),
        muscle_pts_num: wp.array(dtype=int),
        muscle_wrap_adr: wp.array(dtype=int),
        muscle_wrap_num: wp.array(dtype=int),
        site_bodyid: wp.array(dtype=int),
        wrap_bodyid: wp.array(dtype=int),
        wrap_X_BS: wp.array(dtype=wp.transform),
        wrap_segment: wp.array(dtype=int),
        # Data in:
        mob_X_GB_in: wp.array2d(dtype=wp.transform),
        site_pos_G_in: wp.array2d(dtype=wp.vec3),
        wrap_geodesic_in: wp.array2d(dtype=WrapGeodesic),
        # In:
        muscle_id: int,
        # Data out:
        body_F_muscle_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid = wp.tid()
    wrap_begin = muscle_wrap_adr[muscle_id]
    apply_wrap_force_to_bodies(
        1.0, muscle_pts_adr[muscle_id], muscle_pts_num[muscle_id], wrap_begin, wrap_begin + muscle_wrap_num[muscle_id],
        site_bodyid, wrap_bodyid, wrap_X_BS, wrap_segment, mob_X_GB_in[worldid], site_pos_G_in[worldid],
        wrap_geodesic_in[worldid], body_F_muscle_out[worldid]
    )


@wp.kernel
def _reset_wrap_geodesic_kernel(
        # Data in:
        world_reset_in: wp.array(dtype=bool),
        # Data out:
        wrap_geodesic_out: wp.array2d(dtype=WrapGeodesic),
):
    worldid, wrapid = wp.tid()
    if world_reset_in[worldid]:
        wrap_geodesic_out[worldid, wrapid] = WrapGeodesic()


@event_scope
def reset_wrap_geodesics(m: Model, d: Data):
    """ Reset worlds restart their wrapping solution from the obstacles' contact hints """
    if m.nwrap:
        launch(_reset_wrap_geodesic_kernel, dim=(d.nworld, m.nwrap), inputs=[d.world_reset],
               outputs=[d.wrap_geodesic])


@event_scope
def muscle_wrap_path(m: Model, d: Data, length_out: wp.array2d | None = None,
                     velocity_out: wp.array2d | None = None, geodesic_out: wp.array2d | None = None,
                     smoothness_out: wp.array2d | None = None):
    """
    Computes the muscle path length, velocity and smoothness for paths wrapping over obstacles, into d.muscle_length,
    d.muscle_velocity and d.muscle_wrap_smoothness unless other outputs are given. The geodesics are solved in place,
    warm started from their current values: d.wrap_geodesic unless geodesic_out is given
    """
    if not m.muscle_wrap_group_tuple:
        return
    launch(
        _wrap_kernels(m.wrap_max_per_segment),
        dim=(d.nworld, m.muscle_wrap_group.size),
        inputs=[
            m.muscle_pts_adr, m.muscle_pts_num, m.muscle_wrap_adr, m.muscle_wrap_num, m.wrap_type, m.wrap_bodyid,
            m.wrap_X_BS, m.wrap_size, m.wrap_segment, m.wrap_hint_on_surface_S,
            m.opt.wrap_smoothness_tol, m.opt.wrap_max_iterations, m.opt.wrap_geodesic_accuracy,
            d.integration_done, d.mob_X_GB, d.body_V_GB, d.site_pos_G, d.site_vel_G,
            m.muscle_wrap_group
        ],
        outputs=[d.wrap_geodesic if geodesic_out is None else geodesic_out,
                 d.muscle_length if length_out is None else length_out,
                 d.muscle_velocity if velocity_out is None else velocity_out,
                 d.muscle_wrap_smoothness if smoothness_out is None else smoothness_out],
    )


@event_scope
def apply_muscle_force_wrap(m: Model, d: Data):
    if not m.muscle_wrap_group_tuple:
        return
    launch(
        _apply_wrap_force_kernel,
        dim=(d.nworld, m.muscle_wrap_group.size),
        inputs=[
            m.muscle_pts_adr, m.muscle_pts_num, m.muscle_wrap_adr, m.muscle_wrap_num, m.site_bodyid, m.wrap_bodyid,
            m.wrap_X_BS, m.wrap_segment,
            d.integration_done, d.muscle_actuation, d.mob_X_GB, d.site_pos_G, d.wrap_geodesic,
            m.muscle_wrap_group
        ],
        outputs=[d.body_F_muscle],
    )


@event_scope
def apply_unit_force_one_muscle(m: Model, d: Data, body_F_out: wp.array2d, muscle_id: int):
    """ Body forces for a unit muscle tension along the last computed wrapping path """
    launch(
        _apply_unit_wrap_force_one_muscle_kernel,
        dim=(d.nworld,),
        inputs=[
            m.muscle_pts_adr, m.muscle_pts_num, m.muscle_wrap_adr, m.muscle_wrap_num, m.site_bodyid, m.wrap_bodyid,
            m.wrap_X_BS, m.wrap_segment,
            d.mob_X_GB, d.site_pos_G, d.wrap_geodesic, muscle_id
        ],
        outputs=[body_F_out],
    )


# --- Sampling the paths for rendering ---
WRAP_RENDER_SAMPLES = 12  # points along each curve, including its contact points
WRAP_RENDER_CYLINDER_HALF_LENGTH = 3.0  # in radii: cylinder obstacles are infinite, but drawn this long


def wrap_render_layout(m: Model) -> tuple[list[int], list[int]]:
    """
    Start and number of points of each wrapping muscle (in muscle_wrap_group order) in the polyline buffer of
    sample_wrap_paths: its path points, with WRAP_RENDER_SAMPLES points per obstacle in between
    """
    pts_num, wrap_num = m.muscle_pts_num.numpy(), m.muscle_wrap_num.numpy()
    num = [int(pts_num[i] + WRAP_RENDER_SAMPLES * wrap_num[i]) for i in m.muscle_wrap_group_tuple]
    adr = [sum(num[:k]) for k in range(len(num))]
    return adr, num


@wp.kernel
def _sample_wrap_path_kernel(
        # Model:
        muscle_pts_adr: wp.array(dtype=int),
        muscle_pts_num: wp.array(dtype=int),
        muscle_wrap_adr: wp.array(dtype=int),
        muscle_wrap_num: wp.array(dtype=int),
        wrap_type: wp.array(dtype=int),
        wrap_bodyid: wp.array(dtype=int),
        wrap_X_BS: wp.array(dtype=wp.transform),
        wrap_size: wp.array(dtype=wp.vec3),
        wrap_segment: wp.array(dtype=int),
        accuracy: float,
        # Data in:
        mob_X_GB_in: wp.array2d(dtype=wp.transform),
        site_pos_G_in: wp.array2d(dtype=wp.vec3),
        wrap_geodesic_in: wp.array2d(dtype=WrapGeodesic),
        # In:
        muscle_wrap_group: wp.array(dtype=int),
        render_adr: wp.array(dtype=int),
        worlds: wp.array(dtype=int),
        # Out:
        points_out: wp.array2d(dtype=wp.vec3),
):
    w, nodeid = wp.tid()
    worldid = worlds[w]
    muscle_id = muscle_wrap_group[nodeid]
    pts_adr = muscle_pts_adr[muscle_id]
    pts_num = muscle_pts_num[muscle_id]
    wrap_begin = muscle_wrap_adr[muscle_id]
    wrap_end = wrap_begin + muscle_wrap_num[muscle_id]
    X_GB = mob_X_GB_in[worldid]
    k = render_adr[nodeid]

    seg_begin = wrap_begin
    for s in range(pts_num - 1):
        seg_end = segment_obstacle_end(seg_begin, wrap_end, wrap_segment, s)
        last = site_pos_G_in[worldid, pts_adr + s]
        points_out[w, k] = last
        k += 1
        for o in range(seg_begin, seg_end):
            g = wrap_geodesic_in[worldid, o]
            X_GS = surface_frame(o, wrap_bodyid, wrap_X_BS, X_GB, wp.vec3(0.0))
            for i in range(WRAP_RENDER_SAMPLES):
                # Obstacles that are not in contact collapse onto the previous point
                if is_in_contact(g):
                    p = g.p_Q
                    if i == 0:
                        p = g.p_P
                    elif i < WRAP_RENDER_SAMPLES - 1:
                        length = g.length * float(i) / float(WRAP_RENDER_SAMPLES - 1)
                        p = wrap_geometry.shoot_geodesic(wrap_type[o], wrap_size[o], g, g.p_P, g.t_P, length,
                                                         accuracy).p_Q
                    last = wp.transform_point(X_GS, p)
                points_out[w, k] = last
                k += 1
        seg_begin = seg_end
    points_out[w, k] = site_pos_G_in[worldid, pts_adr + pts_num - 1]


def sample_wrap_paths(m: Model, d: Data, render_adr: wp.array, worlds: wp.array, points_out: wp.array2d):
    """
    Polylines along the wrapping paths of the given worlds, laid out as wrap_render_layout. The paths are solved at
    the current positions first (e.g. after only realizing positions), on a copy of the geodesics
    """
    if not m.muscle_wrap_group_tuple:
        return
    scratch = wp.empty((d.nworld, m.nmuscle), dtype=float, device=d.muscle_length.device)
    geodesics = wp.clone(d.wrap_geodesic)
    muscle_wrap_path(m, d, length_out=scratch, velocity_out=scratch, geodesic_out=geodesics, smoothness_out=scratch)
    wp.launch(
        _sample_wrap_path_kernel,
        dim=(worlds.shape[0], m.muscle_wrap_group.size),
        inputs=[
            m.muscle_pts_adr, m.muscle_pts_num, m.muscle_wrap_adr, m.muscle_wrap_num, m.wrap_type, m.wrap_bodyid,
            m.wrap_X_BS, m.wrap_size, m.wrap_segment, m.opt.wrap_geodesic_accuracy,
            d.mob_X_GB, d.site_pos_G, geodesics,
            m.muscle_wrap_group, render_adr, worlds,
        ],
        outputs=[points_out],
        device=points_out.device,
    )
