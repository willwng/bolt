import warp as wp

from . import types
from .consts import MAX_POLY_NUM_DOFS


@wp.func
def sqr(x: float) -> float:
    return x * x


@wp.func
def trans_from_three_shift_axes(
        q0: float, q1: float, q2: float,
        ax0: wp.vec3, ax1: wp.vec3, ax2: wp.vec3
) -> wp.vec3:
    p0 = q0 * ax0
    p1 = q1 * ax1
    p2 = q2 * ax2
    return p0 + p1 + p2


@wp.func
def quat_from_three_angle_axes(
        q0: float, q1: float, q2: float,
        ax0: wp.vec3, ax1: wp.vec3, ax2: wp.vec3
) -> wp.quat:
    qloc0 = wp.quat_from_axis_angle(ax0, q0)
    qloc1 = wp.quat_from_axis_angle(ax1, q1)
    qloc2 = wp.quat_from_axis_angle(ax2, q2)
    return qloc0 * qloc1 * qloc2


@wp.func
def quat_from_xyz(q0: float, q1: float, q2: float) -> wp.quat:
    return quat_from_three_angle_axes(
        q0, q1, q2,
        wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 1.0)
    )


@wp.func
def max_err(x: float, y: float) -> float:
    """Returns the maximum of x and y, treating NaN and Inf as higher priority than any number."""
    if wp.isnan(x) or wp.isnan(y):
        return wp.nan
    if wp.isinf(x) or wp.isinf(y):
        return wp.inf
    return wp.max(x, y)


@wp.func
def quat_normalize_in_place(q: wp.array(dtype=float), adr: int):
    quat = wp.quat(q[adr], q[adr + 1], q[adr + 2], q[adr + 3])
    quat = wp.normalize(quat)
    q[adr] = quat[0]
    q[adr + 1] = quat[1]
    q[adr + 2] = quat[2]
    q[adr + 3] = quat[3]


@wp.func
def quat_swing_twist(q: wp.quat, axis: wp.vec3) -> tuple[wp.quat, wp.quat]:
    """ Decomposes the rotation q (q = q_swing * q_twist) into a swing and a twist about the given axis """
    # based on bullet physics
    q_axis = wp.vec3(q.x, q.y, q.z)
    p = wp.dot(q_axis, axis)

    twist_axis = p * axis
    out_twist = wp.quat(twist_axis.x, twist_axis.y, twist_axis.z, q.w)
    out_twist = wp.normalize(out_twist)
    out_swing = q * wp.quat_inverse(out_twist)
    return out_swing, out_twist


@wp.func
def apply_force_to_body_point(X_GB: wp.transform, point_in_b: wp.vec3, force_in_G: wp.vec3) -> wp.spatial_vector:
    R_GB = wp.transform_get_rotation(X_GB)
    trq = wp.cross(wp.quat_rotate(R_GB, point_in_b), force_in_G)
    return wp.spatial_vector(trq, force_in_G)


@wp.func
def find_station_at_ground_point(X_GB: wp.transform, location_in_G: wp.vec3) -> wp.vec3:
    return wp.transform_point(wp.transform_inverse(X_GB), location_in_G)


@wp.func
def express_vector_in_ground_frame(X_GB: wp.transform, vec_in_B: wp.vec3) -> wp.vec3:
    """
    Re-express a vector expressed in this body B's frame into the same vector in
    G, by applying only a rotation.
    """
    R_GB = wp.transform_get_rotation(X_GB)
    return wp.quat_rotate(R_GB, vec_in_B)


@wp.func
def find_station_velocity_in_ground(X_GB: wp.transform, V_GB: wp.spatial_vector, station_on_B: wp.vec3) -> wp.vec3:
    w, v = wp.spatial_top(V_GB), wp.spatial_bottom(V_GB)  # in G
    r = express_vector_in_ground_frame(X_GB, station_on_B)
    return v + wp.cross(w, r)


@wp.func
def rotate_spatial_vec(q: wp.quat, x: wp.spatial_vector) -> wp.spatial_vector:
    w, v = wp.spatial_top(x), wp.spatial_bottom(x)
    w_rot, v_rot = wp.quat_rotate(q, w), wp.quat_rotate(q, v)
    return wp.spatial_vector(w_rot, v_rot)


@wp.func
def reexpress_inertia(inert: wp.mat33, q: wp.quat) -> wp.mat33:
    R = wp.quat_to_matrix(q)
    return wp.transpose(R) @ inert @ R


@wp.func
def multiply_phi(phi: wp.vec3, sv: wp.spatial_vector) -> wp.spatial_vector:
    w, v = wp.spatial_top(sv), wp.spatial_bottom(sv)
    return wp.spatial_vector(w + wp.cross(phi, v), v)


@wp.func
def multiply_phi_transpose(phi: wp.vec3, sv: wp.spatial_vector) -> wp.spatial_vector:
    w, v = wp.spatial_top(sv), wp.spatial_bottom(sv)
    return wp.spatial_vector(w, v + wp.cross(w, phi))


@wp.func
def multiply_spatial_inertia(Mk_G: types.SpatialInertia, sv: wp.spatial_vector) -> wp.spatial_vector:
    w, v = wp.spatial_top(sv), wp.spatial_bottom(sv)
    m, p, G = Mk_G.m, Mk_G.p, Mk_G.G
    return m * wp.spatial_vector(G @ w + wp.cross(p, v), v - wp.cross(p, w))


@wp.func
def store_mat66(dest: wp.array(dtype=wp.spatial_vector), M: wp.spatial_matrix, adr: int, dofnum: int):
    MT = wp.transpose(M)
    for i in range(dofnum):
        dest[adr + i] = MT[i]
    return


@wp.func
def load_spatial_vec(src: wp.array(dtype=float), adr: int, dofnum: int) -> wp.spatial_vector:
    sv = wp.spatial_vector()
    for i in range(dofnum):
        sv[i] = src[adr + i]
    return sv


@wp.func
def load_mat66(src: wp.array(dtype=wp.spatial_vector), adr: int, dofnum: int) -> wp.spatial_matrix:
    M = wp.spatial_matrix(0.0)
    for i in range(dofnum):
        M[i] = src[adr + i]
    return wp.transpose(M)  # We stored as columns, but filled up the matrix row by row


@wp.func
def print_mat33(M: wp.mat33):
    for i in range(3):
        for j in range(3):
            wp.printf("%f ", M[i, j])
        wp.printf("\n")


@wp.func
def invert_upper_left(D: wp.spatial_matrix, dofnum: int) -> wp.spatial_matrix:
    """ D is a 6x6 matrix but only the top-left dofnum x dofnum block is actually used """
    ret = wp.spatial_matrix(0.0)
    if dofnum == 0:
        return ret
    elif dofnum == 1:
        ret[0, 0] = 1.0 / D[0, 0]
    elif dofnum == 2:
        D_upper_22 = wp.mat22(D[0, 0], D[0, 1], D[1, 0], D[1, 1])
        D_upper_inv_22 = wp.inverse(D_upper_22)
        ret[0, 0] = D_upper_inv_22[0, 0]
        ret[0, 1] = D_upper_inv_22[0, 1]
        ret[1, 0] = D_upper_inv_22[1, 0]
        ret[1, 1] = D_upper_inv_22[1, 1]
    elif dofnum == 3:
        D_upper_33 = wp.mat33(
            D[0, 0], D[0, 1], D[0, 2],
            D[1, 0], D[1, 1], D[1, 2],
            D[2, 0], D[2, 1], D[2, 2],
        )
        D_upper_inv_33 = wp.inverse(D_upper_33)
        for i in range(3):
            for j in range(3):
                ret[i, j] = D_upper_inv_33[i, j]

    elif dofnum == 4:
        D_upper_44 = wp.mat44(
            D[0, 0], D[0, 1], D[0, 2], D[0, 3],
            D[1, 0], D[1, 1], D[1, 2], D[1, 3],
            D[2, 0], D[2, 1], D[2, 2], D[2, 3],
            D[3, 0], D[3, 1], D[3, 2], D[3, 3],
        )
        D_upper_inv_44 = wp.inverse(D_upper_44)
        for i in range(4):
            for j in range(4):
                ret[i, j] = D_upper_inv_44[i, j]
    elif dofnum == 5:
        assert False  # TODO
    else:
        assert dofnum == 6
        return invert_mat66(D)
    return ret


@wp.func
def extract_33_blocks(m: wp.spatial_matrix) -> tuple[wp.mat33, wp.mat33, wp.mat33, wp.mat33]:
    """Extracts 3x3 blocks A, B, C, D from a 6x6 matrix M = [[A, B], [C, D]]"""
    A = wp.mat33(
        m[0, 0], m[0, 1], m[0, 2],
        m[1, 0], m[1, 1], m[1, 2],
        m[2, 0], m[2, 1], m[2, 2],
    )
    B = wp.mat33(
        m[0, 3], m[0, 4], m[0, 5],
        m[1, 3], m[1, 4], m[1, 5],
        m[2, 3], m[2, 4], m[2, 5],
    )
    C = wp.mat33(
        m[3, 0], m[3, 1], m[3, 2],
        m[4, 0], m[4, 1], m[4, 2],
        m[5, 0], m[5, 1], m[5, 2],
    )
    D = wp.mat33(
        m[3, 3], m[3, 4], m[3, 5],
        m[4, 3], m[4, 4], m[4, 5],
        m[5, 3], m[5, 4], m[5, 5],
    )
    return A, B, C, D


@wp.func
def invert_mat66(m: wp.spatial_matrix) -> wp.spatial_matrix:
    # Use block matrix inversion: M = [[A, B], [C, D]]
    # where A, B, C, D are 3x3 blocks
    # M^-1 = [[A-BD^-1C)^-1, -(A-BD^-1C)^-1 BD^-1], [−D^-1C(A−BD^-1C)^-1, D^-1+D^-1C(A−BD^-1C)^-1 BD^-1]]

    A, B, C, D = extract_33_blocks(m)

    D_inv = wp.inverse(D)
    BD_inv = B * D_inv  # B @ D^-1
    BD_inv_C = BD_inv * C  # B @ D^-1 @ C
    schur = A - BD_inv_C  # Schur complement: A - B D^-1 C
    schur_inv = wp.inverse(schur)  # (A - B D^-1 C)^-1

    D_inv_C = D_inv * C  # D^-1 @ C

    # Top-left block
    TL = schur_inv

    # Top-right block: -schur_inv @ B @ D^-1
    TR = wp.mat33(0.0) - schur_inv * BD_inv

    # Bottom-left block: -D^-1 @ C @ schur_inv
    BL = wp.mat33(0.0) - D_inv_C * schur_inv

    # Bottom-right block: D^-1 + D^-1 @ C @ schur_inv @ B @ D^-1
    BR = D_inv + D_inv_C * schur_inv * BD_inv

    return wp.spatial_matrix(
        TL[0, 0], TL[0, 1], TL[0, 2], TR[0, 0], TR[0, 1], TR[0, 2],
        TL[1, 0], TL[1, 1], TL[1, 2], TR[1, 0], TR[1, 1], TR[1, 2],
        TL[2, 0], TL[2, 1], TL[2, 2], TR[2, 0], TR[2, 1], TR[2, 2],
        BL[0, 0], BL[0, 1], BL[0, 2], BR[0, 0], BR[0, 1], BR[0, 2],
        BL[1, 0], BL[1, 1], BL[1, 2], BR[1, 0], BR[1, 1], BR[1, 2],
        BL[2, 0], BL[2, 1], BL[2, 2], BR[2, 0], BR[2, 1], BR[2, 2],
    )


@wp.func
def spatial_inertia_to_articulated_inertia(Mk_G: types.SpatialInertia) -> types.ArticulatedInertia:
    m, p, G = Mk_G.m, Mk_G.p, Mk_G.G
    return types.ArticulatedInertia(
        M=m * wp.identity(3, dtype=float),
        J=m * G,
        F=wp.skew(m * p)
    )


@wp.func
def articulated_inertia_mul(P: types.ArticulatedInertia, sv: wp.spatial_vector) -> wp.spatial_vector:
    M, J, F = P.M, P.J, P.F
    w, v = wp.spatial_top(sv), wp.spatial_bottom(sv)
    return wp.spatial_vector(J @ w + F @ v, wp.transpose(F) @ w + M * v)


@wp.func
def articulated_inertia_shift(P: types.ArticulatedInertia, s: wp.vec3) -> types.ArticulatedInertia:
    """
    Rigid-shift the origin of this Articulated Body Inertia P by a
    shift vector -s to produce a new ABI P'. The calculation is
    <pre>
    P' =  [ J'  F' ]  =  [ 1  sx ] [ J  F ] [ 1  0 ]
          [~F'  M  ]     [ 0  1  ] [~F  M ] [-sx 1 ]
    """
    M, J, F = P.M, P.J, P.F
    sx = wp.skew(s)
    sxM = sx * M

    Fp = F + sxM
    Jp = J - F * sx + sx * wp.transpose(F) - (sxM) * sx

    return types.ArticulatedInertia(M, Jp, Fp)


@wp.func
def symmetrize(M: wp.mat33) -> wp.mat33:
    """ Ensures numerical symmetry of a matrix by averaging it with its transpose. """
    return 0.5 * (M + wp.transpose(M))


@wp.func
def symmetrize_articulated_inertia(P: types.ArticulatedInertia) -> types.ArticulatedInertia:
    """ Ensures numerical symmetry of the inertia and mass moment matrices. """
    return types.ArticulatedInertia(
        M=symmetrize(P.M),
        J=symmetrize(P.J),
        F=P.F,  # not necessarily symmetric
    )


@wp.func
def articulated_inertia_add(P1: types.ArticulatedInertia, P2: types.ArticulatedInertia) -> types.ArticulatedInertia:
    """ Returns P1 + P2 """
    M1, J1, F1 = P1.M, P1.J, P1.F
    M2, J2, F2 = P2.M, P2.J, P2.F
    return types.ArticulatedInertia(
        M=M1 + M2,
        J=J1 + J2,
        F=F1 + F2,
    )


@wp.func
def articulated_inertia_sub(P1: types.ArticulatedInertia, P2: types.ArticulatedInertia) -> types.ArticulatedInertia:
    """ Returns P1 - P2 """
    M1, J1, F1 = P1.M, P1.J, P1.F
    M2, J2, F2 = P2.M, P2.J, P2.F
    return types.ArticulatedInertia(
        M=M1 - M2,
        J=J1 - J2,
        F=F1 - F2,
    )


@wp.func
def orthogonals(a: wp.vec3):
    y = wp.vec3(0.0, 1.0, 0.0)
    z = wp.vec3(0.0, 0.0, 1.0)
    b = wp.where((-0.5 < a[1]) and (a[1] < 0.5), y, z)
    b = b - a * wp.dot(a, b)
    b = wp.normalize(b)
    if wp.length(a) == 0.0:
        b = wp.vec3(0.0, 0.0, 0.0)
    c = wp.cross(a, b)

    return b, c


@wp.func
def make_frame(a: wp.vec3):
    a = wp.normalize(a)
    b, c = orthogonals(a)

    # fmt: off
    return wp.mat33(
        a.x, a.y, a.z,
        b.x, b.y, b.z,
        c.x, c.y, c.z
    )
    # fmt: on


@wp.func
def upper_trid_index(n: int, i: int, j: int) -> int:
    """Returns index of a_ij = a_ji in upper triangular matrix (including diagonal)."""
    if j < i:
        i, j = j, i
    return (i * (2 * n - i - 1)) // 2 + j


@wp.func
def calc_unnormalized_quaternion_N(q: wp.quat) -> types.mat43:
    """
    N*u = q_dot. See https://arxiv.org/pdf/0811.2889
    Note: warp uses quat conventions (x,y,z,w)
    """
    e = q / 2.0
    e0, e1, e2, e3 = e.w, e.x, e.y, e.z
    ne1, ne2, ne3 = -e1, -e2, -e3
    return types.mat43(
        e0, e3, ne2,
        ne3, e0, e1,
        e2, ne1, e0,
        ne1, ne2, ne3,
    )


@wp.func
def calc_unnormalized_quaternion_N_inv(q: wp.quat) -> types.mat34:
    """ N_inv*q_dot = u """
    e = 2.0 * q
    e0, e1, e2, e3 = e.w, e.x, e.y, e.z
    ne1, ne2, ne3 = -e1, -e2, -e3
    return types.mat34(
        e0, ne3, e2, ne1,
        e3, e0, ne1, ne2,
        ne2, e1, e0, ne3,
    )


@wp.func
def mul_body_xyz_N(cosxy: wp.vec2, sinxy: wp.vec2, oocosy: float, w: wp.vec3) -> wp.vec3:
    s0, c0 = sinxy[0], cosxy[0]
    s1 = sinxy[1]
    w0, w1, w2 = w[0], w[1], w[2]

    t = (s0 * w1 - c0 * w2) * oocosy
    return wp.vec3(w0 + t * s1, c0 * w1 + s0 * w2, -t)


@wp.func
def mul_body_xyz_NT(cosxy: wp.vec2, sinxy: wp.vec2, oocosy: float, q: wp.vec3) -> wp.vec3:
    s0, c0 = sinxy[0], cosxy[0]
    s1 = sinxy[1]
    q0, q1, q2 = q[0], q[1], q[2]

    t = (q0 * s1 - q2) * oocosy
    return wp.vec3(q0, c0 * q1 + t * s0, s0 * q1 - t * c0)


@wp.func
def mul_body_xyz_N_inv(cosxy: wp.vec2, sinxy: wp.vec2, qdot: wp.vec3) -> wp.vec3:
    s0, c0 = sinxy[0], cosxy[0]
    s1, c1 = sinxy[1], cosxy[1]
    q0, q1, q2 = qdot[0], qdot[1], qdot[2]
    c1q2 = c1 * q2
    return wp.vec3(q0 + s1 * q2, c0 * q1 - s0 * c1q2, s0 * q1 + c0 * c1q2)


@wp.func
def step_up(x: float) -> float:
    """
    Interpolate smoothly from 0 to 1 as the argument goes from 0
    with first and second derivatives zero at either end
    """
    return x * x * x * (10.0 + x * (6.0 * x - 15.0))


@wp.func
def step_function(
        x: float,
        start_time: float,
        end_time: float,
        start_value: float,
        end_value: float,
) -> float:
    """ A smooth step function that transitions from start_value to end_value between start_time and end_time. """
    if x <= start_time:
        return start_value
    elif x >= end_time:
        return end_value
    else:
        t = (x - start_time) / (end_time - start_time)
        smooth_t = t * t * (3.0 - 2.0 * t)
        return start_value + smooth_t * (end_value - start_value)


@wp.func
def fast_pow_pos(base: float, exp: int) -> float:
    """Fast integer exponentiation for positive, integer exponents."""
    result = float(1.0)
    b = base
    e = exp
    while e > 0:
        if (e & 1) != 0:
            result = result * b
        b = b * b
        e = e >> 1

    return result


@wp.func
def mul_but_i(v: types.PolyVec, i: int) -> float:
    """Multiplies all elements of v except for index i."""
    result = float(1.0)
    for idx in range(wp.static(MAX_POLY_NUM_DOFS)):
        if idx != i:
            result *= v[idx]
    return result


@wp.func
def poly_vec_from_eval(poly_eval: types.PolyEval) -> types.PolyVec:
    """Converts a types.PolyEval to a types.PolyVec by taking the derivative components."""
    ret = types.PolyVec(0.0)
    for i in range(wp.static(MAX_POLY_NUM_DOFS)):
        ret[i] = poly_eval[i + 1]
    return ret


@wp.func
def evaluate_term_and_deriv(
        coeff: float,
        exp: types.PolyInts,
        q: types.PolyVec,
) -> types.PolyEval:
    ret = types.PolyEval(0.0)

    # Compute term, cache powers
    term = float(coeff)
    cache = types.PolyVec(0.0)
    for i in range(wp.static(MAX_POLY_NUM_DOFS)):
        p = fast_pow_pos(q[i], exp[i])
        cache[i] = p
        term *= p
    ret[0] = term

    # Compute derivatives
    for i in range(wp.static(MAX_POLY_NUM_DOFS)):
        if exp[i] > 0:
            ret[i + 1] = coeff * float(exp[i]) * fast_pow_pos(q[i], exp[i] - 1) * mul_but_i(cache, i)

    return ret
