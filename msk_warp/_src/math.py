# Copyright 2025 The Newton Developers
# Modified for MSKWarp by Will Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Any, Tuple

import warp as wp

from msk_warp._src import types
from msk_warp._src import consts


@wp.func
def sqr(x: float) -> float:
    return x * x


@wp.func
def quat_from_xyz(q0: float, q1: float, q2: float) -> wp.quat:
    qloc0 = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), q0)
    qloc1 = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), q1)
    qloc2 = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), q2)
    return qloc0 * qloc1 * qloc2


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
def rotate_spatial_vec(q: wp.quat, x: wp.spatial_vector) -> wp.spatial_vector:
    w, v = wp.spatial_top(x), wp.spatial_bottom(x)
    w_rot, v_rot = wp.quat_rotate(q, w), wp.quat_rotate(q, v)
    return wp.spatial_vector(w_rot, v_rot)


@wp.func
def reexpress_inertia(inert: wp.mat33, q: wp.quat) -> wp.mat33:
    R = wp.quat_to_matrix(q)
    return R @ inert @ wp.transpose(R)


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
    m, p, G = Mk_G.mass, Mk_G.offset, Mk_G.inertia
    return m * wp.spatial_vector(G @ w + wp.cross(p, v), v - wp.cross(p, w))


@wp.func
def invert_upper_left(D: types.mat66, dofnum: int) -> types.mat66:
    """ D is a 6x6 matrix but only the top-left dofnum x dofnum block is actually used """
    ret = types.mat66(0.0)
    if dofnum == 1:
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
def extract_33_blocks(m: types.mat66) -> Tuple[wp.mat33, wp.mat33, wp.mat33, wp.mat33]:
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
def invert_mat66(m: types.mat66) -> types.mat66:
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

    return types.mat66(
        TL[0, 0], TL[0, 1], TL[0, 2], TR[0, 0], TR[0, 1], TR[0, 2],
        TL[1, 0], TL[1, 1], TL[1, 2], TR[1, 0], TR[1, 1], TR[1, 2],
        TL[2, 0], TL[2, 1], TL[2, 2], TR[2, 0], TR[2, 1], TR[2, 2],
        BL[0, 0], BL[0, 1], BL[0, 2], BR[0, 0], BR[0, 1], BR[0, 2],
        BL[1, 0], BL[1, 1], BL[1, 2], BR[1, 0], BR[1, 1], BR[1, 2],
        BL[2, 0], BL[2, 1], BL[2, 2], BR[2, 0], BR[2, 1], BR[2, 2],
    )


@wp.func
def spatial_inertia_to_articulated_inertia(Mk_G: types.SpatialInertia) -> types.ArticulatedInertia:
    m, p, G = Mk_G.mass, Mk_G.offset, Mk_G.inertia
    mass_moment = m * p
    return types.ArticulatedInertia(
        M=m * wp.identity(3, dtype=float),
        J=m * G,
        F=wp.skew(mass_moment)
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
    sx_M = sx * M

    M_new = M
    J_new = J + sx * wp.transpose(F) - F * sx - sx_M * sx
    F_new = F + sx_M

    return types.ArticulatedInertia(M_new, J_new, F_new)



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
def atomic_add_articulated_inertia(P_dest: wp.array(dtype=types.ArticulatedInertia), idx: int,
                                   P: types.ArticulatedInertia):
    """ Atomically adds articulated inertia P to P_dest """
    wp.atomic_add(P_dest, idx, "M", P.M)
    wp.atomic_add(P_dest, idx, "J", P.J)
    wp.atomic_add(P_dest, idx, "F", P.F)


@wp.func
def transform_twist(t: wp.transform, x: wp.spatial_vector) -> wp.spatial_vector:
    """Transform a spatial twist between coordinate frames.

    For transform ``t = (R, p)`` and twist ``x = (w, v)``, the mapped twist is:

    .. math::
       w' = R w,\\quad v' = R v + p \\times w'

    Args:
        t: Rigid transform from source frame to destination frame.
        x: Spatial twist ``(angular, linear)`` expressed in the source frame.

    Returns:
        wp.spatial_vector: Twist expressed in the destination frame.
    """

    q = wp.transform_get_rotation(t)
    p = wp.transform_get_translation(t)

    w = wp.spatial_top(x)
    v = wp.spatial_bottom(x)

    w = wp.quat_rotate(q, w)
    v = wp.quat_rotate(q, v) + wp.cross(p, w)

    return wp.spatial_vector(w, v)


@wp.func
def inert_vec(i: types.vec10, v: wp.spatial_vector) -> wp.spatial_vector:
    """ Multiply spatial vector (rotation, translation) by spatial inertia matrix """
    return wp.spatial_vector(
        i[0] * v[0] + i[3] * v[1] + i[4] * v[2] - i[8] * v[4] + i[7] * v[5],
        i[3] * v[0] + i[1] * v[1] + i[5] * v[2] + i[8] * v[3] - i[6] * v[5],
        i[4] * v[0] + i[5] * v[1] + i[2] * v[2] - i[7] * v[3] + i[6] * v[4],
        i[8] * v[1] - i[7] * v[2] + i[9] * v[3],
        i[6] * v[2] - i[8] * v[0] + i[9] * v[4],
        i[7] * v[0] - i[6] * v[1] + i[9] * v[5],
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
def normalize_with_norm(x: Any):
    norm = wp.length(x)
    if norm == 0.0:
        return x, 0.0
    return x / norm, norm


@wp.func
def closest_segment_point(a: wp.vec3, b: wp.vec3, pt: wp.vec3) -> wp.vec3:
    """Returns the closest point on the a-b line segment to a point pt."""
    ab = b - a
    t = wp.dot(pt - a, ab) / (wp.dot(ab, ab) + 1e-6)
    return a + wp.clamp(t, 0.0, 1.0) * ab


@wp.func
def closest_segment_point_and_dist(a: wp.vec3, b: wp.vec3, pt: wp.vec3) -> \
        Tuple[wp.vec3, float]:
    """Returns closest point on the line segment and the distance squared."""
    closest = closest_segment_point(a, b, pt)
    dist = wp.dot((pt - closest), (pt - closest))
    return closest, dist


@wp.func
def upper_trid_index(n: int, i: int, j: int) -> int:
    """Returns index of a_ij = a_ji in upper triangular matrix (including diagonal)."""
    if j < i:
        i, j = j, i
    return (i * (2 * n - i - 1)) // 2 + j


@wp.func
def calc_unnormalized_quaternion_N(q: wp.quat) -> types.mat43:
    """ N*u = q_dot. See https://arxiv.org/pdf/0811.2889 """
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
    s0, s1, c0 = sinxy[0], sinxy[1], cosxy[0]
    w0, w1, w2 = w[0], w[1], w[2]
    t = (s0 * w1 - c0 * w2) * oocosy
    return wp.vec3(w0 + t * s1, c0 * w1 + s0 * w2, -t)


@wp.func
def mul_body_xyz_N_inv(cosxy: wp.vec2, sinxy: wp.vec2, qdot: wp.vec3) -> wp.vec3:
    s0, c0 = sinxy[0], cosxy[0]
    s1, c1 = sinxy[1], cosxy[1]
    q0, q1, q2 = qdot[0], qdot[1], qdot[2]
    c1q2 = c1 * q2
    return wp.vec3(q0 + s1 * q2, c0 * q1 - s0 * c1q2, s0 * q1 + c0 * c1q2)
