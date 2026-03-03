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
def transform_spatial_inertia(t: wp.transform, I: wp.spatial_matrix):
    """
    Transform a spatial inertia tensor to a new coordinate frame.
    """
    t_inv = wp.transform_inverse(t)

    q = wp.transform_get_rotation(t_inv)
    p = wp.transform_get_translation(t_inv)

    r1 = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    r2 = wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0))
    r3 = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))

    R = wp.matrix_from_cols(r1, r2, r3)
    S = wp.skew(p) @ R

    T = wp.spatial_matrix(
        R[0, 0], R[0, 1], R[0, 2], 0.0,     0.0,     0.0,
        R[1, 0], R[1, 1], R[1, 2], 0.0,     0.0,     0.0,
        R[2, 0], R[2, 1], R[2, 2], 0.0,     0.0,     0.0,
        S[0, 0], S[0, 1], S[0, 2], R[0, 0], R[0, 1], R[0, 2],
        S[1, 0], S[1, 1], S[1, 2], R[1, 0], R[1, 1], R[1, 2],
        S[2, 0], S[2, 1], S[2, 2], R[2, 0], R[2, 1], R[2, 2],
    )

    return wp.mul(wp.mul(wp.transpose(T), I), T)

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
        e0, ne3, e2,
        e3, e0, ne1,
        ne2, e1, e0,
        ne1, ne2, ne3
    )


@wp.func
def calc_unnormalized_quaternion_N_inv(q: wp.quat) -> types.mat34:
    """ N_inv*q_dot = u """
    e = 2.0 * q
    e0, e1, e2, e3 = e.w, e.x, e.y, e.z
    ne1, ne2, ne3 = -e1, -e2, -e3
    return types.mat34(
        e0, e3, ne2, ne1,
        ne3, e0, e1, ne2,
        ne1, e2, e0, ne3,
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
