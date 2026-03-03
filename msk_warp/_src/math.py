# Copyright 2025 The Newton Developers
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
def inert_vec(i: types.vec10, v: wp.spatial_vector) -> wp.spatial_vector:
    """mju_mulInertVec: multiply 6D vector (rotation, translation) by 6D inertia matrix."""
    return wp.spatial_vector(
        i[0] * v[0] + i[3] * v[1] + i[4] * v[2] - i[8] * v[4] + i[7] * v[5],
        i[3] * v[0] + i[1] * v[1] + i[5] * v[2] + i[8] * v[3] - i[6] * v[5],
        i[4] * v[0] + i[5] * v[1] + i[2] * v[2] - i[7] * v[3] + i[6] * v[4],
        i[8] * v[1] - i[7] * v[2] + i[9] * v[3],
        i[6] * v[2] - i[8] * v[0] + i[9] * v[4],
        i[7] * v[0] - i[6] * v[1] + i[9] * v[5],
    )


@wp.func
def motion_cross(u: wp.spatial_vector, v: wp.spatial_vector) -> wp.spatial_vector:
    """Cross product of two motions."""
    u0 = wp.vec3(u[0], u[1], u[2])
    u1 = wp.vec3(u[3], u[4], u[5])
    v0 = wp.vec3(v[0], v[1], v[2])
    v1 = wp.vec3(v[3], v[4], v[5])

    ang = wp.cross(u0, v0)
    vel = wp.cross(u1, v0) + wp.cross(u0, v1)

    return wp.spatial_vector(ang, vel)


@wp.func
def motion_cross_force(v: wp.spatial_vector,
                       f: wp.spatial_vector) -> wp.spatial_vector:
    """Cross product of a motion and a force."""
    v0 = wp.vec3(v[0], v[1], v[2])
    v1 = wp.vec3(v[3], v[4], v[5])
    f0 = wp.vec3(f[0], f[1], f[2])
    f1 = wp.vec3(f[3], f[4], f[5])

    ang = wp.cross(v0, f0) + wp.cross(v1, f1)
    vel = wp.cross(v0, f1)

    return wp.spatial_vector(ang, vel)


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
    """ N*u = q_dot """
    e = q / 2.0
    e0, e1, e2, e3 = e.w, e.x, e.y, e.z
    ne1, ne2, ne3 = -e1, -e2, -e3
    return types.mat43(
        ne1, ne2, ne3,
        e0, e3, ne2,
        ne3, e0, e1,
        e2, ne1, e0
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


@wp.func
def calc_unnormalized_quaternion_N_inv(q: wp.quat) -> types.mat34:
    """ N*u = q_dot """
    e = 2.0 * q
    e0, e1, e2, e3 = e.w, e.x, e.y, e.z
    ne1, ne2, ne3 = -e1, -e2, -e3
    return types.mat34(
        ne1, e0, ne3, e2,
        ne2, e3, e0, ne1,
        ne3, ne2, e1, e0
    )
