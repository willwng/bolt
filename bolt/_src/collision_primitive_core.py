# Copyright 2025 The Newton Developers
# Modified for Bolt by Will Wang
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
from .consts import BOLT_MINVAL

wp.set_module_options({"enable_backward": False})


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
def closest_segment_point_and_dist(a: wp.vec3, b: wp.vec3, pt: wp.vec3) -> Tuple[wp.vec3, float]:
    """Returns closest point on the line segment and the distance squared."""
    closest = closest_segment_point(a, b, pt)
    dist = wp.dot((pt - closest), (pt - closest))
    return closest, dist


class mat23f(wp.types.matrix(shape=(2, 3), dtype=wp.float32)):
    pass


# core
@wp.func
def plane_sphere(plane_normal: wp.vec3, plane_pos: wp.vec3, sphere_pos: wp.vec3, sphere_radius: float) -> Tuple[
    float, wp.vec3]:
    dist = wp.dot(sphere_pos - plane_pos, plane_normal) - sphere_radius
    pos = sphere_pos - plane_normal * (sphere_radius + 0.5 * dist)
    return dist, pos


@wp.func
def sphere_sphere(
        # In:
        pos1: wp.vec3,
        radius1: float,
        pos2: wp.vec3,
        radius2: float,
) -> Tuple[float, wp.vec3, wp.vec3]:
    """Sphere-sphere collision calculation.

    Args:
      pos1: Center position of the first sphere.
      radius1: Radius of the first sphere.
      pos2: Center position of the second sphere.
      radius2: Radius of the second sphere.

    Returns:
      - Distance between sphere surfaces (negative if overlapping).
      - Contact position.
      - Contact normal vector.
    """
    dir = pos2 - pos1
    dist = wp.length(dir)
    if dist == 0.0:
        n = wp.vec3(1.0, 0.0, 0.0)
    else:
        n = dir / dist
    dist = dist - (radius1 + radius2)
    pos = pos1 + n * (radius1 + 0.5 * dist)
    return dist, pos, n


@wp.func
def sphere_capsule(
        # In:
        sphere_pos: wp.vec3,
        sphere_radius: float,
        capsule_pos: wp.vec3,
        capsule_axis: wp.vec3,
        capsule_radius: float,
        capsule_half_length: float,
) -> Tuple[float, wp.vec3, wp.vec3]:
    """Core contact geometry calculation for sphere-capsule collision.

    Args:
      sphere_pos: Center position of the sphere.
      sphere_radius: Radius of the sphere.
      capsule_pos: Center position of the capsule.
      capsule_axis: Axis direction of the capsule.
      capsule_radius: Radius of the capsule.
      capsule_half_length: Half length of the capsule.

    Returns:
      - Vector of contact distances.
      - Matrix of contact positions (one per row).
      - Matrix of contact normal vectors (one per row).
    """
    # Calculate capsule segment
    segment = capsule_axis * capsule_half_length

    # Find closest point on capsule centerline to sphere center
    pt = closest_segment_point(capsule_pos - segment, capsule_pos + segment, sphere_pos)

    # Use sphere-sphere collision between sphere and closest point
    return sphere_sphere(sphere_pos, sphere_radius, pt, capsule_radius)


@wp.func
def capsule_capsule(
        # In:
        cap1_pos: wp.vec3,
        cap1_axis: wp.vec3,
        cap1_radius: float,
        cap1_half_length: float,
        cap2_pos: wp.vec3,
        cap2_axis: wp.vec3,
        cap2_radius: float,
        cap2_half_length: float,
) -> Tuple[wp.vec2, mat23f, mat23f]:
    """Core contact geometry calculation for capsule-capsule collision.

    Args:
      cap1_pos: Center position of the first capsule.
      cap1_axis: Axis direction of the first capsule.
      cap1_radius: Radius of the first capsule.
      cap1_half_length: Half length of the first capsule.
      cap2_pos: Center position of the second capsule.
      cap2_axis: Axis direction of the second capsule.
      cap2_radius: Radius of the second capsule.
      cap2_half_length: Half length of the second capsule.

    Returns:
      - Vector of contact distances.
      - Matrix of contact positions (one per row).
      - Matrix of contact normal vectors (one per row).
    """
    contact_dist = wp.vec2(wp.inf, wp.inf)
    contact_pos = mat23f()
    contact_normal = mat23f()

    # calculate scaled axes and center difference
    axis1 = cap1_axis * cap1_half_length
    axis2 = cap2_axis * cap2_half_length
    dif = cap1_pos - cap2_pos

    # compute matrix coefficients and determinant
    ma = wp.dot(axis1, axis1)
    mb = -wp.dot(axis1, axis2)
    mc = wp.dot(axis2, axis2)
    u = -wp.dot(axis1, dif)
    v = wp.dot(axis2, dif)
    det = ma * mc - mb * mb

    # non-parallel axes: 1 contact
    if wp.abs(det) >= BOLT_MINVAL:
        inv_det = 1.0 / det
        x1 = (mc * u - mb * v) * inv_det
        x2 = (ma * v - mb * u) * inv_det

        if x1 > 1.0:
            x1 = 1.0
            x2 = (v - mb) / mc
        elif x1 < -1.0:
            x1 = -1.0
            x2 = (v + mb) / mc

        if x2 > 1.0:
            x2 = 1.0
            x1 = wp.clamp((u - mb) / ma, -1.0, 1.0)
        elif x2 < -1.0:
            x2 = -1.0
            x1 = wp.clamp((u + mb) / ma, -1.0, 1.0)

        # find nearest points
        vec1 = cap1_pos + axis1 * x1
        vec2 = cap2_pos + axis2 * x2

        dist, pos, normal = sphere_sphere(vec1, cap1_radius, vec2, cap2_radius)
        if dist <= 0.0:
            contact_dist[0] = dist
            contact_pos[0] = pos
            contact_normal[0] = normal

    # parallel axes: test all 4 endpoint pairs, keep first 2 that pass margin check
    else:
        contact_count = 0

        # x1 = 1: test positive end of capsule 1
        vec1 = cap1_pos + axis1
        x2 = wp.clamp((v - mb) / mc, -1.0, 1.0)
        vec2 = cap2_pos + axis2 * x2
        dist, pos, normal = sphere_sphere(vec1, cap1_radius, vec2, cap2_radius)
        if dist <= 0.0:
            contact_dist[contact_count] = dist
            contact_pos[contact_count] = pos
            contact_normal[contact_count] = normal
            contact_count += 1

        # x1 = -1: test negative end of capsule 1
        vec1 = cap1_pos - axis1
        x2 = wp.clamp((v + mb) / mc, -1.0, 1.0)
        vec2 = cap2_pos + axis2 * x2
        dist, pos, normal = sphere_sphere(vec1, cap1_radius, vec2, cap2_radius)
        if dist <= 0.0:
            contact_dist[contact_count] = dist
            contact_pos[contact_count] = pos
            contact_normal[contact_count] = normal
            contact_count += 1

        # x2 = 1: test positive end of capsule 2
        if contact_count < 2:
            vec2 = cap2_pos + axis2
            x1 = wp.clamp((u - mb) / ma, -1.0, 1.0)
            vec1 = cap1_pos + axis1 * x1
            dist, pos, normal = sphere_sphere(vec1, cap1_radius, vec2, cap2_radius)
            if dist <= 0.0:
                contact_dist[contact_count] = dist
                contact_pos[contact_count] = pos
                contact_normal[contact_count] = normal
                contact_count += 1

        # x2 = -1: test negative end of capsule 2
        if contact_count < 2:
            vec2 = cap2_pos - axis2
            x1 = wp.clamp((u + mb) / ma, -1.0, 1.0)
            vec1 = cap1_pos + axis1 * x1
            dist, pos, normal = sphere_sphere(vec1, cap1_radius, vec2, cap2_radius)
            if dist <= 0.0:
                contact_dist[contact_count] = dist
                contact_pos[contact_count] = pos
                contact_normal[contact_count] = normal

    return contact_dist, contact_pos, contact_normal


@wp.func
def plane_capsule(
        # In:
        plane_normal: wp.vec3,
        plane_pos: wp.vec3,
        capsule_pos: wp.vec3,
        capsule_axis: wp.vec3,
        capsule_radius: float,
        capsule_half_length: float,
) -> Tuple[wp.vec2, mat23f, wp.mat33]:
    """Core contact geometry calculation for plane-capsule collision.

    Args:
      plane_normal: Normal vector of the plane.
      plane_pos: Position point on the plane.
      capsule_pos: Center position of the capsule.
      capsule_axis: Axis direction of the capsule.
      capsule_radius: Radius of the capsule.
      capsule_half_length: Half length of the capsule.

    Returns:
      - Vector of contact distances.
      - Matrix of contact positions (one per row).
      - Contact frame for both contacts.
    """
    n = plane_normal
    axis = capsule_axis

    # align contact frames with capsule axis
    b, b_norm = normalize_with_norm(axis - n * wp.dot(n, axis))

    if b_norm < 0.5:
        if -0.5 < n[1] and n[1] < 0.5:
            b = wp.vec3(0.0, 1.0, 0.0)
        else:
            b = wp.vec3(0.0, 0.0, 1.0)

    c = wp.cross(n, b)
    frame = wp.mat33(n[0], n[1], n[2], b[0], b[1], b[2], c[0], c[1], c[2])
    segment = axis * capsule_half_length

    # First contact (positive end of capsule)
    dist1, pos1 = plane_sphere(n, plane_pos, capsule_pos + segment, capsule_radius)

    # Second contact (negative end of capsule)
    dist2, pos2 = plane_sphere(n, plane_pos, capsule_pos - segment, capsule_radius)

    dist = wp.vec2(dist1, dist2)
    pos = mat23f(pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2])

    return dist, pos, frame


@wp.func
def plane_ellipsoid(
        # In:
        plane_normal: wp.vec3,
        plane_pos: wp.vec3,
        ellipsoid_pos: wp.vec3,
        ellipsoid_rot: wp.mat33,
        ellipsoid_size: wp.vec3,
) -> Tuple[float, wp.vec3, wp.vec3]:
    """Core contact geometry calculation for plane-ellipsoid collision.

    Args:
      plane_normal: Normal vector of the plane.
      plane_pos: Position point on the plane.
      ellipsoid_pos: Center position of the ellipsoid.
      ellipsoid_rot: Rotation matrix of the ellipsoid.
      ellipsoid_size: Size (radii) of the ellipsoid along each axis.

    Returns:
      - Vector of contact distances.
      - Matrix of contact positions (one per row).
      - Matrix of contact normal vectors (one per row).
    """
    sphere_support = -wp.normalize(wp.cw_mul(wp.transpose(ellipsoid_rot) @ plane_normal, ellipsoid_size))
    pos = ellipsoid_pos + ellipsoid_rot @ wp.cw_mul(sphere_support, ellipsoid_size)
    dist = wp.dot(plane_normal, pos - plane_pos)
    pos = pos - plane_normal * dist * 0.5

    return dist, pos, plane_normal


