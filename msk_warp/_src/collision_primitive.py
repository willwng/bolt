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

from typing import Tuple

import warp as wp

from .collision_primitive_core import box_box
from .collision_primitive_core import capsule_box
from .collision_primitive_core import capsule_capsule
from .collision_primitive_core import plane_box
from .collision_primitive_core import plane_capsule
from .collision_primitive_core import plane_cylinder
from .collision_primitive_core import plane_ellipsoid
from .collision_primitive_core import plane_sphere
from .collision_primitive_core import sphere_box
from .collision_primitive_core import sphere_capsule
from .collision_primitive_core import sphere_cylinder
from .collision_primitive_core import sphere_sphere
from .math import make_frame
from .math import safe_div
from .math import upper_trid_index
from .consts import MJ_MINMU
from .consts import MJ_MINVAL
from .types import Data
from .types import GeomType
from .types import Model
from .types import vec5
from .warp_util import cache_kernel
from .warp_util import event_scope
from .warp_util import kernel as nested_kernel

wp.set_module_options({"enable_backward": False})


class mat43f(wp.types.matrix(shape=(4, 3), dtype=wp.float32)):
    pass


mat63 = wp.types.matrix(shape=(6, 3), dtype=float)


@wp.struct
class Geom:
    pos: wp.vec3
    rot: wp.mat33
    normal: wp.vec3
    size: wp.vec3
    hfprism: mat63
    vertadr: int
    vertnum: int
    vert: wp.array(dtype=wp.vec3)
    graphadr: int
    graph: wp.array(dtype=int)
    mesh_polynum: int
    mesh_polyadr: int
    mesh_polynormal: wp.array(dtype=wp.vec3)
    mesh_polyvertadr: wp.array(dtype=int)
    mesh_polyvertnum: wp.array(dtype=int)
    mesh_polyvert: wp.array(dtype=int)
    mesh_polymapadr: wp.array(dtype=int)
    mesh_polymapnum: wp.array(dtype=int)
    mesh_polymap: wp.array(dtype=int)
    index: int


@wp.func
def geom_collision_pair(
        # Model:
        geom_type: wp.array(dtype=int),
        geom_size: wp.array(dtype=wp.vec3),
        # Data in:
        geom_xpos_in: wp.array2d(dtype=wp.vec3),
        geom_xmat_in: wp.array2d(dtype=wp.mat33),
        # In:
        geoms: wp.vec2i,
        worldid: int,
) -> Tuple[Geom, Geom]:
    geom1 = Geom()
    geom2 = Geom()

    g1 = geoms[0]
    g2 = geoms[1]
    geom_type1 = geom_type[g1]
    geom_type2 = geom_type[g2]

    geom1.pos = geom_xpos_in[worldid, g1]
    geom1.rot = geom_xmat_in[worldid, g1]
    geom1.size = geom_size[g1]
    geom1.normal = wp.vec3(geom1.rot[0, 1], geom1.rot[1, 1],
                           geom1.rot[2, 1])  # plane

    geom2.pos = geom_xpos_in[worldid, g2]
    geom2.rot = geom_xmat_in[worldid, g2]
    geom2.size = geom_size[g2]
    geom2.normal = wp.vec3(geom2.rot[0, 1], geom2.rot[1, 1],
                           geom2.rot[2, 1])  # plane

    geom1.index = -1
    geom2.index = -1
    return geom1, geom2


@wp.func
def plane_convex(plane_normal: wp.vec3, plane_pos: wp.vec3, convex: Geom) -> \
Tuple[wp.vec4, mat43f, wp.vec3]:
    """Core contact geometry calculation for plane-convex collision.

    Args:
      plane_normal: Normal vector of the plane.
      plane_pos: Position point on the plane.
      convex: Convex geometry object containing position, rotation, and mesh data.

    Returns:
      - Vector of contact distances (wp.inf for unpopulated contacts).
      - Matrix of contact positions (one per row).
      - Matrix of contact normal vectors (one per row).
    """
    _HUGE_VAL = 1e6

    contact_dist = wp.vec4(wp.inf)
    contact_pos = mat43f()
    contact_count = int(0)

    # get points in the convex frame
    plane_pos_local = wp.transpose(convex.rot) @ (plane_pos - convex.pos)
    n = wp.transpose(convex.rot) @ plane_normal

    # Store indices in vec4
    indices = wp.vec4i(-1, -1, -1, -1)

    # exhaustive search over all vertices
    if convex.graphadr == -1 or convex.vertnum < 10:
        # find first support point (a)
        max_support = wp.float32(-_HUGE_VAL)
        a = wp.vec3()
        for i in range(convex.vertnum):
            vert = convex.vert[convex.vertadr + i]
            support = wp.dot(plane_pos_local - vert, n)
            if support > max_support:
                max_support = support
                indices[0] = i
                a = vert

        if max_support < 0:
            return contact_dist, contact_pos, plane_normal

        threshold = max_support - 1e-3

        # find point (b) furthest from a
        b_dist = wp.float32(-_HUGE_VAL)
        b = wp.vec3()
        for i in range(convex.vertnum):
            vert = convex.vert[convex.vertadr + i]
            support = wp.dot(plane_pos_local - vert, n)
            dist_mask = wp.where(support > threshold, 0.0, -_HUGE_VAL)
            dist = wp.length_sq(a - vert) + dist_mask
            if dist > b_dist:
                indices[1] = i
                b_dist = dist
                b = vert

        # find point (c) furthest along axis orthogonal to a-b
        ab = wp.cross(n, a - b)
        c_dist = wp.float32(-_HUGE_VAL)
        c = wp.vec3()
        for i in range(convex.vertnum):
            vert = convex.vert[convex.vertadr + i]
            support = wp.dot(plane_pos_local - vert, n)
            dist_mask = wp.where(support > threshold, 0.0, -_HUGE_VAL)
            ap = a - vert
            dist = wp.abs(wp.dot(ap, ab)) + dist_mask
            if dist > c_dist:
                indices[2] = i
                c_dist = dist
                c = vert

        # find point (d) furthest from other triangle edges
        ac = wp.cross(n, a - c)
        bc = wp.cross(n, b - c)
        d_dist = wp.float32(-_HUGE_VAL)
        for i in range(convex.vertnum):
            vert = convex.vert[convex.vertadr + i]
            support = wp.dot(plane_pos_local - vert, n)
            dist_mask = wp.where(support > threshold, 0.0, -_HUGE_VAL)
            ap = a - vert
            bp = b - vert
            dist_ap = wp.abs(wp.dot(ap, ac)) + dist_mask
            dist_bp = wp.abs(wp.dot(bp, bc)) + dist_mask
            if dist_ap + dist_bp > d_dist:
                indices[3] = i
                d_dist = dist_ap + dist_bp

    else:
        numvert = convex.graph[convex.graphadr]
        vert_edgeadr = convex.graphadr + 2
        vert_globalid = convex.graphadr + 2 + numvert
        edge_localid = convex.graphadr + 2 + 2 * numvert

        # Find support points
        max_support = wp.float32(-_HUGE_VAL)

        # hillclimb until no change
        prev = int(-1)
        imax = int(0)

        while True:
            prev = int(imax)
            i = int(convex.graph[vert_edgeadr + imax])
            while convex.graph[edge_localid + i] >= 0:
                subidx = convex.graph[edge_localid + i]
                idx = convex.graph[vert_globalid + subidx]
                support = wp.dot(
                    plane_pos_local - convex.vert[convex.vertadr + idx], n)
                if support > max_support:
                    max_support = support
                    imax = int(subidx)
                i += int(1)
            if imax == prev:
                break

        threshold = wp.max(0.0, max_support - 1e-3)

        a_dist = wp.float32(-_HUGE_VAL)
        while True:
            prev = int(imax)
            i = int(convex.graph[vert_edgeadr + imax])
            while convex.graph[edge_localid + i] >= 0:
                subidx = convex.graph[edge_localid + i]
                idx = convex.graph[vert_globalid + subidx]
                support = wp.dot(
                    plane_pos_local - convex.vert[convex.vertadr + idx], n)
                dist = wp.where(support > threshold, support, -_HUGE_VAL)
                if dist > a_dist:
                    a_dist = dist
                    imax = int(subidx)
                i += int(1)
            if imax == prev:
                break
        imax_global = convex.graph[vert_globalid + imax]
        a = convex.vert[convex.vertadr + imax_global]
        indices[0] = imax_global

        # Find point b (furthest from a)
        b_dist = wp.float32(-_HUGE_VAL)
        while True:
            prev = int(imax)
            i = int(convex.graph[vert_edgeadr + imax])
            while convex.graph[edge_localid + i] >= 0:
                subidx = convex.graph[edge_localid + i]
                idx = convex.graph[vert_globalid + subidx]
                support = wp.dot(
                    plane_pos_local - convex.vert[convex.vertadr + idx], n)
                dist_mask = wp.where(support > threshold, 0.0, -_HUGE_VAL)
                dist = wp.length_sq(
                    a - convex.vert[convex.vertadr + idx]) + dist_mask
                if dist > b_dist:
                    b_dist = dist
                    imax = int(subidx)
                i += int(1)
            if imax == prev:
                break
        imax_global = convex.graph[vert_globalid + imax]
        b = convex.vert[convex.vertadr + imax_global]
        indices[1] = imax_global

        # Find point c (furthest along axis orthogonal to a-b)
        ab = wp.cross(n, a - b)
        c_dist = wp.float32(-_HUGE_VAL)
        while True:
            prev = int(imax)
            i = int(convex.graph[vert_edgeadr + imax])
            while convex.graph[edge_localid + i] >= 0:
                subidx = convex.graph[edge_localid + i]
                idx = convex.graph[vert_globalid + subidx]
                support = wp.dot(
                    plane_pos_local - convex.vert[convex.vertadr + idx], n)
                dist_mask = wp.where(support > threshold, 0.0, -_HUGE_VAL)
                ap = a - convex.vert[convex.vertadr + idx]
                dist = wp.abs(wp.dot(ap, ab)) + dist_mask
                if dist > c_dist:
                    c_dist = dist
                    imax = int(subidx)
                i += int(1)
            if imax == prev:
                break
        imax_global = convex.graph[vert_globalid + imax]
        c = convex.vert[convex.vertadr + imax_global]
        indices[2] = imax_global

        # Find point d (furthest from other triangle edges)
        ac = wp.cross(n, a - c)
        bc = wp.cross(n, b - c)
        d_dist = wp.float32(-_HUGE_VAL)
        while True:
            prev = int(imax)
            i = int(convex.graph[vert_edgeadr + imax])
            while convex.graph[edge_localid + i] >= 0:
                subidx = convex.graph[edge_localid + i]
                idx = convex.graph[vert_globalid + subidx]
                support = wp.dot(
                    plane_pos_local - convex.vert[convex.vertadr + idx], n)
                dist_mask = wp.where(support > threshold, 0.0, -_HUGE_VAL)
                ap = a - convex.vert[convex.vertadr + idx]
                bp = b - convex.vert[convex.vertadr + idx]
                dist_ap = wp.abs(wp.dot(ap, ac)) + dist_mask
                dist_bp = wp.abs(wp.dot(bp, bc)) + dist_mask
                if dist_ap + dist_bp > d_dist:
                    d_dist = dist_ap + dist_bp
                    imax = int(subidx)
                i += int(1)
            if imax == prev:
                break
        imax_global = convex.graph[vert_globalid + imax]
        indices[3] = imax_global

    # Collect contacts from unique indices
    for i in range(3, -1, -1):
        idx = indices[i]
        count = int(0)
        for j in range(i + 1):
            if indices[j] == idx:
                count = count + 1

        # Check if the index is unique (appears exactly once)
        if count == 1:
            pos = convex.vert[convex.vertadr + idx]
            pos = convex.pos + convex.rot @ pos
            support = wp.dot(
                plane_pos_local - convex.vert[convex.vertadr + idx], n)
            dist = -support
            pos = pos - 0.5 * dist * plane_normal

            contact_dist[contact_count] = dist
            contact_pos[contact_count] = pos
            contact_count = contact_count + 1

    return contact_dist, contact_pos, plane_normal


@wp.func
def write_contact(
        # Data in:
        naconmax_in: int,
        # In:
        id_: int,
        dist_in: float,
        pos_in: wp.vec3,
        frame_in: wp.mat33,
        condim_in: int,
        curvature_in: float,
        friction_in: vec5,
        geoms_in: wp.vec2i,
        pairid_in: wp.vec2i,
        worldid_in: int,
        # Data out:
        contact_dist_out: wp.array(dtype=float),
        contact_pos_out: wp.array(dtype=wp.vec3),
        contact_frame_out: wp.array(dtype=wp.mat33),
        contact_friction_out: wp.array(dtype=vec5),
        contact_dim_out: wp.array(dtype=int),
        contact_curvature_out: wp.array(dtype=float),
        contact_geom_out: wp.array(dtype=wp.vec2i),
        contact_worldid_out: wp.array(dtype=int),
        contact_geomcollisionid_out: wp.array(dtype=int),
        nacon_out: wp.array(dtype=int),
):
    active = dist_in < 0

    # skip contact and no collision sensor
    if (pairid_in[0] == -2 or not active) and pairid_in[1] == -1:
        return

    cid = wp.atomic_add(nacon_out, 0, 1)
    if cid < naconmax_in:
        contact_dist_out[cid] = dist_in
        contact_pos_out[cid] = pos_in
        contact_frame_out[cid] = frame_in
        contact_geom_out[cid] = geoms_in
        contact_worldid_out[cid] = worldid_in
        contact_dim_out[cid] = condim_in
        contact_curvature_out[cid] = curvature_in
        contact_friction_out[cid] = friction_in
        contact_geomcollisionid_out[cid] = id_


@wp.func
def contact_params(
        # Model:
        geom_friction: wp.array(dtype=wp.vec3),
        # Data in:
        collision_pair_in: wp.array(dtype=wp.vec2i),
        collision_pairid_in: wp.array(dtype=wp.vec2i),
        # In:
        cid: int,
        worldid: int,
):
    geoms = collision_pair_in[cid]
    pairid = collision_pairid_in[cid][0]

    g1 = geoms[0]
    g2 = geoms[1]

    condim = 3 # hard coded
    max_geom_friction = wp.max(geom_friction[g1], geom_friction[g2])

    friction = vec5(
        wp.max(MJ_MINMU, max_geom_friction[0]),
        wp.max(MJ_MINMU, max_geom_friction[0]),
        wp.max(MJ_MINMU, max_geom_friction[1]),
        wp.max(MJ_MINMU, max_geom_friction[2]),
        wp.max(MJ_MINMU, max_geom_friction[2]),
    )

    return geoms, condim, friction


@wp.func
def plane_sphere_wrapper(
        # Data in:
        naconmax_in: int,
        # In:
        plane: Geom,
        sphere: Geom,
        worldid: int,
        condim: int,
        friction: vec5,
        geoms: wp.vec2i,
        pairid: wp.vec2i,
        # Data out:
        contact_dist_out: wp.array(dtype=float),
        contact_pos_out: wp.array(dtype=wp.vec3),
        contact_frame_out: wp.array(dtype=wp.mat33),
        contact_friction_out: wp.array(dtype=vec5),
        contact_dim_out: wp.array(dtype=int),
        contact_curvature_out: wp.array(dtype=float),
        contact_geom_out: wp.array(dtype=wp.vec2i),
        contact_worldid_out: wp.array(dtype=int),
        contact_geomcollisionid_out: wp.array(dtype=int),
        nacon_out: wp.array(dtype=int),
):
    """Calculates contact between a sphere and a plane."""
    normal = plane.normal
    dist, pos = plane_sphere(normal, plane.pos, sphere.pos, sphere.size[0])
    curvature = sphere.size[0]

    write_contact(
        naconmax_in,
        0,
        dist,
        pos,
        make_frame(normal),
        condim,
        curvature,
        friction,
        geoms,
        pairid,
        worldid,
        contact_dist_out,
        contact_pos_out,
        contact_frame_out,
        contact_friction_out,
        contact_dim_out,
        contact_curvature_out,
        contact_geom_out,
        contact_worldid_out,
        contact_geomcollisionid_out,
        nacon_out,
    )


@wp.func
def sphere_sphere_wrapper(
        # Data in:
        naconmax_in: int,
        # In:
        sphere1: Geom,
        sphere2: Geom,
        worldid: int,
        condim: int,
        friction: vec5,
        geoms: wp.vec2i,
        pairid: wp.vec2i,
        # Data out:
        contact_dist_out: wp.array(dtype=float),
        contact_pos_out: wp.array(dtype=wp.vec3),
        contact_frame_out: wp.array(dtype=wp.mat33),
        contact_friction_out: wp.array(dtype=vec5),
        contact_dim_out: wp.array(dtype=int),
        contact_curvature_out: wp.array(dtype=float),
        contact_geom_out: wp.array(dtype=wp.vec2i),
        contact_worldid_out: wp.array(dtype=int),
        contact_geomcollisionid_out: wp.array(dtype=int),
        nacon_out: wp.array(dtype=int),
):
    """Calculates contact between two spheres."""
    dist, pos, normal = sphere_sphere(sphere1.pos, sphere1.size[0], sphere2.pos,
                                      sphere2.size[0])
    curvature = wp.sqrt(sphere1.size[0] * sphere2.size[0])

    write_contact(
        naconmax_in,
        0,
        dist,
        pos,
        make_frame(normal),
        condim,
        curvature,
        friction,
        geoms,
        pairid,
        worldid,
        contact_dist_out,
        contact_pos_out,
        contact_frame_out,
        contact_friction_out,
        contact_dim_out,
        contact_curvature_out,
        contact_geom_out,
        contact_worldid_out,
        contact_geomcollisionid_out,
        nacon_out,
    )


@wp.func
def sphere_capsule_wrapper(
        # Data in:
        naconmax_in: int,
        # In:
        sphere: Geom,
        cap: Geom,
        worldid: int,
        condim: int,
        friction: vec5,
        geoms: wp.vec2i,
        pairid: wp.vec2i,
        # Data out:
        contact_dist_out: wp.array(dtype=float),
        contact_pos_out: wp.array(dtype=wp.vec3),
        contact_frame_out: wp.array(dtype=wp.mat33),
        contact_friction_out: wp.array(dtype=vec5),
        contact_dim_out: wp.array(dtype=int),
        contact_curvature_out: wp.array(dtype=float),
        contact_geom_out: wp.array(dtype=wp.vec2i),
        contact_worldid_out: wp.array(dtype=int),
        contact_geomcollisionid_out: wp.array(dtype=int),
        nacon_out: wp.array(dtype=int),
):
    """Calculates one contact between a sphere and a capsule."""
    # capsule axis
    axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])

    dist, pos, normal = sphere_capsule(sphere.pos, sphere.size[0], cap.pos,
                                       axis, cap.size[0], cap.size[1])
    curvature = wp.sqrt(sphere.size[0] * cap.size[0])

    write_contact(
        naconmax_in,
        0,
        dist,
        pos,
        make_frame(normal),
        condim,
        curvature,
        friction,
        geoms,
        pairid,
        worldid,
        contact_dist_out,
        contact_pos_out,
        contact_frame_out,
        contact_friction_out,
        contact_dim_out,
        contact_curvature_out,
        contact_geom_out,
        contact_worldid_out,
        contact_geomcollisionid_out,
        nacon_out,
    )


@wp.func
def capsule_capsule_wrapper(
        # Data in:
        naconmax_in: int,
        # In:
        cap1: Geom,
        cap2: Geom,
        worldid: int,
        condim: int,
        friction: vec5,
        geoms: wp.vec2i,
        pairid: wp.vec2i,
        # Data out:
        contact_dist_out: wp.array(dtype=float),
        contact_pos_out: wp.array(dtype=wp.vec3),
        contact_frame_out: wp.array(dtype=wp.mat33),
        contact_friction_out: wp.array(dtype=vec5),
        contact_dim_out: wp.array(dtype=int),
        contact_curvature_out: wp.array(dtype=float),
        contact_geom_out: wp.array(dtype=wp.vec2i),
        contact_worldid_out: wp.array(dtype=int),
        contact_geomcollisionid_out: wp.array(dtype=int),
        nacon_out: wp.array(dtype=int),
):
    """Calculates contacts between two capsules."""
    # capsule axes
    cap1_axis = wp.vec3(cap1.rot[0, 2], cap1.rot[1, 2], cap1.rot[2, 2])
    cap2_axis = wp.vec3(cap2.rot[0, 2], cap2.rot[1, 2], cap2.rot[2, 2])
    curvature = wp.sqrt(cap1.size[0] * cap2.size[0])

    dist, pos, normal = capsule_capsule(
        cap1.pos,
        cap1_axis,
        cap1.size[0],  # radius1
        cap1.size[1],  # half_length1
        cap2.pos,
        cap2_axis,
        cap2.size[0],  # radius2
        cap2.size[1],  # half_length2
    )

    write_contact(
        naconmax_in,
        0,
        dist,
        pos,
        make_frame(normal),
        condim,
        curvature,
        friction,
        geoms,
        pairid,
        worldid,
        contact_dist_out,
        contact_pos_out,
        contact_frame_out,
        contact_friction_out,
        contact_dim_out,
        contact_curvature_out,
        contact_geom_out,
        contact_worldid_out,
        contact_geomcollisionid_out,
        nacon_out,
    )


@wp.func
def plane_capsule_wrapper(
        # Data in:
        naconmax_in: int,
        # In:
        plane: Geom,
        cap: Geom,
        worldid: int,
        condim: int,
        friction: vec5,
        geoms: wp.vec2i,
        pairid: wp.vec2i,
        # Data out:
        contact_dist_out: wp.array(dtype=float),
        contact_pos_out: wp.array(dtype=wp.vec3),
        contact_frame_out: wp.array(dtype=wp.mat33),
        contact_friction_out: wp.array(dtype=vec5),
        contact_dim_out: wp.array(dtype=int),
        contact_curvature_out: wp.array(dtype=float),
        contact_geom_out: wp.array(dtype=wp.vec2i),
        contact_worldid_out: wp.array(dtype=int),
        contact_geomcollisionid_out: wp.array(dtype=int),
        nacon_out: wp.array(dtype=int),
):
    """Calculates contacts between a capsule and a plane."""
    # capsule axis
    capsule_axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])
    curvature = cap.size[0]

    dist, pos, frame = plane_capsule(
        plane.normal,
        plane.pos,
        cap.pos,
        capsule_axis,
        cap.size[0],  # radius
        cap.size[1],  # half_length
    )

    for i in range(2):
        write_contact(
            naconmax_in,
            i,
            dist[i],
            pos[i],
            frame,
            condim,
            curvature,
            friction,
            geoms,
            pairid,
            worldid,
            contact_dist_out,
            contact_pos_out,
            contact_frame_out,
            contact_friction_out,
            contact_dim_out,
            contact_curvature_out,
            contact_geom_out,
            contact_worldid_out,
            contact_geomcollisionid_out,
            nacon_out,
        )


@wp.func
def plane_ellipsoid_wrapper(
        # Data in:
        naconmax_in: int,
        # In:
        plane: Geom,
        ellipsoid: Geom,
        worldid: int,
        gap: float,
        condim: int,
        friction: vec5,
        solref: wp.vec2,
        solreffriction: wp.vec2,
        solimp: vec5,
        geoms: wp.vec2i,
        pairid: wp.vec2i,
        # Data out:
        contact_dist_out: wp.array(dtype=float),
        contact_pos_out: wp.array(dtype=wp.vec3),
        contact_frame_out: wp.array(dtype=wp.mat33),
        contact_friction_out: wp.array(dtype=vec5),
        contact_solref_out: wp.array(dtype=wp.vec2),
        contact_solreffriction_out: wp.array(dtype=wp.vec2),
        contact_solimp_out: wp.array(dtype=vec5),
        contact_dim_out: wp.array(dtype=int),
        contact_geom_out: wp.array(dtype=wp.vec2i),
        contact_worldid_out: wp.array(dtype=int),
        contact_type_out: wp.array(dtype=int),
        contact_geomcollisionid_out: wp.array(dtype=int),
        nacon_out: wp.array(dtype=int),
):
    """Calculates contacts between an ellipsoid and a plane."""
    dist, pos, normal = plane_ellipsoid(plane.normal, plane.pos, ellipsoid.pos,
                                        ellipsoid.rot, ellipsoid.size)

    write_contact(
        naconmax_in,
        0,
        dist,
        pos,
        make_frame(normal),
        gap,
        condim,
        friction,
        solref,
        solreffriction,
        solimp,
        geoms,
        pairid,
        worldid,
        contact_dist_out,
        contact_pos_out,
        contact_frame_out,
        contact_friction_out,
        contact_solref_out,
        contact_solreffriction_out,
        contact_solimp_out,
        contact_dim_out,
        contact_geom_out,
        contact_worldid_out,
        contact_type_out,
        contact_geomcollisionid_out,
        nacon_out,
    )


@wp.func
def plane_box_wrapper(
        # Data in:
        naconmax_in: int,
        # In:
        plane: Geom,
        box: Geom,
        worldid: int,
        gap: float,
        condim: int,
        friction: vec5,
        solref: wp.vec2,
        solreffriction: wp.vec2,
        solimp: vec5,
        geoms: wp.vec2i,
        pairid: wp.vec2i,
        # Data out:
        contact_dist_out: wp.array(dtype=float),
        contact_pos_out: wp.array(dtype=wp.vec3),
        contact_frame_out: wp.array(dtype=wp.mat33),
        contact_friction_out: wp.array(dtype=vec5),
        contact_solref_out: wp.array(dtype=wp.vec2),
        contact_solreffriction_out: wp.array(dtype=wp.vec2),
        contact_solimp_out: wp.array(dtype=vec5),
        contact_dim_out: wp.array(dtype=int),
        contact_geom_out: wp.array(dtype=wp.vec2i),
        contact_worldid_out: wp.array(dtype=int),
        contact_type_out: wp.array(dtype=int),
        contact_geomcollisionid_out: wp.array(dtype=int),
        nacon_out: wp.array(dtype=int),
):
    """Calculates contacts between a box and a plane."""
    dist, pos, normal = plane_box(plane.normal, plane.pos, box.pos, box.rot,
                                  box.size)
    frame = make_frame(normal)

    for i in range(4):
        write_contact(
            naconmax_in,
            i,
            dist[i],
            pos[i],
            frame,
            gap,
            condim,
            friction,
            solref,
            solreffriction,
            solimp,
            geoms,
            pairid,
            worldid,
            contact_dist_out,
            contact_pos_out,
            contact_frame_out,
            contact_friction_out,
            contact_solref_out,
            contact_solreffriction_out,
            contact_solimp_out,
            contact_dim_out,
            contact_geom_out,
            contact_worldid_out,
            contact_type_out,
            contact_geomcollisionid_out,
            nacon_out,
        )


@wp.func
def plane_convex_wrapper(
        # Data in:
        naconmax_in: int,
        # In:
        plane: Geom,
        convex: Geom,
        worldid: int,
        gap: float,
        condim: int,
        friction: vec5,
        solref: wp.vec2,
        solreffriction: wp.vec2,
        solimp: vec5,
        geoms: wp.vec2i,
        pairid: wp.vec2i,
        # Data out:
        contact_dist_out: wp.array(dtype=float),
        contact_pos_out: wp.array(dtype=wp.vec3),
        contact_frame_out: wp.array(dtype=wp.mat33),
        contact_friction_out: wp.array(dtype=vec5),
        contact_solref_out: wp.array(dtype=wp.vec2),
        contact_solreffriction_out: wp.array(dtype=wp.vec2),
        contact_solimp_out: wp.array(dtype=vec5),
        contact_dim_out: wp.array(dtype=int),
        contact_geom_out: wp.array(dtype=wp.vec2i),
        contact_worldid_out: wp.array(dtype=int),
        contact_type_out: wp.array(dtype=int),
        contact_geomcollisionid_out: wp.array(dtype=int),
        nacon_out: wp.array(dtype=int),
):
    """Calculates contacts between a plane and a convex object."""
    dist, pos, normal = plane_convex(plane.normal, plane.pos, convex)

    frame = make_frame(normal)
    for i in range(4):
        write_contact(
            naconmax_in,
            i,
            dist[i],
            pos[i],
            frame,
            gap,
            condim,
            friction,
            solref,
            solreffriction,
            solimp,
            geoms,
            pairid,
            worldid,
            contact_dist_out,
            contact_pos_out,
            contact_frame_out,
            contact_friction_out,
            contact_solref_out,
            contact_solreffriction_out,
            contact_solimp_out,
            contact_dim_out,
            contact_geom_out,
            contact_worldid_out,
            contact_type_out,
            contact_geomcollisionid_out,
            nacon_out,
        )


@wp.func
def sphere_cylinder_wrapper(
        # Data in:
        naconmax_in: int,
        # In:
        sphere: Geom,
        cylinder: Geom,
        worldid: int,
        gap: float,
        condim: int,
        friction: vec5,
        solref: wp.vec2,
        solreffriction: wp.vec2,
        solimp: vec5,
        geoms: wp.vec2i,
        pairid: wp.vec2i,
        # Data out:
        contact_dist_out: wp.array(dtype=float),
        contact_pos_out: wp.array(dtype=wp.vec3),
        contact_frame_out: wp.array(dtype=wp.mat33),
        contact_friction_out: wp.array(dtype=vec5),
        contact_solref_out: wp.array(dtype=wp.vec2),
        contact_solreffriction_out: wp.array(dtype=wp.vec2),
        contact_solimp_out: wp.array(dtype=vec5),
        contact_dim_out: wp.array(dtype=int),
        contact_geom_out: wp.array(dtype=wp.vec2i),
        contact_worldid_out: wp.array(dtype=int),
        contact_type_out: wp.array(dtype=int),
        contact_geomcollisionid_out: wp.array(dtype=int),
        nacon_out: wp.array(dtype=int),
):
    """Calculates contacts between a sphere and a cylinder."""
    # cylinder axis
    cylinder_axis = wp.vec3(cylinder.rot[0, 2], cylinder.rot[1, 2],
                            cylinder.rot[2, 2])

    dist, pos, normal = sphere_cylinder(
        sphere.pos,
        sphere.size[0],  # sphere radius
        cylinder.pos,
        cylinder_axis,
        cylinder.size[0],  # cylinder radius
        cylinder.size[1],  # cylinder half_height
    )

    write_contact(
        naconmax_in,
        0,
        dist,
        pos,
        make_frame(normal),
        gap,
        condim,
        friction,
        solref,
        solreffriction,
        solimp,
        geoms,
        pairid,
        worldid,
        contact_dist_out,
        contact_pos_out,
        contact_frame_out,
        contact_friction_out,
        contact_solref_out,
        contact_solreffriction_out,
        contact_solimp_out,
        contact_dim_out,
        contact_geom_out,
        contact_worldid_out,
        contact_type_out,
        contact_geomcollisionid_out,
        nacon_out,
    )


@wp.func
def plane_cylinder_wrapper(
        # Data in:
        naconmax_in: int,
        # In:
        plane: Geom,
        cylinder: Geom,
        worldid: int,
        gap: float,
        condim: int,
        friction: vec5,
        solref: wp.vec2,
        solreffriction: wp.vec2,
        solimp: vec5,
        geoms: wp.vec2i,
        pairid: wp.vec2i,
        # Data out:
        contact_dist_out: wp.array(dtype=float),
        contact_pos_out: wp.array(dtype=wp.vec3),
        contact_frame_out: wp.array(dtype=wp.mat33),
        contact_friction_out: wp.array(dtype=vec5),
        contact_solref_out: wp.array(dtype=wp.vec2),
        contact_solreffriction_out: wp.array(dtype=wp.vec2),
        contact_solimp_out: wp.array(dtype=vec5),
        contact_dim_out: wp.array(dtype=int),
        contact_geom_out: wp.array(dtype=wp.vec2i),
        contact_worldid_out: wp.array(dtype=int),
        contact_type_out: wp.array(dtype=int),
        contact_geomcollisionid_out: wp.array(dtype=int),
        nacon_out: wp.array(dtype=int),
):
    """Calculates contacts between a cylinder and a plane."""
    # cylinder axis
    cylinder_axis = wp.vec3(cylinder.rot[0, 2], cylinder.rot[1, 2],
                            cylinder.rot[2, 2])

    dist, pos, normal = plane_cylinder(
        plane.normal,
        plane.pos,
        cylinder.pos,
        cylinder_axis,
        cylinder.size[0],  # radius
        cylinder.size[1],  # half_height
    )

    frame = make_frame(normal)
    for i in range(4):
        write_contact(
            naconmax_in,
            i,
            dist[i],
            pos[i],
            frame,
            gap,
            condim,
            friction,
            solref,
            solreffriction,
            solimp,
            geoms,
            pairid,
            worldid,
            contact_dist_out,
            contact_pos_out,
            contact_frame_out,
            contact_friction_out,
            contact_solref_out,
            contact_solreffriction_out,
            contact_solimp_out,
            contact_dim_out,
            contact_geom_out,
            contact_worldid_out,
            contact_type_out,
            contact_geomcollisionid_out,
            nacon_out,
        )


@wp.func
def sphere_box_wrapper(
        # Data in:
        naconmax_in: int,
        # In:
        sphere: Geom,
        box: Geom,
        worldid: int,
        gap: float,
        condim: int,
        friction: vec5,
        solref: wp.vec2,
        solreffriction: wp.vec2,
        solimp: vec5,
        geoms: wp.vec2i,
        pairid: wp.vec2i,
        # Data out:
        contact_dist_out: wp.array(dtype=float),
        contact_pos_out: wp.array(dtype=wp.vec3),
        contact_frame_out: wp.array(dtype=wp.mat33),
        contact_friction_out: wp.array(dtype=vec5),
        contact_solref_out: wp.array(dtype=wp.vec2),
        contact_solreffriction_out: wp.array(dtype=wp.vec2),
        contact_solimp_out: wp.array(dtype=vec5),
        contact_dim_out: wp.array(dtype=int),
        contact_geom_out: wp.array(dtype=wp.vec2i),
        contact_worldid_out: wp.array(dtype=int),
        contact_type_out: wp.array(dtype=int),
        contact_geomcollisionid_out: wp.array(dtype=int),
        nacon_out: wp.array(dtype=int),
):
    dist, pos, normal = sphere_box(sphere.pos, sphere.size[0], box.pos, box.rot,
                                   box.size)

    write_contact(
        naconmax_in,
        0,
        dist,
        pos,
        make_frame(normal),
        gap,
        condim,
        friction,
        solref,
        solreffriction,
        solimp,
        geoms,
        pairid,
        worldid,
        contact_dist_out,
        contact_pos_out,
        contact_frame_out,
        contact_friction_out,
        contact_solref_out,
        contact_solreffriction_out,
        contact_solimp_out,
        contact_dim_out,
        contact_geom_out,
        contact_worldid_out,
        contact_type_out,
        contact_geomcollisionid_out,
        nacon_out,
    )


@wp.func
def capsule_box_wrapper(
        # Data in:
        naconmax_in: int,
        # In:
        cap: Geom,
        box: Geom,
        worldid: int,
        gap: float,
        condim: int,
        friction: vec5,
        solref: wp.vec2,
        solreffriction: wp.vec2,
        solimp: vec5,
        geoms: wp.vec2i,
        pairid: wp.vec2i,
        # Data out:
        contact_dist_out: wp.array(dtype=float),
        contact_pos_out: wp.array(dtype=wp.vec3),
        contact_frame_out: wp.array(dtype=wp.mat33),
        contact_friction_out: wp.array(dtype=vec5),
        contact_solref_out: wp.array(dtype=wp.vec2),
        contact_solreffriction_out: wp.array(dtype=wp.vec2),
        contact_solimp_out: wp.array(dtype=vec5),
        contact_dim_out: wp.array(dtype=int),
        contact_geom_out: wp.array(dtype=wp.vec2i),
        contact_worldid_out: wp.array(dtype=int),
        contact_type_out: wp.array(dtype=int),
        contact_geomcollisionid_out: wp.array(dtype=int),
        nacon_out: wp.array(dtype=int),
):
    """Calculates contacts between a capsule and a box."""
    # Extract capsule axis
    axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])

    # Call the core function to get contact geometry
    dist, pos, normal = capsule_box(
        cap.pos,
        axis,
        cap.size[0],  # capsule radius
        cap.size[1],  # capsule half length
        box.pos,
        box.rot,
        box.size,
    )

    # Loop over the contacts and write them
    for i in range(2):
        write_contact(
            naconmax_in,
            i,
            dist[i],
            pos[i],
            make_frame(normal[i]),
            gap,
            condim,
            friction,
            solref,
            solreffriction,
            solimp,
            geoms,
            pairid,
            worldid,
            contact_dist_out,
            contact_pos_out,
            contact_frame_out,
            contact_friction_out,
            contact_solref_out,
            contact_solreffriction_out,
            contact_solimp_out,
            contact_dim_out,
            contact_geom_out,
            contact_worldid_out,
            contact_type_out,
            contact_geomcollisionid_out,
            nacon_out,
        )


@wp.func
def box_box_wrapper(
        # Data in:
        naconmax_in: int,
        # In:
        box1: Geom,
        box2: Geom,
        worldid: int,
        gap: float,
        condim: int,
        friction: vec5,
        solref: wp.vec2,
        solreffriction: wp.vec2,
        solimp: vec5,
        geoms: wp.vec2i,
        pairid: wp.vec2i,
        # Data out:
        contact_dist_out: wp.array(dtype=float),
        contact_pos_out: wp.array(dtype=wp.vec3),
        contact_frame_out: wp.array(dtype=wp.mat33),
        contact_friction_out: wp.array(dtype=vec5),
        contact_solref_out: wp.array(dtype=wp.vec2),
        contact_solreffriction_out: wp.array(dtype=wp.vec2),
        contact_solimp_out: wp.array(dtype=vec5),
        contact_dim_out: wp.array(dtype=int),
        contact_geom_out: wp.array(dtype=wp.vec2i),
        contact_worldid_out: wp.array(dtype=int),
        contact_type_out: wp.array(dtype=int),
        contact_geomcollisionid_out: wp.array(dtype=int),
        nacon_out: wp.array(dtype=int),
):
    """Calculates contacts between two boxes."""
    # Call the core function to get contact geometry
    dist, pos, normal = box_box(
        box1.pos,
        box1.rot,
        box1.size,
        box2.pos,
        box2.rot,
        box2.size,
    )

    for i in range(8):
        write_contact(
            naconmax_in,
            i,
            dist[i],
            pos[i],
            make_frame(normal[i]),
            gap,
            condim,
            friction,
            solref,
            solreffriction,
            solimp,
            geoms,
            pairid,
            worldid,
            contact_dist_out,
            contact_pos_out,
            contact_frame_out,
            contact_friction_out,
            contact_solref_out,
            contact_solreffriction_out,
            contact_solimp_out,
            contact_dim_out,
            contact_geom_out,
            contact_worldid_out,
            contact_type_out,
            contact_geomcollisionid_out,
            nacon_out,
        )


_PRIMITIVE_COLLISIONS = {
    (GeomType.PLANE, GeomType.SPHERE): plane_sphere_wrapper,
    (GeomType.PLANE, GeomType.CAPSULE): plane_capsule_wrapper,
    # (GeomType.PLANE, GeomType.ELLIPSOID): plane_ellipsoid_wrapper,
    # (GeomType.PLANE, GeomType.CYLINDER): plane_cylinder_wrapper,
    # (GeomType.PLANE, GeomType.BOX): plane_box_wrapper,
    # (GeomType.PLANE, GeomType.MESH): plane_convex_wrapper,
    (GeomType.SPHERE, GeomType.SPHERE): sphere_sphere_wrapper,
    (GeomType.SPHERE, GeomType.CAPSULE): sphere_capsule_wrapper,
    # (GeomType.SPHERE, GeomType.CYLINDER): sphere_cylinder_wrapper,
    # (GeomType.SPHERE, GeomType.BOX): sphere_box_wrapper,
    (GeomType.CAPSULE, GeomType.CAPSULE): capsule_capsule_wrapper,
    # (GeomType.CAPSULE, GeomType.BOX): capsule_box_wrapper,
    # (GeomType.BOX, GeomType.BOX): box_box_wrapper,
}


def _check_primitive_collisions():
    prev_idx = -1
    for types in _PRIMITIVE_COLLISIONS.keys():
        idx = upper_trid_index(len(GeomType), types[0].value, types[1].value)
        if types[1] < types[0] or idx <= prev_idx:
            return False
        prev_idx = idx
    return True


assert _check_primitive_collisions(), "_PRIMITIVE_COLLISIONS is in invalid order"


@cache_kernel
def _create_narrowphase_kernel(primitive_collisions_types,
                               primitive_collisions_func):
    # AD: no unique here:
    # * we expect this generator to be called only once per model, so no repeated compilation
    # * module="unique" is generating problems because it uses the function name as the key
    #   that in turn will cause multiple kernels to be generated with the same name
    #   this is mostly problematic in cases like the UTs where we don't clear the kernel cache
    #   between different tests.

    @nested_kernel(enable_backward=False)
    def _primitive_narrowphase(
            # Model:
            geom_type: wp.array(dtype=int),
            geom_size: wp.array(dtype=wp.vec3),
            geom_friction: wp.array(dtype=wp.vec3),
            # Data in:
            geom_xpos_in: wp.array2d(dtype=wp.vec3),
            geom_xmat_in: wp.array2d(dtype=wp.mat33),
            naconmax_in: int,
            collision_pair_in: wp.array(dtype=wp.vec2i),
            collision_pairid_in: wp.array(dtype=wp.vec2i),
            collision_worldid_in: wp.array(dtype=int),
            ncollision_in: wp.array(dtype=int),
            # Data out:
            contact_dist_out: wp.array(dtype=float),
            contact_pos_out: wp.array(dtype=wp.vec3),
            contact_frame_out: wp.array(dtype=wp.mat33),
            contact_friction_out: wp.array(dtype=vec5),
            contact_dim_out: wp.array(dtype=int),
            contact_curvature_out: wp.array(dtype=float),
            contact_geom_out: wp.array(dtype=wp.vec2i),
            contact_worldid_out: wp.array(dtype=int),
            contact_geomcollisionid_out: wp.array(dtype=int),
            nacon_out: wp.array(dtype=int),
    ):
        tid = wp.tid()

        if tid >= ncollision_in[0]:
            return

        geoms = collision_pair_in[tid]
        worldid = collision_worldid_in[tid]

        _, condim, friction = contact_params(
            geom_friction,
            collision_pair_in,
            collision_pairid_in,
            tid,
            worldid,
        )

        geom1, geom2 = geom_collision_pair(
            geom_type,
            geom_size,
            geom_xpos_in,
            geom_xmat_in,
            geoms,
            worldid,
        )

        for i in range(wp.static(len(primitive_collisions_func))):
            collision_type1 = wp.static(primitive_collisions_types[i][0])
            collision_type2 = wp.static(primitive_collisions_types[i][1])
            type1 = geom_type[geoms[0]]
            type2 = geom_type[geoms[1]]
            if collision_type1 == type1 and collision_type2 == type2:
                wp.static(primitive_collisions_func[i])(
                    naconmax_in,
                    geom1,
                    geom2,
                    worldid,
                    condim,
                    friction,
                    geoms,
                    collision_pairid_in[tid],
                    contact_dist_out,
                    contact_pos_out,
                    contact_frame_out,
                    contact_friction_out,
                    contact_dim_out,
                    contact_curvature_out,
                    contact_geom_out,
                    contact_worldid_out,
                    contact_geomcollisionid_out,
                    nacon_out,
                )

    return _primitive_narrowphase


def _primitive_narrowphase_builder(m: Model):
    _primitive_collisions_types = []
    _primitive_collisions_func = []

    for types, func in _PRIMITIVE_COLLISIONS.items():
        idx = upper_trid_index(len(GeomType), types[0].value, types[1].value)
        if m.geom_pair_type_count[
            idx] and types not in _primitive_collisions_types:
            _primitive_collisions_types.append(types)
            _primitive_collisions_func.append(func)

    return _create_narrowphase_kernel(_primitive_collisions_types,
                                      _primitive_collisions_func)


@event_scope
def primitive_narrowphase(m: Model, d: Data):
    """Runs collision detection on primitive geom pairs discovered during broadphase.

    This function processes collision pairs involving primitive shapes that were
    identified during the broadphase stage. It computes detailed contact information
    such as distance, position, and frame, and populates the `d.contact` array.

    The primitive geom types: `PLANE`, `SPHERE`, `CAPSULE`, `CYLINDER`, and `BOX`.

    Additionally, collisions between planes and convex hulls.

    To improve performance, it dynamically builds and launches a kernel tailored to
    the specific primitive collision types present in the model, avoiding
    unnecessary checks for non-existent collision pairs.
    """
    # we need to figure out how to keep the overhead of this small - not launching anything
    # for pair types without collisions, as well as updating the launch dimensions.
    wp.launch(
        _primitive_narrowphase_builder(m),
        dim=d.naconmax,
        inputs=[
            m.geom_type,
            m.geom_size,
            m.geom_friction,
            d.geom_xpos,
            d.geom_xmat,
            d.naconmax,
            d.collision_pair,
            d.collision_pairid,
            d.collision_worldid,
            d.ncollision,
        ],
        outputs=[
            d.contact.dist,
            d.contact.pos,
            d.contact.frame,
            d.contact.friction,
            d.contact.dim,
            d.contact.curvature,
            d.contact.geom,
            d.contact.worldid,
            d.contact.geomcollisionid,
            d.nacon,
        ],
    )
