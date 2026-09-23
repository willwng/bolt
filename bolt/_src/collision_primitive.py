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

from typing import Tuple

import warp as wp

from .collision_primitive_core import capsule_capsule
from .collision_primitive_core import plane_capsule
from .collision_primitive_core import plane_ellipsoid
from .collision_primitive_core import plane_sphere
from .collision_primitive_core import sphere_capsule
from .collision_primitive_core import sphere_sphere
from .math import make_frame
from .math import upper_trid_index
from .types import Data
from .types import GeomType
from .types import Model
from .types import vec5
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.struct
class Geom:
    pos: wp.vec3
    rot: wp.mat33
    normal: wp.vec3
    size: wp.vec3


@wp.func
def geom_collision_pair(
        # Model:
        geom_size: wp.array(dtype=wp.vec3),
        # Data in:
        geom_X_in: wp.array2d(dtype=wp.transform),
        # In:
        geoms: wp.vec2i,
        worldid: int,
) -> Tuple[Geom, Geom]:
    geom1 = Geom()
    geom2 = Geom()

    g1 = geoms[0]
    g2 = geoms[1]

    geom_X1 = geom_X_in[worldid, g1]
    geom1.pos = wp.transform_get_translation(geom_X1)
    geom1.rot = wp.quat_to_matrix(wp.transform_get_rotation(geom_X1))
    geom1.size = geom_size[g1]
    geom1.normal = wp.vec3(geom1.rot[0, 1], geom1.rot[1, 1], geom1.rot[2, 1])  # plane

    geom_X2 = geom_X_in[worldid, g2]
    geom2.pos = wp.transform_get_translation(geom_X2)
    geom2.rot = wp.quat_to_matrix(wp.transform_get_rotation(geom_X2))
    geom2.size = geom_size[g2]
    geom2.normal = wp.vec3(geom2.rot[0, 1], geom2.rot[1, 1], geom2.rot[2, 1])  # plane

    return geom1, geom2


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
        stiffness_in: float,
        dissipation_in: float,
        transition_velocity_in: float,
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
        contact_stiffness_out: wp.array(dtype=float),
        contact_dissipation_out: wp.array(dtype=float),
        contact_transition_velocity_out: wp.array(dtype=float),
        contact_geom_out: wp.array(dtype=wp.vec2i),
        contact_worldid_out: wp.array(dtype=int),
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
        contact_stiffness_out[cid] = stiffness_in
        contact_dissipation_out[cid] = dissipation_in
        contact_transition_velocity_out[cid] = transition_velocity_in
        contact_friction_out[cid] = friction_in


@wp.func
def contact_params(
        # Model:
        geom_friction: wp.array(dtype=wp.vec3),
        geom_stiffness: wp.array(dtype=float),
        geom_dissipation: wp.array(dtype=float),
        geom_transition_velocity: wp.array(dtype=float),
        geom_priority: wp.array(dtype=int),
        # Data in:
        collision_pair_in: wp.array(dtype=wp.vec2i),
        collision_pairid_in: wp.array(dtype=wp.vec2i),
        # In:
        cid: int,
        worldid: int,
):
    geoms = collision_pair_in[cid]
    pairid = collision_pairid_in[cid][0]

    g1, g2 = geoms[0], geoms[1]
    p1, p2 = geom_priority[g1], geom_priority[g2]

    if p1 == p2:  # Same priority
        g_friction = wp.max(geom_friction[g1], geom_friction[g2])
        stiffness = wp.max(geom_stiffness[g1], geom_stiffness[g2])
        dissipation = wp.max(geom_dissipation[g1], geom_dissipation[g2])
        transition_vel = wp.max(geom_transition_velocity[g1], geom_transition_velocity[g2])
    elif p1 > p2:  # g1 has higher priority
        g_friction = geom_friction[g1]
        stiffness = geom_stiffness[g1]
        dissipation = geom_dissipation[g1]
        transition_vel = geom_transition_velocity[g1]
    else:
        g_friction = geom_friction[g2]
        stiffness = geom_stiffness[g2]
        dissipation = geom_dissipation[g2]
        transition_vel = geom_transition_velocity[g2]

    friction = vec5(g_friction[0], g_friction[1], g_friction[2], 0.0, 0.0)
    condim = 3  # hard coded for static, dynamic, viscous friction
    return geoms, condim, friction, stiffness, dissipation, transition_vel


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
        stiffness: float,
        dissipation: float,
        transition_velocity: float,
        geoms: wp.vec2i,
        pairid: wp.vec2i,
        # Data out:
        contact_dist_out: wp.array(dtype=float),
        contact_pos_out: wp.array(dtype=wp.vec3),
        contact_frame_out: wp.array(dtype=wp.mat33),
        contact_friction_out: wp.array(dtype=vec5),
        contact_dim_out: wp.array(dtype=int),
        contact_curvature_out: wp.array(dtype=float),
        contact_stiffness_out: wp.array(dtype=float),
        contact_dissipation_out: wp.array(dtype=float),
        contact_transition_velocity_out: wp.array(dtype=float),
        contact_geom_out: wp.array(dtype=wp.vec2i),
        contact_worldid_out: wp.array(dtype=int),
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
        stiffness,
        dissipation,
        transition_velocity,
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
        contact_stiffness_out,
        contact_dissipation_out,
        contact_transition_velocity_out,
        contact_geom_out,
        contact_worldid_out,
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
        stiffness: float,
        dissipation: float,
        transition_velocity: float,
        geoms: wp.vec2i,
        pairid: wp.vec2i,
        # Data out:
        contact_dist_out: wp.array(dtype=float),
        contact_pos_out: wp.array(dtype=wp.vec3),
        contact_frame_out: wp.array(dtype=wp.mat33),
        contact_friction_out: wp.array(dtype=vec5),
        contact_dim_out: wp.array(dtype=int),
        contact_curvature_out: wp.array(dtype=float),
        contact_stiffness_out: wp.array(dtype=float),
        contact_dissipation_out: wp.array(dtype=float),
        contact_transition_velocity_out: wp.array(dtype=float),
        contact_geom_out: wp.array(dtype=wp.vec2i),
        contact_worldid_out: wp.array(dtype=int),
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
        stiffness,
        dissipation,
        transition_velocity,
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
        contact_stiffness_out,
        contact_dissipation_out,
        contact_transition_velocity_out,
        contact_geom_out,
        contact_worldid_out,
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
        stiffness: float,
        dissipation: float,
        transition_velocity: float,
        geoms: wp.vec2i,
        pairid: wp.vec2i,
        # Data out:
        contact_dist_out: wp.array(dtype=float),
        contact_pos_out: wp.array(dtype=wp.vec3),
        contact_frame_out: wp.array(dtype=wp.mat33),
        contact_friction_out: wp.array(dtype=vec5),
        contact_dim_out: wp.array(dtype=int),
        contact_curvature_out: wp.array(dtype=float),
        contact_stiffness_out: wp.array(dtype=float),
        contact_dissipation_out: wp.array(dtype=float),
        contact_transition_velocity_out: wp.array(dtype=float),
        contact_geom_out: wp.array(dtype=wp.vec2i),
        contact_worldid_out: wp.array(dtype=int),
        nacon_out: wp.array(dtype=int),
):
    """Calculates one contact between a sphere and a capsule."""
    # capsule axis
    capsule_axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])
    dist, pos, normal = sphere_capsule(sphere.pos, sphere.size[0], cap.pos, capsule_axis, cap.size[0], cap.size[1])
    curvature = wp.sqrt(sphere.size[0] * cap.size[0])

    write_contact(
        naconmax_in,
        0,
        dist,
        pos,
        make_frame(normal),
        condim,
        curvature,
        stiffness,
        dissipation,
        transition_velocity,
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
        contact_stiffness_out,
        contact_dissipation_out,
        contact_transition_velocity_out,
        contact_geom_out,
        contact_worldid_out,
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
        stiffness: float,
        dissipation: float,
        transition_velocity: float,
        geoms: wp.vec2i,
        pairid: wp.vec2i,
        # Data out:
        contact_dist_out: wp.array(dtype=float),
        contact_pos_out: wp.array(dtype=wp.vec3),
        contact_frame_out: wp.array(dtype=wp.mat33),
        contact_friction_out: wp.array(dtype=vec5),
        contact_dim_out: wp.array(dtype=int),
        contact_curvature_out: wp.array(dtype=float),
        contact_stiffness_out: wp.array(dtype=float),
        contact_dissipation_out: wp.array(dtype=float),
        contact_transition_velocity_out: wp.array(dtype=float),
        contact_geom_out: wp.array(dtype=wp.vec2i),
        contact_worldid_out: wp.array(dtype=int),
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

    for i in range(2):
        write_contact(
            naconmax_in,
            0,
            dist[i],
            wp.vec3(pos[i, 0], pos[i, 1], pos[i, 2]),
            make_frame(wp.vec3(normal[i, 0], normal[i, 1], normal[i, 2])),
            condim,
            curvature,
            stiffness,
            dissipation,
            transition_velocity,
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
            contact_stiffness_out,
            contact_dissipation_out,
            contact_transition_velocity_out,
            contact_geom_out,
            contact_worldid_out,
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
        stiffness: float,
        dissipation: float,
        transition_velocity: float,
        geoms: wp.vec2i,
        pairid: wp.vec2i,
        # Data out:
        contact_dist_out: wp.array(dtype=float),
        contact_pos_out: wp.array(dtype=wp.vec3),
        contact_frame_out: wp.array(dtype=wp.mat33),
        contact_friction_out: wp.array(dtype=vec5),
        contact_dim_out: wp.array(dtype=int),
        contact_curvature_out: wp.array(dtype=float),
        contact_stiffness_out: wp.array(dtype=float),
        contact_dissipation_out: wp.array(dtype=float),
        contact_transition_velocity_out: wp.array(dtype=float),
        contact_geom_out: wp.array(dtype=wp.vec2i),
        contact_worldid_out: wp.array(dtype=int),
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
            stiffness,
            dissipation,
            transition_velocity,
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
            contact_stiffness_out,
            contact_dissipation_out,
            contact_transition_velocity_out,
            contact_geom_out,
            contact_worldid_out,
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
        condim: int,
        friction: vec5,
        stiffness: float,
        dissipation: float,
        transition_velocity: float,
        geoms: wp.vec2i,
        pairid: wp.vec2i,
        # Data out:
        contact_dist_out: wp.array(dtype=float),
        contact_pos_out: wp.array(dtype=wp.vec3),
        contact_frame_out: wp.array(dtype=wp.mat33),
        contact_friction_out: wp.array(dtype=vec5),
        contact_dim_out: wp.array(dtype=int),
        contact_curvature_out: wp.array(dtype=float),
        contact_stiffness_out: wp.array(dtype=float),
        contact_dissipation_out: wp.array(dtype=float),
        contact_transition_velocity_out: wp.array(dtype=float),
        contact_geom_out: wp.array(dtype=wp.vec2i),
        contact_worldid_out: wp.array(dtype=int),
        nacon_out: wp.array(dtype=int),
):
    """Calculates contacts between an ellipsoid and a plane."""
    dist, pos, normal = plane_ellipsoid(plane.normal, plane.pos, ellipsoid.pos,
                                        ellipsoid.rot, ellipsoid.size)
    curvature = wp.cbrt(ellipsoid.size[0] * ellipsoid.size[1] * ellipsoid.size[2]) # todo

    write_contact(
        naconmax_in,
        0,
        dist,
        pos,
        make_frame(normal),
        condim,
        curvature,
        stiffness,
        dissipation,
        transition_velocity,
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
        contact_stiffness_out,
        contact_dissipation_out,
        contact_transition_velocity_out,
        contact_geom_out,
        contact_worldid_out,
        nacon_out,
    )


_PRIMITIVE_COLLISIONS = {
    (GeomType.PLANE, GeomType.SPHERE): plane_sphere_wrapper,
    (GeomType.PLANE, GeomType.CAPSULE): plane_capsule_wrapper,
    (GeomType.PLANE, GeomType.ELLIPSOID): plane_ellipsoid_wrapper,
    (GeomType.SPHERE, GeomType.SPHERE): sphere_sphere_wrapper,
    (GeomType.SPHERE, GeomType.CAPSULE): sphere_capsule_wrapper,
    (GeomType.CAPSULE, GeomType.CAPSULE): capsule_capsule_wrapper,
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


def _create_narrowphase_kernel(primitive_collisions_types,
                               primitive_collisions_func):
    # AD: no unique here:
    # * we expect this generator to be called only once per model, so no repeated compilation
    # * module="unique" is generating problems because it uses the function name as the key
    #   that in turn will cause multiple kernels to be generated with the same name
    #   this is mostly problematic in cases like the UTs where we don't clear the kernel cache
    #   between different tests.

    @wp.kernel(module="unique", enable_backward=False)
    def _primitive_narrowphase(
            # Model:
            geom_type: wp.array(dtype=int),
            geom_size: wp.array(dtype=wp.vec3),
            geom_friction: wp.array(dtype=wp.vec3),
            geom_stiffness: wp.array(dtype=float),
            geom_dissipation: wp.array(dtype=float),
            geom_transition_velocity: wp.array(dtype=float),
            geom_priority: wp.array(dtype=int),
            # Data in:
            geom_X_in: wp.array2d(dtype=wp.transform),
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
            contact_stiffness_out: wp.array(dtype=float),
            contact_dissipation_out: wp.array(dtype=float),
            contact_transition_velocity_out: wp.array(dtype=float),
            contact_geom_out: wp.array(dtype=wp.vec2i),
            contact_worldid_out: wp.array(dtype=int),
            nacon_out: wp.array(dtype=int),
    ):
        tid = wp.tid()

        if tid >= ncollision_in[0]:
            return

        geoms = collision_pair_in[tid]
        worldid = collision_worldid_in[tid]

        _, condim, friction, stiffness, dissipation, transition_vel = contact_params(
            geom_friction,
            geom_stiffness,
            geom_dissipation,
            geom_transition_velocity,
            geom_priority,
            collision_pair_in,
            collision_pairid_in,
            tid,
            worldid,
        )

        geom1, geom2 = geom_collision_pair(
            geom_size,
            geom_X_in,
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
                    stiffness,
                    dissipation,
                    transition_vel,
                    geoms,
                    collision_pairid_in[tid],
                    contact_dist_out,
                    contact_pos_out,
                    contact_frame_out,
                    contact_friction_out,
                    contact_dim_out,
                    contact_curvature_out,
                    contact_stiffness_out,
                    contact_dissipation_out,
                    contact_transition_velocity_out,
                    contact_geom_out,
                    contact_worldid_out,
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

    return _create_narrowphase_kernel(_primitive_collisions_types, _primitive_collisions_func)


@event_scope
def narrowphase(m: Model, d: Data):
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
            m.geom_stiffness,
            m.geom_dissipation,
            m.geom_transition_velocity,
            m.geom_priority,
            d.geom_X,
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
            d.contact.stiffness,
            d.contact.dissipation,
            d.contact.transition_velocity,
            d.contact.geom,
            d.contact.worldid,
            d.nacon,
        ],
    )
