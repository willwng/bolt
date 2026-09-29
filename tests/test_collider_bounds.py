""" The broadphase bounds of a collider must enclose the collider as the narrowphase sees it """

from __future__ import annotations

import numpy as np
import opensim as osim
import pytest
import warp as wp

import bolt
from bolt.loader.converters import geom_helper
from bolt.types import GeomType
from models import model_path


def _capsule(radius: float, half_height: float):
    return bolt.convert_user_collider(bolt.UserGeomData(
        name="capsule", body_name="ground", geom_type=GeomType.CAPSULE,
        transform=wp.transform_identity(), size=wp.vec3(radius, half_height, radius)))


# The narrowphase (collision_primitive.py) takes a capsule's axis to be its local z, so its tips are at
# (0, 0, +/-(half_height + radius)) in the geom frame
CAPSULES = [(0.02, 0.05), (0.05, 0.02), (0.03, 0.0)]


@pytest.mark.parametrize("radius, half_height", CAPSULES)
def test_capsule_rbound_encloses_capsule(radius, half_height):
    """ The sphere and plane filters use rbound as the distance from the center to the farthest point """
    rbound = _capsule(radius, half_height).rbound
    assert rbound >= np.float32(half_height + radius), (
        f"rbound {rbound:.6g} < half_height + radius {half_height + radius:.6g}: "
        "contacts near the capsule's tips can be discarded")


@pytest.mark.parametrize("radius, half_height", CAPSULES)
def test_capsule_box_encloses_capsule(radius, half_height):
    """ The AABB and OBB filters use the box size as half-extents about the box center """
    center, size = _capsule(radius, half_height).aabb
    tip = np.array([0.0, 0.0, half_height + radius])
    for point in (tip, -tip):
        assert np.all(np.abs(point - np.array(center)) <= np.array(size) + 1e-7), (
            f"capsule tip {point.tolist()} lies outside the box of half-extents {[round(float(s), 4) for s in size]}: "
            "contacts near the capsule's tips can be discarded")


def _exact_half_extents(geom) -> np.ndarray:
    """ Half-extents of the collider in its own frame (capsule axis is local z; size = (r, half_height, r)) """
    size = np.array(geom.size, dtype=float)
    if geom.geom_type == GeomType.SPHERE:
        return np.full(3, size[0])
    if geom.geom_type == GeomType.ELLIPSOID:
        return size
    if geom.geom_type == GeomType.CAPSULE:
        return np.array([size[0], size[0], size[1] + size[0]])
    raise ValueError(f"no exact extents for {geom.geom_type}")


def test_boxes_are_tight():
    """ The AABB/OBB filters read the box size as half-extents, so the stored box is exactly the collider's """
    lr = bolt.load_model(model_path=model_path("example_model"), n_worlds=1,
                         integrator=bolt.IntegratorType.EULER_FIXED, requires_visuals=False)
    user_sphere = bolt.convert_user_collider(bolt.UserGeomData(
        name="sphere", body_name="ground", geom_type=GeomType.SPHERE,
        transform=wp.transform_identity(), size=wp.vec3(0.04, 0.04, 0.04)))
    geoms = [g for g in lr.colliders if g.geom_type != GeomType.PLANE] + [user_sphere, _capsule(0.02, 0.05)]
    assert {g.geom_type for g in geoms} >= {GeomType.SPHERE, GeomType.CAPSULE}
    for geom in geoms:
        center, size = geom.aabb
        np.testing.assert_allclose(np.array(size), _exact_half_extents(geom), rtol=1e-6, err_msg=geom.name)
        np.testing.assert_allclose(np.array(center), 0.0, err_msg=geom.name)


def test_ellipsoid_box_is_tight():
    """ example_model has no plain ellipsoids (its ContactEllipsoids are capsules), so build one directly """
    model = osim.Model(model_path("example_model"))
    radii = (0.03, 0.02, 0.01)
    ellipsoid = osim.ContactEllipsoid(osim.Vec3(*radii), osim.Vec3(0.0), osim.Vec3(0.0),
                                      model.getBodySet().get("calcn_r"), "ellipsoid")
    geom_type, _, (center, size), rbound = geom_helper.collect_geom_type_sizes(ellipsoid)
    assert geom_type == GeomType.ELLIPSOID
    np.testing.assert_allclose(np.array(size), radii, rtol=1e-6)
    np.testing.assert_allclose(np.array(center), 0.0)
    assert rbound == pytest.approx(max(radii))
