""" Model-loading checks that don't need an OpenSim oracle """

from __future__ import annotations

import pytest
import warp as wp

import bolt
from bolt.types_consts import GeomType


@pytest.mark.parametrize("geom_type, size, rbound", [
    (GeomType.SPHERE, (0.1, 0.1, 0.1), 0.1),
    (GeomType.CAPSULE, (0.03, 0.04, 0.03), 0.05),
])
def test_convert_user_collider(geom_type, size, rbound):
    user_geom = bolt.UserGeomData(
        name="user_geom",
        body_name="calcn_r",
        geom_type=geom_type,
        transform=wp.transform_identity(),
        size=wp.vec3(size),
    )
    geom = bolt.convert_user_collider(user_geom)
    assert geom.geom_type == geom_type
    assert geom.rbound == pytest.approx(rbound)
