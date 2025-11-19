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

import warp as wp

from .collision_primitive import primitive_narrowphase
from .consts import MJ_MAXVAL
from .types import Data
from .types import Model
from .warp_util import cache_kernel
from .warp_util import event_scope
from .warp_util import kernel as nested_kernel

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _zero_nacon_ncollision(
        # Data out:
        nacon_out: wp.array(dtype=int),
        ncollision_out: wp.array(dtype=int),
):
    ncollision_out[0] = 0
    nacon_out[0] = 0


@wp.func
def _plane_filter(
        size1: float, size2: float,
        xpos1: wp.vec3, xpos2: wp.vec3, xmat1: wp.mat33, xmat2: wp.mat33
) -> bool:
    if size1 == 0.0:
        # geom1 is a plane
        dist = wp.dot(xpos2 - xpos1,
                      wp.vec3(xmat1[0, 2], xmat1[1, 2], xmat1[2, 2]))
        return dist <= size2
    elif size2 == 0.0:
        # geom2 is a plane
        dist = wp.dot(xpos1 - xpos2,
                      wp.vec3(xmat2[0, 2], xmat2[1, 2], xmat2[2, 2]))
        return dist <= size1

    return True


@wp.func
def _sphere_filter(size1: float, size2: float,
                   xpos1: wp.vec3, xpos2: wp.vec3) -> bool:
    bound = size1 + size2
    dif = xpos2 - xpos1
    dist_sq = wp.dot(dif, dif)
    return dist_sq <= bound * bound


# TODO(team): improve performance by precomputing bounding box
@wp.func
def _aabb_filter(
        # In:
        center1: wp.vec3,
        center2: wp.vec3,
        size1: wp.vec3,
        size2: wp.vec3,
        xpos1: wp.vec3,
        xpos2: wp.vec3,
        xmat1: wp.mat33,
        xmat2: wp.mat33,
) -> bool:
    """Axis aligned boxes collision.

    references: see Ericson, Real-time Collision Detection section 4.2.
                filterBox: filter contact based on global AABBs.
    """
    center1 = xmat1 @ center1 + xpos1
    center2 = xmat2 @ center2 + xpos2

    max_x1 = -MJ_MAXVAL
    max_y1 = -MJ_MAXVAL
    max_z1 = -MJ_MAXVAL
    min_x1 = MJ_MAXVAL
    min_y1 = MJ_MAXVAL
    min_z1 = MJ_MAXVAL

    max_x2 = -MJ_MAXVAL
    max_y2 = -MJ_MAXVAL
    max_z2 = -MJ_MAXVAL
    min_x2 = MJ_MAXVAL
    min_y2 = MJ_MAXVAL
    min_z2 = MJ_MAXVAL

    sign = wp.vec2(-1.0, 1.0)

    for i in range(2):
        for j in range(2):
            for k in range(2):
                corner1 = wp.vec3(sign[i] * size1[0], sign[j] * size1[1],
                                  sign[k] * size1[2])
                pos1 = xmat1 @ corner1

                corner2 = wp.vec3(sign[i] * size2[0], sign[j] * size2[1],
                                  sign[k] * size2[2])
                pos2 = xmat2 @ corner2

                if pos1[0] > max_x1:
                    max_x1 = pos1[0]

                if pos1[1] > max_y1:
                    max_y1 = pos1[1]

                if pos1[2] > max_z1:
                    max_z1 = pos1[2]

                if pos1[0] < min_x1:
                    min_x1 = pos1[0]

                if pos1[1] < min_y1:
                    min_y1 = pos1[1]

                if pos1[2] < min_z1:
                    min_z1 = pos1[2]

                if pos2[0] > max_x2:
                    max_x2 = pos2[0]

                if pos2[1] > max_y2:
                    max_y2 = pos2[1]

                if pos2[2] > max_z2:
                    max_z2 = pos2[2]

                if pos2[0] < min_x2:
                    min_x2 = pos2[0]

                if pos2[1] < min_y2:
                    min_y2 = pos2[1]

                if pos2[2] < min_z2:
                    min_z2 = pos2[2]

    if center1[0] + max_x1 < center2[0] + min_x2:
        return False
    if center1[1] + max_y1 < center2[1] + min_y2:
        return False
    if center1[2] + max_z1 < center2[2] + min_z2:
        return False
    if center2[0] + max_x2 < center1[0] + min_x1:
        return False
    if center2[1] + max_y2 < center1[1] + min_y1:
        return False
    if center2[2] + max_z2 < center1[2] + min_z1:
        return False

    return True


mat23 = wp.types.matrix(shape=(2, 3), dtype=float)
mat63 = wp.types.matrix(shape=(6, 3), dtype=float)


# TODO(team): improve performance by precomputing bounding box
@wp.func
def _obb_filter(
        # In:
        center1: wp.vec3,
        center2: wp.vec3,
        size1: wp.vec3,
        size2: wp.vec3,
        xpos1: wp.vec3,
        xpos2: wp.vec3,
        xmat1: wp.mat33,
        xmat2: wp.mat33,
) -> bool:
    """Oriented bounding boxes collision (see Gottschalk et al.), see mj_collideOBB."""
    xcenter = mat23()
    normal = mat63()
    proj = wp.vec2()
    radius = wp.vec2()

    # compute centers in local coordinates
    xcenter[0] = xmat1 @ center1 + xpos1
    xcenter[1] = xmat2 @ center2 + xpos2

    # compute normals in global coordinates
    normal[0] = wp.vec3(xmat1[0, 0], xmat1[1, 0], xmat1[2, 0])
    normal[1] = wp.vec3(xmat1[0, 1], xmat1[1, 1], xmat1[2, 1])
    normal[2] = wp.vec3(xmat1[0, 2], xmat1[1, 2], xmat1[2, 2])
    normal[3] = wp.vec3(xmat2[0, 0], xmat2[1, 0], xmat2[2, 0])
    normal[4] = wp.vec3(xmat2[0, 1], xmat2[1, 1], xmat2[2, 1])
    normal[5] = wp.vec3(xmat2[0, 2], xmat2[1, 2], xmat2[2, 2])

    # check intersections
    for j in range(2):
        for k in range(3):
            for i in range(2):
                proj[i] = wp.dot(xcenter[i], normal[3 * j + k])
                if i == 0:
                    size = size1
                else:
                    size = size2

                # fmt: off
                radius[i] = (
                        wp.abs(size[0] * wp.dot(normal[3 * i + 0],
                                                normal[3 * j + k]))
                        + wp.abs(
                    size[1] * wp.dot(normal[3 * i + 1], normal[3 * j + k]))
                        + wp.abs(
                    size[2] * wp.dot(normal[3 * i + 2], normal[3 * j + k]))
                )
                # fmt: on
            if radius[0] + radius[1] < wp.abs(proj[1] - proj[0]):
                return False

    return True


def _broadphase_filter(m: Model):
    @wp.func
    def func(
            # Model:
            geom_aabb: wp.array2d(dtype=wp.vec3),
            geom_rbound: wp.array(dtype=float),
            # Data in:
            geom_xpos_in: wp.array2d(dtype=wp.vec3),
            geom_xmat_in: wp.array2d(dtype=wp.mat33),
            # In:
            geom1: int,
            geom2: int,
            worldid: int,
    ) -> bool:
        # 1: plane
        # 2: sphere
        # 4: aabb
        # 8: obb

        # Bounding box
        center1, center2 = geom_aabb[geom1, 0], geom_aabb[geom2, 0]
        size1, size2 = geom_aabb[geom1, 1], geom_aabb[geom2, 1]

        # Bounding sphere
        rbound1, rbound2 = geom_rbound[geom1], geom_rbound[geom2]

        xpos1, xpos2 = geom_xpos_in[worldid, geom1], geom_xpos_in[
            worldid, geom2]
        xmat1, xmat2 = geom_xmat_in[worldid, geom1], geom_xmat_in[
            worldid, geom2]

        if rbound1 == 0.0 or rbound2 == 0.0:
            return _plane_filter(rbound1, rbound2, xpos1,
                                 xpos2, xmat1, xmat2)
        else:
            if not _sphere_filter(rbound1, rbound2, xpos1,
                                  xpos2):
                return False
            if not _aabb_filter(center1, center2, size1, size2,
                                xpos1, xpos2, xmat1, xmat2):
                return False
            if not _obb_filter(center1, center2, size1, size2,
                               xpos1, xpos2, xmat1, xmat2):
                return False

        return True

    return func


@wp.func
def _add_geom_pair(
        # Model:
        geom_type: wp.array(dtype=int),
        nxn_pairid: wp.array(dtype=wp.vec2i),
        # Data in:
        naconmax_in: int,
        # In:
        geom1: int,
        geom2: int,
        worldid: int,
        nxnid: int,
        # Data out:
        collision_pair_out: wp.array(dtype=wp.vec2i),
        collision_pairid_out: wp.array(dtype=wp.vec2i),
        collision_worldid_out: wp.array(dtype=int),
        ncollision_out: wp.array(dtype=int),
):
    pairid = wp.atomic_add(ncollision_out, 0, 1)

    if pairid >= naconmax_in:
        return

    type1 = geom_type[geom1]
    type2 = geom_type[geom2]

    if type1 > type2:
        pair = wp.vec2i(geom2, geom1)
    else:
        pair = wp.vec2i(geom1, geom2)

    collision_pair_out[pairid] = pair
    collision_pairid_out[pairid] = nxn_pairid[nxnid]
    collision_worldid_out[pairid] = worldid


@cache_kernel
def _nxn_broadphase(broadphase_filter):
    @nested_kernel(module="unique", enable_backward=False)
    def kernel(
            # Model:
            geom_type: wp.array(dtype=int),
            geom_aabb: wp.array2d(dtype=wp.vec3),
            geom_rbound: wp.array(dtype=float),
            nxn_geom_pair: wp.array(dtype=wp.vec2i),
            nxn_pairid: wp.array(dtype=wp.vec2i),
            # Data in:
            naconmax_in: int,
            geom_xpos_in: wp.array2d(dtype=wp.vec3),
            geom_xmat_in: wp.array2d(dtype=wp.mat33),
            # Data out:
            collision_pair_out: wp.array(dtype=wp.vec2i),
            collision_pairid_out: wp.array(dtype=wp.vec2i),
            collision_worldid_out: wp.array(dtype=int),
            ncollision_out: wp.array(dtype=int),
    ):
        worldid, elementid = wp.tid()

        geom = nxn_geom_pair[elementid]
        geom1 = geom[0]
        geom2 = geom[1]

        if (broadphase_filter(
                geom_aabb, geom_rbound, geom_xpos_in,
                geom_xmat_in, geom1, geom2, worldid)
                or nxn_pairid[elementid][1] >= 0):
            _add_geom_pair(
                geom_type,
                nxn_pairid,
                naconmax_in,
                geom1,
                geom2,
                worldid,
                elementid,
                collision_pair_out,
                collision_pairid_out,
                collision_worldid_out,
                ncollision_out,
            )

    return kernel


@event_scope
def nxn_broadphase(m: Model, d: Data):
    """Runs broadphase collision detection using a brute-force N-squared approach.

    This function iterates through a pre-filtered list of all possible geometry pairs and
    performs a quick bounding sphere check to identify potential collisions.

    For each pair that passes the sphere check, it populates the collision arrays in `d`
    (`d.collision_pair`, `d.collision_pairid`, etc.), which are then consumed by the
    narrowphase.

    The initial list of pairs is filtered at model creation time to exclude pairs based on
    `contype`/`conaffinity`, parent-child relationships, and explicit `<exclude>` tags.
    """
    broadphase_filter = _broadphase_filter(m)
    wp.launch(
        _nxn_broadphase(broadphase_filter),
        dim=(d.nworld, m.nxn_geom_pair_filtered.shape[0]),
        inputs=[
            m.geom_type,
            m.geom_aabb,
            m.geom_rbound,
            m.nxn_geom_pair_filtered,
            m.nxn_pairid_filtered,
            d.naconmax,
            d.geom_xpos,
            d.geom_xmat,
        ],
        outputs=[
            d.collision_pair,
            d.collision_pairid,
            d.collision_worldid,
            d.ncollision,
        ],
    )


def _narrowphase(m, d):
    primitive_narrowphase(m, d)


@event_scope
def collision(m: Model, d: Data):
    """Runs the full collision detection pipeline.

    This function orchestrates the broadphase and narrowphase collision detection stages. It
    first identifies potential collision pairs using a broadphase algorithm (either N-squared
    or Sweep-and-Prune, based on `m.opt.broadphase`). Then, for each potential pair, it
    performs narrowphase collision detection to compute detailed contact information like
    distance, position, and frame.

    The results are used to populate the `d.contact` array, and the total number of contacts
    is stored in `d.nacon`.  If `d.nacon` is larger than `d.naconmax` then an overflow has
    occurred and the remaining contacts will be skipped.  If this happens, raise the `nconmax`
    parameter in `io.make_data` or `io.put_data`.

    This function will do nothing except zero out arrays if collision detection is disabled
    via `m.opt.disableflags` or if `d.nacon` is 0.
    """
    # zero contact and collision counters
    wp.launch(_zero_nacon_ncollision, dim=1, outputs=[d.nacon, d.ncollision])

    if d.naconmax == 0:
        return

    # if m.opt.broadphase == BroadphaseType.NXN:
    nxn_broadphase(m, d)

    _narrowphase(m, d)
