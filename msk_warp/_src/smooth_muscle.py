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

from . import math
from . import support
from .types import Data
from .types import Model
from .types import TILE_SIZE_SITE
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@event_scope
def compute_site_diffs(m: Model, d: Data):
    @wp.kernel
    def _compute_site_diffs_tiled(
            # Data in:
            site_xpos_in: wp.array2d(dtype=wp.vec3),
            site_xvel_in: wp.array2d(dtype=wp.vec3),
            # Data out:
            site_diff_vec_out: wp.array2d(dtype=wp.vec3),
            site_diff_len_out: wp.array2d(dtype=float),
            site_diff_vel_out: wp.array2d(dtype=float),
    ):
        worldid = wp.tid()
        xpos_in = site_xpos_in[worldid]
        xvel_in = site_xvel_in[worldid]

        diff_vec_out = site_diff_vec_out[worldid]
        diff_len_out = site_diff_len_out[worldid]
        diff_vel_out = site_diff_vel_out[worldid]

        n_diffs = wp.static(m.nsite - 1)
        n_tiles = wp.static((n_diffs + TILE_SIZE_SITE - 1) // TILE_SIZE_SITE)
        for i in range(n_tiles - 1):
            offset = i * TILE_SIZE_SITE
            p1 = wp.tile_load(xpos_in, TILE_SIZE_SITE, offset=offset)
            p2 = wp.tile_load(xpos_in, TILE_SIZE_SITE, offset=offset + 1)
            diff = wp.tile_map(wp.sub, p2, p1)
            length = wp.tile_map(wp.length, diff)
            vec = wp.tile_map(math.safe_div, diff, length)

            wp.tile_store(diff_vec_out, vec, offset=offset)
            wp.tile_store(diff_len_out, length, offset=offset)

            # Velocity
            v1 = wp.tile_load(xvel_in, TILE_SIZE_SITE, offset=offset)
            v2 = wp.tile_load(xvel_in, TILE_SIZE_SITE, offset=offset + 1)
            vel_diff = wp.tile_map(wp.sub, v2, v1)
            vel_proj = wp.tile_map(wp.dot, vel_diff, vec)
            wp.tile_store(diff_vel_out, vel_proj, offset=offset)

        # Remaining elements
        offset = wp.static((n_tiles - 1) * TILE_SIZE_SITE)
        rem = wp.static(n_diffs - offset)
        p1 = wp.tile_load(xpos_in, rem, offset=offset)
        p2 = wp.tile_load(xpos_in, rem, offset=offset + 1)
        diff = wp.tile_map(wp.sub, p2, p1)
        length = wp.tile_map(wp.length, diff)
        vec = wp.tile_map(math.safe_div, diff, length)
        wp.tile_store(diff_vec_out, vec, offset=offset)
        wp.tile_store(diff_len_out, length, offset=offset)
        v1 = wp.tile_load(xvel_in, rem, offset=offset)
        v2 = wp.tile_load(xvel_in, rem, offset=offset + 1)
        vel_diff = wp.tile_map(wp.sub, v2, v1)
        vel_proj = wp.tile_map(wp.dot, vel_diff, vec)
        wp.tile_store(diff_vel_out, vel_proj, offset=offset)

    wp.launch_tiled(
        _compute_site_diffs_tiled,
        dim=(d.nworld,),
        inputs=[d.site_xpos, d.site_xvel],
        outputs=[d.site_diff_vec, d.site_diff_len, d.site_diff_vel],
        block_dim=m.block_dim.site_diffs
    )


@wp.kernel
def _compute_path_kernel(
        # Model:
        muscle_pts_adr: wp.array(dtype=int),
        muscle_pts_num: wp.array(dtype=int),
        # Data in:
        site_diff_len_out: wp.array2d(dtype=float),
        site_diff_vel_out: wp.array2d(dtype=float),
        # Data out:
        muscle_length_out: wp.array2d(dtype=float),
        muscle_velocity_out: wp.array2d(dtype=float),
):
    worldid, muscle_id = wp.tid()
    pts_adr = muscle_pts_adr[muscle_id]
    n_pts = muscle_pts_num[muscle_id]
    for i in range(n_pts - 1):
        muscle_length_out[worldid, muscle_id] += site_diff_len_out[
            worldid, pts_adr + i]
        muscle_velocity_out[worldid, muscle_id] += site_diff_vel_out[
            worldid, pts_adr + i]


@event_scope
def compute_path(m: Model, d: Data):
    wp.launch(
        _compute_path_kernel,
        dim=(d.nworld, m.nmuscle),
        inputs=[m.muscle_pts_adr, m.muscle_pts_num, d.site_diff_len,
                d.site_diff_vel],
        outputs=[d.muscle_length, d.muscle_velocity],
    )


@wp.kernel
def _xfrc_muscles(
        # Model:
        muscle_pts_adr: wp.array(dtype=int),
        muscle_pts_num: wp.array(dtype=int),
        site_bodyid: wp.array(dtype=int),
        # Data in:
        muscle_actuation_in: wp.array2d(dtype=float),
        site_diff_vec_in: wp.array2d(dtype=wp.vec3),
        site_diff_len_in: wp.array2d(dtype=float),
        xpos_in: wp.array2d(dtype=wp.vec3),
        site_xpos_in: wp.array2d(dtype=wp.vec3),
        # Data out:
        xfrc_applied_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, muscle_id = wp.tid()
    actuation = muscle_actuation_in[worldid, muscle_id]
    actuation = 0.0
    pt_adr = muscle_pts_adr[muscle_id]
    pt_num = muscle_pts_num[muscle_id]

    for i in range(pt_num - 1):
        length = site_diff_len_in[worldid, pt_adr + i]
        if length < 1e-8:
            continue

        vec = site_diff_vec_in[worldid, pt_adr + i]
        site1, site2 = pt_adr + i, pt_adr + i + 1
        body1, body2 = site_bodyid[site1], site_bodyid[site2]

        p1, p2 = site_xpos_in[worldid, site1], site_xpos_in[worldid, site2]
        com1, com2 = xpos_in[worldid, body1], xpos_in[worldid, body2]

        muscle_frc = actuation * vec
        wp.atomic_add(xfrc_applied_out[worldid], body1,
                      support.force_at_point(muscle_frc, p1 - com1))
        wp.atomic_sub(xfrc_applied_out[worldid], body2,
                      support.force_at_point(muscle_frc, p2 - com2))


@event_scope
def muscle_path(m: Model, d: Data):
    """
    Computes the muscle path length and velocity.
    Length calculations can be done after fwd_position,
        but it's easier to fuse with path velocity calculation
     """
    if not m.nmuscle:
        return
    d.muscle_length.zero_()
    d.muscle_velocity.zero_()

    # Compute diffs between active sites
    compute_site_diffs(m, d)

    # Now we can compute the path
    compute_path(m, d)


@event_scope
def muscle_force(m: Model, d: Data):
    if m.nmuscle:
        wp.launch(
            _xfrc_muscles,
            dim=(d.nworld, m.nmuscle),
            inputs=[
                m.muscle_pts_adr,
                m.muscle_pts_num,
                m.site_bodyid,
                d.muscle_actuation,
                d.site_diff_vec,
                d.site_diff_len,
                d.xpos,
                d.site_xpos,
            ],
            outputs=[d.xfrc_applied],
        )
