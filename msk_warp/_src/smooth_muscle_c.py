import warp as wp

from . import math
from .types import Data
from .types import Model
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _init_sites_arr(
        # Data out:
        site_active_out: wp.array2d(dtype=bool),
        muscle_active_sites_out: wp.array2d(dtype=int),
):
    worldid, site_id = wp.tid()
    site_active_out[worldid, site_id] = True
    muscle_active_sites_out[worldid, site_id] = -1


@wp.kernel
def _conditional_sites(
        # Model:
        site_cond_id: wp.array(dtype=int),
        site_cond_qadr: wp.array(dtype=int),
        site_cond_range: wp.array(dtype=wp.vec2),
        # Data in:
        qpos_in: wp.array2d(dtype=float),
        # Data out:
        site_active_out: wp.array2d(dtype=bool),
):
    worldid, cond_site_id = wp.tid()
    qadr = site_cond_qadr[cond_site_id]
    q = qpos_in[worldid, qadr]
    q_range = site_cond_range[cond_site_id]

    if q < q_range[0] or q > q_range[1]:
        site_id = site_cond_id[cond_site_id]
        site_active_out[worldid, site_id] = False


@wp.kernel
def _collect_active_sites(
        # Model:
        muscle_pts_num: wp.array(dtype=int),
        muscle_pts_adr: wp.array(dtype=int),
        # Data in:
        sites_active_in: wp.array2d(dtype=bool),
        # Data out:
        muscle_active_sites_out: wp.array2d(dtype=int),
        muscle_num_active: wp.array2d(dtype=int),
):
    worldid, muscle_id = wp.tid()

    n_sites = muscle_pts_num[muscle_id]
    pts_adr = muscle_pts_adr[muscle_id]
    num_active = int(0)
    for i in range(n_sites):
        site_id = pts_adr + i
        if sites_active_in[worldid, site_id]:
            muscle_active_sites_out[worldid, pts_adr + num_active] = site_id
            num_active += 1

    muscle_num_active[worldid, muscle_id] = num_active


@wp.kernel
def _compute_active_site_diffs(
        # Data in:
        muscle_active_sites_in: wp.array2d(dtype=int),
        site_xpos_in: wp.array2d(dtype=wp.vec3),
        site_xvel_in: wp.array2d(dtype=wp.vec3),
        # Data out:
        site_diff_vec_out: wp.array2d(dtype=wp.vec3),
        site_diff_len_out: wp.array2d(dtype=float),
        site_diff_vel_out: wp.array2d(dtype=float),
):
    worldid, site_diff_id = wp.tid()
    # Get ids of the two sites
    site_1 = muscle_active_sites_in[worldid, site_diff_id]
    site_2 = muscle_active_sites_in[worldid, site_diff_id + 1]

    # End of active sites
    if site_1 == -1 or site_2 == -1:
        return

    p1, p2 = site_xpos_in[worldid, site_1], site_xpos_in[worldid, site_2]
    v1, v2 = site_xvel_in[worldid, site_1], site_xvel_in[worldid, site_2]
    vec, length = math.normalize_with_norm(p2 - p1)
    site_diff_vec_out[worldid, site_diff_id] = vec
    site_diff_len_out[worldid, site_diff_id] = length

    if length > 1e-8:
        site_diff_vel_out[worldid, site_diff_id] = wp.dot((v2 - v1), vec)
    else:
        site_diff_vel_out[worldid, site_diff_id] = 0.0


@wp.kernel
def _compute_path(
        # Model:
        muscle_pts_adr: wp.array(dtype=int),
        # Data in:
        muscle_num_active: wp.array2d(dtype=int),
        site_diff_len_out: wp.array2d(dtype=float),
        site_diff_vel_out: wp.array2d(dtype=float),
        # Data out:
        muscle_length_out: wp.array2d(dtype=float),
        muscle_velocity_out: wp.array2d(dtype=float),
):
    worldid, muscle_id = wp.tid()
    pts_adr = muscle_pts_adr[muscle_id]
    n_active = muscle_num_active[worldid, muscle_id]
    for i in range(n_active - 1):
        muscle_length_out[worldid, muscle_id] += site_diff_len_out[
            worldid, pts_adr + i]
        muscle_velocity_out[worldid, muscle_id] += site_diff_vel_out[
            worldid, pts_adr + i]


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

    # fill_ doesn't work with graph capture so we just launch a kernel
    wp.launch(
        _init_sites_arr,
        dim=(d.nworld, m.nsite),
        inputs=[],
        outputs=[d.site_active, d.muscle_active_sites],
    )

    # Check whether conditional sites are active
    wp.launch(
        _conditional_sites,
        dim=(d.nworld, m.nsite_cond),
        inputs=[m.site_cond_id, m.site_cond_qadr, m.site_cond_range, d.qpos, ],
        outputs=[d.site_active],
    )

    # Build "compacted" list of active sites for each muscle
    wp.launch(
        _collect_active_sites,
        dim=(d.nworld, m.nmuscle),
        inputs=[m.muscle_pts_num, m.muscle_pts_adr, d.site_active, ],
        outputs=[d.muscle_active_sites, d.muscle_num_active],
    )

    # Compute diffs between active sites
    wp.launch(
        _compute_active_site_diffs,
        dim=(d.nworld, m.nsite - 1),
        inputs=[d.muscle_active_sites, d.site_xpos, d.site_xvel],
        outputs=[d.site_diff_vec, d.site_diff_len, d.site_diff_vel],
    )

    # Now we can compute the path
    wp.launch(
        _compute_path,
        dim=(d.nworld, m.nmuscle),
        inputs=[m.muscle_pts_adr, d.muscle_num_active, d.site_diff_len,
                d.site_diff_vel],
        outputs=[d.muscle_length, d.muscle_velocity],
    )
