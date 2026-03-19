import warp as wp

from .types import Data
from .types import Model
from .consts import MSK_MINVAL
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _compute_path_kernel(
        # Model:
        muscle_pts_adr: wp.array(dtype=int),
        muscle_pts_num: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        site_pos_G_in: wp.array2d(dtype=wp.vec3),
        site_vel_G_in: wp.array2d(dtype=wp.vec3),
        # Data out:
        muscle_length_out: wp.array2d(dtype=float),
        muscle_velocity_out: wp.array2d(dtype=float),
):
    worldid, muscle_id = wp.tid()
    if integration_done_in[worldid]:
        return

    pts_adr = muscle_pts_adr[muscle_id]
    pts_num = muscle_pts_num[muscle_id]

    curr_length = float(0.0)
    curr_vel = float(0.0)
    for i in range(pts_num - 1):
        site1, site2 = pts_adr + i, pts_adr + i + 1
        p1_G, p2_G = site_pos_G_in[worldid, site1], site_pos_G_in[worldid, site2]
        diff = p2_G - p1_G
        dist = wp.length(diff)

        if dist < MSK_MINVAL:
            continue
        direction = diff / dist

        v1_G, v2_G = site_vel_G_in[worldid, site1], site_vel_G_in[worldid, site2]
        vel_diff = v2_G - v1_G

        curr_length += dist
        curr_vel += wp.dot(vel_diff, direction)

    muscle_length_out[worldid, muscle_id] = curr_length
    muscle_velocity_out[worldid, muscle_id] = curr_vel
    return


@wp.kernel
def _apply_muscle_force_kernel(
        # Model:
        muscle_pts_adr: wp.array(dtype=int),
        muscle_pts_num: wp.array(dtype=int),
        site_bodyid: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        muscle_actuation_in: wp.array2d(dtype=float),
        site_pos_G_in: wp.array2d(dtype=wp.vec3),
        site_rel_pos_B_in: wp.array2d(dtype=wp.vec3),
        # Data out:
        body_F_muscle_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, muscle_id = wp.tid()
    if integration_done_in[worldid]:
        return
    actuation = muscle_actuation_in[worldid, muscle_id]

    pts_adr = muscle_pts_adr[muscle_id]
    pts_num = muscle_pts_num[muscle_id]
    for i in range(pts_num - 1):
        site1, site2 = pts_adr + i, pts_adr + i + 1
        p1_G, p2_G = site_pos_G_in[worldid, site1], site_pos_G_in[worldid, site2]
        diff = p2_G - p1_G
        dist = wp.length(diff)

        if dist < MSK_MINVAL:
            continue
        direction = diff / dist
        muscle_frc = actuation * direction

        body1, body2 = site_bodyid[site1], site_bodyid[site2]
        s1_G, s2_G = site_rel_pos_B_in[worldid, site1], site_rel_pos_B_in[worldid, site2]
        wp.atomic_add(body_F_muscle_out[worldid], body1, wp.spatial_vector(wp.cross(s1_G, muscle_frc), muscle_frc))
        wp.atomic_sub(body_F_muscle_out[worldid], body2, wp.spatial_vector(wp.cross(s2_G, muscle_frc), muscle_frc))
    return


@event_scope
def muscle_point_path(m: Model, d: Data):
    """ Computes the muscle path length for point-based paths """
    if m.nmuscle:
        wp.launch(
            _compute_path_kernel,
            dim=(d.nworld, m.nmuscle),
            inputs=[m.muscle_pts_adr, m.muscle_pts_num, d.integration_done, d.site_pos_G, d.site_vel_G],
            outputs=[d.muscle_velocity],
        )


@event_scope
def apply_muscle_force(m: Model, d: Data):
    if m.nmuscle:
        wp.launch(
            _apply_muscle_force_kernel,
            dim=(d.nworld, m.nmuscle),
            inputs=[
                m.muscle_pts_adr, m.muscle_pts_num, m.site_bodyid,
                d.integration_done, d.muscle_actuation, d.site_pos_G, d.site_rel_pos_B
            ],
            outputs=[d.body_F_muscle],
        )
