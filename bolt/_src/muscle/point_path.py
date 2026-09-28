import warp as wp

from bolt.consts import BOLT_MINVAL
from bolt.consts import COND_PATH_POINT_RANGE_TOL
from bolt.types import Data
from bolt.types import Model

from ..warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.func
def site_is_active(
        site: int,
        qpos: wp.array(dtype=float),
        site_cond_qposadr: wp.array(dtype=int),
        site_cond_range: wp.array(dtype=wp.vec2),
) -> bool:
    """ Whether a site is part of its muscle's path (ConditionalPathPoints are only active inside their range) """
    qposadr = site_cond_qposadr[site]
    if qposadr < 0:
        return True
    q = qpos[qposadr]
    q_range = site_cond_range[site]
    return q >= q_range[0] - COND_PATH_POINT_RANGE_TOL and q <= q_range[1] + COND_PATH_POINT_RANGE_TOL


@wp.func
def apply_moving_site_force(
        site: int,
        frc: wp.vec3,
        site_moving_qposadr: wp.array(dtype=wp.vec3i),
        site_moving_jac_G_in: wp.array(dtype=wp.mat33),
        qfrc_out: wp.array(dtype=float),
):
    """ A moving path point also moves relative to its body, so a force on it does work on its coordinates """
    qposadr = site_moving_qposadr[site]
    if qposadr[0] < 0:
        return
    jac = site_moving_jac_G_in[site]
    for i in range(3):
        wp.atomic_add(qfrc_out, qposadr[i], wp.dot(jac[i], frc))


@wp.kernel
def _compute_path_kernel(
        # Model:
        muscle_pts_adr: wp.array(dtype=int),
        muscle_pts_num: wp.array(dtype=int),
        site_cond_qposadr: wp.array(dtype=int),
        site_cond_range: wp.array(dtype=wp.vec2),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qpos_in: wp.array2d(dtype=float),
        site_pos_G_in: wp.array2d(dtype=wp.vec3),
        site_vel_G_in: wp.array2d(dtype=wp.vec3),
        # In:
        muscle_pt_group: wp.array(dtype=int),
        # Data out:
        muscle_length_out: wp.array2d(dtype=float),
        muscle_velocity_out: wp.array2d(dtype=float),
):
    worldid, nodeid = wp.tid()
    if integration_done_in[worldid]:
        return
    muscle_id = muscle_pt_group[nodeid]

    pts_adr = muscle_pts_adr[muscle_id]
    pts_num = muscle_pts_num[muscle_id]

    curr_length = float(0.0)
    curr_vel = float(0.0)
    site1 = int(-1)  # previous active site
    for i in range(pts_num):
        site2 = pts_adr + i
        if not site_is_active(site2, qpos_in[worldid], site_cond_qposadr, site_cond_range):
            continue
        if site1 >= 0:
            p1_G, p2_G = site_pos_G_in[worldid, site1], site_pos_G_in[worldid, site2]
            diff = p2_G - p1_G
            dist = wp.length(diff)

            if dist >= BOLT_MINVAL:
                direction = diff / dist

                v1_G, v2_G = site_vel_G_in[worldid, site1], site_vel_G_in[worldid, site2]
                vel_diff = v2_G - v1_G

                curr_length += dist
                curr_vel += wp.dot(vel_diff, direction)
        site1 = site2

    muscle_length_out[worldid, muscle_id] = curr_length
    muscle_velocity_out[worldid, muscle_id] = curr_vel
    return


@wp.func
def apply_muscle_force_to_bodies(
        actuation: float,
        pts_adr: int,
        pts_num: int,
        site_bodyid: wp.array(dtype=int),
        site_cond_qposadr: wp.array(dtype=int),
        site_cond_range: wp.array(dtype=wp.vec2),
        site_moving_qposadr: wp.array(dtype=wp.vec3i),
        qpos_in: wp.array(dtype=float),
        site_pos_G_in: wp.array(dtype=wp.vec3),
        site_rel_pos_B_in: wp.array(dtype=wp.vec3),
        site_moving_jac_G_in: wp.array(dtype=wp.mat33),
        body_F_muscle_out: wp.array(dtype=wp.spatial_vector),
        qfrc_muscle_out: wp.array(dtype=float),
):
    site1 = int(-1)  # previous active site
    for i in range(pts_num):
        site2 = pts_adr + i
        if not site_is_active(site2, qpos_in, site_cond_qposadr, site_cond_range):
            continue
        if site1 >= 0:
            p1_G, p2_G = site_pos_G_in[site1], site_pos_G_in[site2]
            diff = p2_G - p1_G
            dist = wp.length(diff)

            if dist >= BOLT_MINVAL:
                direction = diff / dist
                muscle_frc = actuation * direction

                # Bodies, position of site relative to body
                body1, body2 = site_bodyid[site1], site_bodyid[site2]
                s1_G, s2_G = site_rel_pos_B_in[site1], site_rel_pos_B_in[site2]
                wp.atomic_add(body_F_muscle_out, body1, wp.spatial_vector(wp.cross(s1_G, muscle_frc), muscle_frc))
                wp.atomic_sub(body_F_muscle_out, body2, wp.spatial_vector(wp.cross(s2_G, muscle_frc), muscle_frc))
                apply_moving_site_force(site1, muscle_frc, site_moving_qposadr, site_moving_jac_G_in, qfrc_muscle_out)
                apply_moving_site_force(site2, -muscle_frc, site_moving_qposadr, site_moving_jac_G_in, qfrc_muscle_out)
        site1 = site2
    return


@wp.kernel
def _apply_muscle_force_kernel(
        # Model:
        muscle_pts_adr: wp.array(dtype=int),
        muscle_pts_num: wp.array(dtype=int),
        site_bodyid: wp.array(dtype=int),
        site_cond_qposadr: wp.array(dtype=int),
        site_cond_range: wp.array(dtype=wp.vec2),
        site_moving_qposadr: wp.array(dtype=wp.vec3i),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        muscle_actuation_in: wp.array2d(dtype=float),
        qpos_in: wp.array2d(dtype=float),
        site_pos_G_in: wp.array2d(dtype=wp.vec3),
        site_rel_pos_B_in: wp.array2d(dtype=wp.vec3),
        site_moving_jac_G_in: wp.array2d(dtype=wp.mat33),
        # In:
        muscle_pt_group: wp.array(dtype=int),
        # Data out:
        body_F_muscle_out: wp.array2d(dtype=wp.spatial_vector),
        qfrc_muscle_out: wp.array2d(dtype=float),
):
    worldid, nodeid = wp.tid()
    if integration_done_in[worldid]:
        return
    muscle_id = muscle_pt_group[nodeid]

    actuation = muscle_actuation_in[worldid, muscle_id]

    pts_adr = muscle_pts_adr[muscle_id]
    pts_num = muscle_pts_num[muscle_id]
    apply_muscle_force_to_bodies(
        actuation, pts_adr, pts_num, site_bodyid, site_cond_qposadr, site_cond_range, site_moving_qposadr,
        qpos_in[worldid], site_pos_G_in[worldid], site_rel_pos_B_in[worldid], site_moving_jac_G_in[worldid],
        body_F_muscle_out[worldid], qfrc_muscle_out[worldid]
    )
    return


@wp.kernel
def _apply_unit_muscle_force_one_muscle_kernel(
        # Model:
        muscle_pts_adr: wp.array(dtype=int),
        muscle_pts_num: wp.array(dtype=int),
        site_bodyid: wp.array(dtype=int),
        site_cond_qposadr: wp.array(dtype=int),
        site_cond_range: wp.array(dtype=wp.vec2),
        site_moving_qposadr: wp.array(dtype=wp.vec3i),
        # Data in:
        qpos_in: wp.array2d(dtype=float),
        site_pos_G_in: wp.array2d(dtype=wp.vec3),
        site_rel_pos_B_in: wp.array2d(dtype=wp.vec3),
        site_moving_jac_G_in: wp.array2d(dtype=wp.mat33),
        # In:
        muscle_id: int,
        # Data out:
        body_F_muscle_out: wp.array2d(dtype=wp.spatial_vector),
        qfrc_muscle_out: wp.array2d(dtype=float),
):
    worldid = wp.tid()

    actuation = 1.0
    pts_adr = muscle_pts_adr[muscle_id]
    pts_num = muscle_pts_num[muscle_id]
    apply_muscle_force_to_bodies(
        actuation, pts_adr, pts_num, site_bodyid, site_cond_qposadr, site_cond_range, site_moving_qposadr,
        qpos_in[worldid], site_pos_G_in[worldid], site_rel_pos_B_in[worldid], site_moving_jac_G_in[worldid],
        body_F_muscle_out[worldid], qfrc_muscle_out[worldid]
    )
    return


@event_scope
def muscle_point_path(m: Model, d: Data):
    """ Computes the muscle path length and velocity for point-based paths """
    wp.launch(
        _compute_path_kernel,
        dim=(d.nworld, m.muscle_pt_group.size),
        inputs=[
            m.muscle_pts_adr, m.muscle_pts_num, m.site_cond_qposadr, m.site_cond_range,
            d.integration_done, d.qpos, d.site_pos_G, d.site_vel_G,
            m.muscle_pt_group
        ],
        outputs=[d.muscle_length, d.muscle_velocity],
    )


@event_scope
def apply_muscle_force_pt(m: Model, d: Data):
    wp.launch(
        _apply_muscle_force_kernel,
        dim=(d.nworld, m.muscle_pt_group.size),
        inputs=[
            m.muscle_pts_adr, m.muscle_pts_num, m.site_bodyid, m.site_cond_qposadr, m.site_cond_range,
            m.site_moving_qposadr,
            d.integration_done, d.muscle_actuation, d.qpos, d.site_pos_G, d.site_rel_pos_B, d.site_moving_jac_G,
            m.muscle_pt_group
        ],
        outputs=[d.body_F_muscle, d.qfrc_muscle],
    )


@event_scope
def apply_unit_force_one_muscle(m: Model, d: Data, body_F_out: wp.array2d, qfrc_out: wp.array2d, muscle_id: int):
    """ Body forces for a unit muscle tension, plus the generalized forces on the coordinates of moving points """
    wp.launch(
        _apply_unit_muscle_force_one_muscle_kernel,
        dim=(d.nworld,),
        inputs=[
            m.muscle_pts_adr, m.muscle_pts_num, m.site_bodyid, m.site_cond_qposadr, m.site_cond_range,
            m.site_moving_qposadr,
            d.qpos, d.site_pos_G, d.site_rel_pos_B, d.site_moving_jac_G, muscle_id
        ],
        outputs=[body_F_out, qfrc_out],
    )


@event_scope
def copy_ufrc_into_moment_arm(m: Model, d: Data, muscle_id: int, qfrc: wp.array2d(dtype=float),
                              qfrc_moving: wp.array2d(dtype=float)):
    """ For point-based paths only. Stores the sum of the body-force and moving-point generalized forces """

    @wp.kernel
    def _copy_ufrc_into_moment_arm_kernel(
            # Data in:
            ufrc_in: wp.array2d(dtype=float),
            qfrc_moving_in: wp.array2d(dtype=float),
            # In:
            mid: int,
            # Data out:
            muscle_moment_arm_out: wp.array3d(dtype=float),
    ):
        worldid = wp.tid()
        muscleid = mid

        nq = wp.static(m.nq)
        ufrc_tile = wp.tile_load(ufrc_in[worldid], shape=nq)
        moving_tile = wp.tile_load(qfrc_moving_in[worldid], shape=nq)
        wp.tile_store(muscle_moment_arm_out[worldid, muscleid], ufrc_tile + moving_tile)
        return

    if m.nmuscle:
        wp.launch_tiled(
            _copy_ufrc_into_moment_arm_kernel,
            dim=(d.nworld,),
            inputs=[qfrc, qfrc_moving, muscle_id, ],
            outputs=[d.muscle_moment_arm],
            block_dim=m.block_dim.muscle_path,
        )
