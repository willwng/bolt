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
from .types import MJ_MINVAL
from .types import Data
from .types import JointType
from .types import Model
from .types import TileSet
from .warp_util import cache_kernel
from .warp_util import event_scope
from .warp_util import kernel as nested_kernel

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _comvel_root(cvel_out: wp.array2d(dtype=wp.spatial_vector)):
    worldid, elementid = wp.tid()
    cvel_out[worldid, 0][elementid] = 0.0


@wp.kernel
def _comvel_level(
        # Model:
        body_parentid: wp.array(dtype=int),
        jnt_dofadr: wp.array(dtype=int),
        jnt_type: wp.array(dtype=int),
        # Data in:
        qvel_in: wp.array2d(dtype=float),
        cdof_in: wp.array2d(dtype=wp.spatial_vector),
        cvel_in: wp.array2d(dtype=wp.spatial_vector),
        # In:
        body_tree_: wp.array(dtype=int),
        # Data out:
        cvel_out: wp.array2d(dtype=wp.spatial_vector),
        cdof_dot_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, nodeid = wp.tid()
    bodyid = body_tree_[nodeid]

    # parent velocity
    pid = body_parentid[bodyid]
    cvel = cvel_in[worldid, pid]

    qvel = qvel_in[worldid]
    cdof = cdof_in[worldid]
    dofid = jnt_dofadr[bodyid]
    jnttype = jnt_type[bodyid]
    if jnttype == JointType.FREE:
        cvel += cdof[dofid + 0] * qvel[dofid + 0]
        cvel += cdof[dofid + 1] * qvel[dofid + 1]
        cvel += cdof[dofid + 2] * qvel[dofid + 2]

        cdof_dot_out[worldid, dofid + 3] = math.motion_cross(cvel,
                                                             cdof[dofid + 3])
        cdof_dot_out[worldid, dofid + 4] = math.motion_cross(cvel,
                                                             cdof[dofid + 4])
        cdof_dot_out[worldid, dofid + 5] = math.motion_cross(cvel,
                                                             cdof[dofid + 5])

        cvel += cdof[dofid + 3] * qvel[dofid + 3]
        cvel += cdof[dofid + 4] * qvel[dofid + 4]
        cvel += cdof[dofid + 5] * qvel[dofid + 5]

        dofid += 6
    elif jnttype == JointType.BALL:
        cdof_dot_out[worldid, dofid + 0] = math.motion_cross(cvel,
                                                             cdof[dofid + 0])
        cdof_dot_out[worldid, dofid + 1] = math.motion_cross(cvel,
                                                             cdof[dofid + 1])
        cdof_dot_out[worldid, dofid + 2] = math.motion_cross(cvel,
                                                             cdof[dofid + 2])

        cvel += cdof[dofid + 0] * qvel[dofid + 0]
        cvel += cdof[dofid + 1] * qvel[dofid + 1]
        cvel += cdof[dofid + 2] * qvel[dofid + 2]

        dofid += 3
    else:
        cdof_dot_out[worldid, dofid] = math.motion_cross(cvel, cdof[dofid])
        cvel += cdof[dofid] * qvel[dofid]

        dofid += 1

    cvel_out[worldid, bodyid] = cvel


@event_scope
def com_vel(m: Model, d: Data):
    """Computes the spatial velocities (cvel) and the derivative `cdof_dot` for all bodies.

    Propagates velocities down the kinematic tree, updating the spatial velocity and
    derivative for each body.
    """
    wp.launch(_comvel_root, dim=(d.nworld, 6), inputs=[], outputs=[d.cvel])

    for i in range(1, len(m.body_tree)):
        body_tree = m.body_tree[i]
        wp.launch(
            _comvel_level,
            dim=(d.nworld, body_tree.size),
            inputs=[m.body_parentid, m.jnt_dofadr, m.jnt_type, d.qvel, d.cdof,
                    d.cvel,
                    body_tree],
            outputs=[d.cvel, d.cdof_dot],
        )


@wp.kernel
def _site_velocity(
        # Model:
        site_bodyid: wp.array(dtype=int),
        body_rootid: wp.array(dtype=int),
        # Data in:
        site_xpos_in: wp.array2d(dtype=wp.vec3),
        subtree_com_in: wp.array2d(dtype=wp.vec3),
        cvel_in: wp.array2d(dtype=wp.spatial_vector),
        # Data out:
        site_xvel: wp.array2d(dtype=wp.vec3),
):
    worldid, siteid = wp.tid()
    # Body COM-velocity
    bodyid = site_bodyid[siteid]
    cvel = cvel_in[worldid, bodyid]
    ang = wp.spatial_top(cvel)
    lin = wp.spatial_bottom(cvel)
    # Transform to site
    pos = site_xpos_in[worldid, siteid]
    subtree_com = subtree_com_in[worldid, body_rootid[bodyid]]
    dif = pos - subtree_com

    site_xvel[worldid, siteid] = lin - wp.cross(dif, ang)


@wp.kernel
def _site_consecutive_diff_vel(
        # Data in:
        site_vel_in: wp.array2d(dtype=wp.vec3),
        site_diff_vec_in: wp.array2d(dtype=wp.vec3),
        site_diff_len_in: wp.array2d(dtype=float),
        # Data out:
        site_diff_vel_out: wp.array2d(dtype=float),
):
    worldid, site_diff_id = wp.tid()
    v1 = site_vel_in[worldid, site_diff_id]
    v2 = site_vel_in[worldid, site_diff_id + 1]

    vec = site_diff_vec_in[worldid, site_diff_id]
    length = site_diff_len_in[worldid, site_diff_id]
    if length > 1e-8:
        site_diff_vel_out[worldid, site_diff_id] = wp.dot((v2 - v1), vec)
    else:
        site_diff_vel_out[worldid, site_diff_id] = 0.0


@wp.kernel
def _compute_path_velocity(
        # Model:
        muscle_pts_adr: wp.array(dtype=int),
        muscle_pts_num: wp.array(dtype=int),
        # Data in:
        site_diff_vel_in: wp.array2d(dtype=float),
        # Data out:
        muscle_velocity_out: wp.array2d(dtype=float),
):
    worldid, muscle_id = wp.tid()

    n_sites = muscle_pts_num[muscle_id]
    pts_adr = muscle_pts_adr[muscle_id]
    for i in range(n_sites - 1):
        muscle_velocity_out[worldid, muscle_id] += site_diff_vel_in[
            worldid, pts_adr + i]


@event_scope
def muscle_path_velocity(m: Model, d: Data):
    """Computes the muscle path velocities. """
    if not m.nmuscle:
        return

    d.muscle_length.zero_()

    wp.launch(
        _site_velocity,
        dim=(d.nworld, m.nsite),
        inputs=[m.site_bodyid, m.body_rootid, d.site_xpos, d.subtree_com,
                d.cvel],
        outputs=[d.site_xvel]
    )

    wp.launch(
        _site_consecutive_diff_vel,
        dim=(d.nworld, m.nsite - 1),
        inputs=[d.site_xvel, d.site_diff_vec, d.site_diff_len],
        outputs=[d.site_diff_vel, ]
    )

    wp.launch(
        _compute_path_velocity,
        dim=(d.nworld, m.nmuscle),
        inputs=[
            m.muscle_pts_adr,
            m.muscle_pts_num,
            d.site_diff_vel,
        ],
        outputs=[d.muscle_velocity],
    )
