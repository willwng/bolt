# Copyright 2025 The Newton Developers
# Modified for MSKWarp by Will Wang
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
from . import mobilizers
from . import support
from .types import Data
from .types import Model
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _link_vel_root(
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        # Data out:
        body_vel_out: wp.array2d(dtype=wp.spatial_vector),
        body_acc_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, elementid = wp.tid()
    if integration_done_in[worldid]:
        return
    body_vel_out[worldid, 0] = wp.spatial_vector()
    body_acc_out[worldid, 0] = wp.spatial_vector()


@wp.kernel
def _link_vel_level(
        # Model:
        body_parentid: wp.array(dtype=int),
        jnt_type: wp.array(dtype=int),
        jnt_dofadr: wp.array(dtype=int),
        jnt_dofnum: wp.array(dtype=int),
        jnt_rel_parent: wp.array(dtype=wp.transform),
        jnt_extra_info: wp.array(dtype=wp.vec3),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qvel_in: wp.array2d(dtype=float),
        body_X_in: wp.array2d(dtype=wp.transform),
        body_vel_in: wp.array2d(dtype=wp.spatial_vector),
        body_acc_in: wp.array2d(dtype=wp.spatial_vector),
        # In:
        body_tree_: wp.array(dtype=int),
        # Data out:
        body_vel_out: wp.array2d(dtype=wp.spatial_vector),
        body_acc_out: wp.array2d(dtype=wp.spatial_vector),
        cdof_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, nodeid = wp.tid()
    if integration_done_in[worldid]:
        return

    bodyid = body_tree_[nodeid]

    # Collect joint information
    jnt_type_ = jnt_type[bodyid]
    dofadr = jnt_dofadr[bodyid]
    X_pj = jnt_rel_parent[bodyid]
    extra_info = jnt_extra_info[bodyid]
    dofnum = jnt_dofnum[bodyid]
    S_out = cdof_out[worldid]

    # Parent world frame
    parentid = body_parentid[bodyid]
    X_wp = body_X_in[worldid, parentid]
    # Parent mobilizer frame
    X_mp = X_wp * X_pj

    # Compute motion subspace and velocity across joint
    v_j = mobilizers.joint_motion(jnt_type_, dofadr, qvel_in[worldid], extra_info, S_out)
    c_j = mobilizers.joint_acc(jnt_type_, dofadr, qvel_in[worldid], extra_info)

    # Transform motion subspace
    for i in range(dofnum):
        S_out[dofadr + i] = math.transform_twist(X_mp, S_out[dofadr + i])

    # Parent velocity, acceleration
    v_p = body_vel_in[worldid, parentid]
    a_p = body_acc_in[worldid, parentid]

    # Child velocity, acceleration
    v_c = v_p + math.transform_twist(X_mp, v_j)
    a_c = a_p + wp.spatial_cross(v_p, v_j) + math.transform_twist(X_mp, c_j)

    body_vel_out[worldid, bodyid] = v_c
    body_acc_out[worldid, bodyid] = a_c


@event_scope
def link_vel(m: Model, d: Data):
    wp.launch(
        _link_vel_root,
        dim=(d.nworld),
        inputs=[d.integration_done],
        outputs=[d.body_vel, d.body_acc]
    )

    for i in range(1, len(m.body_tree)):
        body_tree = m.body_tree[i]
        wp.launch(
            _link_vel_level,
            dim=(d.nworld, body_tree.size),
            inputs=[m.body_parentid, m.jnt_type, m.jnt_dofadr, m.jnt_dofnum, m.jnt_rel_parent, m.jnt_extra_info,
                    d.integration_done, d.qvel, d.body_X, d.body_vel, d.body_acc,
                    body_tree],
            outputs=[d.body_vel, d.body_acc, d.cdof]
        )


@wp.kernel
def _site_velocity(
        # Model:
        site_bodyid: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        site_rpos_in: wp.array2d(dtype=wp.vec3),
        xvel_in: wp.array2d(dtype=wp.spatial_vector),
        # Data out:
        site_xvel_out: wp.array2d(dtype=wp.vec3),
):
    worldid, siteid = wp.tid()
    if integration_done_in[worldid]:
        return
    # Body COM-velocity
    bodyid = site_bodyid[siteid]
    xvel = xvel_in[worldid, bodyid]
    # Transform to site
    site_rel_pos = site_rpos_in[worldid, siteid]
    site_xvel = support.transform_velocity(xvel, site_rel_pos)
    site_xvel_out[worldid, siteid] = wp.spatial_bottom(site_xvel)


@event_scope
def site_velocity(m: Model, d: Data):
    """Computes the velocity of sites. """
    wp.launch(
        _site_velocity,
        dim=(d.nworld, m.nsite),
        inputs=[m.site_bodyid, d.integration_done, d.site_rpos, d.body_acc],
        outputs=[d.site_xvel]
    )
