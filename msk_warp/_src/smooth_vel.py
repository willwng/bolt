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

from . import mobilizers
from . import support
from .types import Data
from .types import Model
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _comvel_root(
        cvel_out: wp.array2d(dtype=wp.spatial_vector),
        xvel_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, elementid = wp.tid()
    cvel_out[worldid, 0][elementid] = 0.0
    xvel_out[worldid, 0][elementid] = 0.0


@wp.kernel
def _comvel_level(
        # Model:
        body_parentid: wp.array(dtype=int),
        jnt_dofadr: wp.array(dtype=int),
        jnt_type: wp.array(dtype=int),
        jnt_dofnum: wp.array(dtype=int),
        jnt_cst_adr: wp.array(dtype=int),
        cst_txfm_dofadr_in: wp.array2d(dtype=int),
        body_rootid: wp.array(dtype=int),
        # Data in:
        qvel_in: wp.array2d(dtype=float),
        cdof_in: wp.array2d(dtype=wp.spatial_vector),
        cdof_tmp_in: wp.array3d(dtype=wp.spatial_vector),
        cvel_in: wp.array2d(dtype=wp.spatial_vector),
        xpos_in: wp.array2d(dtype=wp.vec3),
        subtree_com_in: wp.array2d(dtype=wp.vec3),
        # In:
        body_tree_: wp.array(dtype=int),
        # Data out:
        cvel_out: wp.array2d(dtype=wp.spatial_vector),
        xvel_out: wp.array2d(dtype=wp.spatial_vector),
        cdof_dot_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, nodeid = wp.tid()
    bodyid = body_tree_[nodeid]

    # parent velocity
    pid = body_parentid[bodyid]
    cvel = cvel_in[worldid, pid]

    # Contribution of mobilizer
    qvel = qvel_in[worldid]
    cdof = cdof_in[worldid]
    dofid = jnt_dofadr[bodyid]
    jnttype = jnt_type[bodyid]

    dof_num = jnt_dofnum[bodyid]
    cst_jnt_adr = jnt_cst_adr[bodyid]
    cst_txfm_dofadr = cst_txfm_dofadr_in[cst_jnt_adr]
    cdof_tmp = cdof_tmp_in[worldid, cst_jnt_adr]

    res = cdof_dot_out[worldid]

    # Com-based velocity
    cvel = mobilizers.cvel_joint(cvel, cdof, qvel, jnttype, dofid, dof_num,
                                 cst_txfm_dofadr, cdof_tmp, res)
    cvel_out[worldid, bodyid] = cvel

    # Cartesian velocity
    pos = xpos_in[worldid, bodyid]
    subtree_com = subtree_com_in[worldid, body_rootid[bodyid]]
    dif = pos - subtree_com
    xvel_out[worldid, bodyid] = support.transform_velocity(cvel, dif)


@event_scope
def com_vel(m: Model, d: Data):
    """Computes the spatial velocities (cvel) and the derivative `cdof_dot` for all bodies.

    Propagates velocities down the kinematic tree, updating the spatial velocity and
    derivative for each body.
    """
    wp.launch(
        _comvel_root,
        dim=(d.nworld, 6),
        inputs=[],
        outputs=[d.xvel, d.cvel]
    )

    for i in range(1, len(m.body_tree)):
        body_tree = m.body_tree[i]
        wp.launch(
            _comvel_level,
            dim=(d.nworld, body_tree.size),
            inputs=[m.body_parentid, m.jnt_dofadr, m.jnt_type, m.jnt_dofnum,
                    m.jnt_cst_adr, m.cst_txfm_dofadr, m.body_rootid,
                    d.qvel, d.cdof, d.cdof_tmp, d.cvel, d.xpos, d.subtree_com,
                    body_tree],
            outputs=[d.cvel, d.xvel, d.cdof_dot],
        )


@wp.kernel
def _site_velocity(
        # Model:
        site_bodyid: wp.array(dtype=int),
        # Data in:
        site_rpos_in: wp.array2d(dtype=wp.vec3),
        xvel_in: wp.array2d(dtype=wp.spatial_vector),
        # Data out:
        site_xvel_out: wp.array2d(dtype=wp.vec3),
):
    worldid, siteid = wp.tid()
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
        inputs=[m.site_bodyid, d.site_rpos, d.xvel],
        outputs=[d.site_xvel]
    )
