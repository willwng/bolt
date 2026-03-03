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
from .types import Data
from .types import Model
from .types import TileSet
from .types import vec10
from .warp_util import cache_kernel
from .warp_util import event_scope
from .warp_util import kernel as nested_kernel

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _fix_limits(
        # Model in:
        limit_dof_range: wp.array(dtype=wp.vec2),
        limit_dof_qadr: wp.array(dtype=int),
        # Data in:
        world_reset_in: wp.array(dtype=bool),
        qpos_in: wp.array2d(dtype=float),
        # Data out:
        qpos_out: wp.array2d(dtype=float),
):
    worldid, limitdofid = wp.tid()
    if world_reset_in[worldid]:
        dof_range = limit_dof_range[limitdofid]
        dof_qadr = limit_dof_qadr[limitdofid]
        qpos = qpos_in[worldid, dof_qadr]

        qpos_clamped = wp.clamp(qpos, dof_range[0], dof_range[1])
        qpos_out[worldid, dof_qadr] = qpos_clamped
    return


@event_scope
def fix_qpos_limits(m: Model, d: Data):
    """Clamps qpos values to joint limits."""
    wp.launch(
        _fix_limits,
        dim=(d.nworld, m.ndoflimit),
        inputs=[
            m.limit_dof_range,
            m.limit_dof_qadr,
            d.world_reset,
            d.qpos,
        ],
        outputs=[
            d.qpos,
        ],
    )
    return


@wp.kernel
def _kinematics_root(
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        # Data out:
        body_X_out: wp.array2d(dtype=wp.transform),
        body_X_com_out: wp.array2d(dtype=wp.transform),
):
    worldid = wp.tid()
    if integration_done_in[worldid]:
        return
    body_X_out[worldid, 0] = wp.transform_identity()
    body_X_com_out[worldid, 0] = wp.transform_identity()


@wp.kernel
def _kinematics_level(
        # Model:
        body_parentid: wp.array(dtype=int),
        jnt_type: wp.array(dtype=int),
        jnt_qposadr: wp.array(dtype=int),
        jnt_rel_parent: wp.array(dtype=wp.transform),
        jnt_rel_child: wp.array(dtype=wp.transform),
        jnt_extra_info: wp.array(dtype=wp.vec3),
        body_X_com: wp.array(dtype=wp.transform),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qpos_in: wp.array2d(dtype=float),
        body_X_in: wp.array2d(dtype=wp.transform),
        # In:
        body_tree_: wp.array(dtype=int),
        # Data out:
        body_X_out: wp.array2d(dtype=wp.transform),
        body_X_com_out: wp.array2d(dtype=wp.transform),
):
    worldid, nodeid = wp.tid()
    if integration_done_in[worldid]:
        return

    bodyid = body_tree_[nodeid]

    # Collect joint information
    jnt_type_ = jnt_type[bodyid]
    qpos_start = jnt_qposadr[bodyid]
    X_pj = jnt_rel_parent[bodyid]
    X_cj = jnt_rel_child[bodyid]
    extra_info = jnt_extra_info[bodyid]

    # Parent world frame
    parentid = body_parentid[bodyid]
    X_wp = body_X_in[worldid, parentid]
    # Parent mobilizer frame
    X_mp = X_wp * X_pj

    # Joint transform/child mobilizer frame
    X_j = mobilizers.jcalc_transform(jnt_type_, qpos_start, qpos_in[worldid], extra_info)
    X_mc = X_mp * X_j

    # Child world frame
    X_wc = X_mc * wp.transform_inverse(X_cj)

    # Child COM frame
    X_com_local = body_X_com[bodyid]
    X_com = X_mc * X_com_local

    body_X_out[worldid, bodyid] = X_wc
    body_X_com_out[worldid, bodyid] = X_com
    return


@wp.kernel
def _geom_local_to_global(
        # Model:
        geom_bodyid: wp.array(dtype=int),
        geom_X_loc: wp.array(dtype=wp.transform),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        body_X_in: wp.array2d(dtype=wp.transform),
        # Data out:
        geom_X_out: wp.array2d(dtype=wp.transform),
):
    worldid, geomid = wp.tid()
    if integration_done_in[worldid]:
        return

    bodyid = geom_bodyid[geomid]
    body_X = body_X_in[worldid, bodyid]
    geom_X_out[worldid, geomid] = body_X * geom_X_loc[geomid]


@wp.kernel
def _vis_local_to_global(
        # Model:
        vis_bodyid: wp.array(dtype=int),
        vis_X_loc: wp.array(dtype=wp.transform),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        body_X_in: wp.array2d(dtype=wp.transform),
        # Data out:
        vis_X_out: wp.array2d(dtype=wp.transform),
):
    worldid, visid = wp.tid()
    if integration_done_in[worldid]:
        return

    bodyid = vis_bodyid[visid]
    body_X = body_X_in[worldid, bodyid]
    vis_X_out[worldid, visid] = body_X * vis_X_loc[visid]


@wp.kernel
def _site_local_to_global(
        # Model:
        site_bodyid: wp.array(dtype=int),
        site_pos: wp.array(dtype=wp.vec3),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        body_X_in: wp.array2d(dtype=wp.transform),
        # Data out:
        site_rpos_out: wp.array2d(dtype=wp.vec3),
        site_xpos_out: wp.array2d(dtype=wp.vec3),
):
    worldid, siteid = wp.tid()
    if integration_done_in[worldid]:
        return

    bodyid = site_bodyid[siteid]
    body_X = body_X_in[worldid, bodyid]
    body_quat = wp.transform_get_rotation(body_X)
    body_pos = wp.transform_get_translation(body_X)
    # Relative to body and world positions
    rpos = wp.quat_rotate(body_quat, site_pos[siteid])
    site_rpos_out[worldid, siteid] = rpos
    site_xpos_out[worldid, siteid] = body_pos + rpos


@event_scope
def kinematics(m: Model, d: Data):
    """ Computes forward kinematics for all bodies, sites, geoms. """
    # World body
    wp.launch(
        _kinematics_root,
        dim=(d.nworld),
        inputs=[d.integration_done],
        outputs=[d.body_X, d.body_X_com]
    )

    for i in range(1, len(m.body_tree)):
        body_tree = m.body_tree[i]
        wp.launch(
            _kinematics_level,
            dim=(d.nworld, body_tree.size),
            inputs=[
                m.body_parentid, m.jnt_type, m.jnt_qposadr,
                m.jnt_rel_parent, m.jnt_rel_child, m.jnt_extra_info, m.body_X_com_loc,
                d.integration_done, d.qpos, d.body_X,
                body_tree,
            ],
            outputs=[d.body_X, d.body_X_com],
        )
    wp.launch(
        _geom_local_to_global,
        dim=(d.nworld, m.ngeom),
        inputs=[m.geom_bodyid, m.geom_X_loc, d.integration_done, d.body_X],
        outputs=[d.geom_X],
    )

    if wp.static(m.opt.visuals):
        wp.launch(
            _vis_local_to_global,
            dim=(d.nworld, m.nvis),
            inputs=[m.vis_bodyid, m.vis_X_loc, d.integration_done, d.body_X],
            outputs=[d.vis_X],
        )

    wp.launch(
        _site_local_to_global,
        dim=(d.nworld, m.nsite),
        inputs=[m.site_bodyid, m.site_pos, d.integration_done, d.body_X],
        outputs=[d.site_rpos, d.site_xpos],
    )


@wp.kernel
def _subtree_com_init(
        # Model:
        body_mass: wp.array(dtype=float),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        body_X_in: wp.array2d(dtype=wp.transform),
        # Data out:
        subtree_mass_out: wp.array2d(dtype=float),
        subtree_com_out: wp.array2d(dtype=wp.vec3),
):
    worldid, bodyid = wp.tid()
    if integration_done_in[worldid]:
        return
    body_pos = wp.transform_get_translation(body_X_in[worldid, bodyid])
    subtree_mass_out[worldid, bodyid] = body_mass[bodyid]
    subtree_com_out[worldid, bodyid] = body_pos * body_mass[bodyid]


@wp.kernel
def _subtree_com_acc(
        # Model:
        body_parentid: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        subtree_com_in: wp.array2d(dtype=wp.vec3),
        # In:
        body_tree_: wp.array(dtype=int),
        # Data out:
        subtree_mass_out: wp.array2d(dtype=float),
        subtree_com_out: wp.array2d(dtype=wp.vec3),
):
    worldid, nodeid = wp.tid()
    if integration_done_in[worldid]:
        return

    bodyid = body_tree_[nodeid]
    pid = body_parentid[bodyid]
    if bodyid != 0:
        wp.atomic_add(subtree_mass_out, worldid, pid, subtree_mass_out[worldid, bodyid])
        wp.atomic_add(subtree_com_out, worldid, pid, subtree_com_in[worldid, bodyid])


@wp.kernel
def _subtree_div(
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        subtree_mass_in: wp.array2d(dtype=float),
        subtree_com_in: wp.array2d(dtype=wp.vec3),
        # Data out:
        subtree_com_out: wp.array2d(dtype=wp.vec3),
):
    worldid, bodyid = wp.tid()
    if integration_done_in[worldid]:
        return
    com = subtree_com_in[worldid, bodyid]
    mass = subtree_mass_in[worldid, bodyid]
    if mass != 0.0:
        subtree_com_out[worldid, bodyid] = com / mass


@event_scope
def com_pos(m: Model, d: Data):
    """ Computes subtree center of mass positions. """
    d.subtree_mass.zero_()

    # Initialize subtree_com to (current body com * mass)
    wp.launch(_subtree_com_init,
              dim=(d.nworld, m.nbody),
              inputs=[m.body_mass, d.integration_done, d.body_X_com],
              outputs=[d.subtree_mass, d.subtree_com])

    # Backward pass to propagate subtree com * mass
    for i in reversed(range(len(m.body_tree))):
        body_tree = m.body_tree[i]
        wp.launch(
            _subtree_com_acc,
            dim=(d.nworld, body_tree.size),
            inputs=[m.body_parentid, d.integration_done, d.subtree_com, body_tree],
            outputs=[d.subtree_mass, d.subtree_com],
        )

    # Compute the subtree com of each body
    wp.launch(
        _subtree_div,
        dim=(d.nworld, m.nbody),
        inputs=[d.integration_done, d.subtree_mass, d.subtree_com],
        outputs=[d.subtree_com])
