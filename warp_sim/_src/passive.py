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
from .types import GeomType
from .types import JointType
from .types import Model
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.func
def _pow2(val: float) -> float:
    return val * val


@wp.func
def _pow4(val: float) -> float:
    sq = val * val
    return sq * sq


@wp.func
def _geom_semiaxes(size: wp.vec3, geom_type: int) -> wp.vec3:  # kernel_analyzer: ignore
    if geom_type == GeomType.SPHERE:
        r = size[0]
        return wp.vec3(r, r, r)

    if geom_type == GeomType.CAPSULE:
        radius = size[0]
        half_length = size[1]
        return wp.vec3(radius, radius, half_length + radius)

    if geom_type == GeomType.CYLINDER:
        radius = size[0]
        half_length = size[1]
        return wp.vec3(radius, radius, half_length)

    # ellipsoid, box, mesh, sdf -> use size directly
    return size


@wp.func
def _ellipsoid_max_moment(size: wp.vec3, dir: int) -> float:
    d0 = size[dir]
    d1 = size[(dir + 1) % 3]
    d2 = size[(dir + 2) % 3]
    return wp.static(8.0 / 15.0 * wp.pi) * d0 * _pow4(wp.max(d1, d2))


@wp.kernel
def _spring_damper_dof_passive(
        # Model:
        qpos_spring: wp.array(dtype=float),
        jnt_type: wp.array(dtype=int),
        jnt_qposadr: wp.array(dtype=int),
        jnt_dofadr: wp.array(dtype=int),
        jnt_stiffness: wp.array(dtype=float),
        dof_damping: wp.array(dtype=float),
        # Data in:
        qpos_in: wp.array2d(dtype=float),
        qvel_in: wp.array2d(dtype=float),
        # Data out:
        qfrc_spring_out: wp.array2d(dtype=float),
        qfrc_damper_out: wp.array2d(dtype=float),
):
    worldid, jntid = wp.tid()
    dofid = jnt_dofadr[jntid]
    stiffness = jnt_stiffness[jntid]
    damping = dof_damping[dofid]

    has_stiffness = stiffness != 0.0
    has_damping = damping != 0.0

    if not has_stiffness:
        qfrc_spring_out[worldid, dofid] = 0.0

    if not has_damping:
        qfrc_damper_out[worldid, dofid] = 0.0

    if not (has_stiffness or has_damping):
        return

    jnttype = jnt_type[jntid]
    qposid = jnt_qposadr[jntid]

    if jnttype == JointType.FREE:
        # spring
        if has_stiffness:
            dif = wp.vec3(
                qpos_in[worldid, qposid + 0] - qpos_spring[qposid + 0],
                qpos_in[worldid, qposid + 1] - qpos_spring[qposid + 1],
                qpos_in[worldid, qposid + 2] - qpos_spring[qposid + 2],
            )
            qfrc_spring_out[worldid, dofid + 0] = -stiffness * dif[0]
            qfrc_spring_out[worldid, dofid + 1] = -stiffness * dif[1]
            qfrc_spring_out[worldid, dofid + 2] = -stiffness * dif[2]
            rot = wp.quat(
                qpos_in[worldid, qposid + 3],
                qpos_in[worldid, qposid + 4],
                qpos_in[worldid, qposid + 5],
                qpos_in[worldid, qposid + 6],
            )
            rot = wp.normalize(rot)
            ref = wp.quat(
                qpos_spring[qposid + 3],
                qpos_spring[qposid + 4],
                qpos_spring[qposid + 5],
                qpos_spring[qposid + 6],
            )
            dif = math.quat_sub(rot, ref)
            qfrc_spring_out[worldid, dofid + 3] = -stiffness * dif[0]
            qfrc_spring_out[worldid, dofid + 4] = -stiffness * dif[1]
            qfrc_spring_out[worldid, dofid + 5] = -stiffness * dif[2]

        # damper
        if has_damping:
            qfrc_damper_out[worldid, dofid + 0] = -damping * qvel_in[worldid, dofid + 0]
            qfrc_damper_out[worldid, dofid + 1] = -damping * qvel_in[worldid, dofid + 1]
            qfrc_damper_out[worldid, dofid + 2] = -damping * qvel_in[worldid, dofid + 2]
            qfrc_damper_out[worldid, dofid + 3] = -damping * qvel_in[worldid, dofid + 3]
            qfrc_damper_out[worldid, dofid + 4] = -damping * qvel_in[worldid, dofid + 4]
            qfrc_damper_out[worldid, dofid + 5] = -damping * qvel_in[worldid, dofid + 5]
    elif jnttype == JointType.BALL:
        # spring
        if has_stiffness:
            rot = wp.quat(
                qpos_in[worldid, qposid + 0],
                qpos_in[worldid, qposid + 1],
                qpos_in[worldid, qposid + 2],
                qpos_in[worldid, qposid + 3],
            )
            rot = wp.normalize(rot)
            ref = wp.quat(
                qpos_spring[qposid + 0],
                qpos_spring[qposid + 1],
                qpos_spring[qposid + 2],
                qpos_spring[qposid + 3],
            )
            dif = math.quat_sub(rot, ref)
            qfrc_spring_out[worldid, dofid + 0] = -stiffness * dif[0]
            qfrc_spring_out[worldid, dofid + 1] = -stiffness * dif[1]
            qfrc_spring_out[worldid, dofid + 2] = -stiffness * dif[2]

        # damper
        if has_damping:
            qfrc_damper_out[worldid, dofid + 0] = -damping * qvel_in[worldid, dofid + 0]
            qfrc_damper_out[worldid, dofid + 1] = -damping * qvel_in[worldid, dofid + 1]
            qfrc_damper_out[worldid, dofid + 2] = -damping * qvel_in[worldid, dofid + 2]
    else:  # mjJNT_SLIDE, mjJNT_HINGE
        # spring
        if has_stiffness:
            fdif = qpos_in[worldid, qposid] - qpos_spring[qposid]
            qfrc_spring_out[worldid, dofid] = -stiffness * fdif

        # damper
        if has_damping:
            qfrc_damper_out[worldid, dofid] = -damping * qvel_in[worldid, dofid]

@wp.kernel
def _qfrc_passive(
        # Data in:
        qfrc_spring_in: wp.array2d(dtype=float),
        qfrc_damper_in: wp.array2d(dtype=float),
        # Data out:
        qfrc_passive_out: wp.array2d(dtype=float),
):
    worldid, dofid = wp.tid()
    qfrc_passive = qfrc_spring_in[worldid, dofid]
    qfrc_passive += qfrc_damper_in[worldid, dofid]

    qfrc_passive_out[worldid, dofid] = qfrc_passive


@event_scope
def passive(m: Model, d: Data):
    """Adds all passive forces."""
    wp.launch(
        _spring_damper_dof_passive,
        dim=(d.nworld, m.nbody),
        inputs=[
            m.qpos_spring,
            m.jnt_type,
            m.jnt_qposadr,
            m.jnt_dofadr,
            m.jnt_stiffness,
            m.dof_damping,
            d.qpos,
            d.qvel,
        ],
        outputs=[d.qfrc_spring, d.qfrc_damper],
    )

    wp.launch(
        _qfrc_passive,
        dim=(d.nworld, m.nv),
        inputs=[
            d.qfrc_spring,
            d.qfrc_damper,
        ],
        outputs=[
            d.qfrc_passive,
        ],
    )
