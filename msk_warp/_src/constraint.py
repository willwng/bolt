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

from . import support
from . import types
from .consts import MJ_MINIMP
from .consts import MJ_MAXIMP
from .consts import MJ_MINVAL
from .types import ConstraintType
from .types import vec5
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _zero_constraint_counts(
        # Data out:
        nl_out: wp.array(dtype=int),
        nefc_out: wp.array(dtype=int),
):
    worldid = wp.tid()

    # Zero all constraint counters
    nl_out[worldid] = 0
    nefc_out[worldid] = 0


@wp.func
def _update_efc_row(
        # In:
        worldid: int,
        efcid: int,
        pos_aref: float,
        pos_imp: float,
        invweight: float,
        solref: wp.vec2,
        solimp: vec5,
        vel: float,
        type: int,
        id: int,
        # Data out:
        efc_type_out: wp.array2d(dtype=int),
        efc_id_out: wp.array2d(dtype=int),
        efc_pos_out: wp.array2d(dtype=float),
        efc_D_out: wp.array2d(dtype=float),
        efc_vel_out: wp.array2d(dtype=float),
        efc_aref_out: wp.array2d(dtype=float),
):
    # Calculate kbi
    timeconst = solref[0]
    dampratio = solref[1]
    dmin = solimp[0]
    dmax = solimp[1]
    width = solimp[2]
    mid = solimp[3]
    power = solimp[4]

    dmin = wp.clamp(dmin, MJ_MINIMP, MJ_MAXIMP)
    dmax = wp.clamp(dmax, MJ_MINIMP, MJ_MAXIMP)
    width = wp.max(MJ_MINVAL, width)
    mid = wp.clamp(mid, MJ_MINIMP, MJ_MAXIMP)
    power = wp.max(1.0, power)

    # See https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters
    k = 1.0 / (dmax * dmax * timeconst * timeconst * dampratio * dampratio)
    b = 2.0 / (dmax * timeconst)
    k = wp.where(solref[0] <= 0, -solref[0] / (dmax * dmax), k)
    b = wp.where(solref[1] <= 0, -solref[1] / dmax, b)

    imp_x = wp.abs(pos_imp) / width
    imp_a = (1.0 / wp.pow(mid, power - 1.0)) * wp.pow(imp_x, power)
    imp_b = 1.0 - (1.0 / wp.pow(1.0 - mid, power - 1.0)) * wp.pow(1.0 - imp_x,
                                                                  power)
    imp_y = wp.where(imp_x < mid, imp_a, imp_b)
    imp = dmin + imp_y * (dmax - dmin)
    imp = wp.clamp(imp, dmin, dmax)
    imp = wp.where(imp_x > 1.0, dmax, imp)

    # Update constraints
    efc_D_out[worldid, efcid] = 1.0 / wp.max(invweight * (1.0 - imp) / imp,
                                             MJ_MINVAL)
    efc_vel_out[worldid, efcid] = vel
    efc_aref_out[worldid, efcid] = -k * imp * pos_aref - b * vel
    efc_pos_out[worldid, efcid] = pos_aref
    efc_type_out[worldid, efcid] = type
    efc_id_out[worldid, efcid] = id


@wp.kernel
def _efc_dof_limit(
        # Model:
        nv: int,
        limit_dof_range: wp.array(dtype=wp.vec2),
        limit_dof_adr: wp.array(dtype=int),
        limit_dof_qadr: wp.array(dtype=int),
        dof_invweight0: wp.array(dtype=float),
        # Data in:
        qpos_in: wp.array2d(dtype=float),
        qvel_in: wp.array2d(dtype=float),
        njmax_in: int,
        # In:
        # Data out:
        nl_out: wp.array(dtype=int),
        nefc_out: wp.array(dtype=int),
        efc_type_out: wp.array2d(dtype=int),
        efc_id_out: wp.array2d(dtype=int),
        efc_J_out: wp.array3d(dtype=float),
        efc_pos_out: wp.array2d(dtype=float),
        efc_D_out: wp.array2d(dtype=float),
        efc_vel_out: wp.array2d(dtype=float),
        efc_aref_out: wp.array2d(dtype=float),
):
    worldid, limitdofid = wp.tid()

    dof_range = limit_dof_range[limitdofid]
    dof_adr = limit_dof_adr[limitdofid]
    dof_qadr = limit_dof_qadr[limitdofid]

    qpos = qpos_in[worldid, dof_qadr]

    dist_min, dist_max = qpos - dof_range[0], dof_range[1] - qpos
    pos = wp.min(dist_min, dist_max)
    active = pos < 0

    if active:
        wp.atomic_add(nl_out, worldid, 1)
        efcid = wp.atomic_add(nefc_out, worldid, 1)

        if efcid >= njmax_in:
            return

        for i in range(nv):
            efc_J_out[worldid, efcid, i] = 0.0

        J = float(dist_min < dist_max) * 2.0 - 1.0
        efc_J_out[worldid, efcid, dof_adr] = J
        Jqvel = J * qvel_in[worldid, dof_adr]

        # ref = solref_in[conid]
        ref = wp.vec2(0.02, 1.0)
        solimp = vec5(0.9, 0.95, 0.001, 0.5, 2.0)

        _update_efc_row(
            worldid,
            efcid,
            pos,
            pos,
            dof_invweight0[dof_adr],
            ref,
            solimp,
            Jqvel,
            ConstraintType.LIMIT_JOINT,
            limitdofid,
            efc_type_out,
            efc_id_out,
            efc_pos_out,
            efc_D_out,
            efc_vel_out,
            efc_aref_out,
        )


@wp.kernel
def _efc_contact_elliptic(
        # Model:
        nv: int,
        opt_impratio: float,
        body_parentid: wp.array(dtype=int),
        body_rootid: wp.array(dtype=int),
        body_invweight0: wp.array(dtype=wp.vec2),
        dof_bodyid: wp.array(dtype=int),
        geom_bodyid: wp.array(dtype=int),
        # Data in:
        qvel_in: wp.array2d(dtype=float),
        subtree_com_in: wp.array2d(dtype=wp.vec3),
        cdof_in: wp.array2d(dtype=wp.spatial_vector),
        njmax_in: int,
        nacon_in: wp.array(dtype=int),
        # In:
        dist_in: wp.array(dtype=float),
        condim_in: wp.array(dtype=int),
        worldid_in: wp.array(dtype=int),
        geom_in: wp.array(dtype=wp.vec2i),
        pos_in: wp.array(dtype=wp.vec3),
        frame_in: wp.array(dtype=wp.mat33),
        friction_in: wp.array(dtype=vec5),
        # Data out:
        nefc_out: wp.array(dtype=int),
        contact_efc_address_out: wp.array2d(dtype=int),
        efc_type_out: wp.array2d(dtype=int),
        efc_id_out: wp.array2d(dtype=int),
        efc_J_out: wp.array3d(dtype=float),
        efc_pos_out: wp.array2d(dtype=float),
        efc_D_out: wp.array2d(dtype=float),
        efc_vel_out: wp.array2d(dtype=float),
        efc_aref_out: wp.array2d(dtype=float),
):
    conid, dimid = wp.tid()

    if conid >= nacon_in[0]:
        return

    condim = condim_in[conid]

    if dimid > condim - 1:
        return

    pos = dist_in[conid]
    active = pos < 0.0

    if active:
        worldid = worldid_in[conid]

        efcid = wp.atomic_add(nefc_out, worldid, 1)
        if efcid >= njmax_in:
            contact_efc_address_out[conid, dimid] = -1
            return

        impratio = opt_impratio
        contact_efc_address_out[conid, dimid] = efcid

        geom = geom_in[conid]
        body1 = geom_bodyid[geom[0]]
        body2 = geom_bodyid[geom[1]]

        cpos = pos_in[conid]
        frame = frame_in[conid]

        # TODO(team): parallelize J and Jqvel computation?
        Jqvel = float(0.0)
        for i in range(nv):
            J = float(0.0)
            jac1p, jac1r = support.jac(
                body_parentid,
                body_rootid,
                dof_bodyid,
                subtree_com_in,
                cdof_in,
                cpos,
                body1,
                i,
                worldid,
            )
            jac2p, jac2r = support.jac(
                body_parentid,
                body_rootid,
                dof_bodyid,
                subtree_com_in,
                cdof_in,
                cpos,
                body2,
                i,
                worldid,
            )
            for xyz in range(3):
                if dimid < 3:
                    jac_dif = jac2p[xyz] - jac1p[xyz]
                    J += frame[dimid, xyz] * jac_dif
                else:
                    jac_dif = jac2r[xyz] - jac1r[xyz]
                    J += frame[dimid - 3, xyz] * jac_dif

            efc_J_out[worldid, efcid, i] = J
            Jqvel += J * qvel_in[worldid, i]

        invweight = body_invweight0[body1][0] + body_invweight0[body2][0]

        # ref = solref_in[conid]
        ref = wp.vec2(0.02, 1.0)
        solimp = vec5(0.9, 0.95, 0.001, 0.5, 2.0)
        pos_aref = pos

        if dimid > 0:
            # TODO(team): precompute 1 / impratio
            invweight = invweight / impratio
            friction = friction_in[conid]

            if dimid > 1:
                fri0 = friction[0]
                frii = friction[dimid - 1]
                fri = fri0 * fri0 / (frii * frii)
                invweight *= fri

            pos_aref = 0.0

        if condim == 1:
            efc_type = ConstraintType.CONTACT_FRICTIONLESS
        else:
            efc_type = ConstraintType.CONTACT_ELLIPTIC

        _update_efc_row(
            worldid,
            efcid,
            pos_aref,
            pos,
            invweight,
            ref,
            solimp,
            Jqvel,
            efc_type,
            conid,
            efc_type_out,
            efc_id_out,
            efc_pos_out,
            efc_D_out,
            efc_vel_out,
            efc_aref_out,
        )


@wp.kernel
def _num_equality(
        # Data in:
        ne_connect_in: wp.array(dtype=int),
        ne_weld_in: wp.array(dtype=int),
        ne_jnt_in: wp.array(dtype=int),
        ne_ten_in: wp.array(dtype=int),
        # Data out:
        ne_out: wp.array(dtype=int),
):
    worldid = wp.tid()
    ne = ne_connect_in[worldid] + ne_weld_in[worldid] + ne_jnt_in[worldid] + \
         ne_ten_in[worldid]
    ne_out[worldid] = ne


@event_scope
def make_constraint(m: types.Model, d: types.Data):
    """Creates constraint jacobians and other supporting data."""
    wp.launch(
        _zero_constraint_counts,
        dim=d.nworld,
        inputs=[d.nl, d.nefc],
    )

    # Individual DOF limits
    wp.launch(
        _efc_dof_limit,
        dim=(d.nworld, m.ndoflimit),
        inputs=[
            m.nv,
            m.limit_dof_range,
            m.limit_dof_adr,
            m.limit_dof_qadr,
            m.dof_invweight0,
            d.qpos,
            d.qvel,
            d.njmax,
        ],
        outputs=[
            d.nl,
            d.nefc,
            d.efc.type,
            d.efc.id,
            d.efc.J,
            d.efc.pos,
            d.efc.D,
            d.efc.vel,
            d.efc.aref,
        ],
    )

    # contact
    if wp.static(m.opt.contact_type == types.ContactType.MUJOCO):
        wp.launch(
            _efc_contact_elliptic,
            dim=(d.naconmax, 3),  # assuming max 3 dims per contact
            inputs=[
                m.nv,
                m.opt.impratio,
                m.body_parentid,
                m.body_rootid,
                m.body_invweight0,
                m.dof_bodyid,
                m.geom_bodyid,
                d.qvel,
                d.subtree_com,
                d.cdof,
                d.njmax,
                d.nacon,
                d.contact.dist,
                d.contact.dim,
                d.contact.worldid,
                d.contact.geom,
                d.contact.pos,
                d.contact.frame,
                d.contact.friction,
            ],
            outputs=[
                d.nefc,
                d.contact.efc_address,
                d.efc.type,
                d.efc.id,
                d.efc.J,
                d.efc.pos,
                d.efc.D,
                d.efc.vel,
                d.efc.aref,
            ],
        )
