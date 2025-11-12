from typing import Optional

import warp as wp

from . import math
from . import constraint
from . import passive
from . import smooth
from . import solver
from . import util_misc
from .support import xfrc_accumulate
from .types import Data
from .types import JointType
from .types import Model
from .types import TileSet
from .types import vec10f
from .warp_util import cache_kernel
from .warp_util import event_scope
from .warp_util import kernel as nested_kernel

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _next_position(
        # Model:
        opt_timestep: wp.array(dtype=float),
        jnt_type: wp.array(dtype=int),
        jnt_qposadr: wp.array(dtype=int),
        jnt_dofadr: wp.array(dtype=int),
        # Data in:
        qpos_in: wp.array2d(dtype=float),
        qvel_in: wp.array2d(dtype=float),
        # In:
        qvel_scale_in: float,
        # Data out:
        qpos_out: wp.array2d(dtype=float),
):
    worldid, jntid = wp.tid()
    timestep = opt_timestep[worldid % opt_timestep.shape[0]]

    jnttype = jnt_type[jntid]
    qpos_adr = jnt_qposadr[jntid]
    dof_adr = jnt_dofadr[jntid]
    qpos = qpos_in[worldid]
    qpos_next = qpos_out[worldid]
    qvel = qvel_in[worldid]

    if jnttype == JointType.FREE:
        qpos_pos = wp.vec3(qpos[qpos_adr], qpos[qpos_adr + 1], qpos[qpos_adr + 2])
        qvel_lin = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2]) * qvel_scale_in

        qpos_new = qpos_pos + timestep * qvel_lin

        qpos_quat = wp.quat(
            qpos[qpos_adr + 3],
            qpos[qpos_adr + 4],
            qpos[qpos_adr + 5],
            qpos[qpos_adr + 6],
        )
        qvel_ang = wp.vec3(qvel[dof_adr + 3], qvel[dof_adr + 4], qvel[dof_adr + 5]) * qvel_scale_in
        qpos_quat_new = math.quat_integrate(qpos_quat, qvel_ang, timestep)

        qpos_next[qpos_adr + 0] = qpos_new[0]
        qpos_next[qpos_adr + 1] = qpos_new[1]
        qpos_next[qpos_adr + 2] = qpos_new[2]
        qpos_next[qpos_adr + 3] = qpos_quat_new[0]
        qpos_next[qpos_adr + 4] = qpos_quat_new[1]
        qpos_next[qpos_adr + 5] = qpos_quat_new[2]
        qpos_next[qpos_adr + 6] = qpos_quat_new[3]

    elif jnttype == JointType.BALL:
        qpos_quat = wp.quat(qpos[qpos_adr + 0], qpos[qpos_adr + 1], qpos[qpos_adr + 2], qpos[qpos_adr + 3])
        qvel_ang = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2]) * qvel_scale_in

        qpos_quat_new = math.quat_integrate(qpos_quat, qvel_ang, timestep)

        qpos_next[qpos_adr + 0] = qpos_quat_new[0]
        qpos_next[qpos_adr + 1] = qpos_quat_new[1]
        qpos_next[qpos_adr + 2] = qpos_quat_new[2]
        qpos_next[qpos_adr + 3] = qpos_quat_new[3]

    elif jnttype == JointType.SLIDE or jnttype == JointType.HINGE:
        qpos_next[qpos_adr] = qpos[qpos_adr] + timestep * qvel[dof_adr] * qvel_scale_in
    else:
        assert False, "Unknown joint type"


@wp.kernel
def _next_velocity(
        # Model:
        opt_timestep: wp.array(dtype=float),
        # Data in:
        qvel_in: wp.array2d(dtype=float),
        qacc_in: wp.array2d(dtype=float),
        # In:
        qacc_scale_in: float,
        # Data out:
        qvel_out: wp.array2d(dtype=float),
):
    worldid, dofid = wp.tid()
    timestep = opt_timestep[worldid % opt_timestep.shape[0]]
    qvel_out[worldid, dofid] = qvel_in[worldid, dofid] + qacc_scale_in * qacc_in[worldid, dofid] * timestep


@wp.func
def _next_act(
        # Model:
        opt_timestep: float,  # kernel_analyzer: ignore
        # Data In:
        act_in: float,  # kernel_analyzer: ignore
        act_dot_in: float,  # kernel_analyzer: ignore
) -> float:
    # advance actuation
    act = act_in + act_dot_in * opt_timestep

    # clamp to actrange
    act = wp.clamp(act, 0, 1) # blah

    return act


@wp.kernel
def _next_activation(
        # Model:
        opt_timestep: wp.array(dtype=float),
        # Data in:
        act_in: wp.array2d(dtype=float),
        act_dot_in: wp.array2d(dtype=float),
        # Data out:
        act_out: wp.array2d(dtype=float),
):
    worldid, actid = wp.tid()
    opt_timestep_id = worldid % opt_timestep.shape[0]
    act = _next_act(
        opt_timestep[opt_timestep_id],
        act_in[worldid, actid],
        act_dot_in[worldid, actid],
    )
    act_out[worldid, actid] = act


@wp.kernel
def _next_time(
        # Model:
        opt_timestep: wp.array(dtype=float),
        # Data in:
        time_in: wp.array(dtype=float),
        # Data out:
        time_out: wp.array(dtype=float),
):
    worldid = wp.tid()
    time_out[worldid] = time_in[worldid] + opt_timestep[worldid % opt_timestep.shape[0]]

def _advance(m: Model, d: Data, qacc: wp.array, qvel: Optional[wp.array] = None):
    """Advance state and time given state derivatives"""

    # advance activations
    wp.launch(
        _next_activation,
        dim=(d.nworld, m.na),
        inputs=[
            m.opt.timestep,
            d.act,
            d.act_dot,
        ],
        outputs=[d.act],
    )

    wp.launch(
        _next_velocity,
        dim=(d.nworld, m.nv),
        inputs=[m.opt.timestep, d.qvel, qacc, 1.0],
        outputs=[d.qvel],
    )

    # advance positions with qvel if given, d.qvel otherwise (semi-implicit)
    qvel_in = qvel or d.qvel

    wp.launch(
        _next_position,
        dim=(d.nworld, m.njnt),
        inputs=[m.opt.timestep, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, qvel_in, 1.0],
        outputs=[d.qpos],
    )

    wp.launch(
        _next_time,
        dim=d.nworld,
        inputs=[m.opt.timestep, d.time],
        outputs=[d.time],
    )

    wp.copy(d.qacc_warmstart, d.qacc)


@wp.kernel
def _euler_damp_qfrc_sparse(
        # Model:
        opt_timestep: wp.array(dtype=float),
        dof_Madr: wp.array(dtype=int),
        dof_damping: wp.array2d(dtype=float),
        # Out:
        qM_integration_out: wp.array3d(dtype=float),
):
    worldid, tid = wp.tid()
    timestep = opt_timestep[worldid % opt_timestep.shape[0]]

    adr = dof_Madr[tid]
    qM_integration_out[worldid, 0, adr] += timestep * dof_damping[worldid, tid]


@cache_kernel
def _tile_euler_dense(tile: TileSet):
    @nested_kernel(module="unique", enable_backward=False)
    def euler_dense(
            # Model:
            dof_damping: wp.array2d(dtype=float),
            opt_timestep: wp.array(dtype=float),
            # Data in:
            qM_in: wp.array3d(dtype=float),
            efc_Ma_in: wp.array2d(dtype=float),
            # In:
            adr_in: wp.array(dtype=int),
            # Out:
            qacc_out: wp.array2d(dtype=float),
    ):
        worldid, nodeid = wp.tid()
        timestep = opt_timestep[worldid % opt_timestep.shape[0]]
        TILE_SIZE = wp.static(tile.size)

        dofid = adr_in[nodeid]
        M_tile = wp.tile_load(qM_in[worldid], shape=(TILE_SIZE, TILE_SIZE), offset=(dofid, dofid))
        damping_tile = wp.tile_load(dof_damping[worldid % dof_damping.shape[0]], shape=(TILE_SIZE,), offset=(dofid,))
        damping_scaled = damping_tile * timestep
        qm_integration_tile = wp.tile_diag_add(M_tile, damping_scaled)

        Ma_tile = wp.tile_load(efc_Ma_in[worldid], shape=(TILE_SIZE,), offset=(dofid,))
        L_tile = wp.tile_cholesky(qm_integration_tile)
        qacc_tile = wp.tile_cholesky_solve(L_tile, Ma_tile)
        wp.tile_store(qacc_out[worldid], qacc_tile, offset=(dofid))

    return euler_dense


@event_scope
def euler(m: Model, d: Data):
    """Euler integrator, semi-implicit in velocity."""
    # integrate damping implicitly
    qacc = wp.empty((d.nworld, m.nv), dtype=float)
    if m.opt.is_sparse:
        qM = wp.clone(d.qM)
        qLD = wp.empty((d.nworld, 1, m.nC), dtype=float)
        qLDiagInv = wp.empty((d.nworld, m.nv), dtype=float)
        wp.launch(
            _euler_damp_qfrc_sparse,
            dim=(d.nworld, m.nv),
            inputs=[m.opt.timestep, m.dof_Madr, m.dof_damping],
            outputs=[qM],
        )
        smooth.factor_solve_i(m, d, qM, qLD, qLDiagInv, qacc, d.efc.Ma)
    else:
        for tile in m.qM_tiles:
            wp.launch_tiled(
                _tile_euler_dense(tile),
                dim=(d.nworld, tile.adr.size),
                inputs=[m.dof_damping, m.opt.timestep, d.qM, d.efc.Ma, tile.adr],
                outputs=[qacc],
                block_dim=m.block_dim.euler_dense,
            )
    _advance(m, d, qacc)


@event_scope
def fwd_position(m: Model, d: Data):
    """ Position-dependent computations. """
    smooth.kinematics(m, d)
    smooth.com_pos(m, d)
    smooth.tendon(m, d)
    smooth.crb(m, d)
    smooth.factor_m(m, d)
    collision_driver.collision(m, d)
    constraint.make_constraint(m, d)


def _tendon_velocity(m: Model, d: Data):
    @nested_kernel(module="unique", enable_backward=False)
    def tendon_velocity(
            # Data in:
            qvel_in: wp.array2d(dtype=float),
            ten_J_in: wp.array3d(dtype=float),
            # Data out:
            ten_velocity_out: wp.array2d(dtype=float),
    ):
        worldid, tenid = wp.tid()
        ten_J_tile = wp.tile_load(ten_J_in[worldid, tenid], shape=wp.static(m.nv))
        qvel_tile = wp.tile_load(qvel_in[worldid], shape=wp.static(m.nv))
        ten_J_qvel_tile = wp.tile_map(wp.mul, ten_J_tile, qvel_tile)
        ten_velocity_tile = wp.tile_reduce(wp.add, ten_J_qvel_tile)
        ten_velocity_out[worldid, tenid] = ten_velocity_tile[0]

    wp.launch_tiled(
        tendon_velocity,
        dim=(d.nworld, m.nmuscle),
        inputs=[d.qvel, d.ten_J],
        outputs=[d.ten_velocity],
        block_dim=m.block_dim.tendon_velocity,
    )


@event_scope
def fwd_velocity(m: Model, d: Data):
    """Velocity-dependent computations."""
    _tendon_velocity(m, d)

    smooth.com_vel(m, d)
    passive.passive(m, d)
    smooth.rne(m, d)


@wp.kernel
def _actuator_force(
        # Model:
        na: int,
        opt_timestep: wp.array(dtype=float),
        actuator_dyntype: wp.array(dtype=int),
        actuator_actadr: wp.array(dtype=int),
        actuator_actnum: wp.array(dtype=int),
        actuator_ctrllimited: wp.array(dtype=bool),
        actuator_forcelimited: wp.array(dtype=bool),
        actuator_actlimited: wp.array(dtype=bool),
        actuator_dynprm: wp.array2d(dtype=vec10f),
        actuator_actearly: wp.array(dtype=bool),
        actuator_ctrlrange: wp.array2d(dtype=wp.vec2),
        actuator_forcerange: wp.array2d(dtype=wp.vec2),
        actuator_actrange: wp.array2d(dtype=wp.vec2),
        # Data in:
        act_in: wp.array2d(dtype=float),
        ctrl_in: wp.array2d(dtype=float),
        actuator_length_in: wp.array2d(dtype=float),
        actuator_velocity_in: wp.array2d(dtype=float),
        # In:
        dsbl_clampctrl: int,
        # Data out:
        act_dot_out: wp.array2d(dtype=float),
        actuator_force_out: wp.array2d(dtype=float),
):
    worldid, uid = wp.tid()

    actuator_ctrlrange_id = worldid % actuator_ctrlrange.shape[0]

    ctrl = ctrl_in[worldid, uid]

    if actuator_ctrllimited[uid] and not dsbl_clampctrl:
        ctrlrange = actuator_ctrlrange[actuator_ctrlrange_id, uid]
        ctrl = wp.clamp(ctrl, ctrlrange[0], ctrlrange[1])
    ctrl_act = ctrl

    act_first = actuator_actadr[uid]
    if na and act_first >= 0:
        act_last = act_first + actuator_actnum[uid] - 1
        dyntype = actuator_dyntype[uid]
        dynprm = actuator_dynprm[worldid % actuator_dynprm.shape[0], uid]

        dynprm = actuator_dynprm[worldid, uid]
        act = act_in[worldid, act_last]
        act_dot = util_misc.muscle_dynamics(ctrl, act, dynprm)

        act_dot_out[worldid, act_last] = act_dot

        if actuator_actearly[uid]:
            ctrl_act = _next_act(
                opt_timestep[worldid % opt_timestep.shape[0]],
                dyntype,
                dynprm,
                actuator_actrange[worldid % actuator_actrange.shape[0], uid],
                act,
                act_dot,
                1.0,
                actuator_actlimited[uid],
            )
        else:
            ctrl_act = act_in[worldid, act_last]

    # length = actuator_length_in[worldid, uid]
    # velocity = actuator_velocity_in[worldid, uid]

    # gain
    gain = 0.0
    # bias
    bias = 0.0  # BiasType.NONE
    force = gain * ctrl_act + bias

    if actuator_forcelimited[uid]:
        forcerange = actuator_forcerange[worldid % actuator_forcerange.shape[0], uid]
        force = wp.clamp(force, forcerange[0], forcerange[1])

    actuator_force_out[worldid, uid] = force


@wp.kernel
def _qfrc_actuator(
        # Model:
        nu: int,
        ngravcomp: int,
        jnt_actfrclimited: wp.array(dtype=bool),
        jnt_actgravcomp: wp.array(dtype=int),
        jnt_actfrcrange: wp.array2d(dtype=wp.vec2),
        dof_jntid: wp.array(dtype=int),
        # Data in:
        actuator_moment_in: wp.array3d(dtype=float),
        qfrc_gravcomp_in: wp.array2d(dtype=float),
        actuator_force_in: wp.array2d(dtype=float),
        # Data out:
        qfrc_actuator_out: wp.array2d(dtype=float),
):
    worldid, dofid = wp.tid()

    qfrc = float(0.0)
    for uid in range(nu):
        qfrc += actuator_moment_in[worldid, uid, dofid] * actuator_force_in[worldid, uid]

    jntid = dof_jntid[dofid]

    # actuator-level gravity compensation, skip if added as passive force
    if ngravcomp and jnt_actgravcomp[jntid]:
        qfrc += qfrc_gravcomp_in[worldid, dofid]

    if jnt_actfrclimited[jntid]:
        frcrange = jnt_actfrcrange[worldid % jnt_actfrcrange.shape[0], jntid]
        qfrc = wp.clamp(qfrc, frcrange[0], frcrange[1])

    qfrc_actuator_out[worldid, dofid] = qfrc


@event_scope
def fwd_actuation(m: Model, d: Data):
    """Actuation-dependent computations."""
    if not m.nu:
        d.act_dot.zero_()
        d.qfrc_actuator.zero_()
        return

    wp.launch(
        _actuator_force,
        dim=(d.nworld, m.nu),
        inputs=[
            m.na,
            m.opt.timestep,
            m.actuator_dyntype,
            m.actuator_actadr,
            m.actuator_actnum,
            m.actuator_ctrllimited,
            m.actuator_forcelimited,
            m.actuator_actlimited,
            m.actuator_dynprm,
            m.actuator_actearly,
            m.actuator_ctrlrange,
            m.actuator_forcerange,
            m.actuator_actrange,
            d.act,
            d.ctrl,
            d.actuator_length,
            d.actuator_velocity,
            m.opt.disableflags,
        ],
        outputs=[d.act_dot, d.actuator_force],
    )

    if m.nmuscle:
        # total actuator force at tendon
        ten_actfrc = wp.zeros((d.nworld, m.nmuscle), dtype=float)

    wp.launch(
        _qfrc_actuator,
        dim=(d.nworld, m.nv),
        inputs=[
            m.nu,
            m.ngravcomp,
            m.jnt_actfrclimited,
            m.jnt_actgravcomp,
            m.jnt_actfrcrange,
            m.dof_jntid,
            d.actuator_moment,
            d.qfrc_gravcomp,
            d.actuator_force,
        ],
        outputs=[d.qfrc_actuator],
    )


@wp.kernel
def _qfrc_smooth(
        # Data in:
        qfrc_applied_in: wp.array2d(dtype=float),
        qfrc_bias_in: wp.array2d(dtype=float),
        qfrc_passive_in: wp.array2d(dtype=float),
        qfrc_actuator_in: wp.array2d(dtype=float),
        # Data out:
        qfrc_smooth_out: wp.array2d(dtype=float),
):
    worldid, dofid = wp.tid()
    qfrc_smooth_out[worldid, dofid] = (
            qfrc_passive_in[worldid, dofid]
            - qfrc_bias_in[worldid, dofid]
            + qfrc_actuator_in[worldid, dofid]
            + qfrc_applied_in[worldid, dofid]
    )


@event_scope
def fwd_acceleration(m: Model, d: Data, factorize: bool = False):
    """Add up all non-constraint forces, compute qacc_smooth.

    Args:
      m: The model containing kinematic and dynamic information.
      d: The data object containing the current state and output arrays.
      factorize: Flag to factorize inertia matrix.
    """
    wp.launch(
        _qfrc_smooth,
        dim=(d.nworld, m.nv),
        inputs=[d.qfrc_applied, d.qfrc_bias, d.qfrc_passive, d.qfrc_actuator],
        outputs=[d.qfrc_smooth],
    )
    xfrc_accumulate(m, d, d.qfrc_smooth)

    if factorize:
        smooth.factor_solve_i(m, d, d.qM, d.qLD, d.qLDiagInv, d.qacc_smooth, d.qfrc_smooth)
    else:
        smooth.solve_m(m, d, d.qacc_smooth, d.qfrc_smooth)


@event_scope
def forward(m: Model, d: Data):
    """Forward dynamics."""
    fwd_position(m, d)
    fwd_velocity(m, d)
    fwd_actuation(m, d)
    fwd_acceleration(m, d, factorize=True)
    solver.solve(m, d)


@event_scope
def step(m: Model, d: Data):
    """Advance simulation."""
    forward(m, d)
    euler(m, d)
