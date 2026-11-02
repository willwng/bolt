"""
Coordinate coupler constraints q_dep = scale * f(q_indep), enforced exactly at the acceleration level with
Lagrange multipliers (as Simbody does), with the drift removed by projecting q_dep and u_dep after each step.

With G the constraint Jacobian (one row per coupler: e_dep - scale * f'(q_indep) e_indep), the constrained
accelerations are udot = udot_0 - Y lambda, where udot_0 are the unconstrained accelerations, Y = M^-1 G^T,
and (G Y) lambda = G udot_0 - b with the bias b = scale * f''(q_indep) * u_indep^2.
"""
import warp as wp

from bolt.types import Data
from bolt.types import Model

from ..warp_util import event_scope
from ..warp_util import launch
from . import acceleration

wp.set_module_options({"enable_backward": False})


@wp.func
def _apply_G(
        x: wp.array(dtype=float),
        dep_dofadr: int,
        indep_dofadr: int,
        scale_df: float,
) -> float:
    """ One row of G times x """
    return x[dep_dofadr] - scale_df * x[indep_dofadr]


@wp.kernel
def _coupler_unit_force(
        # Model:
        coupler_fnadr: wp.array(dtype=int),
        coupler_scale: wp.array(dtype=float),
        coupler_indep_dofadr: wp.array(dtype=int),
        coupler_dep_dofadr: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        cst_fn_output_in: wp.array2d(dtype=wp.vec3),
        # In:
        couplerid: int,
        nv: int,
        # Data out:
        coupler_ufrc_out: wp.array2d(dtype=float),
):
    """ Generalized force G^T e_k: column k of G^T """
    worldid = wp.tid()
    if integration_done_in[worldid]:
        return
    for i in range(nv):
        coupler_ufrc_out[worldid, i] = 0.0
    df = cst_fn_output_in[worldid, coupler_fnadr[couplerid]][1]
    coupler_ufrc_out[worldid, coupler_dep_dofadr[couplerid]] = 1.0
    coupler_ufrc_out[worldid, coupler_indep_dofadr[couplerid]] = -coupler_scale[couplerid] * df


@wp.kernel
def _coupler_solve(
        # Model:
        coupler_fnadr: wp.array(dtype=int),
        coupler_scale: wp.array(dtype=float),
        coupler_indep_dofadr: wp.array(dtype=int),
        coupler_dep_dofadr: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qvel_in: wp.array2d(dtype=float),
        cst_fn_output_in: wp.array2d(dtype=wp.vec3),
        coupler_Y_in: wp.array3d(dtype=float),
        coupler_A_in: wp.array3d(dtype=wp.spatial_vector),
        # In:
        ncoupler: int,
        nv: int,
        nbody: int,
        # Data out:
        coupler_GY_out: wp.array3d(dtype=float),
        coupler_rhs_out: wp.array2d(dtype=float),
        qacc_out: wp.array2d(dtype=float),
        body_A_GB_out: wp.array2d(dtype=wp.spatial_vector),
):
    """ Solves (G Y) lambda = G udot_0 - b and applies udot = udot_0 - Y lambda (and likewise to body accelerations) """
    worldid = wp.tid()
    if integration_done_in[worldid]:
        return
    GY = coupler_GY_out[worldid]
    rhs = coupler_rhs_out[worldid]
    qacc = qacc_out[worldid]

    # Assemble G Y and the acceleration error G udot_0 - b
    for k in range(ncoupler):
        dep, indep = coupler_dep_dofadr[k], coupler_indep_dofadr[k]
        fn = cst_fn_output_in[worldid, coupler_fnadr[k]]
        scale = coupler_scale[k]
        for j in range(ncoupler):
            GY[k, j] = _apply_G(coupler_Y_in[j, worldid], dep, indep, scale * fn[1])
        u = qvel_in[worldid, indep]
        rhs[k] = _apply_G(qacc, dep, indep, scale * fn[1]) - scale * fn[2] * u * u

    # G M^-1 G^T is symmetric positive definite: Gaussian elimination without pivoting, lambda overwrites rhs
    for p in range(ncoupler):
        for r in range(p + 1, ncoupler):
            factor = GY[r, p] / GY[p, p]
            for c in range(p, ncoupler):
                GY[r, c] = GY[r, c] - factor * GY[p, c]
            rhs[r] = rhs[r] - factor * rhs[p]
    for p in range(ncoupler - 1, -1, -1):
        x = rhs[p]
        for c in range(p + 1, ncoupler):
            x -= GY[p, c] * rhs[c]
        rhs[p] = x / GY[p, p]

    for i in range(nv):
        a = qacc[i]
        for k in range(ncoupler):
            a -= rhs[k] * coupler_Y_in[k, worldid, i]
        qacc[i] = a
    for bodyid in range(1, nbody):
        A = body_A_GB_out[worldid, bodyid]
        for k in range(ncoupler):
            A -= rhs[k] * coupler_A_in[k, worldid, bodyid]
        body_A_GB_out[worldid, bodyid] = A


@wp.kernel
def _coupler_project(
        # Model:
        coupler_fnadr: wp.array(dtype=int),
        coupler_scale: wp.array(dtype=float),
        coupler_indep_dofadr: wp.array(dtype=int),
        coupler_dep_qposadr: wp.array(dtype=int),
        coupler_dep_dofadr: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        cst_fn_output_in: wp.array2d(dtype=wp.vec3),
        # Data out:
        qpos_out: wp.array2d(dtype=float),
        qvel_out: wp.array2d(dtype=float),
):
    worldid, couplerid = wp.tid()
    if integration_done_in[worldid]:
        return
    fn = cst_fn_output_in[worldid, coupler_fnadr[couplerid]]
    scale = coupler_scale[couplerid]
    qpos_out[worldid, coupler_dep_qposadr[couplerid]] = scale * fn[0]
    qvel_out[worldid, coupler_dep_dofadr[couplerid]] = (
            scale * fn[1] * qvel_out[worldid, coupler_indep_dofadr[couplerid]])


@event_scope
def constrain_accelerations(m: Model, d: Data):
    """ Corrects the unconstrained d.qacc and d.body_A_GB so the couplers' accelerations are satisfied """
    for k in range(m.ncoupler):
        launch(
            _coupler_unit_force,
            dim=d.nworld,
            inputs=[
                m.coupler_fnadr, m.coupler_scale, m.coupler_indep_dofadr, m.coupler_dep_dofadr,
                d.integration_done, d.cst_fn_output,
                k, m.nv,
            ],
            outputs=[d.coupler_ufrc],
        )
        # Y = M^-1 G^T e_k: no bias terms, and coupler_A[k][:, 0] (ground) stays zero
        zero = d.coupler_zero_body
        acceleration.solve_udot(m, d, d.coupler_ufrc, zero, zero, zero, d.coupler_Y[k], d.coupler_A[k])

    launch(
        _coupler_solve,
        dim=d.nworld,
        inputs=[
            m.coupler_fnadr, m.coupler_scale, m.coupler_indep_dofadr, m.coupler_dep_dofadr,
            d.integration_done, d.qvel, d.cst_fn_output, d.coupler_Y, d.coupler_A,
            m.ncoupler, m.nv, m.nbody,
        ],
        outputs=[d.coupler_GY, d.coupler_rhs, d.qacc, d.body_A_GB],
    )


@event_scope
def project(m: Model, d: Data):
    """ Sets q_dep = scale * f(q_indep) and u_dep = scale * f'(q_indep) * u_indep. Needs the evaluated functions """
    launch(
        _coupler_project,
        dim=(d.nworld, m.ncoupler),
        inputs=[
            m.coupler_fnadr, m.coupler_scale, m.coupler_indep_dofadr, m.coupler_dep_qposadr, m.coupler_dep_dofadr,
            d.integration_done, d.cst_fn_output,
        ],
        outputs=[d.qpos, d.qvel],
    )
