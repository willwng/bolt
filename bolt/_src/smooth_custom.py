import warp as wp

from .types import Data
from .types import Model
from .consts import (IDX_SCRATCH_ROT_F, IDX_SCRATCH_ROT_DF, IDX_SCRATCH_ROT_D2F,
                     IDX_SCRATCH_TRANS_F, IDX_SCRATCH_TRANS_DF, IDX_SCRATCH_TRANS_D2F,
                     BOLT_SIG_REAL)
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _eval_const_fn(
        # Model in:
        const_fn_adr: wp.array(dtype=int),
        const_fn_c: wp.array(dtype=float),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        # Data out:
        cst_fn_output: wp.array2d(dtype=wp.vec3),
):
    worldid, constid = wp.tid()
    if integration_done_in[worldid]:
        return
    functionid = const_fn_adr[constid]
    c = const_fn_c[constid]
    cst_fn_output[worldid, functionid] = wp.vec3(c, 0.0, 0.0)
    return


@wp.kernel
def _eval_lin_fn(
        # Model in:
        linear_fn_adr: wp.array(dtype=int),
        linear_fn_qpos_adr: wp.array(dtype=int),
        linear_fn_mb: wp.array(dtype=wp.vec2),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qpos_in: wp.array2d(dtype=float),
        # Data out:
        cst_fn_output: wp.array2d(dtype=wp.vec3),
):
    worldid, linearid = wp.tid()
    if integration_done_in[worldid]:
        return
    mb = linear_fn_mb[linearid]
    qposadr = linear_fn_qpos_adr[linearid]
    functionid = linear_fn_adr[linearid]

    m, b = mb[0], mb[1]
    q = qpos_in[worldid, qposadr]

    cst_fn_output[worldid, functionid] = wp.vec3(m * q + b, m, 0.0)
    return


@wp.kernel
def _eval_poly_fn(
        # Model in:
        poly_fn_adr: wp.array(dtype=int),
        poly_fn_qpos_adr: wp.array(dtype=int),
        poly_fn_coeff: wp.array(dtype=float),
        poly_fn_coeff_adr: wp.array(dtype=int),
        poly_fn_coeff_num: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qpos_in: wp.array2d(dtype=float),
        # Data out:
        cst_fn_output: wp.array2d(dtype=wp.vec3),
):
    worldid, polyid = wp.tid()
    if integration_done_in[worldid]:
        return
    qposadr = poly_fn_qpos_adr[polyid]
    functionid = poly_fn_adr[polyid]
    q = qpos_in[worldid, qposadr]

    # Number of coefficients, address in flatteneed array, order
    coeffs_num = poly_fn_coeff_num[polyid]
    coeffs_adr = poly_fn_coeff_adr[polyid]
    order = coeffs_num - 1

    # Horner's method
    f, df, d2f = float(0.0), float(0.0), float(0.0)
    for i in range(order + 1):
        c = poly_fn_coeff[coeffs_adr + i]
        d2f = d2f * q + df
        df = df * q + f
        f = f * q + c
    d2f *= 2.0

    cst_fn_output[worldid, functionid] = wp.vec3(f, df, d2f)
    return


@wp.kernel
def _eval_spline_fn(
        # Model in:
        spline_fn_adr: wp.array(dtype=int),
        spline_fn_qpos_adr: wp.array(dtype=int),
        spline_fn_xy_y2s: wp.array(dtype=wp.vec3),
        spline_fn_xys_adr: wp.array(dtype=int),
        spline_fn_xys_num: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qpos_in: wp.array2d(dtype=float),
        # Data out:
        cst_fn_output: wp.array2d(dtype=wp.vec3),
):
    worldid, polyid = wp.tid()
    if integration_done_in[worldid]:
        return

    qposadr = spline_fn_qpos_adr[polyid]
    functionid = spline_fn_adr[polyid]
    q = qpos_in[worldid, qposadr]

    # Number of points, address in flattened array
    xys_num = spline_fn_xys_num[polyid]
    adr = spline_fn_xys_adr[polyid]

    pt_first = spline_fn_xy_y2s[adr]
    pt_last = spline_fn_xy_y2s[adr + xys_num - 1]

    # left linear extrapolation
    if q <= pt_first[0]:
        x0, y0, y2_0 = pt_first[0], pt_first[1], pt_first[2]
        pt1 = spline_fn_xy_y2s[adr + 1]
        x1, y1, y2_1 = pt1[0], pt1[1], pt1[2]
        h = x1 - x0
        df = float(0.0)
        if h > BOLT_SIG_REAL:
            df = (y1 - y0) / h - (h / 6.0) * (2.0 * y2_0 + y2_1)
        f = y0 + df * (q - x0)
        cst_fn_output[worldid, functionid] = wp.vec3(f, df, 0.0)
        return

    # right linear extrapolation
    if q >= pt_last[0]:
        x1, y1, y2_1 = pt_last[0], pt_last[1], pt_last[2]
        pt0 = spline_fn_xy_y2s[adr + xys_num - 2]
        x0, y0, y2_0 = pt0[0], pt0[1], pt0[2]
        h = x1 - x0
        df = float(0.0)
        if h > BOLT_SIG_REAL:
            df = (y1 - y0) / h + (h / 6.0) * (y2_0 + 2.0 * y2_1)
        f = y1 + df * (q - x1)
        cst_fn_output[worldid, functionid] = wp.vec3(f, df, 0.0)
        return

    # binary search for interval
    low = int(0)
    high = int(xys_num - 1)
    while low < high - 1:
        mid = low + (high - low) // 2
        pt_mid = spline_fn_xy_y2s[adr + mid]
        if pt_mid[0] <= q:
            low = mid
        else:
            high = mid

    pt0 = spline_fn_xy_y2s[adr + low]
    pt1 = spline_fn_xy_y2s[adr + high]

    x0, y0, y2_0 = pt0[0], pt0[1], pt0[2]
    x1, y1, y2_1 = pt1[0], pt1[1], pt1[2]

    h = x1 - x0
    if h > BOLT_SIG_REAL:
        a = (x1 - q) / h
        b = (q - x0) / h
        f = a * y0 + b * y1 + ((h * h) / 6.0) * ((a * a * a - a) * y2_0 + (b * b * b - b) * y2_1)
        df = (y1 - y0) / h - (h / 6.0) * ((3.0 * a * a - 1.0) * y2_0 - (3.0 * b * b - 1.0) * y2_1)
        d2f = a * y2_0 + b * y2_1
    else:
        f = y0
        df = 0.0
        d2f = 0.0

    cst_fn_output[worldid, functionid] = wp.vec3(f, df, d2f)


@wp.kernel
def _fetch_fn_into_cst(
        # Model in:
        custom_to_mob_id: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        cst_fn_output_in: wp.array2d(dtype=wp.vec3),
        # Data out:
        mob_scratch_out: wp.array3d(dtype=wp.vec3),
):
    worldid, customjointid = wp.tid()
    if integration_done_in[worldid]:
        return
    mobid = custom_to_mob_id[customjointid]
    # Fetch the function output into scratch
    scratch_rot, dscratch_rot, d2scratch_rot = wp.vec3(), wp.vec3(), wp.vec3()
    scratch_trans, dscratch_trans, d2scratch_trans = wp.vec3(), wp.vec3(), wp.vec3()
    for i in range(3):
        rot_fn_idx = 6 * customjointid + i
        trans_fn_idx = 6 * customjointid + 3 + i
        rot_fn_output = cst_fn_output_in[worldid, rot_fn_idx]
        trans_fn_output = cst_fn_output_in[worldid, trans_fn_idx]

        scratch_rot[i] = rot_fn_output[0]
        dscratch_rot[i] = rot_fn_output[1]
        d2scratch_rot[i] = rot_fn_output[2]

        scratch_trans[i] = trans_fn_output[0]
        dscratch_trans[i] = trans_fn_output[1]
        d2scratch_trans[i] = trans_fn_output[2]

    # Store in the mobilizer scratch.
    mob_scratch_out[worldid, mobid, IDX_SCRATCH_ROT_F] = scratch_rot
    mob_scratch_out[worldid, mobid, IDX_SCRATCH_ROT_DF] = dscratch_rot
    mob_scratch_out[worldid, mobid, IDX_SCRATCH_ROT_D2F] = d2scratch_rot
    mob_scratch_out[worldid, mobid, IDX_SCRATCH_TRANS_F] = scratch_trans
    mob_scratch_out[worldid, mobid, IDX_SCRATCH_TRANS_DF] = dscratch_trans
    mob_scratch_out[worldid, mobid, IDX_SCRATCH_TRANS_D2F] = d2scratch_trans
    return


@event_scope
def evaluate_cst_functions(m: Model, d: Data):
    """ Evaluates all custom functions, compute f(x), f'(x) """
    if m.nlinearfn:
        wp.launch(
            _eval_lin_fn,
            dim=(d.nworld, m.nlinearfn),
            inputs=[
                m.linear_fn_adr, m.linear_fn_qpos_adr, m.linear_fn_mb,
                d.integration_done, d.qpos,
            ],
            outputs=[d.cst_fn_output],
        )

    # TODO: do we really need to keep evaluating this? it's cheap but repetitive
    if m.nconstfn:
        wp.launch(
            _eval_const_fn,
            dim=(d.nworld, m.nconstfn),
            inputs=[
                m.const_fn_adr, m.const_fn_c,
                d.integration_done,
            ],
            outputs=[d.cst_fn_output],
        )

    if m.npolyfn:
        wp.launch(
            _eval_poly_fn,
            dim=(d.nworld, m.npolyfn),
            inputs=[
                m.poly_fn_adr, m.poly_fn_qpos_adr, m.poly_fn_coeff, m.poly_fn_coeff_adr, m.poly_fn_coeff_num,
                d.integration_done, d.qpos,
            ],
            outputs=[d.cst_fn_output],
        )

    if m.nsplinefn:
        wp.launch(
            _eval_spline_fn,
            dim=(d.nworld, m.nsplinefn),
            inputs=[
                m.spline_fn_adr, m.spline_fn_qpos_adr, m.spline_fn_xy_y2s, m.spline_fn_xys_adr, m.spline_fn_xys_num,
                d.integration_done, d.qpos,
            ],
            outputs=[d.cst_fn_output],
        )


@event_scope
def prepare_cst_joint(m: Model, d: Data):
    """ Loads the function outputs into the custom function mobilizer scratch """
    if m.njnts_cst:
        wp.launch(
            _fetch_fn_into_cst,
            dim=(d.nworld, m.njnts_cst),
            inputs=[
                m.cst_to_mob_id,
                d.integration_done, d.cst_fn_output,
            ],
            outputs=[d.mob_scratch],
        )
    return
