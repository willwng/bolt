import warp as wp

from .types import Data
from .types import Model
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


# returns f(x), df(x)
@wp.func
def calc_value(
        # data in
        qpos_in: wp.array(dtype=float),
        # in
        fn_type: int,
        txfm_fn_adr: int,
        qadr: int,
        # model in
        const_fns: wp.array(dtype=float),
        linear_fns: wp.array(dtype=wp.vec2),
) -> wp.vec2:
    return wp.vec3(0.0, 0.0)


@wp.kernel
def _eval_cst_fn(
        # Data in:
        integration_done_in: wp.array(dtype=bool),
):
    worldid, cst_jnt_id = wp.tid()
    if integration_done_in[worldid]:
        return
    return


@event_scope
def evaluate_txfm(m: Model, d: Data):
    # evaluate all constant fns
    # evaluate all linear fns
    # evaluate all polynomial fns
    return


@event_scope
def evaluate_cst_functions(m: Model, d: Data):
    """ Evaluates custom functions """
    wp.launch(
        _eval_cst_fn,
        dim=(d.nworld, m.njnts_cst),
        inputs=[
            d.integration_done,
        ],
        outputs=[],
    )
