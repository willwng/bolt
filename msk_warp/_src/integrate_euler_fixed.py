import warp as wp

from . import forward
from . import integrate_common
from . import math
from . import mobilizers
from .consts import MJ_MINVAL
from .types import ActuatorMetadata
from .types import Data
from .types import Model
from .types import MuscleMetadata
from .types import TileSet
from .warp_util import cache_kernel
from .warp_util import event_scope
from .warp_util import kernel as nested_kernel

wp.set_module_options({"enable_backward": False})


@cache_kernel
def _tile_euler_dense(tile: TileSet):
    @nested_kernel(module="unique", enable_backward=False)
    def euler_dense(
            # Model:
            dof_damping: wp.array(dtype=float),
            # Data in:
            actual_step_size_in: wp.array(dtype=float),
            qM_in: wp.array3d(dtype=float),
            efc_Ma_in: wp.array2d(dtype=float),
            # In:
            adr_in: wp.array(dtype=int),
            scale: float,
            # Out:
            qacc_out: wp.array2d(dtype=float),
    ):
        worldid, nodeid = wp.tid()
        timestep = actual_step_size_in[worldid] * scale
        TILE_SIZE = wp.static(tile.size)

        dofid = adr_in[nodeid]
        M_tile = wp.tile_load(qM_in[worldid], shape=(TILE_SIZE, TILE_SIZE),
                              offset=(dofid, dofid))
        damping_tile = wp.tile_load(dof_damping, shape=(TILE_SIZE,),
                                    offset=(dofid,))
        damping_scaled = damping_tile * timestep
        qm_integration_tile = wp.tile_diag_add(M_tile, damping_scaled)

        Ma_tile = wp.tile_load(efc_Ma_in[worldid], shape=(TILE_SIZE,),
                               offset=(dofid,))
        L_tile = wp.tile_cholesky(qm_integration_tile)
        qacc_tile = wp.tile_cholesky_solve(L_tile, Ma_tile)
        wp.tile_store(qacc_out[worldid], qacc_tile, offset=(dofid))

    return euler_dense


@event_scope
def euler(m: Model, d: Data, scale: float):
    """
    Euler integrator, semi-implicit in velocity.
    Requires state derivative is set already
    """
    for tile in m.qM_tiles:
        wp.launch_tiled(
            _tile_euler_dense(tile),
            dim=(d.nworld, tile.adr.size),
            inputs=[m.dof_damping, d.actual_step_size, d.qM, d.efc.Ma, tile.adr, scale],
            outputs=[d.qacc_euler],
            block_dim=m.block_dim.euler_dense,
        )
    integrate_common.advance(m, d, d.qacc_euler, d.qvel, scale)


@event_scope
def integrate(m: Model, d: Data):
    """Steps from d.time to d.next_time using RK4 """
    integrate_common.update_step_size(m, d)
    euler(m, d, 1.0)
    forward.fwd(m, d)  # realize state for next step
