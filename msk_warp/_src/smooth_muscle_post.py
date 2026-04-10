import warp as wp

from .types import Data
from .types import Model
from .types import MuscleMetadata
from .consts import MSK_MINVAL
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@event_scope
def copy_ufrc_into_moment_arm(m: Model, d: Data, nodeid: int, ufrc: wp.array2d(dtype=float)):
    """ For point-based paths only """

    @wp.kernel
    def _copy_ufrc_into_moment_arm_kernel(
            # Model:
            muscle_pt_to_mid: wp.array(dtype=int),
            # Data in:
            ufrc_in: wp.array2d(dtype=float),
            # In:
            nid: int,
            # Data out:
            muscle_moment_arm_out: wp.array3d(dtype=float),
    ):
        worldid = wp.tid()
        muscleid = muscle_pt_to_mid[nid]

        nv = wp.static(m.nv)
        ufrc_tile = wp.tile_load(ufrc_in[worldid], shape=nv)
        wp.tile_store(muscle_moment_arm_out[worldid, muscleid], ufrc_tile)
        return

    if m.nmuscle:
        wp.launch_tiled(
            _copy_ufrc_into_moment_arm_kernel,
            dim=(d.nworld,),
            inputs=[m.muscle_metadata, ufrc, nodeid, ],
            outputs=[d.muscle_moment_arm],
            block_dim=m.block_dim.muscle_path,
        )
