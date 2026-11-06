"""
Tree passes: recursions over the kinematic tree (articulated inertias, accelerations), where each body needs its
children's (inward) or its parent's (outward) result first.

A pass runs in one of three schedules (TreePass). FUSED is one block per world: its threads split each level's
bodies and wait at a barrier (sync_block) before the next level, so the whole pass is one launch, with every world's
tree levels walked in parallel. The kernels take the level widths and offsets from the model's TreeSpec as
compile-time constants (level_tables) and the body ids level by level from Model.body_tree_bodies. The level loop is
deliberately not unrolled: each level would inline its own copy of the heavy per-body functions, and the kernel's
code then outgrows the instruction cache (measured: with few worlds active, one world's pass became slower than one
thread walking the whole tree). Hence the level count is a kernel argument (a run-time trip count), not wp.static.

On CPU, tiled launches run one thread per block (wp.block_dim() is 1), so the same kernel walks each level in turn.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import warp as wp

from ..warp_util import tree_pass_per_world

from .spec import TreeSpec

if TYPE_CHECKING:  # bolt.types imports this package (its specs)
    from bolt.types import Model


class TreePass(enum.Enum):
    PER_WORLD = 0  # one thread per world walks the whole tree; for narrow trees (chains)
    FUSED = 1  # one block per world, threads split each level, a barrier between levels; one launch
    PER_LEVEL = 2  # one launch per tree level, a thread per (world, body); the reference schedule


# Overrides the schedule chosen by tree_pass (tests and benchmarks)
FORCE_TREE_PASS: TreePass | None = None


def tree_pass(m: Model) -> TreePass:
    """ Schedule of the heavy tree passes for model m """
    if FORCE_TREE_PASS is not None:
        return FORCE_TREE_PASS
    return TreePass.PER_WORLD if tree_pass_per_world(m.body_tree) else TreePass.FUSED


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def sync_block():
    """
    Barrier for the threads of a block; orders their global memory writes before the reads after it. All threads of
    the block must reach it, so a kernel may only return early on a condition uniform over the block (the world)
    """
    ...


def level_tables(spec: TreeSpec) -> tuple:
    """ (start, width) of each level in Model.body_tree_bodies, as vector constants for kernels to capture """
    vec = wp.types.vector(length=spec.nlevel, dtype=int)
    return vec(*spec.adr), vec(*spec.widths)
