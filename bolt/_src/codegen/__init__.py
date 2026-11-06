"""
Kernels specialized to the loaded model.

The loader extracts a model's structure once into small hashable specs (spec.py), which kernel factories
(functools.cache over the spec) bake into their kernels with wp.static: trip counts, offsets and tables become
compile-time constants, so a kernel runs a model-specific schedule in one launch instead of a generic one spread over
many. The subsystems define their own specialized kernels next to the generic ones; this package holds the specs and
the primitives they share (tree.py: the tree-pass schedule and the block barrier).
"""

from .spec import TreeSpec
from .tree import TreePass
from .tree import level_tables
from .tree import sync_block
from .tree import tree_pass

__all__ = ["TreeSpec", "TreePass", "level_tables", "sync_block", "tree_pass"]
