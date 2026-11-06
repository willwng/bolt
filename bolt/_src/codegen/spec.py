"""
Structural specs of a model: plain Python values that kernel factories bake into kernels. They must stay hashable
(frozen dataclasses of ints and tuples), as they are the factories' cache keys, and import nothing from Bolt, as
bolt.types holds them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TreeSpec:
    """
    The kinematic tree by level: levels[0] holds ground, levels[k] the bodies k joints away from it. Children have
    larger ids than their parents, so walking the levels in order (or reversed) visits parents before children (or
    after them)
    """
    levels: tuple[tuple[int, ...], ...]

    @staticmethod
    def from_levels(levels) -> TreeSpec:
        return TreeSpec(tuple(tuple(int(b) for b in level) for level in levels))

    @property
    def nlevel(self) -> int:
        return len(self.levels)

    @property
    def nbody(self) -> int:
        return sum(self.widths)

    @property
    def widths(self) -> tuple[int, ...]:
        """ number of bodies in each level """
        return tuple(len(level) for level in self.levels)

    @property
    def adr(self) -> tuple[int, ...]:
        """ start of each level in bodies """
        return tuple(sum(self.widths[:k]) for k in range(self.nlevel))

    @property
    def bodies(self) -> tuple[int, ...]:
        """ body ids level by level (Model.body_tree_bodies) """
        return tuple(b for level in self.levels for b in level)
