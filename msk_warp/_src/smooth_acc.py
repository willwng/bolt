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

from . import math
from . import support
from .types import Data
from .consts import MJ_MINVAL
from .types import Model
from .types import TileSet
from .warp_util import cache_kernel
from .warp_util import event_scope
from .warp_util import kernel as nested_kernel

wp.set_module_options({"enable_backward": False})


@cache_kernel
def _tile_cholesky_solve(tile: TileSet):
    """Returns a kernel for dense Cholesky backsubstitution of a tile."""

    @nested_kernel(module="unique", enable_backward=False)
    def cholesky_solve(
            # In:
            L: wp.array3d(dtype=float),
            y: wp.array2d(dtype=float),
            adr: wp.array(dtype=int),
            # Out:
            x: wp.array2d(dtype=float),
    ):
        worldid, nodeid = wp.tid()
        TILE_SIZE = wp.static(tile.size)

        dofid = adr[nodeid]
        y_slice = wp.tile_load(y[worldid], shape=(TILE_SIZE,), offset=(dofid,))
        L_tile = wp.tile_load(L[worldid], shape=(TILE_SIZE, TILE_SIZE),
                              offset=(dofid, dofid))
        x_slice = wp.tile_cholesky_solve(L_tile, y_slice)
        wp.tile_store(x[worldid], x_slice, offset=(dofid,))

    return cholesky_solve


def _solve_LD_dense(m: Model, d: Data, L: wp.array3d(dtype=float),
                    x: wp.array2d(dtype=float),
                    y: wp.array2d(dtype=float)):
    """Computes dense backsubstitution: x = inv(L'*L)*y."""
    for tile in m.qM_tiles:
        wp.launch_tiled(
            _tile_cholesky_solve(tile),
            dim=(d.nworld, tile.adr.size),
            inputs=[L, y, tile.adr],
            outputs=[x],
            block_dim=m.block_dim.cholesky_solve,
        )


def solve_LD(
        m: Model,
        d: Data,
        L: wp.array3d(dtype=float),
        x: wp.array2d(dtype=float),
        y: wp.array2d(dtype=float),
):
    """Computes backsubstitution to solve a linear system of the form x = inv(L'*D*L) * y.

    L and D are the factors from the Cholesky factorization of the inertia matrix.

    Args:
      m: The model containing factorization and sparsity information.
      d: The data object containing workspace and factorization results.
      L: Lower-triangular factor from the factorization (dense).
      x: Output array for the solution.
      y: Input right-hand side array.
    """
    _solve_LD_dense(m, d, L, x, y)


@event_scope
def solve_m(m: Model, d: Data, x: wp.array2d(dtype=float),
            y: wp.array2d(dtype=float)):
    """Computes backsubstitution: x = qLD * y.

    Args:
      m: The model containing inertia and factorization information.
      d: The data object containing factorization results.
      x: Output array for the solution.
      y: Input right-hand side array.
    """
    solve_LD(m, d, d.qLD, x, y)


@cache_kernel
def _tile_cholesky_factorize_solve(tile: TileSet):
    """Returns a kernel for dense Cholesky factorization and backsubstitution of a tile."""

    @nested_kernel(module="unique", enable_backward=False)
    def cholesky_factorize_solve(
            # In:
            M: wp.array3d(dtype=float),
            y: wp.array2d(dtype=float),
            adr: wp.array(dtype=int),
            # Out:
            x: wp.array2d(dtype=float),
            L: wp.array3d(dtype=float),
    ):
        worldid, nodeid = wp.tid()
        TILE_SIZE = wp.static(tile.size)

        dofid = adr[nodeid]
        M_tile = wp.tile_load(M[worldid], shape=(TILE_SIZE, TILE_SIZE),
                              offset=(dofid, dofid))
        y_slice = wp.tile_load(y[worldid], shape=(TILE_SIZE,), offset=(dofid,))

        L_tile = wp.tile_cholesky(M_tile)
        wp.tile_store(L[worldid], L_tile, offset=(dofid, dofid))
        x_slice = wp.tile_cholesky_solve(L_tile, y_slice)
        wp.tile_store(x[worldid], x_slice, offset=(dofid,))

    return cholesky_factorize_solve


def _factor_solve_i_dense(
        m: Model,
        d: Data,
        M: wp.array3d(dtype=float),
        x: wp.array2d(dtype=float),
        y: wp.array2d(dtype=float),
        L: wp.array3d(dtype=float),
):
    for tile in m.qM_tiles:
        wp.launch_tiled(
            _tile_cholesky_factorize_solve(tile),
            dim=(d.nworld, tile.adr.size),
            inputs=[M, y, tile.adr],
            outputs=[x, L],
            block_dim=m.block_dim.cholesky_factorize_solve,
        )


def factor_solve_i(m, d, M, L, x, y):
    """Factorizes and solves the linear system: x = inv(L'*D*L) * y or x = inv(L'*L) * y.

    M is an inertia-like matrix and L, D are its Cholesky-like factors.

    This function first factorizes the matrix M, then solves the system
    for x given right-hand side y.

    Args:
      m: The model containing factorization and sparsity information.
      d: The data object containing workspace and factorization results.
      M: The inertia-like matrix to factorize.
      L: Output lower-triangular factor from the factorization (dense).
      x: Output array for the solution.
      y: Input right-hand side array.
    """
    _factor_solve_i_dense(m, d, M, x, y, L)
