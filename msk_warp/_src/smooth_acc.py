# Copyright 2025 The Newton Developers
# Modified for MSKWarp by Will Wang
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
from .types import Data
from .types import Model
from .types import TileSet
from .types import vec10
from .warp_util import cache_kernel
from .warp_util import event_scope
from .warp_util import kernel as nested_kernel

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _crb_accumulate(
        # Model:
        body_parentid: wp.array(dtype=int),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        crb_in: wp.array2d(dtype=vec10),
        # In:
        body_tree_: wp.array(dtype=int),
        # Data out:
        crb_out: wp.array2d(dtype=vec10),
):
    worldid, nodeid = wp.tid()
    if integration_done_in[worldid]:
        return
    bodyid = body_tree_[nodeid]
    pid = body_parentid[bodyid]
    if pid == 0:
        return
    wp.atomic_add(crb_out, worldid, pid, crb_in[worldid, bodyid])


@wp.kernel
def _qM_dense(
        # Model:
        dof_bodyid: wp.array(dtype=int),
        dof_parentid: wp.array(dtype=int),
        dof_armature: wp.array(dtype=float),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        cdof_in: wp.array2d(dtype=wp.spatial_vector),
        crb_in: wp.array2d(dtype=vec10),
        # Data out:
        qM_out: wp.array3d(dtype=float),
):
    worldid, dofid = wp.tid()
    if integration_done_in[worldid]:
        return
    bodyid = dof_bodyid[dofid]
    # init M(i,i) with armature inertia.
    M = dof_armature[dofid]

    # precompute buf = crb_body_i * cdof_i
    buf = math.inert_vec(crb_in[worldid, bodyid], cdof_in[worldid, dofid])
    M += wp.dot(cdof_in[worldid, dofid], buf)

    qM_out[worldid, dofid, dofid] = M

    # sparse backward pass over ancestors
    dofidi = dofid
    dofid = dof_parentid[dofid]
    while dofid >= 0:
        qMij = wp.dot(cdof_in[worldid, dofid], buf)
        qM_out[worldid, dofidi, dofid] += qMij
        qM_out[worldid, dofid, dofidi] += qMij
        dofid = dof_parentid[dofid]


@event_scope
def crb(m: Model, d: Data):
    """Computes composite rigid body inertias for each body and the joint-space inertia matrix.

    Accumulates composite rigid body inertias up the kinematic tree and computes the
    joint-space inertia matrix
    """
    wp.copy(d.crb, d.body_inert)

    for i in reversed(range(len(m.body_tree))):
        body_tree = m.body_tree[i]
        wp.launch(_crb_accumulate,
                  dim=(d.nworld, body_tree.size),
                  inputs=[m.body_parentid, d.integration_done, d.crb, body_tree],
                  outputs=[d.crb])

    d.qM.zero_()
    wp.launch(
        _qM_dense,
        dim=(d.nworld, m.nv),
        inputs=[m.dof_bodyid, m.dof_parentid, m.dof_armature, d.integration_done, d.cdof, d.crb],
        outputs=[d.qM]
    )


@cache_kernel
def _tile_cholesky_factorize(tile: TileSet):
    """Returns a kernel for dense Cholesky factorization of a tile."""

    @nested_kernel(module="unique", enable_backward=False)
    def cholesky_factorize(
            # Data In:
            integration_done_in: wp.array(dtype=bool),
            qM_in: wp.array3d(dtype=float),
            # In:
            adr: wp.array(dtype=int),
            # Out:
            L_out: wp.array3d(dtype=float),
    ):
        worldid, nodeid = wp.tid()
        if integration_done_in[worldid]:
            return
        TILE_SIZE = wp.static(tile.size)

        dofid = adr[nodeid]
        M_tile = wp.tile_load(qM_in[worldid], shape=(TILE_SIZE, TILE_SIZE),
                              offset=(dofid, dofid))
        L_tile = wp.tile_cholesky(M_tile)
        wp.tile_store(L_out[worldid], L_tile, offset=(dofid, dofid))

    return cholesky_factorize


def _factor_i_dense(m: Model, d: Data, M: wp.array, L: wp.array):
    """Dense Cholesky factorization of inertia-like matrix M, assumed spd."""
    for tile in m.qM_tiles:
        wp.launch_tiled(
            _tile_cholesky_factorize(tile),
            dim=(d.nworld, tile.adr.size),
            inputs=[d.integration_done, M, tile.adr],
            outputs=[L],
            block_dim=m.block_dim.cholesky_factorize,
        )


@event_scope
def factor_m(m: Model, d: Data):
    """Factorization of inertia-like matrix M, assumed spd."""
    _factor_i_dense(m, d, d.qM, d.qLD)


@cache_kernel
def _tile_cholesky_factorize_solve(tile: TileSet):
    """Returns a kernel for dense Cholesky factorization and backsubstitution of a tile."""

    @nested_kernel(module="unique", enable_backward=False)
    def cholesky_factorize_solve(
            # In:
            integration_done_in: wp.array(dtype=bool),
            M: wp.array3d(dtype=float),
            y: wp.array2d(dtype=float),
            adr: wp.array(dtype=int),
            # Out:
            x: wp.array2d(dtype=float),
            L: wp.array3d(dtype=float),
    ):
        worldid, nodeid = wp.tid()
        if integration_done_in[worldid]:
            return
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
            inputs=[d.integration_done, M, y, tile.adr],
            outputs=[x, L],
            block_dim=m.block_dim.cholesky_factorize_solve,
        )


@event_scope
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
