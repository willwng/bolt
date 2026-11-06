"""
The schedules of the tree passes (codegen.TreePass) must agree: the fused kernels (one block per world, specialized to
the model's tree) and the one-thread-per-world kernels against the reference one launch per tree level. All three
run the same per-body functions in an order the tree fixes, so on CPU the results are bitwise the same. On CUDA the
compiler contracts the same expressions into FMAs differently in each kernel, so the accelerations differ by a few
float32 ulps (as the one-thread-per-world kernels always did): there they are compared to RTOL_CUDA.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import bolt
from bolt._src import codegen
from bolt._src.codegen import tree as codegen_tree
from bolt._src.pipeline import forward
from models import MODEL_NAMES, STANDING_PELVIS_HEIGHT, model_path

N_WORLDS = 8
POSE_NOISE = 0.2  # rad, on every coordinate but the pelvis's
MAX_SPEED = 1.0
RTOL_CUDA = 1e-6  # relative to the largest magnitude of each field; measured up to ~1e-7

FIELDS = [
    # articulated inertias
    "body_P", "body_PPlus", "mob_G", "mob_DI", "body_articulated_centrifugal_force",
    # accelerations
    "body_eps", "body_zPlus", "body_A_GB", "qacc",
]


@pytest.fixture
def force_tree_pass():
    """ Sets codegen_tree.FORCE_TREE_PASS, restoring it afterwards """
    yield lambda schedule: setattr(codegen_tree, "FORCE_TREE_PASS", schedule)
    codegen_tree.FORCE_TREE_PASS = None


def _setup(name: str):
    lr = bolt.load_model(model_path=model_path(name), n_worlds=N_WORLDS, integrator=bolt.IntegratorType.EULER_ADAPTIVE,
                         requires_visuals=False)
    m, d = lr.model, lr.data
    rng = np.random.default_rng(5)
    qpos = d.qpos.numpy()
    qpos[:, lr.qpos_id_lookup["pelvis_ty"]] = STANDING_PELVIS_HEIGHT[name]
    ids = [i for n, i in lr.qpos_id_lookup.items() if not n.startswith("pelvis")]
    qpos[:, ids] += POSE_NOISE * rng.standard_normal((N_WORLDS, len(ids)))
    d.qpos.assign(qpos.astype(np.float32))
    d.world_reset.fill_(True)
    bolt.reset(m, d)
    d.qvel.assign(rng.uniform(-MAX_SPEED, MAX_SPEED, size=(N_WORLDS, m.nv)).astype(np.float32))
    if m.nmuscle:
        d.m_excitations.assign(rng.uniform(0.0, 1.0, size=(N_WORLDS, m.nmuscle)).astype(np.float32))
    return m, d


def _outputs(m, d) -> dict:
    forward.fwd(m, d)
    return {f: getattr(d, f).numpy().copy() for f in FIELDS}  # on CPU numpy() is a view


def _differs(got: np.ndarray, ref: np.ndarray) -> bool:
    if wp.get_device().is_cpu:
        return got.tobytes() != ref.tobytes()
    got, ref = got.view(np.float32).astype(np.float64), ref.view(np.float32).astype(np.float64)  # struct fields too
    return np.abs(got - ref).max() > RTOL_CUDA * max(np.abs(ref).max(), 1.0)


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("schedule", [codegen.TreePass.FUSED, codegen.TreePass.PER_WORLD], ids=lambda s: s.name)
def test_tree_pass_matches_per_level(name, schedule, force_tree_pass):
    m, d = _setup(name)
    force_tree_pass(codegen.TreePass.PER_LEVEL)
    ref = _outputs(m, d)
    assert np.isfinite(d.qacc.numpy()).all(), "reference accelerations are not finite"

    force_tree_pass(schedule)
    got = _outputs(m, d)
    differ = [f for f in FIELDS if _differs(got[f], ref[f])]
    assert not differ, f"{schedule.name} differs from PER_LEVEL in {differ}"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_fused_skips_finished_worlds(name, force_tree_pass):
    """ Worlds done integrating keep their outputs (the whole block returns, before any barrier) """
    m, d = _setup(name)
    force_tree_pass(codegen.TreePass.FUSED)
    forward.fwd(m, d)
    before = d.qacc.numpy().copy()

    done = np.zeros(N_WORLDS, dtype=bool)
    done[::2] = True
    d.integration_done.assign(done)
    d.ufrc_total.assign(d.ufrc_total.numpy() + 1.0)  # new generalized forces in every world
    forward.compute_accelerations(m, d)
    after = d.qacc.numpy()
    d.integration_done.zero_()

    assert np.array_equal(after[done], before[done]), "a finished world's accelerations changed"
    assert not np.isclose(after[~done], before[~done]).all(axis=1).any(), "an unfinished world was not updated"
