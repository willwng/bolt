""" Validate Bolt forward dynamics (gravity + inertia only) against OpenSim """

from __future__ import annotations

import numpy as np
import pytest

import opensim_oracle
from bolt._src.pipeline import forward
from conftest import N_STATES
from models import MODEL_NAMES, model_path
from tolerances import ACCELERATION_ATOL, ACCELERATION_RTOL

# Near gimbal lock of 3-axis joints (e.g. hip adduction ~ pi/2) the float32 articulated-body solve is
# ill-conditioned, so keep rotational coordinates away from +/- pi/2
MAX_ROTATION = 1.2


@pytest.fixture(scope="module", params=MODEL_NAMES)
def case(request, load_case, tmp_path_factory):
    """ Skeleton-only copy of the model: no muscles, contacts, springs, or limits """
    path = opensim_oracle.skeleton_only_model(model_path(request.param), str(tmp_path_factory.mktemp("skel")))
    return load_case(path)


@pytest.fixture(scope="module", params=[0.0, 1.0, 3.0], ids=["static", "moving", "fast"])
def dynamics(request, case):
    """ Bolt and OpenSim accelerations over a batch of random states """
    q, u = opensim_oracle.random_states(case.osim_model, N_STATES, seed=1, speed_scale=request.param,
                                        max_rotation=MAX_ROTATION)
    case.set_states(q, u)
    forward.fwd(case.m, case.d)

    refs = [opensim_oracle.forward_dynamics(case.osim_model, case.osim_state, qi, ui) for qi, ui in zip(q, u)]
    return case, refs, case.d.body_A_GB.numpy(), case.d.qacc.numpy()


def _assert_close(mine: np.ndarray, ref: np.ndarray, labels: list[str], world: int):
    """
    float32 error accumulates relative to the largest accelerations in the system, so small
    entries are compared against the scale of the whole world's acceleration vector.
    """
    tol = ACCELERATION_ATOL + ACCELERATION_RTOL * np.max(np.abs(ref))
    err = np.abs(mine - ref).max(axis=tuple(range(1, mine.ndim)))
    worst = int(np.argmax(err))
    assert err[worst] < tol, (f"world {world}, {labels[worst]}: error {err[worst]:.3e} > tol {tol:.3e} "
                              f"(bolt {mine[worst]}, opensim {ref[worst]})")


def test_body_accelerations(dynamics):
    case, refs, A, _ = dynamics
    for world, (body_acc, _) in enumerate(refs):
        bodies = list(body_acc)
        ids = [case.load_result.body_id_lookup[b] for b in bodies]
        _assert_close(A[world, ids], np.array([body_acc[b] for b in bodies]), bodies, world)


def test_coordinate_accelerations(dynamics):
    """ qacc for non-root coordinates, where Bolt's u matches OpenSim's coordinate speeds """
    case, refs, _, qacc = dynamics
    lr = case.load_result
    root_coords = opensim_oracle.root_free_coordinates(case.osim_model, lr)  # covered by the root body acceleration
    for world, (_, udot) in enumerate(refs):
        coords = [c for c in udot if c not in root_coords]
        ids = [lr.dof_id_lookup[c] for c in coords]
        _assert_close(qacc[world, ids], np.array([udot[c] for c in coords]), coords, world)


def test_model_gravity_is_respected(load_case, tmp_path_factory):
    path = opensim_oracle.skeleton_only_model(model_path(MODEL_NAMES[0]), str(tmp_path_factory.mktemp("mars")),
                                              gravity=(0.0, -3.71, 0.0))
    case = load_case(path)
    q, u = opensim_oracle.random_states(case.osim_model, N_STATES, seed=4, speed_scale=0.0,
                                        max_rotation=MAX_ROTATION)
    case.set_states(q, u)
    forward.fwd(case.m, case.d)
    A = case.d.body_A_GB.numpy()
    for world, (qi, ui) in enumerate(zip(q, u)):
        body_acc, _ = opensim_oracle.forward_dynamics(case.osim_model, case.osim_state, qi, ui)
        bodies = list(body_acc)
        ids = [case.load_result.body_id_lookup[b] for b in bodies]
        _assert_close(A[world, ids], np.array([body_acc[b] for b in bodies]), bodies, world)
