""" Validate Bolt position/velocity kinematics against OpenSim """

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import opensim_oracle
from bolt._src import forward
from conftest import N_STATES
from models import MODEL_NAMES, model_path
from tolerances import POSITION_M, ROTATION, VELOCITY


@pytest.fixture(scope="module", params=MODEL_NAMES)
def case(request, load_case):
    return load_case(model_path(request.param))


@pytest.fixture(scope="module")
def kinematics(case):
    """ Bolt and OpenSim body kinematics over a batch of random states """
    q, u = opensim_oracle.random_states(case.osim_model, N_STATES, seed=0)
    case.set_states(q, u)
    forward.realize_position(case.m, case.d)
    forward.realize_velocity(case.m, case.d)

    refs = [opensim_oracle.body_kinematics(case.osim_model, case.osim_state, qi, ui) for qi, ui in zip(q, u)]
    X = case.d.mob_X_GB.numpy()
    V = case.d.body_V_GB.numpy()
    return case, refs, X, V


def _max_err(case, refs, fn):
    err, worst = 0.0, None
    for world, ref in enumerate(refs):
        for body, r in ref.items():
            bid = case.load_result.body_id_lookup[body]
            e = float(np.max(np.abs(fn(world, bid) - r)))
            if e > err:
                err, worst = e, (world, body)
    return err, worst


def test_body_positions(kinematics):
    case, refs, X, _ = kinematics
    err, worst = _max_err(case, [{b: v[0] for b, v in r.items()} for r in refs],
                          lambda w, b: X[w, b, :3])
    assert err < POSITION_M, f"max body-origin error {err:.3e} m at {worst}"


def test_body_orientations(kinematics):
    case, refs, X, _ = kinematics
    err, worst = _max_err(case, [{b: v[1] for b, v in r.items()} for r in refs],
                          lambda w, b: Rotation.from_quat(X[w, b, 3:]).as_matrix())
    assert err < ROTATION, f"max rotation-matrix error {err:.3e} at {worst}"


def test_body_velocities(kinematics):
    case, refs, _, V = kinematics
    err, worst = _max_err(case, [{b: v[2] for b, v in r.items()} for r in refs],
                          lambda w, b: V[w, b])
    assert err < VELOCITY, f"max spatial-velocity error {err:.3e} at {worst}"
