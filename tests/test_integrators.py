"""
Integration tests for bolt.step with each integrator.

Positions and velocities are checked loosely against the RK-Merson adaptive integrator
Muscle activation with constant excitation is an independent scalar ODE per muscle, checked against scipy
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import bolt
from bolt.types_consts import ActivationType
from models import MODEL_NAMES, STANDING_PELVIS_HEIGHT, model_path
from tolerances import INTEGRATOR_ACTIVATION, INTEGRATOR_QPOS, INTEGRATOR_QVEL

I = bolt.IntegratorType

N_WORLDS = 4
DT = 1e-3
N_STEPS = 5
INITIAL_SPEED = 0.1  # max |initial generalized velocity|
REFERENCE = I.RK_MERSON_ADAPTIVE
STATE_FIELDS = ["qpos", "qvel", "m_act", "m_state", "a_act", "stl_contact_state"]


def _simulate(path: str, integrator: bolt.IntegratorType, height: float) -> dict:
    lr = bolt.load_model(model_path=path, n_worlds=N_WORLDS, integrator=integrator, requires_visuals=False)
    m, d = lr.model, lr.data

    qpos = d.qpos.numpy()
    qpos[:, lr.qpos_id_lookup["pelvis_ty"]] = height
    d.qpos.assign(qpos)
    rng = np.random.default_rng(8)
    d.qvel.assign(rng.uniform(-INITIAL_SPEED, INITIAL_SPEED, size=(N_WORLDS, m.nv)).astype(np.float32))
    d.world_reset.fill_(True)
    bolt.reset(m, d)

    out = {"m_act_initial": d.m_act.numpy().copy(), "max_grf": 0.0}
    for _ in range(N_STEPS):
        bolt.increment_next_time(m, d, DT)
        bolt.step(m, d)
        out["max_grf"] = max(out["max_grf"], float(np.abs(d.grf.numpy()).max()))
    out.update({f: getattr(d, f).numpy().copy() for f in STATE_FIELDS})
    out["time"] = d.time.numpy().copy()
    out["m_excitations"] = d.m_excitations.numpy().copy()
    out["muscle_metadata"] = m.muscle_metadata.numpy().copy()
    out["activation_type"] = m.opt.activation_type
    return out


def _activation_derivative(a, e, mm, activation_type):
    """ Mirrors bolt/_src/muscle/activation.py in float64 """
    ta, td = mm["activation_time_const"], mm["deactivation_time_const"]
    if activation_type == ActivationType.MILLARD:
        tau = np.where(e > a, ta * (0.5 + 1.5 * a), td / (0.5 + 1.5 * a))
        return (e - a) / tau
    factor = 0.5 + 1.5 * a
    f = 0.5 * np.tanh(mm["activation_dynamics_smoothing"] * (e - a))
    return (1.0 / (ta * factor) * (f + 0.5) + factor / td * (0.5 - f)) * (e - a)


def _exact_activation(run: dict) -> np.ndarray:
    """ Float64 solution of each muscle's activation ODE from the post-reset activation """
    a0, e, mm = run["m_act_initial"].astype(float), run["m_excitations"].astype(float), run["muscle_metadata"]
    out = np.empty_like(a0)
    for world in range(a0.shape[0]):
        sol = solve_ivp(lambda t, a: _activation_derivative(a, e[world], mm, run["activation_type"]),
                        (0.0, DT * N_STEPS), a0[world], method="DOP853", rtol=1e-10, atol=1e-12)
        out[world] = np.clip(sol.y[:, -1], mm["min_activation"], mm["max_activation"])
    return out


@pytest.fixture(scope="module", params=MODEL_NAMES)
def runs(request):
    path, height = model_path(request.param), STANDING_PELVIS_HEIGHT[request.param]
    return request.param, {integrator: _simulate(path, integrator, height) for integrator in I}


@pytest.mark.parametrize("integrator", list(I), ids=lambda i: i.name)
def test_state_is_valid(runs, integrator):
    _, results = runs
    run = results[integrator]
    for field in STATE_FIELDS:
        assert np.isfinite(run[field]).all(), f"{field} is not finite"
    np.testing.assert_allclose(run["time"], DT * N_STEPS, rtol=1e-6, err_msg="worlds did not reach the target time")
    assert run["max_grf"] > 0.0, "contacts were never engaged"


def test_reset_gives_valid_activation(runs):
    """ Reset must leave activations inside [min_activation, max_activation] (the loader starts them at 0) """
    _, results = runs
    run = results[REFERENCE]
    mm, a0 = run["muscle_metadata"], run["m_act_initial"]
    assert ((a0 >= mm["min_activation"]) & (a0 <= mm["max_activation"])).all()


@pytest.mark.parametrize("integrator", list(I), ids=lambda i: i.name)
def test_activation_matches_exact_solution(runs, integrator):
    _, results = runs
    run = results[integrator]
    if run["m_act"].size == 0:
        pytest.skip("no muscles")
    err = np.abs(run["m_act"] - _exact_activation(run)).max()
    assert err < INTEGRATOR_ACTIVATION[integrator.name], f"max activation error {err:.3e}"


@pytest.mark.parametrize("integrator", [i for i in I if i != REFERENCE], ids=lambda i: i.name)
def test_mechanics_match_reference(runs, integrator):
    _, results = runs
    run, ref = results[integrator], results[REFERENCE]
    for field, tol in [("qpos", INTEGRATOR_QPOS[integrator.name]), ("qvel", INTEGRATOR_QVEL[integrator.name])]:
        err = np.abs(run[field] - ref[field]).max()
        assert err < tol, f"max {field} difference from {REFERENCE.name}: {err:.3e}"
