""" Validate coordinate coupler constraints: accelerations against OpenSim, and the projection while stepping """

from __future__ import annotations

import numpy as np
import opensim as osim
import pytest

import bolt
import opensim_oracle
from bolt._src.pipeline import forward
from conftest import N_STATES
from models import STANDING_PELVIS_HEIGHT, model_path
from tolerances import ACCELERATION_ATOL, ACCELERATION_RTOL

MODEL = "example_model2"
MAX_ROTATION = 1.2  # see test_dynamics


def _couplers():
    """ (independent, dependent, function, scale): a scaled linear coupler, and a spline across three joints """
    spline = osim.SimmSpline()
    for x, y in [(-2.2, 0.3), (-1.4, -0.2), (-0.6, 0.1), (0.0, 0.25), (0.4, 0.4)]:
        spline.addPoint(x, y)
    return [
        ("ankle_angle_r", "subtalar_angle_r", osim.LinearFunction(0.6, 0.1), 0.5),
        ("knee_angle_l", "mtp_angle_l", spline, 1.0),
    ]


def _residuals(lr, qpos, qvel):
    """ Position and velocity constraint errors (nworld, ncoupler), evaluated in float64 from the OpenSim functions """
    pos, vel = [], []
    for independent, dependent, function, scale in _couplers():
        qi = qpos[:, lr.qpos_id_lookup[independent]].astype(float)
        ui = qvel[:, lr.dof_id_lookup[independent]].astype(float)
        f = np.array([function.calcValue(osim.Vector(1, x)) for x in qi])
        h = 1e-6
        df = np.array([(function.calcValue(osim.Vector(1, x + h)) - function.calcValue(osim.Vector(1, x - h))) / (2 * h)
                       for x in qi])
        pos.append(qpos[:, lr.qpos_id_lookup[dependent]] - scale * f)
        vel.append(qvel[:, lr.dof_id_lookup[dependent]] - scale * df * ui)
    return np.array(pos).T, np.array(vel).T


@pytest.fixture(scope="module")
def case(load_case, tmp_path_factory):
    path = opensim_oracle.coupled_model(model_path(MODEL), str(tmp_path_factory.mktemp("coupled")), _couplers(),
                                        skeleton_only=True)
    return load_case(path)


def test_couplers_loaded(case):
    assert case.m.ncoupler == len(_couplers())


@pytest.mark.parametrize("speed_scale", [0.0, 1.0, 3.0], ids=["static", "moving", "fast"])
def test_constrained_accelerations(case, speed_scale):
    """ Off the constraint manifold too: OpenSim enforces only the acceleration constraint during realizeAcceleration """
    q, u = opensim_oracle.random_states(case.osim_model, N_STATES, seed=3, speed_scale=speed_scale,
                                        max_rotation=MAX_ROTATION)
    case.set_states(q, u)
    forward.fwd(case.m, case.d)
    qacc, A = case.d.qacc.numpy(), case.d.body_A_GB.numpy()

    lr = case.load_result
    root_coords = opensim_oracle.root_free_coordinates(case.osim_model, lr)
    for world, (qi, ui) in enumerate(zip(q, u)):
        body_acc, udot = opensim_oracle.forward_dynamics(case.osim_model, case.osim_state, qi, ui)
        ref = np.array([udot[c] for c in udot if c not in root_coords])
        mine = qacc[world, [lr.dof_id_lookup[c] for c in udot if c not in root_coords]]
        tol = ACCELERATION_ATOL + ACCELERATION_RTOL * np.abs(ref).max()
        assert np.abs(mine - ref).max() < tol, f"world {world}: qacc error {np.abs(mine - ref).max():.3e}"

        ref = np.array(list(body_acc.values()))
        mine = A[world, [lr.body_id_lookup[b] for b in body_acc]]
        tol = ACCELERATION_ATOL + ACCELERATION_RTOL * np.abs(ref).max()
        assert np.abs(mine - ref).max() < tol, f"world {world}: body acceleration error {np.abs(mine - ref).max():.3e}"


@pytest.mark.parametrize("integrator", list(bolt.IntegratorType), ids=lambda i: i.name)
def test_projection_keeps_constraints(integrator, tmp_path_factory):
    """ Full model (muscles, contacts): reset assembles the constraints, and stepping keeps them satisfied """
    path = opensim_oracle.coupled_model(model_path(MODEL), str(tmp_path_factory.mktemp("coupled_full")), _couplers())
    nworld = 4
    lr = bolt.load_model(model_path=path, n_worlds=nworld, integrator=integrator, requires_visuals=False)
    m, d = lr.model, lr.data

    qpos = d.qpos.numpy()
    qpos[:, lr.qpos_id_lookup["pelvis_ty"]] = STANDING_PELVIS_HEIGHT[MODEL]
    qpos[:, lr.qpos_id_lookup["knee_angle_l"]] = -0.8
    d.qpos.assign(qpos)
    rng = np.random.default_rng(2)
    d.qvel.assign(rng.uniform(-1.0, 1.0, size=(nworld, m.nv)).astype(np.float32))
    d.world_reset.fill_(True)
    bolt.reset(m, d)

    pos, vel = _residuals(lr, d.qpos.numpy(), d.qvel.numpy())
    assert np.abs(pos).max() < 1e-6 and np.abs(vel).max() < 1e-5

    fixed = integrator in (bolt.IntegratorType.EULER_FIXED, bolt.IntegratorType.RK4_FIXED)
    dt = 2e-4 if fixed else 1e-3
    for _ in range(10):
        bolt.increment_next_time(m, d, dt)
        bolt.step(m, d)
    assert np.allclose(d.time.numpy(), 10 * dt)
    pos, vel = _residuals(lr, d.qpos.numpy(), d.qvel.numpy())
    assert np.abs(pos).max() < 1e-6 and np.abs(vel).max() < 1e-5
    # The independent coordinates moved, so the projection is doing something
    knee = d.qpos.numpy()[:, lr.qpos_id_lookup["knee_angle_l"]]
    assert np.abs(knee + 0.8).max() > 1e-4
