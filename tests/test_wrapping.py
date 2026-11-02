""" Validate Bolt's wrapping (Scholz2015GeometryPath) muscle paths against OpenSim

The wrapping solver starts from the previous solution, so a path depends on the states that led to it: two states far
apart can converge to different locally shortest paths (e.g. around opposite sides of an obstacle). Both Bolt and
OpenSim therefore approach each random state gradually from the default pose, as a simulation would.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import bolt
import opensim_oracle
from bolt._src.dynamics import operators
from bolt._src.pipeline import forward
from bolt._src.muscle import wrap_path
from bolt.loader.converters import wrap_helper
from bolt.types import WrapStatus
from bolt.types import WrapType
from conftest import N_STATES
from models import model_path
from tolerances import MUSCLE_LENGTH_M, MUSCLE_MOMENT_ARM_M, MUSCLE_SPEED_MS

N_CONTINUATION_STEPS = 50
N_MOMENT_ARM_STATES = 4
WRAPPING_MODELS = ["example_model", "example_model2"]


@pytest.fixture(scope="session")
def wrapping_test_model(tmp_path_factory) -> str:
    """ A two-link model wrapping over every obstacle type, two obstacles in a row, and past a via point """
    return opensim_oracle.wrapping_test_model(str(tmp_path_factory.mktemp("wrapping") / "wrapping_test.osim"))


def _reset_to_default_pose(case):
    """ Resets every Bolt world at the default pose, which restarts the wrapping from the contact hints """
    q0 = opensim_oracle.default_coordinates(case.osim_model, case.osim_state)
    case.set_states(np.tile(q0, (N_STATES, 1)), np.zeros((N_STATES, len(q0))))
    case.d.world_reset.fill_(True)
    bolt.reset(case.m, case.d)


def _continue_to(case, q, u):
    """ Moves every Bolt world from the default pose to its state (q, u) in N_CONTINUATION_STEPS """
    q0 = opensim_oracle.default_coordinates(case.osim_model, case.osim_state)
    for k in range(1, N_CONTINUATION_STEPS + 1):
        case.set_states(q0 + (q - q0) * k / N_CONTINUATION_STEPS, u)
        forward.realize_position(case.m, case.d)
        forward.realize_velocity(case.m, case.d)
        wrap_path.muscle_wrap_path(case.m, case.d)


def _compare(load_case, path, seed):
    case = load_case(path)
    lr = case.load_result
    muscles = sorted(opensim_oracle.scholz_wrapped_muscles(path))
    assert muscles and {lr.muscle_id_lookup[mu] for mu in muscles} == set(case.m.muscle_wrap_group_tuple)

    q, u = opensim_oracle.random_states(case.osim_model, N_STATES, seed=seed)
    _reset_to_default_pose(case)
    _continue_to(case, q, u)
    L, V = case.d.muscle_length.numpy(), case.d.muscle_velocity.numpy()
    bolt.compute_muscle_moments(case.m, case.d)
    MA = case.d.muscle_moment_arm.numpy()
    status = case.d.wrap_geodesic.numpy()["status"]

    root = opensim_oracle.root_free_coordinates(case.osim_model, lr)
    coords = [c for c in opensim_oracle.coordinate_names(case.osim_model) if c not in root]
    err = {"length": (0.0, None), "speed": (0.0, None), "moment arm": (0.0, None)}

    def record(kind, e, where):
        if e > err[kind][0]:
            err[kind] = (e, where)

    for w in range(N_STATES):
        ma_coords = {mu: coords for mu in muscles} if w < N_MOMENT_ARM_STATES else None
        out = opensim_oracle.continuation_muscle_paths(path, q[w], u[w], N_CONTINUATION_STEPS, ma_coords)
        ref, ref_ma = out if ma_coords else (out, {})
        for mu in muscles:
            i = lr.muscle_id_lookup[mu]
            record("length", abs(L[w, i] - ref[mu][0]), (w, mu))
            record("speed", abs(V[w, i] - ref[mu][1]), (w, mu))
            for coord, r in ref_ma.get(mu, {}).items():
                record("moment arm", abs(MA[w, i, lr.qpos_id_lookup[coord]] - r), (w, mu, coord))
    return err, status


@pytest.mark.parametrize("name", WRAPPING_MODELS)
def test_wrapping_paths(load_case, name):
    err, status = _compare(load_case, model_path(name), seed=2)
    assert (status == WrapStatus.IN_CONTACT).any() and (status == WrapStatus.LIFTED).any()
    assert err["length"][0] < MUSCLE_LENGTH_M, f"max length error {err['length']}"
    assert err["speed"][0] < MUSCLE_SPEED_MS, f"max lengthening-speed error {err['speed']}"
    assert err["moment arm"][0] < MUSCLE_MOMENT_ARM_M, f"max moment-arm error {err['moment arm']}"


def test_wrapping_obstacle_types(load_case, wrapping_test_model):
    """ Sphere, cylinder and torus obstacles, two obstacles between path points, and a via point """
    err, status = _compare(load_case, wrapping_test_model, seed=3)
    # Every obstacle wraps in some of the states
    assert (status == WrapStatus.IN_CONTACT).any(axis=0).all()
    assert err["length"][0] < MUSCLE_LENGTH_M, f"max length error {err['length']}"
    assert err["speed"][0] < MUSCLE_SPEED_MS, f"max lengthening-speed error {err['speed']}"
    assert err["moment arm"][0] < MUSCLE_MOMENT_ARM_M, f"max moment-arm error {err['moment arm']}"


def test_wrapping_forces_match_moment_arms(load_case, wrapping_test_model):
    """ The muscle body forces of the forward pipeline map to the moment arms times the tension """
    case = load_case(wrapping_test_model)
    q, u = opensim_oracle.random_states(case.osim_model, N_STATES, seed=4)
    _reset_to_default_pose(case)
    _continue_to(case, q, u)
    forward.fwd(case.m, case.d)
    ufrc = wp.zeros((case.d.nworld, case.m.nv), dtype=float)
    operators.multiply_by_jacobian_transpose(case.m, case.d, X_in=case.d.body_F_muscle, JtX_out=ufrc)
    actual = ufrc.numpy()  # pin joints: qpos and qvel spaces coincide
    bolt.compute_muscle_moments(case.m, case.d)
    expected = np.einsum("wmq,wm->wq", case.d.muscle_moment_arm.numpy(), case.d.muscle_actuation.numpy())
    scale = np.abs(expected).max()
    assert scale > 1.0
    # Both sum body moments of hundreds of N m in float32, so coordinates the muscles don't cross (where these cancel
    # to zero) are left with rounding errors that depend on the platform: compare relative to the largest moment
    np.testing.assert_allclose(actual, expected, atol=1e-5 * scale, rtol=1e-4)


def test_wrapping_reset_restarts_from_hints(load_case, wrapping_test_model):
    """ A reset world forgets its previous path, so it gives the same path as a freshly loaded model """
    case = load_case(wrapping_test_model)
    _reset_to_default_pose(case)
    fresh = case.d.muscle_length.numpy().copy()
    q, u = opensim_oracle.random_states(case.osim_model, N_STATES, seed=5)
    _continue_to(case, q, u)
    _reset_to_default_pose(case)
    np.testing.assert_allclose(case.d.muscle_length.numpy(), fresh, atol=1e-6)


def test_wrapping_render_polylines(load_case, wrapping_test_model):
    """ The polylines the renderers draw follow the wrapping paths: same length, curves on the obstacle surfaces """
    case = load_case(wrapping_test_model)
    m, d = case.m, case.d
    q, u = opensim_oracle.random_states(case.osim_model, N_STATES, seed=6)
    _reset_to_default_pose(case)
    _continue_to(case, q, u)
    adr, num = wrap_path.wrap_render_layout(m)
    points = wp.empty((N_STATES, sum(num)), dtype=wp.vec3)
    wrap_path.sample_wrap_paths(m, d, wp.array(adr, dtype=int), wp.array(np.arange(N_STATES), dtype=int), points)
    points = points.numpy()

    L = d.muscle_length.numpy()
    geodesics = d.wrap_geodesic.numpy()
    X_GB = d.mob_X_GB.numpy()
    wrap_adr, wrap_num, wrap_segment = m.muscle_wrap_adr.numpy(), m.muscle_wrap_num.numpy(), m.wrap_segment.numpy()
    samples = wrap_path.WRAP_RENDER_SAMPLES
    for k, muscle in enumerate(m.muscle_wrap_group_tuple):
        polyline = points[:, adr[k]:adr[k] + num[k]]
        # Chords cut the curves short by at most ~ length * (angle per sample)^2 / 24
        chords = np.linalg.norm(np.diff(polyline, axis=1), axis=2).sum(axis=1)
        np.testing.assert_allclose(chords, L[:, muscle], rtol=2e-3, atol=1e-6)

        # Curve samples of obstacles in contact lie on their surface
        for j, o in enumerate(range(wrap_adr[muscle], wrap_adr[muscle] + wrap_num[muscle])):
            start = wrap_segment[o] + 1 + samples * j
            for w in range(N_STATES):
                if geodesics["status"][w, o] != WrapStatus.IN_CONTACT:
                    continue
                X_GB_o = X_GB[w, m.wrap_bodyid.numpy()[o]]
                X_BS = m.wrap_X_BS.numpy()[o]
                X_GS = wp.transform_multiply(wp.transform(X_GB_o[:3], X_GB_o[3:]), wp.transform(X_BS[:3], X_BS[3:]))
                X_SG = wp.transform_inverse(X_GS)
                curve_S = np.array([wp.transform_point(X_SG, wp.vec3(p)) for p in polyline[w, start:start + samples]])
                ends_S = np.array([geodesics["p_P"][w, o], geodesics["p_Q"][w, o]])
                np.testing.assert_allclose(curve_S[[0, -1]], ends_S, atol=1e-6)
                value = _surface_value(m.wrap_type.numpy()[o], m.wrap_size.numpy()[o], curve_S)
                assert np.abs(value).max() < 1e-4, (muscle, o, w, value)


def _surface_value(wrap_type, size, x):
    """ Implicit surface functions of the obstacles, zero on the surface and made dimensionless """
    size = np.asarray(size, dtype=float)
    value = np.array([wrap_helper._surface_value(WrapType(wrap_type), size, np.asarray(p, dtype=float)) for p in x])
    if wrap_type in (WrapType.SPHERE, WrapType.CYLINDER):
        value /= size[0] ** 2
    return value
