""" Validate Bolt muscle path lengths, lengthening speeds, and moment arms against OpenSim """

from __future__ import annotations

import pytest

import opensim_oracle
from bolt._src import forward, smooth_muscle_fn, smooth_muscle_pt
from conftest import N_STATES
from models import FN_PATH_FILE, FN_PATH_MODEL, MODEL_NAMES, model_path
from tolerances import MUSCLE_LENGTH_M, MUSCLE_MOMENT_ARM_M, MUSCLE_SPEED_MS


def _realize_paths(case, q, u):
    case.set_states(q, u)
    forward.realize_position(case.m, case.d)
    forward.realize_velocity(case.m, case.d)
    smooth_muscle_pt.muscle_point_path(case.m, case.d)
    smooth_muscle_fn.muscle_fn_path(case.m, case.d)
    return case.d.muscle_length.numpy(), case.d.muscle_velocity.numpy()


def _max_err(case, names, refs, mine, index):
    err, worst = 0.0, None
    for world, ref in enumerate(refs):
        for name in names:
            e = abs(float(mine[world, case.load_result.muscle_id_lookup[name]]) - ref[name][index])
            if e > err:
                err, worst = e, (world, name)
    return err, worst


# --- Point (geometry) paths ---
@pytest.fixture(scope="module", params=MODEL_NAMES)
def point_paths(request, load_case):
    case = load_case(model_path(request.param))
    q, u = opensim_oracle.random_states(case.osim_model, N_STATES, seed=2)
    L, V = _realize_paths(case, q, u)
    refs = [opensim_oracle.muscle_paths(case.osim_model, case.osim_state, qi, ui) for qi, ui in zip(q, u)]
    wrapped = opensim_oracle.wrapped_muscles(case.path)
    return case, refs, L, V, wrapped


def _unwrapped(names, wrapped):
    names = [n for n in names if n not in wrapped]
    if not names:
        pytest.skip("no unwrapped muscles")
    return names


def test_point_path_lengths(point_paths):
    case, refs, L, _, wrapped = point_paths
    err, worst = _max_err(case, _unwrapped(refs[0], wrapped), refs, L, 0)
    assert err < MUSCLE_LENGTH_M, f"max length error {err:.3e} m at {worst}"


def test_point_path_speeds(point_paths):
    case, refs, _, V, wrapped = point_paths
    err, worst = _max_err(case, _unwrapped(refs[0], wrapped), refs, V, 1)
    assert err < MUSCLE_SPEED_MS, f"max lengthening-speed error {err:.3e} m/s at {worst}"


@pytest.mark.xfail(strict=True, reason="Bolt point paths do not model wrap surfaces/obstacles")
def test_wrapped_point_path_lengths(point_paths):
    case, refs, L, _, wrapped = point_paths
    if not wrapped:
        pytest.skip("no wrapped muscles")
    err, worst = _max_err(case, sorted(wrapped), refs, L, 0)
    assert err < MUSCLE_LENGTH_M, f"max length error {err:.3e} m at {worst}"


# --- Function-based (polynomial) paths ---
@pytest.fixture(scope="module")
def fn_paths(load_case):
    case = load_case(model_path(FN_PATH_MODEL), muscle_fn_path=FN_PATH_FILE)
    osim_fn_model = opensim_oracle.function_based_path_model(model_path(FN_PATH_MODEL), FN_PATH_FILE)
    osim_fn_state = osim_fn_model.initSystem()
    coords = opensim_oracle.function_based_path_coordinates(osim_fn_model)
    assert coords, "no function-based paths were loaded"

    q, u = opensim_oracle.random_states(osim_fn_model, N_STATES, seed=3)
    L, V = _realize_paths(case, q, u)
    refs, moment_arms = zip(*[
        opensim_oracle.muscle_paths(osim_fn_model, osim_fn_state, qi, ui, moment_arm_coords=coords)
        for qi, ui in zip(q, u)
    ])
    return case, coords, refs, moment_arms, L, V, case.d.muscle_moment_arm.numpy()


def test_fn_path_lengths(fn_paths):
    case, coords, refs, _, L, _, _ = fn_paths
    err, worst = _max_err(case, coords, refs, L, 0)
    assert err < MUSCLE_LENGTH_M, f"max length error {err:.3e} m at {worst}"


def test_fn_path_speeds(fn_paths):
    case, coords, refs, _, _, V, _ = fn_paths
    err, worst = _max_err(case, coords, refs, V, 1)
    assert err < MUSCLE_SPEED_MS, f"max lengthening-speed error {err:.3e} m/s at {worst}"


def test_fn_path_moment_arms(fn_paths):
    case, _, _, moment_arms, _, _, MA = fn_paths
    lr = case.load_result
    err, worst = 0.0, None
    for world, ref in enumerate(moment_arms):
        for muscle, per_coord in ref.items():
            for coord, r in per_coord.items():
                e = abs(float(MA[world, lr.muscle_id_lookup[muscle], lr.qpos_id_lookup[coord]]) - r)
                if e > err:
                    err, worst = e, (world, muscle, coord)
    assert err < MUSCLE_MOMENT_ARM_M, f"max moment-arm error {err:.3e} m at {worst}"
