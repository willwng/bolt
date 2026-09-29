""" Validate Bolt muscle path lengths, lengthening speeds, and moment arms against OpenSim """

from __future__ import annotations

import pytest

import bolt
import opensim_oracle
from bolt._src.pipeline import forward
from bolt._src.muscle import function_path
from bolt._src.muscle import point_path
from conftest import N_STATES
from models import FN_PATH_FILE, FN_PATH_MODEL, MODEL_NAMES, model_path
from tolerances import MUSCLE_LENGTH_M, MUSCLE_MOMENT_ARM_M, MUSCLE_SPEED_MS

N_MOMENT_ARM_STATES = 4  # computing moment arms of point-based paths is slow, so just check a few

UNSUPPORTED = {
    "wrapping": ("Bolt point paths do not model wrap surfaces/obstacles", opensim_oracle.wrapped_muscles),
}


def _max_err(case, names, refs, mine, index):
    err, worst = 0.0, None
    for world, ref in enumerate(refs):
        for name in names:
            e = abs(float(mine[world, case.load_result.muscle_id_lookup[name]]) - ref[name][index])
            if e > err:
                err, worst = e, (world, name)
    return err, worst


def _realize_paths(case, q, u):
    case.set_states(q, u)
    forward.realize_position(case.m, case.d)
    forward.realize_velocity(case.m, case.d)
    point_path.muscle_point_path(case.m, case.d)
    function_path.muscle_fn_path(case.m, case.d)
    return case.d.muscle_length.numpy(), case.d.muscle_velocity.numpy()


# --- Point (geometry) paths ---
@pytest.fixture(scope="module", params=MODEL_NAMES)
def point_paths(request, load_case):
    case = load_case(model_path(request.param))
    q, u = opensim_oracle.random_states(case.osim_model, N_STATES, seed=2)
    L, V = _realize_paths(case, q, u)
    refs = [opensim_oracle.muscle_paths(case.osim_model, case.osim_state, qi, ui) for qi, ui in zip(q, u)]
    unsupported = {feature: find(case.path) for feature, (_, find) in UNSUPPORTED.items()}
    return case, refs, L, V, unsupported


def _supported(names, unsupported):
    excluded = set().union(*unsupported.values())
    names = [n for n in names if n not in excluded]
    if not names:
        pytest.skip("no muscles with fully supported point paths")
    return names


def test_point_path_lengths(point_paths):
    case, refs, L, _, unsupported = point_paths
    err, worst = _max_err(case, _supported(refs[0], unsupported), refs, L, 0)
    assert err < MUSCLE_LENGTH_M, f"max length error {err:.3e} m at {worst}"


def test_point_path_speeds(point_paths):
    case, refs, _, V, unsupported = point_paths
    err, worst = _max_err(case, _supported(refs[0], unsupported), refs, V, 1)
    assert err < MUSCLE_SPEED_MS, f"max lengthening-speed error {err:.3e} m/s at {worst}"


@pytest.mark.parametrize("feature", [
    pytest.param(feature, marks=pytest.mark.xfail(strict=True, reason=reason))
    for feature, (reason, _) in UNSUPPORTED.items()
])
def test_unsupported_point_path_lengths(point_paths, feature):
    case, refs, L, _, unsupported = point_paths
    if not unsupported[feature]:
        pytest.skip(f"no muscles with {feature}")
    err, worst = _max_err(case, sorted(unsupported[feature]), refs, L, 0)
    assert err < MUSCLE_LENGTH_M, f"max length error {err:.3e} m at {worst}"


@pytest.fixture(scope="module", params=MODEL_NAMES)
def point_moment_arms(request, load_case):
    case = load_case(model_path(request.param))
    lr = case.load_result
    q, u = opensim_oracle.random_states(case.osim_model, N_STATES, seed=5)
    case.set_states(q, u)
    forward.realize_position(case.m, case.d)
    bolt.compute_muscle_moments(case.m, case.d)
    MA = case.d.muscle_moment_arm.numpy()

    # The free root is a quaternion in Bolt (not comparable to OpenSim's coordinates), and muscles are internal
    # forces, so they apply no net generalized force to it anyway
    root = opensim_oracle.root_free_coordinates(case.osim_model, lr)
    coords = [c for c in opensim_oracle.coordinate_names(case.osim_model) if c not in root]
    muscles = [lr_name for lr_name in lr.muscle_id_lookup if
               lr.muscle_id_lookup[lr_name] in case.m.muscle_pt_group_tuple]
    refs = [opensim_oracle.muscle_paths(case.osim_model, case.osim_state, q[w], u[w],
                                        moment_arm_coords={mu: coords for mu in muscles})[1]
            for w in range(N_MOMENT_ARM_STATES)]
    unsupported = {feature: find(case.path) for feature, (_, find) in UNSUPPORTED.items()}
    return case, MA, refs, muscles, unsupported


def _moment_arm_err(case, MA, refs, muscles):
    lr = case.load_result
    err, worst = 0.0, None
    for world, ref in enumerate(refs):
        for muscle in muscles:
            for coord, r in ref[muscle].items():
                e = abs(float(MA[world, lr.muscle_id_lookup[muscle], lr.qpos_id_lookup[coord]]) - r)
                if e > err:
                    err, worst = e, (world, muscle, coord, r)
    return err, worst


def test_point_path_moment_arms(point_moment_arms):
    case, MA, refs, muscles, unsupported = point_moment_arms
    err, worst = _moment_arm_err(case, MA, refs, _supported(muscles, unsupported))
    assert err < MUSCLE_MOMENT_ARM_M, f"max moment-arm error {err:.3e} m at {worst}"


@pytest.mark.parametrize("feature", [
    pytest.param(feature, marks=pytest.mark.xfail(strict=True, reason=reason))
    for feature, (reason, _) in UNSUPPORTED.items()
])
def test_unsupported_point_path_moment_arms(point_moment_arms, feature):
    case, MA, refs, muscles, unsupported = point_moment_arms
    affected = sorted(set(muscles) & unsupported[feature])
    if not affected:
        pytest.skip(f"no muscles with {feature}")
    err, worst = _moment_arm_err(case, MA, refs, affected)
    assert err < MUSCLE_MOMENT_ARM_M, f"max moment-arm error {err:.3e} m at {worst}"


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
