"""
Validate Bolt's muscle metabolics against OpenSim's Umberger2010MuscleMetabolicsProbe.

OpenSim's muscle state (activation, excitation, fiber length/velocity, active force) is fed into Bolt's kernel,
so this checks the metabolic model itself independent of Bolt's muscle dynamics.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import warp as wp

import opensim_oracle
from bolt._src.muscle.metabolics import MetabolicOptions, compute_muscle_metabolics, \
    default_metabolic_muscle_parameters
from bolt._src.types import FiberVelocityInfo, MuscleDynamicsInfo, MuscleLengthInfo
from conftest import N_STATES
from models import MODEL_NAMES, model_path
from tolerances import METABOLIC_POWER_ATOL, METABOLIC_POWER_RTOL

# OpenSim's default MetabolicMuscleParameter values, matching default_metabolic_muscle_parameters
SPECIFIC_TENSION, DENSITY, SLOW_TWITCH_RATIO = 0.25e6, 1059.7, 0.5

OPTION_SETS = {
    "opensim_defaults": MetabolicOptions(),
    "umberger2010_published": MetabolicOptions(
        use_bhargava_recruitment=False, include_negative_mechanical_work=False, forbid_negative_total_power=False),
    "activation_maintenance_only": MetabolicOptions(
        shortening_rate_on=False, mechanical_work_rate_on=False, forbid_negative_total_power=False),
    "shortening_only": MetabolicOptions(
        activation_maintenance_rate_on=False, mechanical_work_rate_on=False, forbid_negative_total_power=False),
    "mechanical_work_only": MetabolicOptions(
        activation_maintenance_rate_on=False, shortening_rate_on=False, forbid_negative_total_power=False),
    "no_minimum_heat_rate": MetabolicOptions(enforce_minimum_heat_rate=False, aerobic_factor=1.0,
                                             muscle_effort_scaling_factor=0.8),
}

# Bolt option name -> OpenSim probe property name
_PROBE_PROPERTY = {
    "enforce_minimum_heat_rate": "enforce_minimum_heat_rate_per_muscle",
    "use_bhargava_recruitment": "use_Bhargava_recruitment_model",
}


def _probe_options(options: MetabolicOptions) -> dict:
    return {_PROBE_PROPERTY.get(k, k): v for k, v in dataclasses.asdict(options).items()}


def _struct_array(struct, values: dict[str, np.ndarray]) -> wp.array:
    shape = next(iter(values.values())).shape
    arr = np.zeros(shape, dtype=struct.numpy_dtype())
    for field, v in values.items():
        arr[field] = v
    return wp.array(arr, dtype=struct)


@pytest.fixture(scope="module", params=list(OPTION_SETS), ids=list(OPTION_SETS))
def metabolics(request, load_case):
    options = OPTION_SETS[request.param]
    case = load_case(model_path(MODEL_NAMES[0]))
    lr = case.load_result
    model, state, probe = opensim_oracle.umberger_probe_model(
        case.path, _probe_options(options), SPECIFIC_TENSION, DENSITY, SLOW_TWITCH_RATIO)
    names = sorted(lr.muscle_id_lookup, key=lr.muscle_id_lookup.get)

    rng = np.random.default_rng(5)
    q, u = opensim_oracle.random_states(model, N_STATES, seed=5, speed_scale=3.0, max_rotation=1.2)
    refs, inputs = [], []
    for world in range(N_STATES):
        activation = dict(zip(names, rng.uniform(0.01, 1.0, len(names))))
        excitation = dict(zip(names, rng.uniform(0.0, 1.0, len(names))))
        excitation[names[0]] = 0.0  # exercise the zero-excitation recruitment branch
        power, inp = opensim_oracle.umberger_metabolics(model, state, probe, q[world], u[world], activation, excitation)
        refs.append([power[n] for n in names])
        inputs.append(inp)

    def gather(key):
        return np.array([[inp[n][key] for n in names] for inp in inputs])

    # Bolt's muscle parameters must agree with OpenSim for the comparison to be meaningful
    metadata = case.m.muscle_metadata.numpy()
    for key, bolt_key in [("optimal_fiber_length", "optimal_fiber_length"),
                          ("max_isometric_force", "max_isometric_force"),
                          ("max_contraction_velocity", "v_max")]:
        np.testing.assert_allclose(metadata[bolt_key], gather(key)[0], rtol=1e-6, err_msg=key)

    d = case.d
    d.m_act.assign(gather("activation").astype(np.float32))
    d.m_excitations.assign(gather("excitation").astype(np.float32))
    wp.copy(d.muscle_length_info, _struct_array(MuscleLengthInfo, {
        "norm_fiber_length": gather("norm_fiber_length"),
        "fiber_active_force_length_multiplier": gather("active_force_length_multiplier"),
    }))
    wp.copy(d.muscle_velocity_info, _struct_array(FiberVelocityInfo, {"fiber_velocity": gather("fiber_velocity")}))
    wp.copy(d.muscle_dynamics_info, _struct_array(MuscleDynamicsInfo, {
        "active_fiber_force": gather("active_fiber_force"),
    }))

    out = wp.zeros((N_STATES, case.m.nmuscle), dtype=float)
    params = default_metabolic_muscle_parameters(case.m.nmuscle)
    compute_muscle_metabolics(case.m, case.d, params, options, out)
    return names, out.numpy(), np.array(refs), gather("fiber_velocity")


def test_metabolic_power_matches_opensim(metabolics):
    names, mine, ref, _ = metabolics
    err = np.abs(mine - ref) - METABOLIC_POWER_RTOL * np.abs(ref)
    world, muscle = np.unravel_index(np.argmax(err), err.shape)
    assert err[world, muscle] < METABOLIC_POWER_ATOL, (
        f"world {world}, {names[muscle]}: bolt {mine[world, muscle]:.6g} W, opensim {ref[world, muscle]:.6g} W")


def test_states_cover_both_contraction_directions(metabolics):
    """ Guard against a vacuous comparison: both shortening and lengthening fibers are exercised """
    *_, fiber_velocity = metabolics
    assert (fiber_velocity < 0).any() and (fiber_velocity > 0).any()
