"""
Muscle metabolic power following OpenSim's Umberger2010MuscleMetabolicsProbe (Umberger et al., 2003; 2010).
The basal heat rate is not included.
"""
from dataclasses import dataclass

import warp as wp

from ..types import Data
from ..types import FiberVelocityInfo
from ..types import Model
from ..types import MuscleDynamicsInfo
from ..types import MuscleLengthInfo
from ..types import MuscleMetadata
from ..warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@dataclass
class MetabolicOptions:
    """ Mirrors the Umberger2010MuscleMetabolicsProbe properties (defaults match OpenSim) """
    activation_maintenance_rate_on: bool = True
    shortening_rate_on: bool = True
    mechanical_work_rate_on: bool = True
    enforce_minimum_heat_rate: bool = True
    aerobic_factor: float = 1.5
    muscle_effort_scaling_factor: float = 1.0
    use_bhargava_recruitment: bool = True
    include_negative_mechanical_work: bool = True
    forbid_negative_total_power: bool = True


@wp.struct
class MetabolicMuscleParameters:
    specific_tension: float  # N/m^2
    density: float  # kg/m^3
    slow_twitch_ratio: float  # ratio of slow-twitch fibers


def default_metabolic_muscle_parameters(nmuscle: int) -> wp.array:
    """ OpenSim's default parameters for every muscle """
    params = MetabolicMuscleParameters()
    params.specific_tension = 0.25e6
    params.density = 1059.7
    params.slow_twitch_ratio = 0.5
    return wp.array([params] * nmuscle, dtype=MetabolicMuscleParameters)


@wp.kernel
def _metabolics_kernel(
        # Model:
        muscle_metadata: wp.array(dtype=MuscleMetadata),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        m_act_in: wp.array2d(dtype=float),
        m_excitations_in: wp.array2d(dtype=float),
        muscle_length_info_in: wp.array2d(dtype=MuscleLengthInfo),
        muscle_velocity_info_in: wp.array2d(dtype=FiberVelocityInfo),
        muscle_dynamics_info_in: wp.array2d(dtype=MuscleDynamicsInfo),
        # In:
        metabolic_params: wp.array(dtype=MetabolicMuscleParameters),
        activation_maintenance_rate_on: bool,
        shortening_rate_on: bool,
        mechanical_work_rate_on: bool,
        enforce_minimum_heat_rate: bool,
        aerobic_factor: float,
        muscle_effort_scaling_factor: float,
        use_bhargava_recruitment: bool,
        include_negative_mechanical_work: bool,
        forbid_negative_total_power: bool,
        # Out:
        muscle_metabolic_out: wp.array2d(dtype=float),
):
    """ Metabolic power (W) of each muscle: activation/maintenance and shortening heat, and mechanical work """
    worldid, muscle_id = wp.tid()
    if integration_done_in[worldid]:
        return
    mm = muscle_metadata[muscle_id]
    mp = metabolic_params[muscle_id]
    mli = muscle_length_info_in[worldid, muscle_id]
    fvi = muscle_velocity_info_in[worldid, muscle_id]
    mdi = muscle_dynamics_info_in[worldid, muscle_id]

    muscle_mass = (mm.max_isometric_force / mp.specific_tension) * mp.density * mm.optimal_fiber_length
    activation = muscle_effort_scaling_factor * m_act_in[worldid, muscle_id]
    excitation = muscle_effort_scaling_factor * m_excitations_in[worldid, muscle_id]
    fiber_force_active = wp.max(muscle_effort_scaling_factor * mdi.active_fiber_force, 0.0)
    fiber_length_normalized = mli.norm_fiber_length
    fiber_velocity = fvi.fiber_velocity
    fiber_velocity_normalized = fiber_velocity / mm.optimal_fiber_length
    F_iso = mli.fiber_active_force_length_multiplier

    # Activation dependence scaling parameter
    A = excitation
    if excitation <= activation:
        A = (excitation + activation) / 2.0

    slow_twitch_ratio = mp.slow_twitch_ratio
    if use_bhargava_recruitment:
        u_slow = slow_twitch_ratio * wp.sin(0.5 * wp.pi * excitation)
        # 1 - cos(pi/2 * e), written as 2 sin^2(pi/4 * e) to avoid float32 cancellation at small excitations
        u_fast = (1.0 - slow_twitch_ratio) * 2.0 * wp.pow(wp.sin(0.25 * wp.pi * excitation), 2.0)
        slow_twitch_ratio = wp.where(excitation == 0.0, 1.0, u_slow / (u_slow + u_fast))

    # Activation and maintenance heat rate (W/kg)
    AM_dot = float(0.0)
    if forbid_negative_total_power or activation_maintenance_rate_on:
        unscaled_AM_dot = 128.0 * (1.0 - slow_twitch_ratio) + 25.0
        if fiber_length_normalized <= 1.0:
            AM_dot = aerobic_factor * wp.pow(A, 0.6) * unscaled_AM_dot
        else:
            AM_dot = aerobic_factor * wp.pow(A, 0.6) * (0.4 * unscaled_AM_dot + 0.6 * unscaled_AM_dot * F_iso)

    # Shortening heat rate (W/kg)
    S_dot = float(0.0)
    if forbid_negative_total_power or shortening_rate_on:
        v_max_fast_twitch = mm.v_max
        v_max_slow_twitch = mm.v_max / 2.5
        alpha_shortening_fast_twitch = 153.0 / v_max_fast_twitch
        alpha_shortening_slow_twitch = 100.0 / v_max_slow_twitch

        if fiber_velocity_normalized <= 0.0:  # concentric contraction
            max_shortening_rate = 100.0
            tmp_slow_twitch = wp.min(-alpha_shortening_slow_twitch * fiber_velocity_normalized, max_shortening_rate)
            tmp_fast_twitch = alpha_shortening_fast_twitch * fiber_velocity_normalized * (1.0 - slow_twitch_ratio)
            unscaled_S_dot = tmp_slow_twitch * slow_twitch_ratio - tmp_fast_twitch
            S_dot = aerobic_factor * wp.pow(A, 2.0) * unscaled_S_dot
        else:  # eccentric contraction
            alpha_lengthening = wp.where(include_negative_mechanical_work, 4.0, 0.3)
            unscaled_S_dot = alpha_lengthening * alpha_shortening_slow_twitch * fiber_velocity_normalized
            S_dot = aerobic_factor * A * unscaled_S_dot

        if fiber_length_normalized > 1.0:
            S_dot *= F_iso

    # Mechanical work rate (W/kg)
    W_dot = float(0.0)
    if forbid_negative_total_power or mechanical_work_rate_on:
        if include_negative_mechanical_work or fiber_velocity <= 0.0:
            W_dot = -fiber_force_active * fiber_velocity
        W_dot /= muscle_mass

    # Increase the shortening heat rate so the total power is non-negative
    if forbid_negative_total_power:
        E_dot_before_clamp = AM_dot + S_dot + W_dot
        if E_dot_before_clamp < 0.0:
            S_dot -= E_dot_before_clamp

    # The total heat rate cannot fall below 1.0 W/kg
    total_heat_rate = AM_dot + S_dot
    if enforce_minimum_heat_rate and total_heat_rate < 1.0 and activation_maintenance_rate_on and shortening_rate_on:
        total_heat_rate = 1.0

    E_dot = float(0.0)
    if activation_maintenance_rate_on and shortening_rate_on:
        E_dot += total_heat_rate
    else:
        if activation_maintenance_rate_on:
            E_dot += AM_dot
        if shortening_rate_on:
            E_dot += S_dot
    if mechanical_work_rate_on:
        E_dot += W_dot
    muscle_metabolic_out[worldid, muscle_id] = E_dot * muscle_mass


@event_scope
def compute_muscle_metabolics(
        m: Model,
        d: Data,
        metabolic_params: wp.array,
        options: MetabolicOptions,
        muscle_metabolic_out: wp.array2d,
):
    """ Writes the metabolic power (W) of each muscle to muscle_metabolic_out """
    if not m.nmuscle:
        return
    wp.launch(
        _metabolics_kernel,
        dim=(d.nworld, m.nmuscle),
        inputs=[
            m.muscle_metadata,
            d.integration_done, d.m_act, d.m_excitations,
            d.muscle_length_info, d.muscle_velocity_info, d.muscle_dynamics_info,
            metabolic_params,
            options.activation_maintenance_rate_on,
            options.shortening_rate_on,
            options.mechanical_work_rate_on,
            options.enforce_minimum_heat_rate,
            options.aerobic_factor,
            options.muscle_effort_scaling_factor,
            options.use_bhargava_recruitment,
            options.include_negative_mechanical_work,
            options.forbid_negative_total_power,
        ],
        outputs=[muscle_metabolic_out],
    )
