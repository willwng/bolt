import warp as wp

from . import dgf
from .types import Data
from .types import Model
from .types import ResidualResult
from .types import MuscleMetadata
from .types import MuscleLengthInfo
from .types import FiberVelocityInfo
from .types import MuscleDynamicsInfo
from .consts import M_MIN_NORM_TENDON_FORCE
from .consts import M_MAX_NORM_TENDON_FORCE
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _metabolics_kernel(
        # Model:
        muscle_metadata: wp.array(dtype=MuscleMetadata),
        # In:
        activation_maintenance_rate_on: bool,
        shortening_rate_on: bool,
        mechanical_work_rate_on: bool,
        enforce_minimum_heat_rate: bool,
        aerobic_factor: float,
        muscle_effort_scaling_factor: float,
        use_bhargava_recruitment: bool,
        include_negative_mechanical_work: bool,
        forbid_negative_total_power: bool,
        # Data in:
        m_activation_in: wp.array2d(dtype=float),
        m_excitation_in: wp.array2d(dtype=float),
        muscle_length_info_in: wp.array2d(dtype=MuscleLengthInfo),
        fiber_velocity_info_in: wp.array2d(dtype=FiberVelocityInfo),
        muscle_dynamics_info_in: wp.array2d(dtype=MuscleDynamicsInfo),
        # Data out:
        muscle_metabolic_out: wp.array2d(dtype=float),
):
    """ Computes muscle activation, shortening, mechanical heat rate. Does not compute basal heat rate """
    worldid, muscle_id = wp.tid()
    mm = muscle_metadata[muscle_id]
    mli = muscle_length_info_in[worldid, muscle_id]
    fvi = fiber_velocity_info_in[worldid, muscle_id]
    mdi = muscle_dynamics_info_in[worldid, muscle_id]

    # Get some muscle properties
    muscle_mass = (mm.max_isometric_force / mm.specific_tension) * mm.density * mm.optimal_fiber_length
    slow_twitch_ratio = mm.slow_twitch_ratio
    max_shortening_velocity = mm.v_max
    activation = muscle_effort_scaling_factor * m_activation_in[worldid, muscle_id]
    excitation = muscle_effort_scaling_factor * m_excitation_in[worldid, muscle_id]
    fiber_force_active = muscle_effort_scaling_factor * mdi.active_fiber_force
    fiber_force_active = wp.max(fiber_force_active, 0.0)  # should not be happening, but just in case
    fiber_length_normalized = mli.norm_fiber_length
    fiber_velocity = fvi.fiber_velocity
    fiber_velocity_normalized = fiber_velocity / mm.optimal_fiber_length
    F_iso = mli.fiber_active_force_length_multiplier

    # Set activation dependence scaling parameter: A
    if excitation > activation:
        A = excitation
    else:
        A = (excitation + activation) / 2.0

    if use_bhargava_recruitment:
        u_slow = slow_twitch_ratio * wp.sin(0.5 * wp.pi * excitation)
        u_fast = (1.0 - slow_twitch_ratio) * (1.0 - wp.cos(0.5 * wp.pi * excitation))
        slow_twitch_ratio = 1.0 if excitation == 0.0 else u_slow / (u_slow + u_fast)

    if forbid_negative_total_power or activation_maintenance_rate_on:
        unscaled_AM_dot = 128.0 * (1.0 - slow_twitch_ratio) + 25.0

        if fiber_length_normalized <= 1.0:
            AM_dot = aerobic_factor * wp.pow(A, 0.6) * unscaled_AM_dot
        else:
            AM_dot = wp.pow(A, 0.6) * ((0.4 * unscaled_AM_dot) + (0.6 * unscaled_AM_dot * F_iso))

    # Shortening Heart Rate
    if forbid_negative_total_power or shortening_rate_on:
        v_max_fast_twitch = max_shortening_velocity
        v_max_slow_twitch = max_shortening_velocity / 2.5
        alpha_shortening_fast_twitch = 153.0 / v_max_fast_twitch;
        alpha_shortening_slow_twitch = 100.0 / v_max_slow_twitch;

        if fiber_length_normalized <= 0.0:  # Concentric contraction, Vm < 0
            max_shortening_rate = 100.0
            tmp_slow_twitch = -alpha_shortening_slow_twitch * fiber_length_normalized
            # Apply upper limit to unscaled slow twitch shortening rate
            tmp_slow_twitch = wp.min(tmp_slow_twitch, max_shortening_rate)

            tmp_fast_twitch = alpha_shortening_fast_twitch * fiber_length_normalized * (1.0 - slow_twitch_ratio)
            unscaled_Sdot = (tmp_slow_twitch * slow_twitch_ratio) - tmp_fast_twitch
            S_dot = aerobic_factor * wp.pow(A, 2.0) * unscaled_Sdot
        else:  # Eccentric contraction, Vm >= 0
            unscaled_Sdot = ((4.0 if include_negative_mechanical_work else 0.3) *
                             alpha_shortening_slow_twitch * fiber_length_normalized)
            S_dot = aerobic_factor * A * unscaled_Sdot

        # Fiber length dependence on scaled shortening heat rate
        if fiber_length_normalized > 1.0:
            S_dot *= F_iso

    # Mechanical Work Rate
    if forbid_negative_total_power or mechanical_work_rate_on:
        if include_negative_mechanical_work or fiber_velocity <= 0.0:
            W_dot = -fiber_force_active * fiber_velocity
        else:
            W_dot = 0.0

        W_dot /= muscle_mass

    # If necessary, increase the shortening heat rate so total power is non-negative
    if forbid_negative_total_power:
        E_dot_Wkg_before_clamp = AM_dot + S_dot + W_dot
        if E_dot_Wkg_before_clamp < 0.0:
            S_dot -= E_dot_Wkg_before_clamp

    # Check from Umberger, total heat rate cannot fall below 1.0 W/kg
    total_heat_rate = AM_dot + S_dot
    if enforce_minimum_heat_rate and total_heat_rate < 1.0 and activation_maintenance_rate_on and shortening_rate_on:
        total_heat_rate = 1.0

    # Total Metabolic Energy Rate
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
    E_dot *= muscle_mass
    muscle_metabolic_out[worldid, muscle_id] = E_dot
    return


@event_scope
def compute_muscle_metabolics(m: Model, d: Data):
    """ Muscle dynamics """
    if not m.nmuscle:
        return

    mo = m.opt.metabolic_options
    wp.launch(
        _metabolics_kernel,
        dim=(d.nworld, m.nmuscle),
        inputs=[m.muscle_metadata,
                mo.activation_maintenance_rate_on,
                mo.shortening_rate_on,
                mo.mechanical_work_rate_on,
                mo.enforce_minimum_heat_rate,
                mo.aerobic_factor,
                mo.muscle_effort_scaling_factor,
                mo.use_bhargava_recruitment,
                mo.include_negative_mechanical_work,
                mo.forbid_negative_total_power,
                d.m_act, d.m_excitations,
                d.muscle_length_info, d.muscle_velocity_info, d.muscle_dynamics_info],
        outputs=[d.muscle_metabolic],
    )
