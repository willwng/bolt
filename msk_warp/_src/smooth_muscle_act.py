import warp as wp

from . import consts
from .types import ActivationType
from .types import Data
from .types import Model
from .types import MuscleMetadata
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.func
def calc_activation_derivative_dgf(
        activation: float,
        excitation: float,
        activation_time_const: float,
        deactivation_time_const: float,
        activation_dynamics_smoothing: float
) -> float:
    time_const_fact = 0.5 + 1.5 * activation
    tmp_act = 1.0 / (activation_time_const * time_const_fact)
    tmp_deact = time_const_fact / deactivation_time_const
    f = 0.5 * wp.tanh(activation_dynamics_smoothing * (excitation - activation))
    time_const = tmp_act * (f + 0.5) + tmp_deact * (-f + 0.5)
    return time_const * (excitation - activation)


@wp.func
def calc_activation_derivative_millard(
        activation: float,
        excitation: float,
        activation_time_const: float,
        deactivation_time_const: float,
) -> float:
    if excitation > activation:
        tau = activation_time_const * (0.5 + 1.5 * activation)
    else:
        tau = deactivation_time_const / (0.5 + 1.5 * activation)
    return (excitation - activation) / tau


@wp.kernel
def _compute_activation_dot_dgf(
        # Model:
        muscle_metadata: wp.array(dtype=MuscleMetadata),
        # Data in:
        mexcitation_in: wp.array2d(dtype=float),
        act_in: wp.array2d(dtype=float),
        # Data out:
        act_dot_out: wp.array2d(dtype=float),
):
    worldid, muscle_id = wp.tid()
    mm = muscle_metadata[muscle_id]

    excitation = mexcitation_in[worldid, muscle_id]
    activation = act_in[worldid, muscle_id]
    act_dot = calc_activation_derivative_dgf(
        activation,
        excitation,
        mm.activation_time_const,
        mm.deactivation_time_const,
        mm.activation_dynamics_smoothing
    )
    act_dot_out[worldid, muscle_id] = act_dot
    return


@wp.kernel
def _compute_activation_dot_millard(
        # Model:
        muscle_metadata: wp.array(dtype=MuscleMetadata),
        # Data in:
        mexcitation_in: wp.array2d(dtype=float),
        act_in: wp.array2d(dtype=float),
        # Data out:
        act_dot_out: wp.array2d(dtype=float),
):
    worldid, muscle_id = wp.tid()
    mm = muscle_metadata[muscle_id]

    excitation = mexcitation_in[worldid, muscle_id]
    activation = act_in[worldid, muscle_id]
    act_dot = calc_activation_derivative_millard(
        activation,
        excitation,
        mm.activation_time_const,
        mm.deactivation_time_const,
    )
    act_dot_out[worldid, muscle_id] = act_dot
    return


@event_scope
def compute_act_dot(m: Model, d: Data):
    if m.opt.activation_type == ActivationType.DGF:
        wp.launch(
            _compute_activation_dot_dgf,
            dim=(d.nworld, m.nmuscle),
            inputs=[m.muscle_metadata, d.m_excitations, d.m_act],
            outputs=[d.m_act_dot],
        )
    elif m.opt.activation_type == ActivationType.MILLARD:
        wp.launch(
            _compute_activation_dot_millard,
            dim=(d.nworld, m.nmuscle),
            inputs=[m.muscle_metadata, d.m_excitations, d.m_act],
            outputs=[d.m_act_dot],
        )