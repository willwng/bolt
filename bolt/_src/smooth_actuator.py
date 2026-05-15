import warp as wp

from .types import Data
from .types import Model
from .types import ActuatorMetadata
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _reset_to_default_activation(
        # Model:
        actuator_metadata: wp.array(dtype=ActuatorMetadata),
        # Data in:
        world_reset_in: wp.array(dtype=bool),
        # Data out:
        act_out: wp.array2d(dtype=float)
):
    worldid, act_id = wp.tid()
    if not world_reset_in[worldid]:
        return
    act_out[worldid, act_id] = actuator_metadata[act_id].default_activation
    return


@wp.kernel
def _compute_activation_dot(
        # Model in:
        actuator_metadata: wp.array(dtype=ActuatorMetadata),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        a_excitation_in: wp.array2d(dtype=float),
        act_in: wp.array2d(dtype=float),
        # Data out:
        act_out: wp.array2d(dtype=float),
        act_dot_out: wp.array2d(dtype=float),
):
    worldid, actuator_id = wp.tid()
    if integration_done_in[worldid]:
        return
    excitation = a_excitation_in[worldid, actuator_id]
    activation = act_in[worldid, actuator_id]
    tau = actuator_metadata[actuator_id].activation_time_constant

    if tau == 0.0:  # zero time constant (instantaneous activation)
        act_out[worldid, actuator_id] = excitation
        act_dot_out[worldid, actuator_id] = 0.0
    else:
        act_dot_out[worldid, actuator_id] = (excitation - activation) / tau
    return


@wp.kernel
def _ufrc_actuators(
        # Model in:
        actuator_metadata: wp.array(dtype=ActuatorMetadata),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        a_act_in: wp.array2d(dtype=float),
        # Data out:
        ufrc_actuator_out: wp.array2d(dtype=float),
):
    worldid, actuator_id = wp.tid()
    if integration_done_in[worldid]:
        return
    am = actuator_metadata[actuator_id]
    activation = a_act_in[worldid, actuator_id]
    actuation = (activation - 0.5) * 2.0 * am.optimal_force
    wp.atomic_add(ufrc_actuator_out[worldid], am.coordinate, actuation)
    return


@event_scope
def reset_to_default_activation(m: Model, d: Data):
    wp.launch(
        _reset_to_default_activation,
        dim=(d.nworld, m.nactuator),
        inputs=[m.actuator_metadata, d.world_reset],
        outputs=[d.a_act],
    )


@event_scope
def activation_dynamics(m: Model, d: Data):
    wp.launch(
        _compute_activation_dot,
        dim=(d.nworld, m.nactuator),
        inputs=[m.actuator_metadata, d.integration_done, d.a_excitations, d.a_act],
        outputs=[d.a_act, d.a_act_dot],
    )


@event_scope
def actuator_force(m: Model, d: Data):
    wp.launch(
        _ufrc_actuators,
        dim=(d.nworld, m.nactuator),
        inputs=[m.actuator_metadata, d.integration_done, d.a_act],
        outputs=[d.ufrc_actuator],
    )
