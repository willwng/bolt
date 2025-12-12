import warp as wp

from .types import Data
from .types import Model
from .types import ActuatorMetadata
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _reset_actuator(
        # Model in:
        actuator_metadata: wp.array(dtype=ActuatorMetadata),
        # Data in:
        world_reset_in: wp.array(dtype=bool),
        # Data out:
        act_out: wp.array2d(dtype=float),
):
    worldid, actuator_id = wp.tid()
    if world_reset_in[worldid]:
        act_out[worldid, actuator_id] = actuator_metadata[actuator_id].default_activation
    return


@wp.kernel
def _compute_activation_dot(
        # Model in:
        actuator_metadata: wp.array(dtype=ActuatorMetadata),
        # Data in:
        a_excitation_in: wp.array2d(dtype=float),
        act_in: wp.array2d(dtype=float),
        # Data out:
        act_dot_out: wp.array2d(dtype=float),
):
    worldid, actuator_id = wp.tid()
    excitation = a_excitation_in[worldid, actuator_id]
    activation = act_in[worldid, actuator_id]
    tau = actuator_metadata[actuator_id].activation_time_constant
    act_dot_out[worldid, actuator_id] = (excitation - activation) / tau
    return


@wp.kernel
def _qfrc_actuators(
        # Model in:
        actuator_metadata: wp.array(dtype=ActuatorMetadata),
        # Data in:
        a_act_in: wp.array2d(dtype=float),
        # Data out:
        qfrc_applied_out: wp.array2d(dtype=float),
):
    worldid, actuator_id = wp.tid()
    am = actuator_metadata[actuator_id]
    activation = a_act_in[worldid, actuator_id]
    actuation = (activation - 0.5) * 2.0 * am.optimal_force
    wp.atomic_add(qfrc_applied_out[worldid], am.coordinate, actuation)
    return


@event_scope
def actuator_reset(m: Model, d: Data):
    wp.launch(
        _reset_actuator,
        dim=(d.nworld, m.nactuator),
        inputs=[m.actuator_metadata, d.world_reset],
        outputs=[d.a_act],
    )


@event_scope
def compute_act_dot(m: Model, d: Data):
    wp.launch(
        _compute_activation_dot,
        dim=(d.nworld, m.nactuator),
        inputs=[m.actuator_metadata, d.a_excitations, d.a_act],
        outputs=[d.a_act_dot],
    )


@event_scope
def actuator_force(m: Model, d: Data):
    wp.launch(
        _qfrc_actuators,
        dim=(d.nworld, m.nactuator),
        inputs=[m.actuator_metadata, d.a_act],
        outputs=[d.qfrc_applied],
    )
