from bolt.types_consts import ActuatorMetadata
from bolt.load_utils.converted_objects import ActivationCoordinateActuatorData
from bolt.load_utils.osim_types import OSimType


def convert_activation_actuators(model: OSimType.Model) -> list[ActivationCoordinateActuatorData]:
    actuator_data = []
    force_set = model.getForceSet()
    activation_actuators = filter(lambda f: f.getConcreteClassName() == "ActivationCoordinateActuator", force_set)
    for actuator in activation_actuators:
        actuator = OSimType.ActivationCoordinateActuator.safeDownCast(actuator)
        actuator_name = actuator.getName()

        actuator_data.append(
            ActivationCoordinateActuatorData(
                name=actuator_name,
                coordinate=actuator.get_coordinate(),
                optimal_force=actuator.get_optimal_force(),
                activation_time_constant=actuator.get_activation_time_constant(),
                default_activation=actuator.get_default_activation(),
            )
        )

    return actuator_data


def create_actuator_metadata(
        actuators: list[ActivationCoordinateActuatorData],
        dof_ordering: dict[str, int]
) -> list[ActuatorMetadata]:
    actuator_metadata = []
    for actuator in actuators:
        actuator_meta = ActuatorMetadata()
        actuator_meta.optimal_force = actuator.optimal_force
        actuator_meta.activation_time_constant = actuator.activation_time_constant
        actuator_meta.default_activation = actuator.default_activation

        actuator_meta.coordinate = dof_ordering[actuator.coordinate]

        actuator_meta.min_activation = 0.0
        actuator_meta.max_activation = 1.0
        actuator_metadata.append(actuator_meta)
    return actuator_metadata


def get_actuator_ordering(actuators: list[ActivationCoordinateActuatorData]) -> dict[str, int]:
    return {actuator.name: i for i, actuator in enumerate(actuators)}
