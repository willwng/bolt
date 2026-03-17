import opensim as osim
import warp as wp

from msk_warp.utils.python_util import apply_map_to_list
from msk_warp.utils.converted_objects import CoordinateLinearDamperData, CoordinateLinearSpringData, \
    CoordinateLinearStopData
from msk_warp.utils.osim_types import OSimType


def convert_coordinate_linear_damper(model: osim.Model) -> list[CoordinateLinearDamperData]:
    """ Returns the all the converted CoordinateLinearDamper in the model """
    coordinate_linear_damper_data = []
    force_set = model.getForceSet()
    # During parsing these are only parsed as a force?
    coordinate_linear_dampers = filter(lambda f: f.getConcreteClassName() == "CoordinateLinearDamper", force_set)
    for coordinate_linear_damper in coordinate_linear_dampers:
        coordinate_linear_damper = OSimType.CoordinateLinearDamper.safeDownCast(coordinate_linear_damper)
        coordinate_linear_damper_data.append(
            CoordinateLinearDamperData(
                name=coordinate_linear_damper.getName(),
                coordinate=coordinate_linear_damper.get_coordinate(),
                damping=coordinate_linear_damper.get_damping(),
            )
        )

    return coordinate_linear_damper_data


def convert_coordinate_linear_spring(model: osim.Model) -> list[CoordinateLinearSpringData]:
    """ Returns the all the converted CoordinateLinearSpring in the model """
    coordinate_linear_spring_data = []
    force_set = model.getForceSet()
    coordinate_linear_springs = filter(lambda f: f.getConcreteClassName() == "CoordinateLinearSpring", force_set)
    for coordinate_linear_spring in coordinate_linear_springs:
        coordinate_linear_spring = OSimType.CoordinateLinearSpring.safeDownCast(coordinate_linear_spring)
        coordinate_linear_spring_data.append(
            CoordinateLinearSpringData(
                name=coordinate_linear_spring.getName(),
                coordinate=coordinate_linear_spring.get_coordinate(),
                default_stiffness=coordinate_linear_spring.get_default_stiffness(),
                rest_length=coordinate_linear_spring.get_rest_length(),
            )
        )
    return coordinate_linear_spring_data


def convert_coordinate_linear_stop(model: osim.Model) -> list[CoordinateLinearStopData]:
    """ Returns the all the converted CoordinateLinearStop in the model """
    coordinate_linear_stop_data = []
    force_set = model.getForceSet()
    coordinate_linear_stops = filter(lambda f: f.getConcreteClassName() == "CoordinateLinearStop", force_set)
    for coordinate_linear_stop in coordinate_linear_stops:
        coordinate_linear_stop = OSimType.CoordinateLinearStop.safeDownCast(coordinate_linear_stop)
        coordinate_linear_stop_data.append(
            CoordinateLinearStopData(
                name=coordinate_linear_stop.getName(),
                coordinate=coordinate_linear_stop.get_coordinate(),
                stiffness_damping=wp.vec2(coordinate_linear_stop.get_stiffness(), coordinate_linear_stop.get_damping()),
                range=wp.vec2(coordinate_linear_stop.get_q_low(), coordinate_linear_stop.get_q_high()),
            )
        )
    return coordinate_linear_stop_data


def get_dof_damping(
        coordinate_linear_damper_data: list[CoordinateLinearDamperData],
        dof_ordering: dict[str, int]
) -> list[float]:
    """ Returns the damping value for each dof """
    dof_damping = [0.0 for _ in range(len(dof_ordering))]
    for damper in coordinate_linear_damper_data:
        if damper.coordinate not in dof_ordering:
            raise ValueError(f"Coordinate {damper.coordinate} in CoordinateLinearDamper {damper.name} not found")
        dof_idx = dof_ordering[damper.coordinate]
        dof_damping[dof_idx] = damper.damping
    return dof_damping


def get_dof_stiffness(
        coordinate_linear_spring_data: list[CoordinateLinearSpringData],
        dof_ordering: dict[str, int]
) -> list[float]:
    """ Returns the stiffness value for each dof """
    dof_stiffness = [0.0 for _ in range(len(dof_ordering))]
    for damper in coordinate_linear_spring_data:
        if damper.coordinate not in dof_ordering:
            raise ValueError(f"Coordinate {damper.coordinate} in CoordinateLinearDamper {damper.name} not found")
        dof_idx = dof_ordering[damper.coordinate]
        dof_stiffness[dof_idx] = damper.default_stiffness
    return dof_stiffness


def get_qpos_spring_rest(
        coordinate_linear_spring_data: list[CoordinateLinearSpringData],
        qpos_ordering: dict[str, int]
) -> list[float]:
    """ Returns the stiffness value for each dof """
    qpos_rest_length = [0.0 for _ in range(len(qpos_ordering))]
    for damper in coordinate_linear_spring_data:
        if damper.coordinate not in qpos_ordering:
            raise ValueError(f"Coordinate {damper.coordinate} in CoordinateLinearDamper {damper.name} not found")
        qpos_idx = qpos_ordering[damper.coordinate]
        qpos_rest_length[qpos_idx] = damper.rest_length
    return qpos_rest_length


def get_stop_coordinates_adr(
        coordinate_linear_stop_data: list[CoordinateLinearStopData],
        ordering: dict[str, int]
) -> list[int]:
    """ Returns the address of the stop coordinate for each dof """
    stop_coordinates = [stop.coordinate for stop in coordinate_linear_stop_data]
    return apply_map_to_list(stop_coordinates, ordering)


def get_stop_qpos_range(coordinate_linear_stop_data: list[CoordinateLinearStopData], ) -> list[wp.vec2]:
    return [stop.range for stop in coordinate_linear_stop_data]


def get_stop_dof_stiffness_damping(coordinate_linear_stop_data: list[CoordinateLinearStopData]) -> list[wp.vec2]:
    return [stop.stiffness_damping for stop in coordinate_linear_stop_data]
