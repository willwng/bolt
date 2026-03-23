import warp as wp
import numpy as np

from msk_warp import CoordinateLinearStop, CoordinateLimitForce
from msk_warp.utils.converted_objects import CoordinateLinearDamperData, CoordinateLinearSpringData, \
    CoordinateLinearStopData, SpringGeneralizedForceData, CoordinateLimitForceData
from msk_warp.utils.osim_types import OSimType


def convert_coordinate_linear_damper(model: OSimType.Model) -> list[CoordinateLinearDamperData]:
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


def convert_coordinate_linear_spring(model: OSimType.Model) -> list[CoordinateLinearSpringData]:
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


def convert_spring_generalized_force(model: OSimType.Model) -> list[SpringGeneralizedForceData]:
    """ Returns the all the converted SpringGeneralizedForce in the model """
    spring_generalized_force_data = []
    force_set = model.getForceSet()
    spring_generalized_forces = filter(lambda f: f.getConcreteClassName() == "SpringGeneralizedForce", force_set)
    for spring_generalized_force in spring_generalized_forces:
        spring_generalized_force = OSimType.SpringGeneralizedForce.safeDownCast(spring_generalized_force)
        spring_generalized_force_data.append(
            SpringGeneralizedForceData(
                name=spring_generalized_force.getName(),
                coordinate=spring_generalized_force.get_coordinate(),
                stiffness=spring_generalized_force.get_stiffness(),
                viscosity=spring_generalized_force.get_viscosity(),
            )
        )
    return spring_generalized_force_data


def convert_coordinate_linear_stop(model: OSimType.Model) -> list[CoordinateLinearStopData]:
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
                stiffness=coordinate_linear_stop.get_stiffness(),
                damping=coordinate_linear_stop.get_damping(),
                range=wp.vec2(coordinate_linear_stop.get_q_low(), coordinate_linear_stop.get_q_high()),
            )
        )
    return coordinate_linear_stop_data


def convert_coordinate_limit_force(model: OSimType.Model) -> list[CoordinateLimitForceData]:
    """ Returns the all the converted CoordinateLinearStop in the model """
    coordinate_limit_force_data = []
    force_set = model.getForceSet()
    coordinate_limits_forces = filter(lambda f: f.getConcreteClassName() == "CoordinateLimitForce", force_set)
    for coordinate_limit_force in coordinate_limits_forces:
        coordinate_limit_force = OSimType.CoordinateLimitForce.safeDownCast(coordinate_limit_force)
        # Note: we need to convert from deg to rad (reverse for inverse degrees)
        coordinate_limit_force_data.append(
            CoordinateLimitForceData(
                name=coordinate_limit_force.getName(),
                coordinate=coordinate_limit_force.get_coordinate(),
                stiffness=wp.vec2(
                    np.rad2deg(coordinate_limit_force.get_lower_stiffness()),
                    np.rad2deg(coordinate_limit_force.get_upper_stiffness())
                ),
                range=wp.vec2(
                    np.deg2rad(coordinate_limit_force.get_lower_limit()),
                    np.deg2rad(coordinate_limit_force.get_upper_limit())
                ),
                damping=np.rad2deg(coordinate_limit_force.get_damping()),
                transition=np.deg2rad(coordinate_limit_force.get_transition()),
            )
        )
    return coordinate_limit_force_data


def get_dof_stiffness_damping(
        coordinate_linear_spring_data: list[CoordinateLinearSpringData],
        coordinate_linear_damper_data: list[CoordinateLinearDamperData],
        spring_generalized_force_data: list[SpringGeneralizedForceData],
        dof_ordering: dict[str, int]
) -> tuple[list[float], list[float]]:
    """ Returns the stiffness and damping value for each dof """
    dof_stiffness = [0.0 for _ in range(len(dof_ordering))]
    for spring in coordinate_linear_spring_data:
        if spring.coordinate not in dof_ordering:
            raise ValueError(f"Coordinate {spring.coordinate} in CoordinateLinearSpring {spring.name} not found")
        dof_idx = dof_ordering[spring.coordinate]
        dof_stiffness[dof_idx] = spring.default_stiffness

    dof_damping = [0.0 for _ in range(len(dof_ordering))]
    for damper in coordinate_linear_damper_data:
        if damper.coordinate not in dof_ordering:
            raise ValueError(f"Coordinate {damper.coordinate} in CoordinateLinearDamper {damper.name} not found")
        dof_idx = dof_ordering[damper.coordinate]
        dof_damping[dof_idx] = damper.damping

    for spring_damper in spring_generalized_force_data:
        if spring_damper.coordinate not in dof_ordering:
            raise ValueError(
                f"Coordinate {spring_damper.coordinate} in SpringGeneralizedForce {spring_damper.name} not found")
        dof_idx = dof_ordering[spring_damper.coordinate]
        dof_stiffness[dof_idx] += spring_damper.stiffness
        dof_damping[dof_idx] += spring_damper.viscosity

    return dof_stiffness, dof_damping


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


# --- Linear Stop ---
def create_coordinate_linear_stop(
        coordinate_linear_stop_data: list[CoordinateLinearStopData],
        qpos_ordering: dict[str, int],
        dof_ordering: dict[str, int]
) -> list[CoordinateLinearStop]:
    coordinate_linear_stops = []
    for stop_data in coordinate_linear_stop_data:
        stop = CoordinateLinearStop()

        stop.qpos_adr = qpos_ordering[stop_data.coordinate]
        stop.dof_adr = dof_ordering[stop_data.coordinate]
        stop.qpos_range = stop_data.range
        stop.stiffness = stop_data.stiffness
        stop.damping = stop_data.damping

        coordinate_linear_stops.append(stop)
    return coordinate_linear_stops


# --- Limit Force ---
def create_coordinate_limit_force(
        coordinate_limit_force_data: list[CoordinateLimitForceData],
        qpos_ordering: dict[str, int],
        dof_ordering: dict[str, int]
) -> list[CoordinateLimitForce]:
    coordinate_limit_forces = []
    for limit_force_data in coordinate_limit_force_data:
        limit_force = CoordinateLimitForce()

        limit_force.qpos_adr = qpos_ordering[limit_force_data.coordinate]
        limit_force.dof_adr = dof_ordering[limit_force_data.coordinate]
        limit_force.qpos_range = limit_force_data.range
        limit_force.stiffness = limit_force_data.stiffness
        limit_force.damping = limit_force_data.damping
        limit_force.transition = limit_force_data.transition

        coordinate_limit_forces.append(limit_force)
    return coordinate_limit_forces


def create_limit_id_lookup(
        coordinate_limit_forces: list[CoordinateLimitForceData],
        coordinate_linear_stops: list[CoordinateLinearStopData],
        qpos_ordering: dict[str, int]
) -> dict[str, tuple[float, float]]:
    """ For each coordinate, return the associated qpos range """
    limit_id_lookup = {}

    def update_limit_id(c: str, rng: wp.vec2):
        """ If the coordinate already has a limit, update the range to be the min/max of the existing and new range """
        if c in limit_id_lookup:
            existing_range = limit_id_lookup[c]
            new_range = wp.vec2(
                min(existing_range[0], rng[0]),
                max(existing_range[1], rng[1])
            )
            limit_id_lookup[c] = (new_range[0], new_range[1])
        else:
            limit_id_lookup[c] = (rng[0], rng[1])

    # Coordinate limit forces
    for limit_force in coordinate_limit_forces:
        coord = limit_force.coordinate
        update_limit_id(coord, limit_force.range)
    # Then linear stops
    for linear_stop in coordinate_linear_stops:
        coord = linear_stop.coordinate
        update_limit_id(coord, linear_stop.range)

    # For any coordinate that doesn't have a limit force or linear stop, set to inf
    for coordinate in qpos_ordering.keys():
        if coordinate not in limit_id_lookup:
            limit_id_lookup[coordinate] = (-float('inf'), float('inf'))

    return limit_id_lookup
