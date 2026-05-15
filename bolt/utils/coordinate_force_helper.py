import warp as wp
import numpy as np

from bolt import CoordinateLimitForce
from bolt.utils.converted_objects import SpringGeneralizedForceData, CoordinateLimitForceData
from bolt.utils.osim_types import OSimType


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


def convert_coordinate_limit_force(model: OSimType.Model) -> list[CoordinateLimitForceData]:
    """ Returns the all the converted CoordinateLimitForce in the model """
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
        spring_generalized_force_data: list[SpringGeneralizedForceData],
        dof_ordering: dict[str, int]
) -> tuple[list[float], list[float]]:
    """ Returns the stiffness and damping value for each dof """
    dof_stiffness = [0.0 for _ in range(len(dof_ordering))]
    dof_damping = [0.0 for _ in range(len(dof_ordering))]
    for gen_force in spring_generalized_force_data:
        if gen_force.coordinate not in dof_ordering:
            raise ValueError(
                f"Coordinate {gen_force.coordinate} in SpringGeneralizedForce {gen_force.name} not found")
        dof_idx = dof_ordering[gen_force.coordinate]
        dof_stiffness[dof_idx] += gen_force.stiffness
        dof_damping[dof_idx] += gen_force.viscosity

    return dof_stiffness, dof_damping


def get_qpos_spring_rest(
        qpos_ordering: dict[str, int]
) -> list[float]:
    """ Returns the stiffness value for each dof """
    qpos_rest_length = [0.0 for _ in range(len(qpos_ordering))]
    return qpos_rest_length


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

    # For any coordinate that doesn't have a limit force or linear stop, set to inf
    for coordinate in qpos_ordering.keys():
        if coordinate not in limit_id_lookup:
            limit_id_lookup[coordinate] = (-float('inf'), float('inf'))

    return limit_id_lookup
