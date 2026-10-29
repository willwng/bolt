import warnings

from bolt.loader.converters.converted_objects import CoordinateCouplerData
from bolt.loader.converters.function_helper import convert_function
from bolt.loader.converters.osim_types import OSimType
from bolt.loader.converters.property_helper import extract_value_from_prop


def convert_coordinate_couplers(model: OSimType.Model) -> list[CoordinateCouplerData]:
    """ Returns the converted, enforced CoordinateCouplerConstraints in the model. Other constraints are skipped """
    couplers = []
    for constraint in model.getConstraintSet():
        constraint_class = constraint.getConcreteClassName()
        if constraint_class != "CoordinateCouplerConstraint":
            warnings.warn(f"Unsupported constraint {constraint.getName()} ({constraint_class}) is ignored")
            continue
        coupler = OSimType.CoordinateCouplerConstraint.safeDownCast(constraint)
        if not coupler.get_isEnforced():
            continue
        independent = coupler.getIndependentCoordinateNames()
        if independent.getSize() != 1:
            raise ValueError(f"CoordinateCouplerConstraint {coupler.getName()} must have exactly one "
                             f"independent coordinate, has {independent.getSize()}")
        couplers.append(
            CoordinateCouplerData(
                name=coupler.getName(),
                independent_coordinate=independent.get(0),
                dependent_coordinate=coupler.getDependentCoordinateName(),
                function=convert_function(coupler.getFunction()),
                scale=extract_value_from_prop(coupler.getPropertyByName("scale_factor")),
            )
        )
    return couplers
