from .osim_types import OSimType


def extract_vec3(v: OSimType.Vec3) -> tuple[float, float, float]:
    return v.get(0), v.get(1), v.get(2)


def extract_quat(q: OSimType.Quat) -> tuple[float, float, float, float]:
    return q.get(0), q.get(1), q.get(2), q.get(3)


def extract_value_from_prop(prop: OSimType.Property) -> float:
    """ Extract the value of an OpenSim property as a float """
    return float(prop.toString())


def extract_vec3_from_prop(prop: OSimType.Property) -> tuple[float, float, float]:
    """ Extract the value of an OpenSim property as a tuple of 3 floats """
    return tuple(map(float, prop.toString().strip("()").split(" ")))
