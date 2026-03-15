from .osim_types import OSimType


def extract_vector(v: OSimType.Vector) -> list[int]:
    return v.to_numpy().tolist()


def extract_quat(q: OSimType.Quat) -> tuple[float, float, float, float]:
    return q.get(0), q.get(1), q.get(2), q.get(3)


def extract_vec3(v: OSimType.Vec3) -> tuple[float, float, float]:
    return v.get(0), v.get(1), v.get(2)


def extract_value_from_prop(prop: OSimType.Property) -> float:
    """ Extract the value of an OpenSim property as a float """
    return float(prop.toString())


def extract_property_string_list(l: OSimType.PropertyStringList) -> list[str]:
    return [l.getValue(i) for i in range(l.size())]
