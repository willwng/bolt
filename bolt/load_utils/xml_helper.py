import xml.etree.ElementTree as ET
from typing import Iterable


def extract_bolt_only_objects(model_path: str) -> Iterable:
    # OpenSim doesn't support some features, so we put them in a custom set called BoltOnlySet
    tree = ET.parse(model_path)
    root = tree.getroot()
    model = root.find("Model")
    extra_set = model.find("BoltOnlySet")
    if extra_set is None:
        return []
    objects = extra_set.find("objects")
    return objects if objects is not None else []


def extract_string_from_element(element: ET.Element, field_name: str) -> str:
    field = element.find(field_name)
    return field.text


def extract_float_from_element(element: ET.Element, field_name: str) -> float:
    return float(extract_string_from_element(element, field_name))


def extract_vec3_from_element(element: ET.Element, field_name: str) -> tuple[float, float, float]:
    field = element.find(field_name)
    vec3_str = field.text
    vec3_values = vec3_str.split()
    return float(vec3_values[0]), float(vec3_values[1]), float(vec3_values[2])
