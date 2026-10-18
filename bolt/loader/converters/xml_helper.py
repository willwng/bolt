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


def extract_optional_float_from_element(element: ET.Element, field_name: str, default: float) -> float:
    field = element.find(field_name)
    return default if field is None else float(field.text)


def extract_optional_bool_from_element(element: ET.Element, field_name: str, default: bool) -> bool:
    field = element.find(field_name)
    return default if field is None else field.text.strip().lower() in ("true", "1")


