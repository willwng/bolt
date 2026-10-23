""" Merges several parsed models into one, so that they share a world """
import copy
import dataclasses
import types
import typing
from collections import Counter
from typing import Optional

import numpy as np
import warp as wp

from bolt.loader.converters.converted_objects import GROUND
from bolt.loader.converters.converted_objects import GROUND_BODY
from bolt.loader.converters.converted_objects import GROUND_COLLIDER
from bolt.loader.converters.converted_objects import GROUND_JOINT
from bolt.loader.converters.converted_objects import GROUND_PARENT
from bolt.loader.converters.converted_objects import NO_DOF
from bolt.loader.converters.converted_objects import NameMarker
from bolt.loader.converters.converted_objects import PADDED_DOF
from bolt.loader.converters.converted_objects import USE_POINT_PATH
from bolt.loader.model_parser import ParsedModel

# Names shared by every model, which are never prefixed
_SHARED_NAMES = {GROUND, GROUND_PARENT, NO_DOF, PADDED_DOF}


def merge_parsed_models(
        parsed_models: list[ParsedModel],
        prefixes: list[str],
        gravity: Optional[wp.vec3] = None,
) -> ParsedModel:
    """
    Merges the models into one ParsedModel with a single ground.
    Every name of model i is prefixed with "<prefixes[i]>/"
    """
    if len(parsed_models) != len(prefixes):
        raise ValueError(f"Got {len(parsed_models)} models but {len(prefixes)} prefixes")
    if len(set(prefixes)) != len(prefixes):
        raise ValueError(f"Model names must be unique, got {prefixes}")
    for prefix in prefixes:
        if not prefix or "/" in prefix:
            raise ValueError(f"Model name '{prefix}' must be non-empty and not contain '/'")

    # Check that all models agree with gravity
    if gravity is None:
        gravity = parsed_models[0].gravity
        for prefix, parsed in zip(prefixes, parsed_models):
            if not np.allclose(np.array(parsed.gravity), np.array(gravity)):
                raise ValueError(f"Model '{prefix}' has gravity {parsed.gravity}, but '{prefixes[0]}' has {gravity}. "
                                 f"Set the scene's gravity to override it")

    merged = ParsedModel(
        gravity=wp.vec3(gravity),
        bodies=[GROUND_BODY],
        joints=[GROUND_JOINT],
        geoms=[GROUND_COLLIDER],
        stl_contacts=[],
        visuals=[],
        spatial_transforms=[],
        spring_gen_forces=[],
        limit_forces=[],
        swing_twists=[],
        activation_actuators=[],
        muscles=[],
        function_paths=[],
        sites_muscle=[],
        sites_contact=[],
        sites_marker=[],
        sites_rem=[],
    )
    for prefix, parsed in zip(prefixes, parsed_models):
        _append(merged, _prefixed(_without_ground(parsed), prefix))

    _check_unique_names(merged)
    return merged


def _without_ground(parsed: ParsedModel) -> ParsedModel:
    """ Shallow copy of parsed without its ground body, joint and collider (the merged model has its own) """
    if parsed.bodies[0] is not GROUND_BODY or parsed.joints[0] is not GROUND_JOINT or parsed.geoms[0] is not GROUND_COLLIDER:
        raise ValueError("Expected the parsed model to start with the ground body, joint and collider")
    return dataclasses.replace(parsed, bodies=parsed.bodies[1:], joints=parsed.joints[1:], geoms=parsed.geoms[1:])


def _prefixed(parsed: ParsedModel, prefix: str) -> ParsedModel:
    """ Deep copy of parsed with every Name field prefixed. The copy keeps the sharing of sites between lists """
    # Point-path muscles share the USE_POINT_PATH placeholder, which is compared by value so must keep its name
    parsed = copy.deepcopy(parsed, memo={id(USE_POINT_PATH): USE_POINT_PATH})
    _prefix_names(parsed, prefix, seen={id(USE_POINT_PATH)})
    return parsed


def _prefix_names(obj, prefix: str, seen: set[int]):
    """ Prefixes, in place, every Name field of the dataclass obj and of the dataclasses it contains """
    if id(obj) in seen:  # e.g. a muscle path point, which is also in sites_muscle
        return
    seen.add(id(obj))
    hints = typing.get_type_hints(type(obj), include_extras=True)
    for f in dataclasses.fields(obj):
        setattr(obj, f.name, _prefix_value(getattr(obj, f.name), hints[f.name], prefix, seen))


def _prefix_value(value, hint, prefix: str, seen: set[int]):
    if value is None:
        return None
    hint = _strip_optional(hint)
    if is_name(hint):
        return value if value in _SHARED_NAMES else f"{prefix}/{value}"
    if dataclasses.is_dataclass(value):
        _prefix_names(value, prefix, seen)
        return value
    if isinstance(value, (list, tuple)):
        args = typing.get_args(hint)
        if isinstance(value, tuple) and args and args[-1] is not Ellipsis:
            elem_hints = args  # fixed-length tuple, e.g. tuple[Name, Name, Name]
        else:
            elem_hints = [args[0] if args else None] * len(value)  # list[X], tuple[X, ...], or bare list
        return type(value)(_prefix_value(v, h, prefix, seen) for v, h in zip(value, elem_hints))
    return value


def is_name(hint) -> bool:
    """ Whether the type hint is Name (Annotated with NameMarker) """
    return typing.get_origin(hint) is typing.Annotated and any(isinstance(a, NameMarker) for a in hint.__metadata__)


def _strip_optional(hint):
    """ X | None -> X """
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return hint


def _append(merged: ParsedModel, parsed: ParsedModel):
    """ Concatenates every list of parsed onto merged """
    for name, value in vars(parsed).items():
        if isinstance(value, list):
            getattr(merged, name).extend(value)


def _check_unique_names(merged: ParsedModel):
    """ Names are used as ids by the loader, ensure no duplicates """
    groups = {
        "body": [body.name for body in merged.bodies],
        "joint": [joint.name for joint in merged.joints],
        "coordinate": [c for joint in merged.joints for c in joint.coordinates],
        "collider": [geom.name for geom in merged.geoms],
        "muscle": [muscle.name for muscle in merged.muscles],
        "actuator": [actuator.name for actuator in merged.activation_actuators],
    }
    for kind, names in groups.items():
        duplicates = [name for name, count in Counter(names).items() if count > 1]
        if duplicates:
            raise ValueError(f"Duplicate {kind} names after merging: {duplicates}")
    return
