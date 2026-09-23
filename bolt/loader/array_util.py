import dataclasses

import numpy as np
import warp as wp


def check_zero(arr: wp.array):
    # if any the dimensions are zero, replace with a 1
    shape = list(arr.shape)
    found_zero = False
    for i in range(len(shape)):
        if shape[i] == 0:
            shape[i] = 1
            found_zero = True
    if not found_zero:
        return arr
    return wp.zeros(shape, dtype=arr.dtype)


def to_warp_array(lst, dtype):
    arr = np.array(lst)
    return check_zero(wp.from_numpy(arr, dtype=dtype))


def make_zero(shape, dtype):
    return check_zero(wp.zeros(shape, dtype=dtype))


def dataclass_sizes(*objs) -> dict[str, int]:
    """ The int-valued fields of the given dataclass instances, usable as array(...) dim names """
    sizes = {}
    for obj in objs:
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            if isinstance(value, int) and not isinstance(value, bool):
                sizes[f.name] = value
    return sizes


def _annotated_shape(cls, field: dataclasses.Field, sizes: dict[str, int]) -> tuple[int, ...]:
    """ Resolves the dims of a array(...) annotation """
    ann = field.type
    shape = getattr(ann, "shape", None)
    if not isinstance(ann, wp.array) or shape is None or all(s == 0 for s in shape):
        raise TypeError(f"{cls.__name__}.{field.name} has no array(...) annotation, so it must be passed explicitly")
    resolved = []
    for dim in shape:
        if isinstance(dim, int):
            resolved.append(dim)
        elif dim.isdigit():
            resolved.append(int(dim))
        elif dim in sizes:
            resolved.append(sizes[dim])
        else:
            raise TypeError(f"{cls.__name__}.{field.name}: cannot resolve dim {dim!r}, so it must be passed explicitly")
    return tuple(resolved)


def allocate_field(cls, name: str, sizes: dict[str, int]) -> wp.array:
    """ Allocates zeros for field `name` of dataclass cls, with the shape of its types.array(...) annotation """
    field = next(f for f in dataclasses.fields(cls) if f.name == name)
    return make_zero(_annotated_shape(cls, field, sizes), dtype=field.type.dtype)


def allocate_from_annotations(cls, sizes: dict[str, int], **values):
    """
    Constructs the dataclass cls; fields must have a array(...) annotation whose dims resolve from sizes
    """
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - field_names
    if unknown:
        raise TypeError(f"{cls.__name__} has no fields {sorted(unknown)}")
    kwargs = dict(values)
    for f in dataclasses.fields(cls):
        if f.name not in kwargs:
            kwargs[f.name] = allocate_field(cls, f.name, sizes)
    return cls(**kwargs)


def new_unset_class(cls):
    """ An instance of dataclass cls with no fields set yet (see assert_all_fields_set) """
    return object.__new__(cls)


def assert_all_fields_set(obj):
    """ Checks that every field of dataclass instance obj was set, and nothing that isn't a field """
    names = {f.name for f in dataclasses.fields(obj)}
    missing = sorted(names - set(vars(obj)))
    unknown = sorted(set(vars(obj)) - names)
    if missing or unknown:
        raise TypeError(f"{type(obj).__name__}: missing fields {missing}, unknown fields {unknown}")
