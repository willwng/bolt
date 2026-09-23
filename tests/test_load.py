""" Model-loading checks that don't need an OpenSim oracle """

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import warp as wp

import bolt
from bolt.types import array as types_array
from bolt.loader.array_util import allocate_from_annotations
from bolt.types import GeomType
from models import FN_PATH_FILE, FN_PATH_MODEL, MODEL_NAMES, model_path


@pytest.mark.parametrize("geom_type, size, rbound", [
    (GeomType.SPHERE, (0.1, 0.1, 0.1), 0.1),
    (GeomType.CAPSULE, (0.03, 0.04, 0.03), 0.05),
])
def test_convert_user_collider(geom_type, size, rbound):
    user_geom = bolt.UserGeomData(
        name="user_geom",
        body_name="calcn_r",
        geom_type=geom_type,
        transform=wp.transform_identity(),
        size=wp.vec3(size),
    )
    geom = bolt.convert_user_collider(user_geom)
    assert geom.geom_type == geom_type
    assert geom.rbound == pytest.approx(rbound)


# --- Every array allocated by the loader matches its annotation in bolt/_src/types.py ---
_SCALAR_DTYPES = {float: wp.float32, int: wp.int32, bool: wp.bool}


def _cases():
    cases = [(name, None) for name in MODEL_NAMES]
    cases.append((FN_PATH_MODEL, FN_PATH_FILE))
    return cases


@pytest.fixture(scope="module", params=_cases(), ids=lambda c: c[0] + ("+fn" if c[1] else ""))
def loaded(request):
    name, fn_path = request.param
    lr = bolt.load_model(model_path=model_path(name), n_worlds=3, integrator=bolt.IntegratorType.RK_MERSON_ADAPTIVE,
                         requires_visuals=False, muscle_fn_path=fn_path)
    return lr.model, lr.data


def _sizes(m, d) -> dict[str, int]:
    """ Every int-valued attribute of Model, Data and Option, usable as a dim name """
    sizes = {}
    for obj in (m.opt, m, d):
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            if isinstance(value, int) and not isinstance(value, bool):
                sizes[f.name] = value
    return sizes


def _expected(dim, sizes):
    """ Returns (exact size, upper bound); either may be None """
    if isinstance(dim, int):
        return dim, None
    if dim == "*":
        return None, None
    if dim.startswith("<="):
        return None, eval(dim[2:], {}, dict(sizes))
    if dim.isdigit():
        return int(dim), None
    if dim not in sizes:
        raise KeyError(f"unknown dim name {dim!r}")
    return sizes[dim], None


def _arrays(obj, prefix=""):
    """ Yields (path, annotation, value) for the array fields of a dataclass, recursing into nested dataclasses """
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        path = f"{prefix}{f.name}"
        if isinstance(value, wp.array):
            yield path, f.type, value
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            yield from _arrays(value, prefix=f"{path}.")
        elif isinstance(value, list) and value and dataclasses.is_dataclass(value[0]):
            for i, item in enumerate(value):
                yield from _arrays(item, prefix=f"{path}[{i}].")


def _check(obj, m, d):
    sizes = _sizes(m, d)
    problems = []
    for path, ann, arr in _arrays(obj):
        shape = getattr(ann, "shape", None)
        # a plain wp.array(...) annotation has an all-zero shape; types.array(...) sets symbolic dims
        if not isinstance(ann, wp.array) or shape is None or all(s == 0 for s in shape):
            problems.append(f"{path}: not annotated with array(...)")
            continue
        if len(shape) != arr.ndim:
            problems.append(f"{path}: annotated ndim {len(shape)} but allocated shape {arr.shape}")
            continue
        for axis, (dim, actual) in enumerate(zip(shape, arr.shape)):
            try:
                exact, bound = _expected(dim, sizes)
            except KeyError as e:
                problems.append(f"{path}: {e.args[0]}")
                continue
            # the loader allocates at least 1 along every axis (loader.arrays.check_zero)
            if exact is not None and actual != exact and not (exact == 0 and actual == 1):
                problems.append(f"{path}: axis {axis} is {actual}, annotation {dim!r} = {exact}")
            if bound is not None and actual > max(bound, 1):
                problems.append(f"{path}: axis {axis} is {actual}, annotation {dim!r} = {bound}")
        dtype = _SCALAR_DTYPES.get(ann.dtype, ann.dtype)
        if not wp.types.types_equal(dtype, arr.dtype):
            problems.append(
                f"{path}: annotated dtype {wp.types.type_repr(dtype)}, allocated {wp.types.type_repr(arr.dtype)}")
    return problems


@pytest.mark.parametrize("which", ["model", "data"])
def test_arrays_match_annotations(loaded, which):
    m, d = loaded
    problems = _check(m if which == "model" else d, m, d)
    assert not problems, "\n".join(problems)


# --- allocate_from_annotations ---
_Example = dataclasses.make_dataclass("_Example", [
    ("n", int),
    ("a", types_array("nworld", "n", wp.vec3)),
    ("b", types_array("*", float)),
])


def test_allocate_from_annotations():
    obj = allocate_from_annotations(_Example, {"nworld": 2, "n": 0}, n=0, b=wp.zeros(5, dtype=float))
    assert obj.a.shape == (2, 1) and obj.a.dtype == wp.vec3  # zero-sized dims are padded to 1
    assert obj.b.shape == (5,)


@pytest.mark.parametrize("values, message", [
    ({"n": 3}, "cannot resolve dim '\\*'"),  # b has no resolvable size
    ({"b": None}, "has no array"),  # n is not an array
    ({"n": 3, "b": None, "c": 1}, "has no fields"),
])
def test_allocate_from_annotations_errors(values, message):
    with pytest.raises(TypeError, match=message):
        allocate_from_annotations(_Example, {"nworld": 2, "n": 3}, **values)


# --- update_colliders ---
def test_update_colliders_adds_user_collider():
    lr = bolt.load_model(model_path=model_path(MODEL_NAMES[0]), n_worlds=2, integrator=bolt.IntegratorType.EULER_FIXED,
                         requires_visuals=False)
    m, d = lr.model, lr.data
    ngeom = m.ngeom
    lr.colliders.append(bolt.convert_user_collider(bolt.UserGeomData(
        name="user_ball", body_name="hand_r", geom_type=GeomType.SPHERE,
        transform=wp.transform_identity(), size=wp.vec3(0.05, 0.05, 0.05))))
    bolt.update_colliders(lr)

    assert m.ngeom == ngeom + 1
    assert lr.collider_id_lookup["user_ball"] == ngeom
    assert not _check(m, m, d) and not _check(d, m, d)  # geom arrays were resized consistently

    d.world_reset.fill_(True)
    bolt.reset(m, d)
    bolt.increment_next_time(m, d, 1e-3)
    bolt.step(m, d)
    assert np.isfinite(d.qpos.numpy()).all()
