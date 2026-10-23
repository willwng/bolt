"""
Loads an OpenSim model or scene into Bolt's Model and Data:
    parse -> (merge) -> model topology -> create Model -> allocate Data
"""
from typing import Optional

import warp as wp

from bolt.loader.allocate_data import allocate_data
from bolt.loader.allocate_data import data_sizes
from bolt.loader.array_util import allocate_field
from bolt.loader.array_util import assert_all_fields_set
from bolt.loader.converters import geom_helper
from bolt.loader.create_model import create_model
from bolt.loader.create_model import pack_geoms
from bolt.loader.model_merge import merge_parsed_models
from bolt.loader.model_parser import ParsedModel
from bolt.loader.model_parser import parse_osim_model
from bolt.loader.model_load_result import ModelLoadResult
from bolt.loader.model_load_result import make_load_result
from bolt.loader.model_topology import build_topology
from bolt.loader.converters.converted_objects import GROUND
from bolt.loader.scene import InitialPose
from bolt.loader.scene import Scene
from bolt.types import Data
from bolt.types import IntegratorType
from bolt.types import MobilizerType


def load_model(
        model_path: str,
        n_worlds: int,
        integrator: IntegratorType,
        requires_visuals: bool,
        muscle_fn_path: Optional[str],
        render_kinematic_tree: bool,
) -> ModelLoadResult:
    parsed = parse_osim_model(model_path, requires_visuals, muscle_fn_path)
    return _build(parsed, n_worlds, integrator, requires_visuals, render_kinematic_tree)


def load_scene(
        scene: Scene,
        n_worlds: int,
        integrator: IntegratorType,
        requires_visuals: bool,
        render_kinematic_tree: bool,
) -> ModelLoadResult:
    """ Loads every model of the scene into a single Model; names are prefixed with "<model name>/" """
    if not scene.models:
        raise ValueError("Scene has no models")
    parsed = merge_parsed_models(
        parsed_models=[parse_osim_model(model.path, requires_visuals, model.muscle_fn_path) for model in scene.models],
        prefixes=[model.name for model in scene.models],
        gravity=None if scene.gravity is None else wp.vec3(scene.gravity),
    )
    load_result = _build(parsed, n_worlds, integrator, requires_visuals, render_kinematic_tree)
    for model in scene.models:
        if model.initial is not None:
            _set_initial_pose(load_result, parsed, model.name, model.initial)
    return load_result


def _set_initial_pose(load_result: ModelLoadResult, parsed: ParsedModel, prefix: str, initial: InitialPose):
    """ Writes the initial pose into the qpos of the model's root joint, which must be free """
    roots = [joint for joint in parsed.joints
             if joint.parent_body_name == GROUND and joint.child_body_name.startswith(f"{prefix}/")]
    if len(roots) != 1 or roots[0].mob_type != MobilizerType.FREE:
        raise ValueError(f"Model '{prefix}': an initial pose needs exactly one free root joint, got "
                         f"{[(joint.name, joint.mob_type.name) for joint in roots]}")
    adr = load_result.model.mob_qposadr.numpy()[load_result.body_id_lookup[roots[0].child_body_name]]
    w, x, y, z = initial.quat_wxyz
    # Free joint qpos: quaternion (x, y, z, w) of the root mobilizer frame in ground, then its position
    load_result.initial_qpos[adr:adr + 7] = (x, y, z, w, *initial.pos)


def _build(
        parsed: ParsedModel,
        n_worlds: int,
        integrator: IntegratorType,
        requires_visuals: bool,
        render_kinematic_tree: bool,
) -> ModelLoadResult:
    topology = build_topology(parsed, render_kinematic_tree)
    m = create_model(parsed, topology, integrator, requires_visuals)
    d = allocate_data(m, n_worlds, integrator)
    return make_load_result(parsed, topology, m, d)


def update_colliders(load_result: ModelLoadResult):
    """ Re-packs the colliders in load_result.colliders into the Model (e.g. after adding or editing colliders) """
    m, d = load_result.model, load_result.data
    pack_geoms(m, load_result.colliders, load_result.body_id_lookup, m.body_parentid.list())
    assert_all_fields_set(m)
    load_result.collider_id_lookup = geom_helper.get_geom_ordering(load_result.colliders)

    # Reallocate the per-geom Data arrays
    sizes = data_sizes(m, d.nworld, d.naconmax)
    for name in ("geom_X", "geom_cforce", "geom_self_cforce"):
        setattr(d, name, allocate_field(Data, name, sizes))
