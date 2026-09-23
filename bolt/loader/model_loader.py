"""
Loads an OpenSim model into Bolt's Model and Data:
    parse -> model topology -> create Model -> allocate Data
"""
from typing import Optional

from bolt.loader.allocate_data import allocate_data
from bolt.loader.allocate_data import data_sizes
from bolt.loader.array_util import allocate_field
from bolt.loader.array_util import assert_all_fields_set
from bolt.loader.converters import geom_helper
from bolt.loader.create_model import create_model
from bolt.loader.create_model import pack_geoms
from bolt.loader.model_parser import parse_osim_model
from bolt.loader.model_load_result import ModelLoadResult
from bolt.loader.model_load_result import make_load_result
from bolt.loader.model_topology import build_topology
from bolt.types_consts import Data
from bolt.types_consts import IntegratorType


def load_model(
        model_path: str,
        n_worlds: int,
        integrator: IntegratorType,
        requires_visuals: bool,
        muscle_fn_path: Optional[str],
        render_kinematic_tree: bool,
) -> ModelLoadResult:
    parsed = parse_osim_model(model_path, requires_visuals, muscle_fn_path)
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
