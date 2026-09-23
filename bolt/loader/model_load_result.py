from dataclasses import dataclass

from bolt.loader.converters import actuator_helper
from bolt.loader.converters import coordinate_force_helper
from bolt.loader.converters import geom_helper
from bolt.loader.converters import joint_helper
from bolt.loader.converters import muscle_helper
from bolt.loader.converters import visual_helper
from bolt.loader.converters.converted_objects import GeomData
from bolt.loader.model_parser import ParsedModel
from bolt.loader.model_topology import ModelTopology
from bolt.types import Data
from bolt.types import MeshLoadResult
from bolt.types import Model


@dataclass
class ModelLoadResult:
    model: Model
    data: Data
    root_free: bool
    body_id_lookup: dict[str, int]
    dof_id_lookup: dict[str, int]
    qpos_id_lookup: dict[str, int]
    limit_id_lookup: dict[str, tuple[float, float]]
    muscle_id_lookup: dict[str, int]
    actuator_id_lookup: dict[str, int]
    collider_id_lookup: dict[str, int]
    mesh_load_results: list[MeshLoadResult]
    colliders: list[GeomData]


def make_load_result(parsed: ParsedModel, topology: ModelTopology, m: Model, d: Data) -> ModelLoadResult:
    return ModelLoadResult(
        model=m,
        data=d,
        root_free=joint_helper.check_root_free(topology.joints),
        body_id_lookup=topology.body_ordering,
        qpos_id_lookup=topology.qpos_ordering,
        dof_id_lookup=topology.dof_ordering,
        limit_id_lookup=coordinate_force_helper.create_limit_id_lookup(parsed.limit_forces, topology.qpos_ordering),
        muscle_id_lookup=muscle_helper.get_muscle_ordering(parsed.muscles),
        actuator_id_lookup=actuator_helper.get_actuator_ordering(parsed.activation_actuators),
        collider_id_lookup=geom_helper.get_geom_ordering(parsed.geoms),
        mesh_load_results=visual_helper.create_mesh_load_results(parsed.visuals),
        colliders=parsed.geoms,
    )
