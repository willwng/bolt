from msk_warp import Model, Data, MeshLoadResult
from msk_warp.utils.converted_objects import GeomData
from dataclasses import dataclass


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
