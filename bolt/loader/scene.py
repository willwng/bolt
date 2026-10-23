"""
A scene: several OpenSim models sharing every world (e.g. a subject and a ball), sharing one ground and floor.
Each model's names are prefixed with "<model name>/" (e.g. "ball/ball"), so models may reuse names.

Scenes are built in Python, or loaded from YAML:
    gravity: [0, -9.80665, 0]    # optional; otherwise all models must agree
    models:
      - name: subject
        path: example_model.osim  # relative to the YAML file
        muscle_fn_path: example_model_fn.xml  # optional
        initial:                  # optional initial root pose (free roots only), in ground
          pos: [0, 1.05, 0]       # root mobilizer frame position, i.e. the root's translation qpos
          rot: {euler_deg: [0, 90, 0]}  # or {quat_wxyz: [1, 0, 0, 0]}; euler order defaults to "xyz"
      - name: ball
        path: ball.osim
"""
import os
from dataclasses import dataclass
from dataclasses import field
from typing import Optional

import yaml
from scipy.spatial.transform import Rotation


@dataclass
class InitialPose:
    """
    Initial pose of a model's free root joint: written into qpos at load and kept in
    ModelLoadResult.initial_qpos (e.g. for resets). It isn't part of the Model
    """
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @staticmethod
    def from_euler_deg(pos: tuple[float, float, float], euler_deg: tuple[float, float, float],
                       order: str = "xyz") -> "InitialPose":
        """ order follows scipy: lowercase is extrinsic (fixed axes), uppercase is intrinsic """
        x, y, z, w = Rotation.from_euler(order, euler_deg, degrees=True).as_quat()
        return InitialPose(pos=tuple(pos), quat_wxyz=(w, x, y, z))


@dataclass
class SceneModel:
    name: str  # prefix for every name in the model
    path: str  # .osim file
    muscle_fn_path: Optional[str] = None  # fitted function-based muscle paths
    initial: Optional[InitialPose] = None  # initial root pose; otherwise qpos starts at zero


@dataclass
class Scene:
    models: list[SceneModel] = field(default_factory=list)
    gravity: Optional[tuple[float, float, float]] = None  # overrides the models' gravity


_SCENE_KEYS = {"models", "gravity"}
_MODEL_KEYS = {"name", "path", "muscle_fn_path", "initial"}
_INITIAL_KEYS = {"pos", "rot"}
_ROT_KEYS = {"quat_wxyz", "euler_deg", "order"}


def load_scene_file(scene_path: str) -> Scene:
    """ Reads a Scene from a YAML file. Model paths are relative to the file """
    with open(scene_path) as f:
        desc = yaml.safe_load(f) or {}
    base_dir = os.path.dirname(os.path.abspath(scene_path))

    def resolve(path: Optional[str]) -> Optional[str]:
        return None if path is None else os.path.join(base_dir, os.path.expanduser(path))

    _check_keys(desc, _SCENE_KEYS, scene_path)
    models = []
    for model in desc.get("models", []):
        _check_keys(model, _MODEL_KEYS, scene_path)
        if "name" not in model or "path" not in model:
            raise ValueError(f"{scene_path}: every model needs a 'name' and a 'path', got {model}")
        models.append(SceneModel(
            name=str(model["name"]),
            path=resolve(model["path"]),
            muscle_fn_path=resolve(model.get("muscle_fn_path")),
            initial=None if model.get("initial") is None else _parse_initial(model["initial"], scene_path),
        ))

    gravity = desc.get("gravity")
    if gravity is not None:
        gravity = _floats(gravity, 3, "gravity", scene_path)
    return Scene(models=models, gravity=gravity)


def _parse_initial(desc: dict, scene_path: str) -> InitialPose:
    _check_keys(desc, _INITIAL_KEYS, scene_path)
    pos = _floats(desc.get("pos", (0.0, 0.0, 0.0)), 3, "initial pos", scene_path)
    rot = desc.get("rot", {"quat_wxyz": (1.0, 0.0, 0.0, 0.0)})
    _check_keys(rot, _ROT_KEYS, scene_path)
    if ("quat_wxyz" in rot) == ("euler_deg" in rot):
        raise ValueError(f"{scene_path}: initial rot needs exactly one of 'quat_wxyz' or 'euler_deg', got {rot}")
    if "quat_wxyz" in rot:
        if "order" in rot:
            raise ValueError(f"{scene_path}: 'order' only applies to 'euler_deg'")
        return InitialPose(pos=pos, quat_wxyz=_floats(rot["quat_wxyz"], 4, "initial quat_wxyz", scene_path))
    euler = _floats(rot["euler_deg"], 3, "initial euler_deg", scene_path)
    return InitialPose.from_euler_deg(pos, euler, order=rot.get("order", "xyz"))


def _floats(values, n: int, what: str, scene_path: str) -> tuple:
    if len(values) != n:
        raise ValueError(f"{scene_path}: {what} must have {n} components, got {values}")
    return tuple(float(v) for v in values)


def _check_keys(desc: dict, allowed: set[str], scene_path: str):
    unknown = set(desc) - allowed
    if unknown:
        raise ValueError(f"{scene_path}: unknown keys {sorted(unknown)} (allowed: {sorted(allowed)})")
