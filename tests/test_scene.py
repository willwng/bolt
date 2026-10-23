""" Scenes: several models merged into every world """

from __future__ import annotations

import dataclasses
import inspect
import typing

import numpy as np
import pytest

import bolt
from bolt._src.pipeline import forward
from bolt.loader.converters import converted_objects
from bolt.loader.model_merge import is_name
from conftest import N_STATES
import opensim_oracle
from models import FN_PATH_FILE, FN_PATH_MODEL, MODELS_DIR, model_path

# (prefix, model, muscle_fn_path, x offset of the root): the models are spread out so they don't touch each other
SCENE = [("a", FN_PATH_MODEL, FN_PATH_FILE, 0.0), ("b", "hamner_model", None, 10.0)]
BALL_RADIUS = 0.11

# str fields of the converted objects that aren't names, so aren't prefixed when merging
NOT_NAMES = {("VisualData", "mesh_file")}


def _scene() -> bolt.Scene:
    return bolt.Scene(models=[bolt.SceneModel(prefix, model_path(name), fn) for prefix, name, fn, _ in SCENE])


def _ids(lookup: dict[str, int]) -> dict[str, int]:
    return {name: i for name, i in lookup.items() if i >= 0}  # skip the NO_DOF placeholder


@pytest.fixture(scope="module")
def merged(load_case):
    """ The merged scene and each model loaded alone, all in the same random states """
    lr = bolt.load_scene(_scene(), n_worlds=N_STATES, integrator=bolt.IntegratorType.EULER_FIXED,
                         requires_visuals=False)
    qpos, qvel = np.zeros((N_STATES, lr.model.nq)), np.zeros((N_STATES, lr.model.nv))
    singles = {}
    for seed, (prefix, name, fn, x_offset) in enumerate(SCENE):
        case = load_case(model_path(name), fn)
        case.set_states(*opensim_oracle.random_states(case.osim_model, N_STATES, seed=seed, speed_scale=0.5))
        q = case.d.qpos.numpy()
        q[:, case.m.mob_qposadr.numpy()[1] + 4] += x_offset  # body 1 is the free root: quat (4), then translation
        case.d.qpos.assign(q)
        forward.fwd(case.m, case.d)

        for coord, i in _ids(case.load_result.qpos_id_lookup).items():
            qpos[:, lr.qpos_id_lookup[f"{prefix}/{coord}"]] = case.d.qpos.numpy()[:, i]
        for coord, i in _ids(case.load_result.dof_id_lookup).items():
            qvel[:, lr.dof_id_lookup[f"{prefix}/{coord}"]] = case.d.qvel.numpy()[:, i]
        singles[prefix] = (case.load_result, case.d.mob_X_GB.numpy(), case.d.muscle_length.numpy(),
                           case.d.qacc.numpy())

    lr.data.qpos.assign(qpos.astype(np.float32))
    lr.data.qvel.assign(qvel.astype(np.float32))
    forward.fwd(lr.model, lr.data)
    return lr, singles


def test_sizes(merged):
    lr, singles = merged
    m = lr.model
    assert m.nbody == 1 + sum(s[0].model.nbody - 1 for s in singles.values())  # one shared ground
    assert m.ngeom == 1 + sum(s[0].model.ngeom - 1 for s in singles.values())  # one shared floor
    for field in ("nq", "nv", "nmuscle", "nactuator", "nsite", "nstlcontact", "nlimitforce", "njnts_cst"):
        assert getattr(m, field) == sum(getattr(s[0].model, field) for s in singles.values()), field


def test_names_are_prefixed(merged):
    lr, singles = merged
    for prefix, (single, *_) in singles.items():
        for lookup in ("body_id_lookup", "qpos_id_lookup", "dof_id_lookup", "muscle_id_lookup",
                       "actuator_id_lookup", "collider_id_lookup"):
            names = set(_ids(getattr(single, lookup))) - {bolt.GROUND, bolt.GROUND_COLLIDER.name}
            assert {f"{prefix}/{n}" for n in names} <= set(getattr(lr, lookup)), lookup


def test_body_transforms_match(merged):
    lr, singles = merged
    X = lr.data.mob_X_GB.numpy()
    for prefix, (single, X_single, _, _) in singles.items():
        for body, i in _ids(single.body_id_lookup).items():
            if body != bolt.GROUND:
                np.testing.assert_allclose(X[:, lr.body_id_lookup[f"{prefix}/{body}"]], X_single[:, i],
                                           atol=1e-5, err_msg=body)


def test_muscle_lengths_match(merged):
    lr, singles = merged
    lengths = lr.data.muscle_length.numpy()
    for prefix, (single, _, lengths_single, _) in singles.items():
        ids = [lr.muscle_id_lookup[f"{prefix}/{name}"] for name in single.muscle_id_lookup]
        np.testing.assert_allclose(lengths[:, ids], lengths_single[:, list(single.muscle_id_lookup.values())],
                                   atol=1e-6)


def test_accelerations_match(merged):
    lr, singles = merged
    qacc = lr.data.qacc.numpy()
    for prefix, (single, _, _, qacc_single) in singles.items():
        for coord, i in _ids(single.dof_id_lookup).items():
            ref = qacc_single[:, i]
            tol = 1e-3 * (1.0 + np.abs(qacc_single).max(axis=1))
            err = np.abs(qacc[:, lr.dof_id_lookup[f"{prefix}/{coord}"]] - ref)
            assert np.all(err < tol), f"{coord}: max error {err.max():.3e}"


def test_ball_rests_on_floor():
    """ The ball's 6-dof CustomJoint root becomes a free joint, and it collides with the shared floor """
    scene = bolt.Scene(models=[bolt.SceneModel("ball", f"{MODELS_DIR}/ball.osim")])
    lr = bolt.load_scene(scene, n_worlds=1, integrator=bolt.IntegratorType.RK_MERSON_ADAPTIVE,
                         requires_visuals=False)
    m, d = lr.model, lr.data
    assert m.mob_type.numpy()[lr.body_id_lookup["ball/ball"]] == bolt.MobilizerType.FREE

    ty = lr.qpos_id_lookup["ball/ball_ty"]
    qpos = np.zeros(m.nq, dtype=np.float32)
    qpos[lr.qpos_id_lookup["ball/ball_quat_w"]] = 1.0
    qpos[ty] = 0.5
    d.qpos.assign(qpos[None])
    bolt.reset(m, d)
    for _ in range(200):
        bolt.increment_next_time(m, d, 0.01)
        bolt.step(m, d)
    assert d.qpos.numpy()[0, ty] == pytest.approx(BALL_RADIUS, abs=0.01)


def test_scene_file(tmp_path):
    (tmp_path / "scene.yaml").write_text(
        f"gravity: [0, -3.7, 0]\n"
        f"models:\n"
        f"  - name: ball\n"
        f"    path: {MODELS_DIR}/ball.osim\n"
        f"  - name: ball2\n"
        f"    path: ../{tmp_path.name}/ball.osim\n"  # relative to the scene file
    )
    (tmp_path / "ball.osim").write_text(open(f"{MODELS_DIR}/ball.osim").read())
    lr = bolt.load_scene(str(tmp_path / "scene.yaml"), n_worlds=1, integrator=bolt.IntegratorType.EULER_FIXED,
                         requires_visuals=False)
    assert {"ball/ball", "ball2/ball"} <= set(lr.body_id_lookup)
    assert lr.model.opt.gravity[1] == pytest.approx(-3.7)


def test_scene_file_rejects_unknown_keys(tmp_path):
    path = tmp_path / "scene.yaml"
    path.write_text("models:\n  - name: ball\n    path: ball.osim\n    pos: [0, 1, 0]\n")
    with pytest.raises(ValueError, match="unknown keys"):
        bolt.load_scene(str(path), n_worlds=1, integrator=bolt.IntegratorType.EULER_FIXED, requires_visuals=False)


def test_duplicate_model_names_rejected():
    scene = bolt.Scene(models=[bolt.SceneModel("ball", f"{MODELS_DIR}/ball.osim")] * 2)
    with pytest.raises(ValueError, match="unique"):
        bolt.load_scene(scene, n_worlds=1, integrator=bolt.IntegratorType.EULER_FIXED, requires_visuals=False)


def _plain_str(hint) -> bool:
    """ Whether the hint contains a str that isn't a Name, e.g. str, list[str], or str | None """
    if is_name(hint):
        return False
    return hint is str or any(_plain_str(arg) for arg in typing.get_args(hint))


def test_every_str_field_is_classified():
    """ A new str field must be either a Name (prefixed when merging) or listed in NOT_NAMES """
    unclassified = []
    for cls_name, cls in inspect.getmembers(converted_objects, dataclasses.is_dataclass):
        if not isinstance(cls, type):
            continue
        hints = typing.get_type_hints(cls, include_extras=True)
        for f in dataclasses.fields(cls):
            if _plain_str(hints[f.name]) and (cls_name, f.name) not in NOT_NAMES:
                unclassified.append(f"{cls_name}.{f.name}")
    assert not unclassified, f"Annotate as Name, or add to NOT_NAMES: {unclassified}"


def test_initial_pose(tmp_path):
    """ The scene's initial pose ends up in qpos, initial_qpos, and the root body's transform """
    path = tmp_path / "scene.yaml"
    path.write_text(
        f"models:\n"
        f"  - name: ball\n"
        f"    path: {MODELS_DIR}/ball.osim\n"
        f"    initial:\n"
        f"      pos: [1.0, 0.5, -2.0]\n"
        f"      rot: {{euler_deg: [0, 90, 0]}}\n"
        f"  - name: still\n"  # no initial pose: zero, with an identity quaternion
        f"    path: {MODELS_DIR}/ball.osim\n"
    )
    lr = bolt.load_scene(str(path), n_worlds=2, integrator=bolt.IntegratorType.EULER_FIXED, requires_visuals=False)
    forward.realize_position(lr.model, lr.data)
    X = lr.data.mob_X_GB.numpy()

    yaw = (0.0, np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4))  # (x, y, z, w)
    np.testing.assert_allclose(X[:, lr.body_id_lookup["ball/ball"]], [[1.0, 0.5, -2.0, *yaw]] * 2, atol=1e-6)
    np.testing.assert_allclose(X[:, lr.body_id_lookup["still/ball"]], [[0, 0, 0, 0, 0, 0, 1]] * 2, atol=1e-6)
    np.testing.assert_allclose(lr.data.qpos.numpy(), np.tile(lr.initial_qpos, (2, 1)), atol=1e-6)


def test_initial_pose_rejects_two_rotations(tmp_path):
    path = tmp_path / "scene.yaml"
    path.write_text(f"models:\n  - name: ball\n    path: {MODELS_DIR}/ball.osim\n"
                    f"    initial: {{rot: {{quat_wxyz: [1, 0, 0, 0], euler_deg: [0, 0, 0]}}}}\n")
    with pytest.raises(ValueError, match="exactly one"):
        bolt.load_scene(str(path), n_worlds=1, integrator=bolt.IntegratorType.EULER_FIXED, requires_visuals=False)
