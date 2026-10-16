"""
Implicit joint damping must give the accelerations of (M + h B) qacc = f.

At zero velocity there are no Coriolis forces and damping is linear in qvel, so central differences of the
explicit accelerations in qvel give K = M^-1 B exactly. The implicitly damped accelerations are then
qacc_implicit = (I + h K)^-1 qacc_explicit, with no mass matrix needed.
"""

from __future__ import annotations

import os

import numpy as np
import opensim as osim
import pytest

import bolt
import opensim_oracle
from bolt._src.pipeline import forward
from models import MODEL_NAMES, model_path

STEP = 0.01       # s
VISCOSITY = 1.0   # N m s / rad (or N s / m) on every non-root coordinate
STIFFNESS = 10.0  # N m / rad, so that every joint carries a torque (a force-free skeleton just falls)
SPEED = 1.0       # velocity perturbation for the central differences
MAX_ROTATION = 1.2
TOL = 1e-3        # relative to the largest acceleration


def _damped_skeleton(path: str, out_dir: str) -> tuple[str, list[str]]:
    """ Skeleton-only copy of the model with a spring-damper on every coordinate except the root's """
    model = osim.Model(opensim_oracle.skeleton_only_model(path, out_dir))
    root = opensim_oracle._root_free_joint(model)
    root_coords = {root.get_coordinates(i).getName() for i in range(root.numCoordinates())} if root else set()
    damped = [c.getName() for c in model.getCoordinateSet() if c.getName() not in root_coords]
    for name in damped:
        force = osim.SpringGeneralizedForce(name)
        force.setName(f"damping_{name}")
        force.setStiffness(STIFFNESS)
        force.setViscosity(VISCOSITY)
        model.addForce(force)
    model.initSystem()
    out_path = os.path.join(out_dir, "damped_" + os.path.basename(path))
    model.printToXML(out_path)
    return out_path, damped


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_implicit_damping_solves_m_plus_hb(name, tmp_path):
    path, damped = _damped_skeleton(model_path(name), str(tmp_path))
    osim_model = osim.Model(path)
    osim_state = osim_model.initSystem()

    # World 0 at rest; worlds 2k+1 / 2k+2 with +/- SPEED on the k-th damped coordinate
    n_worlds = 1 + 2 * len(damped)
    lr = bolt.load_model(model_path=path, n_worlds=n_worlds, integrator=bolt.IntegratorType.EULER_FIXED,
                         requires_visuals=False)
    m, d = lr.model, lr.data
    dofs = [lr.dof_id_lookup[c] for c in damped]
    np.testing.assert_allclose(m.dof_damping.numpy()[dofs], VISCOSITY)

    q, u = opensim_oracle.random_states(osim_model, 1, seed=5, speed_scale=0.0, max_rotation=MAX_ROTATION)
    qpos, _ = opensim_oracle.bolt_state(osim_model, osim_state, lr, q[0], u[0])
    qvel = np.zeros((n_worlds, m.nv))
    for k, dof in enumerate(dofs):
        qvel[2 * k + 1, dof], qvel[2 * k + 2, dof] = SPEED, -SPEED

    def accelerations(implicit: bool) -> np.ndarray:
        bolt.set_implicit_damping(m, implicit)
        d.qpos.assign(np.tile(qpos, (n_worlds, 1)).astype(np.float32))
        d.qvel.assign(qvel.astype(np.float32))
        d.actual_step_size.fill_(STEP)
        forward.fwd(m, d)
        return d.qacc.numpy().astype(np.float64)

    explicit = accelerations(False)
    K = np.zeros((m.nv, m.nv))
    for k, dof in enumerate(dofs):
        K[:, dof] = -(explicit[2 * k + 1] - explicit[2 * k + 2]) / (2.0 * SPEED)
    expected = np.linalg.solve(np.eye(m.nv) + STEP * K, explicit[0])

    implicit = accelerations(True)[0]
    err = np.abs(implicit - expected)
    worst = int(np.argmax(err))
    assert err[worst] < TOL * np.abs(expected).max(), (
        f"dof {worst}: implicit qacc {implicit[worst]:.6g}, (M + hB)^-1 f gives {expected[worst]:.6g} "
        f"(explicit {explicit[0, worst]:.6g})")
