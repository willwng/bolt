"""
Validate Bolt's aerodynamic drag, F = -k |v| v at the center of mass, using OpenSim's center-of-mass kinematics.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import opensim_oracle
from bolt._src.pipeline import forward
from bolt._src.dynamics.drag import DEFAULT_DRAG_FACTOR, Drag, drag_forces
from conftest import N_STATES
from models import MODEL_NAMES, model_path
from tolerances import BODY_FORCE_ATOL, BODY_FORCE_RTOL

# (body, drag factor). pelvis has two drags to exercise accumulation.
DRAGS = [
    ("pelvis", DEFAULT_DRAG_FACTOR),
    ("pelvis", 0.1),
    ("femur_r", 0.05),
    ("hand_l", 0.02),
]


@pytest.fixture(scope="module")
def drag(load_case):
    case = load_case(model_path(MODEL_NAMES[0]))
    lr = case.load_result

    drags = []
    for body, factor in DRAGS:
        d = Drag()
        d.bodyid = lr.body_id_lookup[body]
        d.drag_factor = factor
        drags.append(d)

    q, u = opensim_oracle.random_states(case.osim_model, N_STATES, seed=7, speed_scale=3.0)
    case.set_states(q, u)
    forward.realize_position(case.m, case.d)
    forward.realize_velocity(case.m, case.d)
    body_F = wp.zeros((N_STATES, case.m.nbody), dtype=wp.spatial_vector)
    drag_forces(case.m, case.d, wp.array(drags, dtype=Drag), body_F)

    bodies = sorted({b for b, _ in DRAGS})
    refs = []
    for qi, ui in zip(q, u):
        com = opensim_oracle.body_com_kinematics(case.osim_model, case.osim_state, qi, ui, bodies)
        ref = {b: np.zeros(6) for b in bodies}
        for body, factor in DRAGS:
            offset, velocity = com[body]
            force = -factor * np.linalg.norm(velocity) * velocity
            ref[body] += np.concatenate([np.cross(offset, force), force])  # torque about body origin, force
        refs.append(ref)
    return lr, body_F.numpy(), refs


def test_drag_forces_match_expected(drag):
    lr, body_F, refs = drag
    for world, ref in enumerate(refs):
        for body, expected in ref.items():
            np.testing.assert_allclose(body_F[world, lr.body_id_lookup[body]], expected,
                                       rtol=BODY_FORCE_RTOL, atol=BODY_FORCE_ATOL, err_msg=f"world {world}, {body}")


def test_other_bodies_untouched(drag):
    lr, body_F, _ = drag
    dragged = {lr.body_id_lookup[b] for b, _ in DRAGS}
    others = [i for i in range(body_F.shape[1]) if i not in dragged]
    assert not body_F[:, others].any()
