"""
Ensures folded wrapping paths are not considered smooth

A path folds when a straight line runs backward along a curve, so that the cable doubles back on the obstacle. The
normal and binormal path errors vanish for a reversed line, so the solver used to converge to such a fold and report it
as smooth.
"""

from __future__ import annotations

import numpy as np
import opensim as osim
import pytest
from scipy.spatial.transform import Rotation

import bolt
from bolt._src.muscle import wrap_path
from bolt._src.pipeline import forward
from bolt.loader.converters import wrap_helper
from bolt.types import WrapStatus
from bolt.types import WrapType

MAX_ITERATIONS = 100  # as the Simbody tests
LENGTH_TOL = 1e-4  # m
POINT_TOL = 1e-4  # m


def _muscle(model: osim.Model, name: str, elements: list):
    """ A muscle along path points (frame, location) and obstacles (geometry, contact hint) """
    muscle = osim.Millard2012EquilibriumMuscle(name, 1000.0, 0.2, 0.3, 0.0)
    muscle.upd_FiberForceLengthCurve().setOptionalProperties(0.2, 2.0 / 0.7, 0.75)  # the loader reads them
    muscle.set_path(osim.Scholz2015GeometryPath())
    model.addForce(muscle)
    path = osim.Scholz2015GeometryPath.safeDownCast(muscle.updPath())
    for element, vec in elements:
        if isinstance(element, osim.ContactGeometry):
            path.appendObstacle(element, osim.Vec3(*vec))
        else:
            path.appendPathPoint(element, osim.Vec3(*vec))


def _body(model: osim.Model, name: str) -> osim.Body:
    body = osim.Body(name, 1.0, osim.Vec3(0), osim.Inertia(0.01, 0.01, 0.01))
    model.addBody(body)
    return body


def _cable_model(path: str, origin, geometry: str, size, hint, start) -> str:
    """
    A cable from a point on ground, over an obstacle at the ground origin (a cylinder along z, or an ellipsoid), to a
    termination point that three sliders move along x, y and z (coordinates tx, ty, tz), by default at start (OpenSim
    requires the path points to be outside the obstacle)
    """
    model = osim.Model()
    model.setName("cable")
    model.setGravity(osim.Vec3(0))
    ground = model.getGround()
    parent = ground
    for axis, orientation, default in (("x", (0, 0, 0), start[0]), ("y", (0, 0, np.pi / 2), start[1]),
                                       ("z", (0, -np.pi / 2, 0), start[2])):
        body = _body(model, f"slider_{axis}")
        joint = osim.SliderJoint(f"slide_{axis}", parent, osim.Vec3(0), osim.Vec3(*orientation), body, osim.Vec3(0),
                                 osim.Vec3(*orientation))
        coord = joint.upd_coordinates(0)
        coord.setName(f"t{axis}")
        coord.setRangeMin(-100.0)
        coord.setRangeMax(100.0)
        coord.setDefaultValue(float(default))
        model.addJoint(joint)
        parent = body
    if geometry == "cylinder":
        obstacle = osim.ContactCylinder(size, osim.Vec3(0), osim.Vec3(0), ground)
    else:
        obstacle = osim.ContactEllipsoid(osim.Vec3(*size), osim.Vec3(0), osim.Vec3(0), ground)
    obstacle.setName("obstacle")
    ground.addComponent(obstacle)
    _muscle(model, "cable", [(ground, origin), (obstacle, hint), (parent, (0, 0, 0))])
    model.finalizeConnections()
    model.initSystem()
    model.printToXML(path)
    return path


def _pendulum_model(path: str, hint) -> str:
    """
    Simbody's PendulumSystem: a base welded to ground at (0, 1.5, 0), a head pinned 1 below it, and a cable from the head
    to the base over an ellipsoid on the base (coordinate angle)
    """
    model = osim.Model()
    model.setName("pendulum")
    model.setGravity(osim.Vec3(0))
    base, head = _body(model, "base"), _body(model, "head")
    model.addJoint(osim.WeldJoint("weld", model.getGround(), osim.Vec3(0, 1.5, 0), osim.Vec3(0), base, osim.Vec3(0),
                                  osim.Vec3(0)))
    pin = osim.PinJoint("pin", base, osim.Vec3(0), osim.Vec3(0), head, osim.Vec3(0, 1, 0), osim.Vec3(0))
    pin.upd_coordinates(0).setName("angle")
    pin.upd_coordinates(0).setRangeMin(-10.0)
    pin.upd_coordinates(0).setRangeMax(10.0)
    model.addJoint(pin)
    obstacle = osim.ContactEllipsoid(osim.Vec3(0.1, 0.1, 0.5), osim.Vec3(0, -0.5, 0), osim.Vec3(0), base)
    obstacle.setName("obstacle")
    base.addComponent(obstacle)
    _muscle(model, "cable", [(head, (0, 0, 0)), (obstacle, hint), (base, (0, 0, 0))])
    model.finalizeConnections()
    model.initSystem()
    model.printToXML(path)
    return path


class Cable:
    """ A loaded cable model, solved at given coordinates as a realization would """

    def __init__(self, path: str):
        self.lr = bolt.load_model(model_path=path, n_worlds=1, integrator=bolt.IntegratorType.EULER_FIXED,
                                  requires_visuals=False)
        self.m, self.d = self.lr.model, self.lr.data
        self.m.opt.wrap_max_iterations = MAX_ITERATIONS
        self.mu = self.lr.muscle_id_lookup["cable"]
        self.o = int(self.m.muscle_wrap_adr.numpy()[self.mu])

    def _set(self, coords: dict):
        qpos = self.d.qpos.numpy()
        for name, value in coords.items():
            qpos[0, self.lr.qpos_id_lookup[name]] = value
        self.d.qpos.assign(qpos)

    def reset(self, **coords):
        """ Restarts the path from the contact hint at these coordinates """
        self._set(coords)
        self.d.world_reset.fill_(True)
        bolt.reset(self.m, self.d)
        self.solve()

    def solve(self, **coords):
        self._set(coords)
        forward.realize_position(self.m, self.d)
        wrap_path.muscle_wrap_path(self.m, self.d)

    def length(self) -> float:
        return float(self.d.muscle_length.numpy()[0, self.mu])

    def converged(self) -> bool:
        return float(self.d.muscle_wrap_smoothness.numpy()[0, self.mu]) <= self.m.opt.wrap_smoothness_tol

    def in_contact(self) -> bool:
        return int(self.d.wrap_geodesic.numpy()[0, self.o]["status"]) == WrapStatus.IN_CONTACT

    def frames_G(self):
        """ The curve's initial and final contact points and tangents, in ground """
        g = self.d.wrap_geodesic.numpy()[0, self.o]
        X_GB = self.d.mob_X_GB.numpy()[0, int(self.m.wrap_bodyid.numpy()[self.o])]
        X_BS = self.m.wrap_X_BS.numpy()[self.o]
        R_GB, R_BS = Rotation.from_quat(X_GB[3:7]), Rotation.from_quat(X_BS[3:7])  # xyzw, as Warp
        R_GS, p_GS = R_GB * R_BS, X_GB[:3] + R_GB.apply(X_BS[:3])
        point = lambda p: p_GS + R_GS.apply(p)
        return point(g["p_P"]), R_GS.apply(g["t_P"]), point(g["p_Q"]), R_GS.apply(g["t_Q"])

    def site_G(self, k: int) -> np.ndarray:
        return self.d.site_pos_G.numpy()[0, int(self.m.muscle_pts_adr.numpy()[self.mu]) + k]


def _termination(point) -> dict:
    return dict(tx=float(point[0]), ty=float(point[1]), tz=float(point[2]))


def _assert_forward(cable: Cable, origin, termination):
    """ The curve leaves the line from the origin forward, and heads forward to the termination: no fold """
    p_P, t_P, p_Q, t_Q = cable.frames_G()
    assert np.dot(t_P, p_P - np.asarray(origin)) > 0.0, "the initial wrap tangent should point away from the origin"
    assert np.dot(t_Q, np.asarray(termination) - p_Q) > 0.0, "the final wrap tangent should point toward the termination"


@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("wrap_folds")


# --- testUseWarmStart ---
_PENDULUM_RADIUS = 0.1
_PENDULUM_LENGTH = 2.0 * np.sqrt(0.25 - _PENDULUM_RADIUS ** 2) + 2.0 * _PENDULUM_RADIUS * np.arcsin(_PENDULUM_RADIUS / 0.5)


def _pendulum_initial_point(tmp, warm_start: bool, default_angle: float, hint_z: float) -> np.ndarray:
    cable = Cable(_pendulum_model(str(tmp / f"pendulum_{hint_z:g}.osim"), (0.01, 0.2, hint_z)))
    cable.reset(angle=default_angle)
    origin, termination = cable.site_G(0), cable.site_G(1)
    assert cable.converged()
    _assert_forward(cable, origin, termination)
    if warm_start:
        cable.solve(angle=2.0 * np.pi)
    else:
        cable.reset(angle=2.0 * np.pi)
    origin = cable.site_G(0)
    assert cable.converged(), "the pendulum cable path should converge"
    _assert_forward(cable, origin, termination)
    assert abs(cable.length() - _PENDULUM_LENGTH) < LENGTH_TOL, "the cable should follow the short, forward-facing wrap"
    return cable.frames_G()[0]


def test_use_warm_start(tmp):
    """ Swinging the pendulum to a full turn: the warm-started path must not converge to a fold (Simbody issue #877) """
    # Warm start: similar paths on opposite sides of the ellipsoid
    first = _pendulum_initial_point(tmp, True, 0.05 * np.pi, 0.0)
    second = _pendulum_initial_point(tmp, True, 1.95 * np.pi, 0.0)
    assert abs(first[0] + second[0]) < POINT_TOL
    assert abs(first[1] - second[1]) < POINT_TOL
    assert abs(first[2]) < POINT_TOL and abs(second[2]) < POINT_TOL
    # Restarting from the contact hint: the same path
    first = _pendulum_initial_point(tmp, False, 0.05 * np.pi, 0.0)
    second = _pendulum_initial_point(tmp, False, 1.95 * np.pi, 0.0)
    assert np.linalg.norm(first - second) < POINT_TOL
    # A tiny out-of-plane perturbation must not send the path toward the semicircular wrap with a reversed tangent
    for warm_start in (False, True):
        assert abs(_pendulum_initial_point(tmp, warm_start, 1.95 * np.pi, 1e-8)[2]) < POINT_TOL


def test_initial_point_is_nearest_hint_projection(tmp):
    """ With no solver iterations, the curve starts at the hint's nearest point on the surface """
    hint = np.array([0.01, 0.2, 0.1])
    cable = Cable(_pendulum_model(str(tmp / "pendulum_hint.osim"), hint))
    cable.m.opt.wrap_max_iterations = 0
    cable.reset(angle=0.05 * np.pi)
    expected = wrap_helper.project_to_nearest_point(WrapType.ELLIPSOID, np.array([0.1, 0.1, 0.5]), hint) + (0, 1, 0)
    assert np.linalg.norm(cable.frames_G()[0] - expected) < POINT_TOL


def test_warm_start_normal_restart_direction(tmp):
    """
    A fold that can't be undone: no part of the folded curve heads toward the termination point, and no zero-length
    curve at the initial contact point can align with both lines. The curve is left as is, and not reported as smooth
    """
    origin = np.array([-2.0, 1.0, 0.0])
    cable = Cable(_cable_model(str(tmp / "normal_restart.osim"), origin, "ellipsoid", (1, 1, 2), (0, 1, 0),
                                (1, -2, 0)))
    cable.reset(**_termination((1, -2, 0)))
    p_P, t_P, p_Q, _ = cable.frames_G()
    assert np.linalg.norm(p_P - (0, 1, 0)) < 1e-3, "the cable should start its wrap at the top of the ellipsoid"
    assert np.linalg.norm(p_Q - (1, 0, 0)) < 1e-3, "the cable should end its wrap at the side of the ellipsoid"

    # Reverse the outgoing tangent, with the termination minus the origin along the surface normal at P
    g = cable.d.wrap_geodesic.numpy()[0, cable.o]
    cable.m.opt.wrap_max_iterations = 0
    cable.solve(**_termination(origin + 2.0 * np.asarray(g["n_P"])))
    assert np.isfinite(cable.length())
    assert cable.in_contact(), "the folded curve should be left in contact with the obstacle"
    assert np.dot(cable.frames_G()[1], (1, 0, 0)) > 0.99, "the curve should keep its finite initial tangent"
    assert not cable.converged(), "a folded path must not be reported as smooth"


# --- testFoldRecoveryAfterLargeMotion ---
_HOOK_ORIGIN = np.array([-0.5, 2.0, 0.0])


def _hooked_length(termination) -> float:
    """ Leaving the origin along the left of the unit cylinder, wrapping counterclockwise under it, to the termination """
    def tangent_point_angle(p, side):
        return np.arctan2(p[1], p[0]) + side * np.arccos(1.0 / np.linalg.norm(p[:2]))
    wrap = np.fmod(tangent_point_angle(termination, -1.0) - tangent_point_angle(_HOOK_ORIGIN, 1.0), 2.0 * np.pi)
    wrap += 2.0 * np.pi if wrap < 0.0 else 0.0
    return np.sqrt(_HOOK_ORIGIN @ _HOOK_ORIGIN - 1.0) + wrap + np.sqrt(np.dot(termination, termination) - 1.0)


@pytest.mark.parametrize("new_termination", [(2.0, 0.0, 0.0), (1.5, -1.5, 0.0), (0.0, -2.5, 0.0)])
def test_fold_recovery_after_large_motion(tmp, new_termination):
    """
    A cable hooked more than 180 degrees under a cylinder (a valid, locally shortest path) has its termination point
    moved far in one step. The previous curve then wraps too far, and folds back: the solver must find the hooked path
    """
    # Hook the cable by moving the termination point counterclockwise around the cylinder, from below on the left
    radius = np.linalg.norm([0.5, 2.0])
    angles = np.radians(np.arange(240.0, 360.0 + np.degrees(np.arctan2(2.0, 0.5)), 2.0))
    start = radius * np.array([np.cos(angles[0]), np.sin(angles[0]), 0.0])
    cable = Cable(_cable_model(str(tmp / "hook.osim"), _HOOK_ORIGIN, "cylinder", 1.0, (-1, 0, 0), start))
    cable.reset(**_termination(radius * np.array([np.cos(angles[0]), np.sin(angles[0]), 0.0])))
    for angle in angles[1:]:
        cable.solve(**_termination(radius * np.array([np.cos(angle), np.sin(angle), 0.0])))
    hooked = np.array([0.5, 2.0, 0.0])
    cable.solve(**_termination(hooked))
    assert cable.converged() and cable.in_contact()
    assert abs(cable.length() - _hooked_length(hooked)) < LENGTH_TOL, "the cable should be hooked under the cylinder"

    new_termination = np.array(new_termination)
    cable.solve(**_termination(new_termination))
    assert cable.converged(), "the hooked cable path should converge"
    assert cable.in_contact(), "the hooked cable should remain in contact with the cylinder"
    _assert_forward(cable, _HOOK_ORIGIN, new_termination)
    assert abs(cable.length() - _hooked_length(new_termination)) < LENGTH_TOL, "the cable should follow the hooked path"


def test_fold_recovery_keeps_windings(tmp):
    """ A cable wound about 3.2 pi around a cylinder, unwound by 0.9 pi in one step, must keep its remaining windings """
    origin = np.array([1.0, -2.0, 0.0])
    termination = lambda angle: 2.0 * np.array([np.cos(angle), np.sin(angle), 0.0])
    angle = 0.4 * np.pi
    cable = Cable(_cable_model(str(tmp / "windings.osim"), origin, "cylinder", 1.0, (1, 0, 0), termination(angle)))
    cable.reset(**_termination(termination(angle)))
    while angle < 3.5 * np.pi:
        angle += np.radians(5.0)
        cable.solve(**_termination(termination(angle)))
        assert cable.converged(), "the cable should converge while winding around the cylinder"
    wound = cable.length()
    cable.solve(**_termination(termination(angle - 0.9 * np.pi)))
    assert cable.converged(), "the unwound cable path should converge"
    assert abs(cable.length() - (wound - 0.9 * np.pi)) < LENGTH_TOL, "undoing the fold should keep the windings"


def test_exact_fold(tmp):
    """
    A zero-length curve whose lines run exactly opposite along its tangent: a fold with zero path error that can't be
    undone by a restart. The solver must give a finite path, and either lift off or report it as not smooth
    """
    origin = np.array([-2.0, 1.0, 0.0])
    cable = Cable(_cable_model(str(tmp / "exact_fold.osim"), origin, "cylinder", 1.0, (0, 1, 0), (-1, 1, 0)))
    cable.m.opt.wrap_max_iterations = 20
    cable.reset(**_termination((-1, 1, 0)))
    assert np.isfinite(cable.length())
    if cable.converged():
        assert not cable.in_contact() and abs(cable.length() - 1.0) < LENGTH_TOL


def test_fold_restart_with_steep_line(tmp):
    """
    A contact hint whose initial tangent points backward along the steep incoming line: the fold is undone by a
    restart aligned with both lines, after which the cable lifts off to the straight line
    """
    origin, termination = np.array([11.0, -1.0, 0.0]), np.array([1.5, -1.2, 1.732])
    cable = Cable(_cable_model(str(tmp / "steep.osim"), origin, "cylinder", 1.0, (1, 0, 0), termination))
    cable.reset(**_termination(termination))
    assert cable.converged()
    assert not cable.in_contact(), "the cable should lift off the cylinder"
    assert abs(cable.length() - np.linalg.norm(termination - origin)) < LENGTH_TOL


def test_fold_recovery_keeps_helix_windings(tmp):
    """
    A steep helix (pitch 3) wound 2.5 turns around a cylinder, with its termination point moved further along and
    around it in one step. A forward-facing stretch of the folded curve spans more than a turn: only the part beyond
    its nearest local alignment maximum may be trimmed, and the cable must keep two full turns
    """
    pitch, wind = 3.0, 5.0 * np.pi
    helix = lambda t: np.array([np.cos(t), np.sin(t), pitch * t])
    tangent = lambda t: np.array([-np.sin(t), np.cos(t), pitch]) / np.sqrt(1.0 + pitch ** 2)
    line = 2.0 * np.sqrt(1.0 + pitch ** 2)
    origin = helix(0.0) - line * tangent(0.0)
    t = 0.3
    cable = Cable(_cable_model(str(tmp / "helix.osim"), origin, "cylinder", 1.0, (1, 0, 0.1),
                               helix(t) + line * tangent(t)))
    cable.reset(**_termination(helix(t) + line * tangent(t)))
    while t < wind:
        t = min(wind, t + np.radians(3.0))
        cable.solve(**_termination(helix(t) + line * tangent(t)))
        assert cable.converged(), "the cable should converge while winding around the cylinder"

    new_termination = np.array([-1.0, 8.0, 15.0 * np.pi])
    cable.solve(**_termination(new_termination))
    assert cable.converged(), "the cable path should converge after the jump"
    assert cable.in_contact(), "the cable should remain in contact with the cylinder"
    # On a cylinder the curve is a helix, which winds by its arc length projected onto the plane normal to the axis
    p_P, _, p_Q, _ = cable.frames_G()
    arc = cable.length() - np.linalg.norm(p_P - origin) - np.linalg.norm(new_termination - p_Q)
    wind_angle = np.sqrt(max(0.0, arc ** 2 - (p_Q[2] - p_P[2]) ** 2))
    assert 4.0 * np.pi < wind_angle < 5.0 * np.pi, "the cable should keep two windings around the cylinder"
