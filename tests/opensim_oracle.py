""" OpenSim ground-truth helpers used for validation testing """

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np
import opensim as osim
from scipy.spatial.transform import Rotation


def coordinate_names(model: osim.Model) -> list[str]:
    return [c.getName() for c in model.getCoordinateSet()]


def random_states(model: osim.Model, n: int, seed: int, speed_scale: float = 1.0, max_rotation: float = np.inf):
    """
    Random (q, u) in coordinate-set order, with q sampled inside each coordinate's range.
    max_rotation additionally clips rotational coordinates, e.g. to stay away from the gimbal-lock
    singularity (middle angle of +/- pi/2) of 3-axis joints, where float32 dynamics is ill-conditioned.
    """
    rng = np.random.default_rng(seed)
    cs = model.getCoordinateSet()
    lo = np.array([c.getRangeMin() for c in cs])
    hi = np.array([c.getRangeMax() for c in cs])
    rotational = np.array([c.getMotionType() == osim.Coordinate.Rotational for c in cs])
    lo = np.where(rotational, np.maximum(lo, -max_rotation), lo)
    hi = np.where(rotational, np.minimum(hi, max_rotation), hi)
    q = rng.uniform(lo, hi, size=(n, len(lo)))
    u = rng.uniform(-1.0, 1.0, size=(n, len(lo))) * speed_scale
    return q, u


def set_state(model: osim.Model, state: osim.State, q, u):
    """ Sets coordinate values/speeds (coordinate-set order) without enforcing constraints """
    for i, c in enumerate(model.getCoordinateSet()):
        c.setValue(state, float(q[i]), False)
        c.setSpeedValue(state, float(u[i]))


def _root_free_joint(model: osim.Model):
    """ The ground->root CustomJoint, which Bolt converts to a quaternion free joint """
    for joint in model.getJointSet():
        parent = joint.getParentFrame().findBaseFrame().getName()
        if parent == "ground" and joint.getConcreteClassName() == "CustomJoint":
            return joint
    return None


def root_free_coordinates(model: osim.Model, load_result) -> set[str]:
    """ Coordinates of the root joint if Bolt made it a free joint (their speeds are not OpenSim's u) """
    root = _root_free_joint(model) if load_result.root_free else None
    if root is None:
        return set()
    return {root.get_coordinates(i).getName() for i in range(root.numCoordinates())}


def _mat33(R: osim.Rotation) -> np.ndarray:
    m = R.asMat33()
    return np.array([[m.get(i, j) for j in range(3)] for i in range(3)])


def _spatial(v) -> np.ndarray:
    """ SimTK SpatialVec -> (angular, linear) """
    return np.concatenate([v.get(0).to_numpy(), v.get(1).to_numpy()])


def bolt_state(model: osim.Model, state: osim.State, load_result, q, u):
    """
    Maps an OpenSim (q, u) to Bolt's (qpos, qvel). relevant for free root coordinates
    """
    m = load_result.model
    qpos, qvel = np.zeros(m.nq), np.zeros(m.nv)

    root_coords = root_free_coordinates(model, load_result)

    for i, name in enumerate(coordinate_names(model)):
        if name in root_coords:
            continue
        qpos[load_result.qpos_id_lookup[name]] = q[i]
        qvel[load_result.dof_id_lookup[name]] = u[i]

    if root_coords:
        root = _root_free_joint(model)
        set_state(model, state, q, u)
        model.realizeVelocity(state)
        child = root.getChildFrame()
        X = child.getTransformInGround(state)
        V = _spatial(child.getVelocityInGround(state))
        qadr = min(load_result.qpos_id_lookup[c] for c in root_coords)
        dadr = min(load_result.dof_id_lookup[c] for c in root_coords)
        qpos[qadr:qadr + 4] = Rotation.from_matrix(_mat33(X.R())).as_quat()  # xyzw
        qpos[qadr + 4:qadr + 7] = X.p().to_numpy()
        qvel[dadr:dadr + 6] = V
    return qpos, qvel


def body_kinematics(model: osim.Model, state: osim.State, q, u):
    """ Per body name: (position, rotation matrix, spatial velocity) in ground """
    set_state(model, state, q, u)
    model.realizeVelocity(state)
    out = {}
    for body in model.getBodySet():
        X = body.getTransformInGround(state)
        out[body.getName()] = (
            X.p().to_numpy(),
            _mat33(X.R()),
            _spatial(body.getVelocityInGround(state)),
        )
    return out


def forward_dynamics(model: osim.Model, state: osim.State, q, u):
    """ Per body name spatial acceleration in ground, and per coordinate udot """
    set_state(model, state, q, u)
    model.realizeAcceleration(state)
    body_acc = {b.getName(): _spatial(b.getAccelerationInGround(state)) for b in model.getBodySet()}
    udot = {c.getName(): c.getAccelerationValue(state) for c in model.getCoordinateSet()}
    return body_acc, udot


def skeleton_only_model(model_path: str, out_dir: str, gravity=None) -> str:
    """
    Writes a copy of the model with all forces, contact geometry and markers removed,
    so forward dynamics reflects gravity + inertia only. Returns the new path.
    """
    model = osim.Model(model_path)
    if gravity is not None:
        model.setGravity(osim.Vec3(*gravity))
    model.updForceSet().clearAndDestroy()
    model.updContactGeometrySet().clearAndDestroy()
    model.updMarkerSet().clearAndDestroy()
    model.initSystem()
    out_path = os.path.join(out_dir, os.path.basename(model_path))
    model.printToXML(out_path)
    return out_path


def wrapped_muscles(model_path: str) -> set[str]:
    """
    Names of muscles whose path wraps (GeometryPath wrap objects or Scholz2015 obstacles).
    Bolt point paths do not model wrapping.
    """
    wrapped = set()
    for el in ET.parse(model_path).getroot().iter():
        if not el.tag.endswith("Muscle"):
            continue
        for path in el:
            if path.tag == "GeometryPath" and path.findall("./PathWrapSet/objects/*"):
                wrapped.add(el.get("name"))
            elif path.tag == "Scholz2015GeometryPath" and path.findall(".//Scholz2015GeometryPathObstacle"):
                wrapped.add(el.get("name"))
    return wrapped


def function_based_path_model(model_path: str, paths_xml: str) -> osim.Model:
    """ Loads a model and replaces its muscle paths with function-based (polynomial) paths """
    processor = osim.ModelProcessor(model_path)
    processor.append(osim.ModOpReplacePathsWithFunctionBasedPaths(paths_xml))
    model = processor.process()
    model.initSystem()
    return model


def _muscles(model: osim.Model) -> dict[str, osim.Muscle]:
    fs = model.getForceSet()
    muscles = {}
    for i in range(fs.getSize()):
        muscle = osim.Muscle.safeDownCast(fs.get(i))
        if muscle is not None:
            muscles[muscle.getName()] = muscle
    return muscles


def function_based_path_coordinates(model: osim.Model) -> dict[str, list[str]]:
    """ Muscle name -> coordinate names for muscles that use a FunctionBasedPath """
    out = {}
    for name, muscle in _muscles(model).items():
        path = osim.FunctionBasedPath.safeDownCast(muscle.updPath())
        if path is not None:
            out[name] = [c.split("/")[-1] for c in path.getCoordinatePaths()]
    return out


def muscle_paths(model: osim.Model, state: osim.State, q, u, moment_arm_coords=None):
    """
    Per muscle name: (length, lengthening speed). If moment_arm_coords (muscle -> coordinate names)
    is given, also returns muscle -> {coordinate: moment arm}.
    """
    set_state(model, state, q, u)
    model.realizeVelocity(state)
    muscles = _muscles(model)
    paths = {n: (m.getLength(state), m.getLengtheningSpeed(state)) for n, m in muscles.items()}
    if moment_arm_coords is None:
        return paths

    cs = model.getCoordinateSet()
    moment_arms = {
        n: {c: muscles[n].computeMomentArm(state, cs.get(c)) for c in coords}
        for n, coords in moment_arm_coords.items()
    }
    return paths, moment_arms


def body_com_kinematics(model: osim.Model, state: osim.State, q, u, body_names) -> dict:
    """ Per body name: (center-of-mass offset from the body origin, center-of-mass velocity), in ground """
    set_state(model, state, q, u)
    model.realizeVelocity(state)
    out = {}
    for name in body_names:
        body = model.getBodySet().get(name)
        com_B = body.getMassCenter()
        offset_G = body.expressVectorInGround(state, com_B).to_numpy()
        velocity_G = body.findStationVelocityInGround(state, com_B).to_numpy()
        out[name] = (offset_G, velocity_G)
    return out


def umberger_probe_model(model_path: str, options: dict, specific_tension: float, density: float,
                         slow_twitch_ratio: float):
    """
    Loads the model with an Umberger2010MuscleMetabolicsProbe on every muscle (basal rate off, per-muscle outputs).
    options: probe property name -> value. Returns (model, state, probe).
    """
    model = osim.Model(model_path)
    probe = osim.Umberger2010MuscleMetabolicsProbe()
    probe.setName("oracle_metabolics")
    for name, value in options.items():
        getattr(probe, f"set_{name}")(value)
    probe.set_basal_rate_on(False)
    probe.set_report_total_metabolics_only(False)
    for name in _muscles(model):
        probe.addMuscle(name, slow_twitch_ratio)
    model.addProbe(probe)
    model.initSystem()  # the per-muscle setters need the probe connected
    for name in _muscles(model):
        probe.setSpecificTension(name, specific_tension)
        probe.setDensity(name, density)
    state = model.initSystem()
    return model, state, probe


def umberger_metabolics(model: osim.Model, state: osim.State, probe, q, u, activations: dict, excitations: dict):
    """
    Sets the state (coordinates, muscle activations with equilibrated fibers, excitations) and returns
    (per-muscle metabolic power from the probe, per-muscle probe inputs).
    """
    set_state(model, state, q, u)
    muscles = _muscles(model)
    for name, muscle in muscles.items():
        muscle.setActivation(state, float(activations[name]))
    model.equilibrateMuscles(state)

    model.realizeVelocity(state)
    controls = model.updControls(state)
    for name, muscle in muscles.items():
        muscle.setControls(osim.Vector(1, float(excitations[name])), controls)
    model.setControls(state, controls)
    model.realizeDynamics(state)

    labels = probe.getProbeOutputLabels()
    labels = [labels.get(i) for i in range(labels.getSize())]
    values = probe.getProbeOutputs(state).to_numpy()
    power = {label.replace("oracle_metabolics_", ""): v for label, v in zip(labels, values)}
    inputs = {
        name: dict(
            activation=muscle.getActivation(state),
            excitation=muscle.getExcitation(state),
            norm_fiber_length=muscle.getNormalizedFiberLength(state),
            fiber_velocity=muscle.getFiberVelocity(state),
            active_fiber_force=muscle.getActiveFiberForce(state),
            active_force_length_multiplier=muscle.getActiveForceLengthMultiplier(state),
            optimal_fiber_length=muscle.getOptimalFiberLength(),
            max_isometric_force=muscle.getMaxIsometricForce(),
            max_contraction_velocity=muscle.getMaxContractionVelocity(),
        )
        for name, muscle in muscles.items()
    }
    return power, inputs
