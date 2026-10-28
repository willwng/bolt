"""
Converts the muscles of an OpenSim model whose GeometryPaths wrap over WrapCylinders into Scholz2015GeometryPaths,
Each WrapCylinder becomes an elongated ContactEllipsoid (default) or a ContactCylinder (infinite length)

Usage: python -m bolt.tools.convert_wrap_cylinders <model.osim> <out.osim> [--shape cylinder]
"""
import argparse

import numpy as np
import opensim as osim
from scipy.spatial.transform import Rotation

QUADRANT_DIRECTION = {"x": (1.0, 0.0), "+x": (1.0, 0.0), "-x": (-1.0, 0.0),
                      "y": (0.0, 1.0), "+y": (0.0, 1.0), "-y": (0.0, -1.0)}


def _transform_in_ground(state: osim.State, frame: osim.Frame) -> tuple[np.ndarray, np.ndarray]:
    X = frame.getTransformInGround(state)
    R = np.array([[X.R().get(i, j) for j in range(3)] for i in range(3)])
    return R, X.p().to_numpy()


def _cylinder_frame(state: osim.State, cylinder: osim.WrapCylinder) -> tuple[np.ndarray, np.ndarray]:
    """ Rotation and origin of the cylinder's frame in ground """
    R_GF, p_GF = _transform_in_ground(state, cylinder.getFrame())
    R_FS = Rotation.from_euler("XYZ", cylinder.get_xyz_body_rotation().to_numpy()).as_matrix()
    return R_GF @ R_FS, p_GF + R_GF @ cylinder.get_translation().to_numpy()


def _closest_to_axis(a: np.ndarray, b: np.ndarray) -> tuple[float, float, np.ndarray]:
    """ For the segment ab (in the cylinder frame): distance to the z-axis, segment parameter, and closest point """
    d = b[:2] - a[:2]
    dd = d @ d
    t = 0.0 if dd < 1e-16 else float(np.clip(-(a[:2] @ d) / dd, 0.0, 1.0))
    point = a + t * (b - a)
    return float(np.linalg.norm(point[:2])), t, point


def convert(model_path: str, out_path: str, shape: str = "ellipsoid", length_scale: float = 2.0,
            muscles: set[str] | None = None) -> list[str]:
    """
    Writes the converted model (only converting the given muscles, if any), and returns the converted muscles.
    shape is "ellipsoid" or "cylinder"; an ellipsoid's semi-axis along the cylinder is length_scale * length / 2.
    """
    model = osim.Model(model_path)
    state = model.initSystem()

    def semi_axis(cylinder: osim.WrapCylinder) -> float:
        return length_scale * 0.5 * cylinder.get_length()

    # One obstacle per WrapCylinder, attached to the same frame
    contact_cylinders = {}

    def contact_cylinder(cylinder: osim.WrapCylinder) -> osim.ContactGeometry:
        key = cylinder.getAbsolutePathString()
        if key not in contact_cylinders:
            r = cylinder.get_radius()
            if shape == "cylinder":
                geom = osim.ContactCylinder(r, cylinder.get_translation(), cylinder.get_xyz_body_rotation(),
                                            cylinder.getFrame())
            else:
                geom = osim.ContactEllipsoid(osim.Vec3(r, r, semi_axis(cylinder)), cylinder.get_translation(),
                                             cylinder.get_xyz_body_rotation(), cylinder.getFrame())
            geom.setName(f"{cylinder.getName()}_obstacle")
            contact_cylinders[key] = (geom, cylinder.getFrame().getAbsolutePathString())
        return contact_cylinders[key][0]

    # Decide the new paths in the default pose, before changing the model
    new_paths = {}
    for muscle in model.getMuscles():
        path = osim.GeometryPath.safeDownCast(muscle.getPath())
        if path is None or path.getWrapSet().getSize() == 0 or (muscles is not None and muscle.getName() not in muscles):
            continue
        point_set = path.getPathPointSet()
        points = []
        for i in range(point_set.getSize()):
            point = point_set.get(i)
            if point.getConcreteClassName() != "PathPoint":
                raise ValueError(f"{muscle.getName()}: cannot convert {point.getConcreteClassName()} {point.getName()}")
            point = osim.PathPoint.safeDownCast(point)
            # Copies: replacing the path below frees the old path's points
            points.append((point.getParentFrame().getAbsolutePathString(), point.get_location().to_numpy(),
                           point.getLocationInGround(state).to_numpy()))

        obstacles = []  # (segment, position along it, ContactCylinder, hint)
        wrap_set = path.getWrapSet()
        for w in range(wrap_set.getSize()):
            path_wrap = wrap_set.get(w)
            cylinder = osim.WrapCylinder.safeDownCast(path_wrap.getWrapObject())
            if cylinder is None:
                raise ValueError(f"{muscle.getName()}: cannot convert {path_wrap.getWrapObject().getConcreteClassName()}")
            # Legacy ranges are 1-based and inclusive, -1 for the first/last point
            first = 0 if path_wrap.getStartPoint() < 1 else path_wrap.getStartPoint() - 1
            last = len(points) - 1 if path_wrap.getEndPoint() < 1 else path_wrap.getEndPoint() - 1
            R_GS, p_GS = _cylinder_frame(state, cylinder)
            to_S = [R_GS.T @ (p_G - p_GS) for _, _, p_G in points]
            candidates = [(*_closest_to_axis(to_S[k], to_S[k + 1]), k) for k in range(first, max(last, first + 1))]
            dist, t, closest, segment = min(candidates, key=lambda c: c[0])
            direction = QUADRANT_DIRECTION.get(cylinder.get_quadrant().strip(), None)
            if direction is None:  # "all": the side the straight line passes on
                direction = closest[:2] / max(dist, 1e-12)
            half_length = 0.5 * cylinder.get_length()
            z = float(np.clip(closest[2], -half_length, half_length))
            radius = cylinder.get_radius()
            if shape != "cylinder":  # the ellipsoid's cross-section at z
                radius *= np.sqrt(1.0 - (z / semi_axis(cylinder)) ** 2)
            hint = osim.Vec3(radius * direction[0], radius * direction[1], z)
            if dist >= radius:  # the straight line misses: start lifted, and touch down once the line hits it
                hint = osim.Vec3(np.nan)
            obstacles.append((segment, t, contact_cylinder(cylinder), hint))
        obstacles.sort(key=lambda o: (o[0], o[1]))
        new_paths[muscle.getName()] = (points, obstacles)

    def physical_frame(frame_path: str) -> osim.PhysicalFrame:
        return osim.PhysicalFrame.safeDownCast(model.updComponent(frame_path))

    for geom, frame_path in contact_cylinders.values():
        physical_frame(frame_path).addComponent(geom)
    for name, (points, obstacles) in new_paths.items():
        muscle = model.updMuscles().get(name)
        muscle.set_path(osim.Scholz2015GeometryPath())
        path = osim.Scholz2015GeometryPath.safeDownCast(muscle.updPath())
        path.setName("path")
        for k, (frame_path, location, _) in enumerate(points):
            path.appendPathPoint(physical_frame(frame_path), osim.Vec3(*location))
            for segment, _, geom, hint in obstacles:
                if segment == k:
                    path.appendObstacle(geom, hint)
    model.finalizeConnections()
    model.initSystem()
    model.printToXML(out_path)
    return sorted(new_paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("--shape", choices=["ellipsoid", "cylinder"], default="ellipsoid")
    parser.add_argument("--length-scale", type=float, default=2.0,
                        help="ellipsoid semi-axis along the cylinder, relative to half the cylinder's length")
    args = parser.parse_args()
    converted = convert(args.model, args.out, args.shape, args.length_scale)
    print(f"Converted {len(converted)} muscles: {', '.join(converted)}")


if __name__ == "__main__":
    main()
