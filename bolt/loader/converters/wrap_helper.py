""" Converts the wrapping obstacles of Scholz2015GeometryPaths """
import numpy as np
import opensim as osim
import warp as wp

from bolt.loader.converters.converted_objects import WrapObstacleData
from bolt.loader.converters.geom_helper import compute_contact_geom_transform
from bolt.loader.converters.osim_types import OSimType
from bolt.loader.converters.physical_frame_helper import get_body_name_of_frame
from bolt.loader.converters.property_helper import extract_vec3
from bolt.types import WrapType


def collect_wrap_type_size(geom: osim.ContactGeometry) -> tuple[WrapType, wp.vec3]:
    """ Returns the obstacle surface type and its size parameters (see WrapType) """
    geom_cls = geom.getConcreteClassName()
    if geom_cls == "ContactSphere":
        return WrapType.SPHERE, wp.vec3(OSimType.ContactSphere.safeDownCast(geom).getRadius(), 0.0, 0.0)
    elif geom_cls == "ContactCylinder":
        return WrapType.CYLINDER, wp.vec3(OSimType.ContactCylinder.safeDownCast(geom).getRadius(), 0.0, 0.0)
    elif geom_cls == "ContactEllipsoid":
        return WrapType.ELLIPSOID, wp.vec3(extract_vec3(OSimType.ContactEllipsoid.safeDownCast(geom).getRadii()))
    elif geom_cls == "ContactTorus":
        torus = OSimType.ContactTorus.safeDownCast(geom)
        return WrapType.TORUS, wp.vec3(torus.getTorusRadius(), torus.getTubeRadius(), 0.0)
    raise NotImplementedError(f"Unsupported wrapping obstacle: {geom_cls} ({geom.getAbsolutePathString()})")


def convert_obstacles(muscle_path: OSimType.ScholzPath) -> list[WrapObstacleData]:
    """ Converts the obstacles of a Scholz2015GeometryPath, recording the path point that precedes each """
    obstacles = []
    path_elements = muscle_path.getPropertyByName("path_elements")
    num_points = 0
    for i in range(path_elements.size()):
        element = path_elements.getValueAsObject(i)
        if OSimType.ScholzPathPoint.safeDownCast(element):
            num_points += 1
            continue
        obstacle = OSimType.ScholzPathObstacle.safeDownCast(element)
        geom = obstacle.getContactGeometry()
        wrap_type, size = collect_wrap_type_size(geom)
        hint = np.array(extract_vec3(obstacle.getContactHint()))
        if np.isnan(hint).all():
            hint_on_surface = hint
        else:
            hint_on_surface = project_to_nearest_point(wrap_type, np.array(size), hint)
        obstacles.append(WrapObstacleData(
            name=geom.getName(),
            body_name=get_body_name_of_frame(geom.getFrame()),
            wrap_type=wrap_type,
            transform=compute_contact_geom_transform(geom),
            size=size,
            segment=num_points - 1,
            hint=wp.vec3(hint),
            hint_on_surface=wp.vec3(hint_on_surface),
        ))
    return obstacles


# --- Implicit surface functions (positive inside, zero on the surface), as in Simbody's ContactGeometry ---
def _surface_value(wrap_type: WrapType, size: np.ndarray, x: np.ndarray) -> float:
    if wrap_type == WrapType.SPHERE:
        return size[0] ** 2 - x @ x
    elif wrap_type == WrapType.CYLINDER:
        return size[0] ** 2 - x[0] ** 2 - x[1] ** 2
    elif wrap_type == WrapType.ELLIPSOID:
        return 1.0 - np.sum((x / size) ** 2)
    rho = np.hypot(x[0], x[1])
    return 1.0 - ((size[0] - rho) ** 2 + x[2] ** 2) / size[1] ** 2


def _surface_gradient(wrap_type: WrapType, size: np.ndarray, x: np.ndarray) -> np.ndarray:
    if wrap_type == WrapType.SPHERE:
        return -2.0 * x
    elif wrap_type == WrapType.CYLINDER:
        return np.array([-2.0 * x[0], -2.0 * x[1], 0.0])
    elif wrap_type == WrapType.ELLIPSOID:
        return -2.0 * x / size ** 2
    rho = np.hypot(x[0], x[1])
    r2 = size[1] ** 2
    return np.array([2.0 * x[0] * (size[0] - rho) / (r2 * rho), 2.0 * x[1] * (size[0] - rho) / (r2 * rho),
                     -2.0 * x[2] / r2])


def _surface_hessian(wrap_type: WrapType, size: np.ndarray, x: np.ndarray) -> np.ndarray:
    if wrap_type == WrapType.SPHERE:
        return -2.0 * np.eye(3)
    elif wrap_type == WrapType.CYLINDER:
        return np.diag([-2.0, -2.0, 0.0])
    elif wrap_type == WrapType.ELLIPSOID:
        return np.diag(-2.0 / size ** 2)
    rho2 = x[0] ** 2 + x[1] ** 2
    r2 = size[1] ** 2
    den = r2 * rho2 * np.sqrt(rho2)
    fxy = -2.0 * size[0] * x[0] * x[1] / den
    return np.array([[2.0 * size[0] * x[1] ** 2 / den - 2.0 / r2, fxy, 0.0],
                     [fxy, 2.0 * size[0] * x[0] ** 2 / den - 2.0 / r2, 0.0],
                     [0.0, 0.0, -2.0 / r2]])


def project_to_nearest_point(wrap_type: WrapType, size: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """ Port of Simbody's ContactGeometry::projectDownhillToNearestPoint: Newton's method for the nearest point """
    ftol, xtol, minlam, max_newton_iterations, max_move = 1e-14, 1e-12, 1e-9, 30, 0.25

    def value(x):
        return _surface_value(wrap_type, size, x)

    def unit_normal(x):
        g = _surface_gradient(wrap_type, size, x)
        return -g / np.linalg.norm(g)

    def curvature(x, d):
        g, H = _surface_gradient(wrap_type, size, x), _surface_hessian(wrap_type, size, x)
        return (d @ H @ d) / (g @ unit_normal(x))

    if abs(value(Q)) <= ftol:
        return Q

    # Arbitrary tangent frame at Q
    nP = unit_normal(Q)
    tP = np.cross(nP, np.eye(3)[np.argmin(np.abs(nP))])
    tP /= np.linalg.norm(tP)
    bP = np.cross(tP, nP)
    # Limit each move to a fraction of the local radius of curvature
    max_k = max(abs(curvature(Q, tP)), abs(curvature(Q, bP)))
    scale = min(1.0 / max_k, 1000.0) if max_k > 0.0 else 1000.0

    def error(x):
        n = unit_normal(x)
        r = Q - x
        return np.array([value(x), r @ (tP - n * (tP @ n)), r @ (bP - n * (bP @ n))])

    x = Q.copy()
    rms = abs(value(x)) / np.sqrt(3.0)
    for _ in range(max_newton_iterations + 1):
        if rms <= ftol:
            break
        g, H = _surface_gradient(wrap_type, size, x), _surface_hessian(wrap_type, size, x)
        g_norm = np.linalg.norm(g)
        n = -g / g_norm
        r = Q - x
        t, b = tP - n * (tP @ n), bP - n * (bP @ n)
        f = np.array([value(x), r @ t, r @ b])
        Nx = -((np.eye(3) - np.outer(n, n)) / g_norm) @ H
        Tx = -Nx * (tP @ n) - np.outer(n, tP @ Nx)
        Bx = -Nx * (bP @ n) - np.outer(n, bP @ Nx)
        J = np.stack([g, -t + r @ Tx, -b + r @ Bx])
        dx = np.linalg.solve(J, f)
        dx_rms = np.linalg.norm(dx) / np.sqrt(3.0)

        rms_old, x_old = rms, x
        lam = min(1.0, max_move * scale / dx_rms)
        while True:
            x = x_old - lam * dx
            rms = np.linalg.norm(error(x)) / np.sqrt(3.0)
            if rms > rms_old and lam > minlam:
                lam /= 2.0
            else:
                break
        if rms > ftol and dx_rms * lam < xtol:
            break
    return x
