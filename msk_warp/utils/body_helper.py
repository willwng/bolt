import numpy as np
import warp as wp
from .converted_objects import BodyData, GROUND
from .osim_types import OSimType
from .property_helper import extract_vec3


def _shift_inertia_from_com_to_body(mass: float, mass_center: np.ndarray, inertia_in_com: np.ndarray) -> np.ndarray:
    """ Shift inertia from COM frame to body frame parallel axis theorem """
    to_body_center = -mass_center

    mp = to_body_center * mass
    mxx = mp[0] * mass_center[0]
    myy = mp[1] * mass_center[1]
    mzz = mp[2] * mass_center[2]
    nmx = -mp[0]
    nmy = -mp[1]
    point_mass = np.array([[myy + mzz, nmx * mass_center[1], nmx * mass_center[2]],
                           [nmx * mass_center[1], mxx + mzz, nmy * mass_center[2]],
                           [nmx * mass_center[2], nmy * mass_center[2], mxx + myy]])
    inertia_in_B = inertia_in_com + point_mass
    return inertia_in_B


def convert_body(body: OSimType.Body) -> BodyData:
    """ Converts an OpenSim body to the relevant BodyData """
    name = body.getName()
    mass = float(body.getMass())
    mass_com = body.getMassCenter()
    mass_center = np.array(extract_vec3(mass_com))

    # Model provides the inertia in COM frame
    inertia = body.getInertia()
    moments, products = inertia.getMoments(), inertia.getProducts()
    ixx, iyy, izz = extract_vec3(moments)
    ixy, ixz, iyz = extract_vec3(products)
    inertia_in_com = np.array([[ixx, ixy, ixz, ], [ixy, iyy, iyz, ], [ixz, iyz, izz, ], ])
    # Shift inertia from COM frame to body frame
    inertia_in_B = _shift_inertia_from_com_to_body(mass, mass_center, inertia_in_com)
    # Convert to unit inertia
    unit_inertia_in_B = inertia_in_B / mass
    unit_inertia_OB_B = wp.mat33(
        unit_inertia_in_B[0, 0], unit_inertia_in_B[0, 1], unit_inertia_in_B[0, 2],
        unit_inertia_in_B[1, 0], unit_inertia_in_B[1, 1], unit_inertia_in_B[1, 2],
        unit_inertia_in_B[2, 0], unit_inertia_in_B[2, 1], unit_inertia_in_B[2, 2],
    )

    return BodyData(name=name, mass=mass, mass_center=wp.vec3(*mass_center), unit_inertia_OB_B=unit_inertia_OB_B)
