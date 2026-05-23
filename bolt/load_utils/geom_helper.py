import opensim as osim
import warp as wp
import numpy as np

from bolt import GeomType
from bolt.load_utils.converted_objects import GeomData, AABB, UserGeomData
from bolt.load_utils.osim_types import OSimType
from bolt.load_utils.physical_frame_helper import get_body_name_of_frame, wp_transform_from_osim_transform
from bolt.load_utils.property_helper import extract_vec3


def collect_geom_type_sizes(geom: osim.ContactGeometry) -> tuple[GeomType, wp.vec3, AABB, float]:
    """
    Returns the converted geometry type, size, aabb (center, size), radius bound
    """
    geom_cls = geom.getConcreteClassName()
    if geom_cls == "ContactSphere":
        geom = osim.ContactSphere.safeDownCast(geom)
        geom_type = GeomType.SPHERE
        radius = geom.getRadius()
        size = wp.vec3(radius, radius, radius)
        aabb = (wp.vec3(0.0), wp.vec3(2.0 * radius, 2.0 * radius, 2.0 * radius))
        rbound = radius
        return geom_type, size, aabb, rbound

    elif geom_cls == "ContactEllipsoid":
        # This is quite the hack, but OpenSim doesn't support capsules, so we just put it in the name
        geom = osim.ContactEllipsoid.safeDownCast(geom)
        geom_name = geom.getName()
        radii = extract_vec3(geom.getRadii())
        if "capsule" not in geom_name.lower():  # parse as ellipsoid
            geom_type = GeomType.ELLIPSOID
            size = wp.vec3(radii)
            aabb = (wp.vec3(0.0), wp.vec3(radii[0] * 2, radii[1] * 2, radii[2] * 2))
            rbound = max(radii)
            return geom_type, size, aabb, rbound
        else:  # parse as capsule
            if radii[0] != radii[2]:
                raise ValueError("Converting the ellipsoid to a capsule, x and z radii should be the same")
            geom_type = GeomType.CAPSULE
            radius, half_height = radii[0], radii[1]
            height = 2.0 * half_height
            size = wp.vec3(radius, half_height, radius)
            aabb = (wp.vec3(0.0), wp.vec3(radii[0] * 2, height + 2 * radius, radii[2] * 2))
            rbound = wp.sqrt(half_height ** 2 + radius ** 2)
            return geom_type, size, aabb, rbound
    else:
        raise NotImplementedError(f"Unsupported contact geometry: {geom_cls}")


def collect_user_geom_aabb(geom: UserGeomData) -> tuple[AABB, float]:
    geom_type = geom.geom_type
    if geom_type == GeomType.SPHERE:
        geom = osim.ContactSphere.safeDownCast(geom)
        radius = geom.size[0]
        aabb = (wp.vec3(0.0), wp.vec3(2.0 * radius, 2.0 * radius, 2.0 * radius))
        rbound = radius
        return aabb, rbound
    elif geom_type == GeomType.CAPSULE:
        radius, half_height = geom.size[0], geom.size[1]
        height = 2.0 * half_height
        aabb = (wp.vec3(0.0), wp.vec3(2.0 * radius, height + 2.0 * radius, 2.0 * radius))
        rbound = wp.sqrt(half_height ** 2 + radius ** 2)
        return aabb, rbound
    else:
        raise NotImplementedError(f"Unsupported contact geometry: {geom_type}")


def _compute_contact_geom_transform(geom: osim.ContactGeometry) -> wp.transform:
    """ Computes the transform of the contact geometry relative to the body frame it is attached to. """
    # Transform of geometry relative to the frame it is attached to
    X_FP = geom.getTransform()
    # Transform of the frame this geometry is attached to, relative to body
    frame = geom.getFrame()
    X_BF = frame.findTransformInBaseFrame()
    # Transform of geometry relative to body
    X_BP = X_BF.compose(X_FP)
    transform = wp_transform_from_osim_transform(X_BP)
    return transform


def convert_contact_geometry(geom: osim.ContactGeometry) -> GeomData:
    geom_name = geom.getName()

    frame = geom.getFrame()
    body_name = get_body_name_of_frame(frame)

    transform = _compute_contact_geom_transform(geom)

    geom_type, size, aabb, rbound = collect_geom_type_sizes(geom)

    return GeomData(
        name=geom_name,
        body_name=body_name,
        geom_type=geom_type,
        transform=transform,
        size=size,
        aabb=aabb,
        rbound=rbound,

        # Defaults
        friction=wp.vec3(0.8, 0.8, 0.0),
        stiffness=(5e6 ** (2 / 3)),
        dissipation=1.0,
        transition_velocity=0.1,
        priority=0
    )


def convert_user_contact_geometry(geom: UserGeomData) -> GeomData:
    aabb, rbound = collect_user_geom_aabb(geom)
    return GeomData(
        name=geom.name,
        body_name=geom.body_name,
        geom_type=geom.geom_type,
        transform=geom.transform,
        size=geom.size,
        aabb=aabb,
        rbound=rbound,
        friction=geom.friction,
        stiffness=geom.stiffness,
        dissipation=geom.dissipation,
        transition_velocity=geom.transition_velocity,
        priority=geom.priority,
    )


def convert_geoms(model: OSimType.Model, include_body_components: bool) -> list[GeomData]:
    """ Returns the all the converted contact geometries in the model """
    collider_data = []

    # Check for contact geometry within ContactGeometrySet
    for contact_geom in model.getContactGeometrySet():
        contact_geom_data = convert_contact_geometry(contact_geom)
        collider_data.append(contact_geom_data)

    # Check for contact geometry within <components> element
    if include_body_components:
        for body in model.getBodyList():
            components = body.getComponentsList()
            contact_geom_components = filter(lambda c: isinstance(c, OSimType.ContactGeometry), components)
            for contact_geom in contact_geom_components:
                contact_geom_data = convert_contact_geometry(contact_geom)
                collider_data.append(contact_geom_data)
    return collider_data


def _upper_trid_index(n: int, i: int, j: int) -> int:
    """Returns index of a_ij = a_ji in upper triangular matrix (including diagonal)."""
    if j < i:
        i, j = j, i
    return (i * (2 * n - i - 1)) // 2 + j


def prepare_contacts(
        geom_types: list[int],
        geom_body_id: list[int],
        body_parent_ids: list[int],
        ngeom: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # precalculated geom pairs
    geom1, geom2 = np.triu_indices(ngeom, k=1)
    nxn_geom_pair = np.stack((geom1, geom2), axis=1)

    # Contact pair id: -1 if not pre-defined, -2 if skipped, id otherwise
    nxn_pairid_contact = -1 * np.ones(len(geom1), dtype=int)

    # filter out parent-child collisions and self-collisions
    geom_bodyid = np.array(geom_body_id)
    body_parentid = np.array(body_parent_ids)
    bodyid1, bodyid2 = geom_bodyid[geom1], geom_bodyid[geom2]
    parentid1, parentid2 = body_parentid[bodyid1], body_parentid[bodyid2]

    self_collision = (bodyid1 == bodyid2)
    # Mask for whether collision is between parent-child
    parent_child_collision = ((bodyid1 == parentid2) & (bodyid1 != 0)) | ((bodyid2 == parentid1) & (bodyid2 != 0))

    nxn_pairid_contact[parent_child_collision | self_collision] = -2
    nxn_pairid_collision = -1 * np.ones(len(geom1), dtype=int)
    include = (nxn_pairid_contact > -2) | (nxn_pairid_collision >= 0)
    nxn_pairid = np.hstack([nxn_pairid_contact.reshape((-1, 1)), nxn_pairid_collision.reshape((-1, 1))])
    nxn_pairid_filtered = nxn_pairid[include]
    nxn_geom_pair_filtered = nxn_geom_pair[include]

    # count contact pair types
    geom_type_pair_count = np.bincount([
        _upper_trid_index(len(GeomType), int(geom_types[geom1[i]]), int(geom_types[geom2[i]]))
        for i in np.arange(len(geom1))
        if nxn_pairid_contact[i] > -2 or nxn_pairid_collision[i] > -1
    ], minlength=len(GeomType) * (len(GeomType) + 1) // 2, )
    return geom_type_pair_count, nxn_geom_pair_filtered, nxn_pairid_filtered


def get_geom_ordering(geom_data_list: list[GeomData]) -> dict[str, int]:
    return {geom_data.name: i for i, geom_data in enumerate(geom_data_list)}


def get_geom_body_name(geom_data_list: list[GeomData]) -> list[str]:
    return [geom_data.body_name for geom_data in geom_data_list]


def get_geom_type(geom_data_list: list[GeomData]) -> list[GeomType]:
    return [geom_data.geom_type for geom_data in geom_data_list]


def get_geom_size(geom_data_list: list[GeomData]) -> list[wp.vec3]:
    return [geom_data.size for geom_data in geom_data_list]


def get_geom_transform(geom_data_list: list[GeomData]) -> list[wp.transform]:
    return [geom_data.transform for geom_data in geom_data_list]


def get_geom_aabb(geom_data_list: list[GeomData]) -> list[AABB]:
    return [geom_data.aabb for geom_data in geom_data_list]


def get_geom_rbound(geom_data_list: list[GeomData]) -> list[float]:
    return [geom_data.rbound for geom_data in geom_data_list]


def get_geom_friction(geom_data_list: list[GeomData]) -> list[wp.vec3]:
    return [geom_data.friction for geom_data in geom_data_list]


def get_geom_stiffness(geom_data_list: list[GeomData]) -> list[float]:
    return [geom_data.stiffness for geom_data in geom_data_list]


def get_geom_dissipation(geom_data_list: list[GeomData]) -> list[float]:
    return [geom_data.dissipation for geom_data in geom_data_list]


def get_geom_transition_velocity(geom_data_list: list[GeomData]) -> list[float]:
    return [geom_data.transition_velocity for geom_data in geom_data_list]


def get_geom_priority(geom_data_list: list[GeomData]) -> list[int]:
    return [geom_data.priority for geom_data in geom_data_list]
