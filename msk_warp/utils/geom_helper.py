import opensim as osim

from msk_warp.utils.converted_objects import *
from msk_warp.utils.physical_frame_helper import get_body_name_of_frame, wp_transform_from_osim_transform


def collect_geom_type_sizes(geom: osim.ContactGeometry) -> tuple[GeomType, wp.vec3, wp.vec3, float]:
    """ Returns the converted geometry type, size, aabb, radius bound """
    geom_cls = geom.getConcreteClassName()
    raise NotImplementedError


def convert_contact_geometry(geom: osim.ContactGeometry) -> GeomData:
    geom_name = geom.getName()

    frame = geom.getFrame()
    body_name = get_body_name_of_frame(frame)

    # Need to compute the transform of the geometry relative to the body frame
    X_BF = frame.findTransformInBaseFrame()
    X_FP = geom.getTransform()
    X_BP = X_BF.compose(X_FP)
    transform = wp_transform_from_osim_transform(X_BP)

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
        friction=wp.vec3(0.95, 0.6, 0.0),
        stiffness=(1e6 ** (2 / 3)),
        dissipation=1.0,
        transition_velocity=0.1,
        priority=0
    )


def convert_geoms(model: osim.Model) -> list[GeomData]:
    """ Returns the all the converted contact geometries in the model """
    collider_data = []

    # Check for contact geometry within ContactGeometrySet
    contact_geom_set = model.getContactGeometrySet()
    for contact_geom in contact_geom_set:
        contact_geom_data = convert_contact_geometry(contact_geom)
        collider_data.append(contact_geom_data)

    # Check for contact geometry within <components> element
    body_list = model.getBodyList()
    for body in body_list:
        components = body.getComponentsList()
        contact_geom_components = filter(lambda c: isinstance(c, osim.ContactGeometry), components)
        for contact_geom in contact_geom_components:
            contact_geom_data = convert_contact_geometry(contact_geom)
            collider_data.append(contact_geom_data)
    return collider_data
