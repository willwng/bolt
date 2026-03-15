import opensim as osim

from msk_warp.utils.converted_objects import *
from msk_warp.utils.physical_frame_helper import get_body_name_of_frame, transform_from_osim_transform


def convert_contact_geometry(geom: osim.ContactGeometry) -> ColliderData:
    geom_name = geom.getName()

    frame = geom.getFrame()
    body_name = get_body_name_of_frame(frame)

    # Need to compute the transform of the geometry relative to the body frame
    X_BF = frame.findTransformInBaseFrame()
    X_FP = geom.getTransform()
    X_BP = X_BF.compose(X_FP)
    transform = transform_from_osim_transform(X_BP)
    return ColliderData(
        name=geom_name,
        body_name=body_name,
        transform=transform
    )


def convert_geoms(model: osim.Model) -> list[ColliderData]:
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
