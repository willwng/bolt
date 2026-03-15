import opensim as osim

from msk_warp.utils.converted_objects import *
from msk_warp.utils.physical_frame_helper import get_body_name_of_frame, extract_transform_from_frame
from msk_warp.utils.property_helper import extract_vec3


def convert_visuals(model: osim.Model) -> list[VisualData]:
    """ Returns the all the converted visual geometry in the model """
    visual_data = []

    # Check for contact geometry within <components> element
    for body in model.getBodyList():
        n_attached_geom = body.getPropertyByName("attached_geometry").size()
        for i in range(n_attached_geom):
            geom = body.get_attached_geometry(i)
            geom_name = geom.getName()

            # Get the body and transform of the attached geometry
            frame = geom.getFrame()
            body_name = get_body_name_of_frame(frame)
            transform = extract_transform_from_frame(frame)

            # Mesh file, scale factors
            mesh_file = geom.getPropertyByName("mesh_file").toString()
            scale_factors = wp.vec3(extract_vec3(geom.get_scale_factors()))

            visual_data.append(
                VisualData(
                    name=geom_name,
                    body_name=body_name,
                    mesh_file=mesh_file,

                    transform=transform,
                    scale_factors=scale_factors,
                )
            )

    return visual_data
