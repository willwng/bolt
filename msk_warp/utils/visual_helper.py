import opensim as osim
import warp as wp
from msk_warp import MeshLoadResult

from msk_warp.utils.converted_objects import VisualData
from msk_warp.utils.physical_frame_helper import get_body_name_of_frame, extract_frame_transform_from_base_frame
from msk_warp.utils.property_helper import extract_vec3
from msk_warp.utils.osim_types import OSimType


def convert_visuals(model: osim.Model) -> list[VisualData]:
    """ Returns the all the converted visual geometry in the model """
    visual_data = []

    # Check for visual within <attached_geometry> element of bodies
    for body in model.getBodyList():
        n_attached_geom = body.getPropertyByName("attached_geometry").size()  # todo: is there a beter way to get this?
        for i in range(n_attached_geom):
            geom = body.get_attached_geometry(i)
            mesh = OSimType.Mesh.safeDownCast(geom)  # must be mesh

            geom_name = geom.getName()

            # Get the body and transform of the attached geometry
            frame = geom.getFrame()
            body_name = get_body_name_of_frame(frame)
            transform = extract_frame_transform_from_base_frame(frame)

            # Mesh file, scale factors
            mesh_file = mesh.get_mesh_file()
            scale_factors = extract_vec3(geom.get_scale_factors())

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


def create_mesh_load_results(visual_data_list: list[VisualData]) -> list[MeshLoadResult]:
    """ Converts the visual data into MeshLoadResults, which can be used to load meshes into the renderer. """
    mesh_load_results = []
    for visual in visual_data_list:
        mesh_load_results.append(
            MeshLoadResult(
                file=visual.mesh_file,
                scale=visual.scale_factors
            )
        )
    return mesh_load_results


def get_vis_body_name(visual_data_list: list[VisualData]) -> list[str]:
    return [visual.body_name for visual in visual_data_list]


def get_vis_transform(visual_data_list: list[VisualData]) -> list[wp.transform]:
    return [visual.transform for visual in visual_data_list]
