import opensim as osim
import warp as wp

from bolt.utils.converted_objects import SpatialTransformData, TransformAxisData, NO_DOF
from bolt.utils.joint_helper import is_free_joint
from bolt.utils.function_helper import convert_function
from bolt.utils.property_helper import extract_vec3, extract_property_string_list
from bolt.utils.osim_types import OSimType


def convert_transform_axis(txfm: OSimType.TransformAxis) -> TransformAxisData:
    axis = extract_vec3(txfm.getAxis())
    function = convert_function(txfm.getFunction())
    coordinates = extract_property_string_list(txfm.getCoordinateNames())
    if len(coordinates) > 1:
        raise ValueError(f"Unsupported: transform axis has more than 1 coordinate: {coordinates}")
    elif len(coordinates) == 1:
        coordinate = coordinates[0]
    else:  # no dof provided, just use -1
        coordinate = NO_DOF

    return TransformAxisData(
        coordinate=coordinate,
        axis=wp.vec3(axis),
        function=function
    )


def convert_spatial_transforms(model: OSimType.Model) -> list[SpatialTransformData]:
    """ Returns the all the converted spatial transforms in the model """
    spatial_transform_data = []

    # Look through all custom joints
    custom_joints = filter(lambda j: j.getConcreteClassName() == "CustomJoint", model.getJointList())
    custom_joints = [OSimType.CustomJoint.safeDownCast(j) for j in custom_joints]
    for joint in custom_joints:
        # Skip the "CustomJoint" representing the free joint between ground and root
        if is_free_joint(joint):
            continue
        spatial_transform = joint.getSpatialTransform()
        spatial_transform_data.append(
            SpatialTransformData(
                joint_name=joint.getName(),
                rotation_1=convert_transform_axis(spatial_transform.get_rotation1()),
                rotation_2=convert_transform_axis(spatial_transform.get_rotation2()),
                rotation_3=convert_transform_axis(spatial_transform.get_rotation3()),
                translation_1=convert_transform_axis(spatial_transform.get_translation1()),
                translation_2=convert_transform_axis(spatial_transform.get_translation2()),
                translation_3=convert_transform_axis(spatial_transform.get_translation3()),
            )
        )
    return spatial_transform_data


def order_spatial_transforms(
        spatial_transforms: list[SpatialTransformData],
        joint_ordering: dict[str, int]
) -> list[SpatialTransformData]:
    """ Re-orders the spatial transforms to match the order of the joints in the model. """
    ordered_spatial_transforms = sorted(spatial_transforms, key=lambda spt: joint_ordering[spt.joint_name])
    return list(ordered_spatial_transforms)


def get_flattened_transform_axes(spatial_transforms: list[SpatialTransformData]) -> list[TransformAxisData]:
    """ Flattens the spatial transform data into a list of transform axes """
    flattened_axes = []
    for spt in spatial_transforms:
        flattened_axes.extend([
            spt.rotation_1,
            spt.rotation_2,
            spt.rotation_3,
            spt.translation_1,
            spt.translation_2,
            spt.translation_3,
        ])
    return flattened_axes


def get_txfm_coordinate_names(transform_axes: list[TransformAxisData]) -> list[str]:
    """ Returns a list of the coordinate names for each transform axis """
    return [txfm.coordinate for txfm in transform_axes]


def get_txfm_axes(transform_axes: list[TransformAxisData]) -> list[wp.vec3]:
    """ Returns a list of the axis for each transform axis """
    return [txfm.axis for txfm in transform_axes]