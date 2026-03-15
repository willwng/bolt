import opensim as osim

from msk_warp.utils.converted_objects import SpatialTransformData, TransformAxisData
from msk_warp.utils.physical_frame_helper import get_body_name_of_frame, extract_transform_from_frame
from msk_warp.utils.property_helper import extract_vec3
from msk_warp.utils.osim_types import OSimType

def convert_transform_axis(txfm: OSimType.TransformAxis) -> TransformAxisData:
    return


def convert_spatial_transforms(model: osim.Model) -> list[SpatialTransformData]:
    """ Returns the all the converted spatial transforms in the model """
    spatial_transform_data = []

    # Look through all custom joints
    custom_joints = filter(lambda j: j.getConcreteClassName() == "CustomJoint", model.getJointList())
    custom_joints = [OSimType.CustomJoint.safeDownCast(j) for j in custom_joints]
    for joint in custom_joints:
        joint_name = joint.getName()
        spatial_transform = joint.getSpatialTransform()

        # 3 rotations followed by 3 translations
        rotation_1 = convert_transform_axis(spatial_transform.get_rotation1())
        rotation_2 = convert_transform_axis(spatial_transform.get_rotation2())
        rotation_3 = convert_transform_axis(spatial_transform.get_rotation3())
        translation_1 = convert_transform_axis(spatial_transform.get_translation1())
        translation_2 = convert_transform_axis(spatial_transform.get_translation2())
        translation_3 = convert_transform_axis(spatial_transform.get_translation3())

        print(dir(rotation_1))
        quit()
    quit()
