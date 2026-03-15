import warp as wp
import msk_warp.utils.physical_frame_helper as physical_frame_helper
from msk_warp import MobilizerType
from .converted_objects import JointData, GROUND, GROUND_PARENT
from .osim_types import OSimType
from .physical_frame_helper import extract_transform_from_frame
from .property_helper import extract_value_from_prop, extract_vec3

# Converts an OpenSim joint class to (MobilizerType, num_coordinates, num_speeds)
joint_conversion = {
    "WeldJoint": (MobilizerType.WELD, 0, 0),
    "PinJoint": (MobilizerType.PIN, 1, 1),
    "SliderJoint": (MobilizerType.SLIDER, 1, 1),
    "UniversalJoint": (MobilizerType.UNIVERSAL, 2, 2),
    "GimbalJoint": (MobilizerType.GIMBAL, 3, 3),
    "CantileverFreeBeamJoint": (MobilizerType.BEAM, 3, 3),
    "EllipsoidJoint": (MobilizerType.ELLIPSOID, 3, 3),
    "BallJoint": (MobilizerType.BALL, 4, 3),
    # "CustomJoint": (MobilizerType.CUSTOM, -1, -1),  # CustomJoint requires special handling
    "FreeJoint": (MobilizerType.FREE, 7, 6),
}


def get_joint_parent_body_name(joint: OSimType.Joint) -> str:
    """ Get the name of the parent body of a joint """
    parent_frame = joint.getParentFrame()
    parent_body_name = physical_frame_helper.get_body_name_of_frame(parent_frame)
    return parent_body_name


def get_joint_child_body_name(joint: OSimType.Joint) -> str:
    """ Get the name of the child body of a joint """
    child_frame = joint.getChildFrame()
    child_body_name = physical_frame_helper.get_body_name_of_frame(child_frame)
    return child_body_name


def get_mobilizer_type(joint: OSimType.Joint, root_free: bool) -> tuple[MobilizerType, int, int]:
    joint_class = joint.getConcreteClassName()
    # Custom joint from ground to root may be a free or weld joint
    if joint_class == "CustomJoint" and get_joint_parent_body_name(joint) == GROUND:
        return joint_conversion["FreeJoint"] if root_free else joint_conversion["WeldJoint"]

    # Get proper number of coordinates and speeds for custom joints
    if joint_class == "CustomJoint":
        return MobilizerType.CUSTOM, joint.numCoordinates(), joint.numCoordinates()

    if joint_class in joint_conversion:
        return joint_conversion[joint_class]
    raise ValueError(f"Unsupported joint type: {joint_class}")


def get_extra_info(joint: OSimType.Joint) -> wp.vec3:
    joint_class = joint.getConcreteClassName()
    if joint_class == "CantileverFreeBeamJoint":
        # We don't downcast this since OpenSim API may not support this beam yet
        beam_length = extract_value_from_prop(joint.getPropertyByName("beam_length"))
        deflection_coeff = (2.0 / 3.0) * beam_length
        displacement_coeff = (4.0 / 15.0) * beam_length
        return wp.vec3(beam_length, deflection_coeff, displacement_coeff)
    elif joint_class == "EllipsoidJoint":
        ellipsoid_joint = OSimType.EllipsoidJoint.safeDownCast(joint)
        radii_x_y_z = extract_vec3(ellipsoid_joint.get_radii_x_y_z())
        return wp.vec3(radii_x_y_z)
    return wp.vec3()


def convert_joint(joint: OSimType.Joint, root_free: bool) -> JointData:
    """ Converts an OpenSim joint to the relevant JointData """
    mob_type, num_coordinates, num_speeds = get_mobilizer_type(joint, root_free)

    return JointData(
        name=joint.getName(),
        parent_body_name=get_joint_parent_body_name(joint),
        child_body_name=get_joint_child_body_name(joint),

        mob_type=mob_type,
        num_coordinates=num_coordinates,
        num_speeds=num_speeds,

        transform_PF=extract_transform_from_frame(joint.getParentFrame()),
        transform_MB=wp.transform_inverse(extract_transform_from_frame(joint.getChildFrame())),
        extra_info=get_extra_info(joint)
    )


def get_joint_parent_id(joint: JointData, ordered_bodies_names: list[str]):
    """ Returns the id of the parent body of a joint, or -1 if the body is ground """
    if joint.parent_body_name == GROUND_PARENT:
        return -1
    return ordered_bodies_names.index(joint.parent_body_name)


def compute_qpos_dof_adr(joints: list[JointData]) -> tuple[list[int], list[int]]:
    """ Computes the starting address of each joint q, u values in qpos, qvel """
    qpos_adr, dof_adr = [], []
    curr_qpos, curr_dof = 0, 0
    for joint in joints:
        qpos_adr.append(curr_qpos)
        dof_adr.append(curr_dof)

        curr_qpos += joint.num_coordinates
        curr_dof += joint.num_speeds
    return qpos_adr, dof_adr
