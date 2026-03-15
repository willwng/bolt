import warp as wp
import msk_warp.utils.physical_frame_helper as physical_frame_helper
from msk_warp import MobilizerType
from .converted_objects import JointData, GROUND, GROUND_PARENT, NO_DOF
from .osim_types import OSimType
from .physical_frame_helper import extract_transform_from_frame
from .property_helper import extract_value_from_prop, extract_vec3
from .python_util import string_list_to_ordering

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

# For mobilizers that use quaternions, we need the index of the scalar part of the quaternion in the coordinates
mobilizer_quaternion_scalar_index = {
    MobilizerType.FREE: 3,
    MobilizerType.BALL: 3,
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


def get_coordinates(joint: OSimType.Joint) -> list[str]:
    """ Get the coordinate names of a joint """
    coordinates = []
    for i in range(joint.numCoordinates()):
        coordinates.append(joint.get_coordinates(i).getName())
    return coordinates


def convert_joint(joint: OSimType.Joint, root_free: bool) -> JointData:
    """ Converts an OpenSim joint to the relevant JointData """
    mob_type, num_coordinates, num_speeds = get_mobilizer_type(joint, root_free)

    return JointData(
        name=joint.getName(),
        parent_body_name=get_joint_parent_body_name(joint),
        child_body_name=get_joint_child_body_name(joint),

        mob_type=mob_type,
        coordinates=get_coordinates(joint),
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


def compute_joint_name_ordering(ordered_joints: list[JointData]) -> dict[str, int]:
    """ Computes the ordering of the joints by name """
    return string_list_to_ordering([joint.name for joint in ordered_joints])


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


def compute_qpos_dof_lookups(joints: list[JointData]) -> tuple[dict[str, int], dict[str, int]]:
    """ Computes the lookup for coordinate_name -> address in qpos, address in qvel """
    qpos_adr, dof_adr = [], []
    curr_qpos, curr_dof = 0, 0
    for joint in joints:
        qpos_adr.append(curr_qpos)
        dof_adr.append(curr_dof)

        curr_qpos += joint.num_coordinates
        curr_dof += joint.num_speeds
    return qpos_adr, dof_adr


def compute_mobilizer_custom_joint_index(joints: list[JointData]) -> tuple[list[int], list[int]]:
    """
    Returns:
        1. mapping from mobilizer index to custom joint index (-1 if not a custom joint). (size num mobilizers)
        2. mapping from custom joint index to mobilizer index. (size num custom joints)
    """
    mob_to_cst_index = [-1 for _ in joints]
    cst_to_mob_index = []
    curr_cst_idx = 0
    for i, joint in enumerate(joints):
        if joint.mob_type == MobilizerType.CUSTOM:
            mob_to_cst_index[curr_cst_idx] = curr_cst_idx
            cst_to_mob_index.append(i)
            curr_cst_idx += 1
    return mob_to_cst_index, cst_to_mob_index


def _quat_scalar_coordinate_name(joint: JointData) -> str:
    """ Coordinate name for the scalar part of the quaternion for a joint """
    return f"{joint.child_body_name}_quat_w"


def _retrieve_all_qpos(joint: JointData) -> list[str]:
    """ Get the qpos coordinate names for a joint, including the scalar part of the quaternion if applicable """
    joint_coordinates = [v for v in joint.coordinates]
    if joint.mob_type in mobilizer_quaternion_scalar_index:
        scalar_index = mobilizer_quaternion_scalar_index[joint.mob_type]
        joint_coordinates.insert(scalar_index, _quat_scalar_coordinate_name(joint))
    return joint_coordinates


def get_dof_coordinates(joints: list[JointData]) -> list[str]:
    """ Get the coordinate names of all the dofs in the model, in order """
    dof_coordinates = []
    for joint in joints:
        dof_coordinates.extend(joint.coordinates)
    return dof_coordinates


def get_qpos_coordinates(joints: list[JointData]) -> list[str]:
    """ Get the coordinate names of all the qpos in the model, in order """
    qpos_coordinates = []
    for joint in joints:
        qpos_coordinates.extend(_retrieve_all_qpos(joint))
    return qpos_coordinates


def get_relative_qpos_ordering_lookup(joints: list[JointData]) -> dict[str, int]:
    """ Map from coordinate name -> "relative to joint" qpos coordinate ordering """
    relative_qpos_ordering = {}
    for joint in joints:
        for i, coord in enumerate(_retrieve_all_qpos(joint)):
            relative_qpos_ordering[coord] = i
    relative_qpos_ordering[NO_DOF] = -1
    return relative_qpos_ordering


def get_relative_dof_ordering_lookup(joints: list[JointData]) -> dict[str, int]:
    """ Map from coordinate name -> "relative to joint" dof coordinate ordering """
    relative_dof_ordering = {}
    for joint in joints:
        for i, coord in enumerate(joint.coordinates):
            relative_dof_ordering[coord] = i
    relative_dof_ordering[NO_DOF] = -1
    return relative_dof_ordering
