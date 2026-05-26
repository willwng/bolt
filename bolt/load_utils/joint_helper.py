import warp as wp
import bolt.load_utils.physical_frame_helper as physical_frame_helper
from bolt.types_consts import MobilizerType
from .converted_objects import JointData, GROUND, GROUND_PARENT, NO_DOF
from .osim_types import OSimType
from .physical_frame_helper import extract_frame_transform_from_base_frame
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


def is_free_joint(joint: OSimType.Joint) -> bool:
    """ Returns true if the joint is a free joint (i.e. a custom joint from ground to root), false otherwise """
    joint_class = joint.getConcreteClassName()
    return joint_class == "CustomJoint" and get_joint_parent_body_name(joint) == GROUND


def get_mobilizer_type(joint: OSimType.Joint) -> tuple[MobilizerType, int, int]:
    joint_class = joint.getConcreteClassName()
    # Custom joint from ground to root may be a free or weld joint
    if joint_class == "CustomJoint" and get_joint_parent_body_name(joint) == GROUND:
        return joint_conversion["FreeJoint"]

    # Get proper number of coordinates for custom joints. num coordinates should be equal to num speeds
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


def convert_joint(joint: OSimType.Joint) -> JointData:
    """ Converts an OpenSim joint to the relevant JointData """
    mob_type, num_coordinates, num_speeds = get_mobilizer_type(joint)

    return JointData(
        name=joint.getName(),
        parent_body_name=get_joint_parent_body_name(joint),
        child_body_name=get_joint_child_body_name(joint),

        mob_type=mob_type,
        coordinates=get_coordinates(joint),
        num_coordinates=num_coordinates,
        num_speeds=num_speeds,

        transform_PF=extract_frame_transform_from_base_frame(joint.getParentFrame()),
        transform_MB=wp.transform_inverse(extract_frame_transform_from_base_frame(joint.getChildFrame())),
        extra_info=get_extra_info(joint)
    )


def get_joint_parent_id(joint: JointData, ordered_bodies_names: list[str]):
    """ Returns the id of the parent body of a joint, or 0 if the body is ground """
    if joint.parent_body_name == GROUND_PARENT:
        return 0
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


def compute_num_joints_of_type(joints: list[JointData], mob_type: MobilizerType) -> int:
    """ Computes the number of joints with the specified mobilizer in the model """
    return len(list(filter(lambda joint: joint.mob_type == mob_type, joints)))


def compute_mobilizer_index_of_type(
        joints: list[JointData], mob_type: MobilizerType
) -> tuple[list[int], list[int]]:
    """
    Returns:
        1. mapping from mobilizer index to (specified type) joint index (-1 if not specific type). (size num mobilizers)
        2. mapping from (specified type) joint index to mobilizer index. (size num joints of specified type)
    """
    mob_to_spc_index = [-1 for _ in joints]
    spc_to_mob_index = []
    curr_spc_idx = 0
    for mob_idx, joint in enumerate(joints):
        if joint.mob_type == mob_type:
            mob_to_spc_index[mob_idx] = curr_spc_idx
            spc_to_mob_index.append(mob_idx)
            curr_spc_idx += 1
    return mob_to_spc_index, spc_to_mob_index


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


def get_global_qpos_ordering_lookup(joints: list[JointData]) -> dict[str, int]:
    """ Map from coordinate name -> global qpos coordinate ordering """
    global_qpos_ordering = {}
    curr_idx = 0
    for joint in joints:
        for coord in _retrieve_all_qpos(joint):
            global_qpos_ordering[coord] = curr_idx
            curr_idx += 1
    global_qpos_ordering[NO_DOF] = -1
    return global_qpos_ordering


def get_relative_qpos_ordering_lookup(joints: list[JointData]) -> dict[str, int]:
    """ Map from coordinate name -> "relative to joint" qpos coordinate ordering """
    relative_qpos_ordering = {}
    for joint in joints:
        for i, coord in enumerate(_retrieve_all_qpos(joint)):
            relative_qpos_ordering[coord] = i
    relative_qpos_ordering[NO_DOF] = -1
    return relative_qpos_ordering


def get_global_dof_ordering_lookup(joints: list[JointData]) -> dict[str, int]:
    """ Map from coordinate name -> global dof coordinate ordering """
    global_qpos_ordering = {}
    curr_idx = 0
    for joint in joints:
        for coord in joint.coordinates:
            global_qpos_ordering[coord] = curr_idx
            curr_idx += 1
    global_qpos_ordering[NO_DOF] = -1
    return global_qpos_ordering


def get_relative_dof_ordering_lookup(joints: list[JointData]) -> dict[str, int]:
    """ Map from coordinate name -> "relative to joint" dof coordinate ordering """
    relative_dof_ordering = {}
    for joint in joints:
        for i, coord in enumerate(joint.coordinates):
            relative_dof_ordering[coord] = i
    relative_dof_ordering[NO_DOF] = -1
    return relative_dof_ordering


def check_root_free(joints: list[JointData]) -> bool:
    """ Returns whether if the root joint is free """
    for joint in joints:
        if joint.parent_body_name == GROUND and joint.mob_type == MobilizerType.FREE:
            return True
    return False


def get_mob_type(joints: list[JointData]) -> list[MobilizerType]:
    return [joint.mob_type for joint in joints]


def get_mob_dofnum(joints: list[JointData]) -> list[int]:
    return [joint.num_speeds for joint in joints]


def get_mob_X_PF(joints: list[JointData]) -> list[wp.transform]:
    return [joint.transform_PF for joint in joints]


def get_mob_X_MB(joints: list[JointData]) -> list[wp.transform]:
    return [joint.transform_MB for joint in joints]


def get_mob_extra_info(joints: list[JointData]) -> list[wp.vec3]:
    return [joint.extra_info for joint in joints]
