import warp as wp

from bolt.load_utils.converted_objects import SwingTwistLimitData
from bolt.load_utils.xml_helper import extract_bolt_only_objects, extract_float_from_element, \
    extract_string_from_element
from bolt.types_consts import SwingTwistLimit


def convert_swing_twist_limits(model_path: str) -> list[SwingTwistLimitData]:
    """ Extracts swing twist limit data from the BoltOnlySet in the OpenSim model XML. """
    swing_twist_data = []
    bolt_only_objects = extract_bolt_only_objects(model_path)
    if bolt_only_objects is None:
        return swing_twist_data

    for obj in bolt_only_objects:
        if obj.tag == "SwingTwistLimit":
            name = obj.get("name")
            joint = extract_string_from_element(obj, "joint")

            twist_low = extract_float_from_element(obj, "twist_low")
            twist_high = extract_float_from_element(obj, "twist_high")

            swing_1_neg_limit = extract_float_from_element(obj, "swing1_neg_limit")
            swing_1_pos_limit = extract_float_from_element(obj, "swing1_pos_limit")
            swing_2_neg_limit = extract_float_from_element(obj, "swing2_neg_limit")
            swing_2_pos_limit = extract_float_from_element(obj, "swing2_pos_limit")

            stiffness = extract_float_from_element(obj, "stiffness")
            damping = extract_float_from_element(obj, "damping")

            swing_twist_data.append(SwingTwistLimitData(
                name=name,
                joint=joint,
                stiffness=stiffness,
                damping=damping,
                twist_limits=wp.vec2(twist_low, twist_high),
                swing1_limits=wp.vec2(swing_1_neg_limit, swing_1_pos_limit),
                swing2_limits=wp.vec2(swing_2_neg_limit, swing_2_pos_limit)
            ))
    return swing_twist_data


def create_swing_twist_data(
        swing_twist_data: list[SwingTwistLimitData],
        joint_ordering: dict[str, int],
        mob_qpos_adr: list[int],
        mob_dof_adr: list[int]
) -> list[SwingTwistLimit]:
    swing_twist_limits = []
    for data in swing_twist_data:
        limit = SwingTwistLimit()
        joint_id = joint_ordering[data.joint]

        limit.qpos_adr = mob_qpos_adr[joint_id]
        limit.dof_adr = mob_dof_adr[joint_id]

        limit.twist_range = data.twist_limits
        limit.swing1_range = data.swing1_limits
        limit.swing2_range = data.swing2_limits

        limit.stiffness = data.stiffness
        limit.damping = data.damping

        swing_twist_limits.append(limit)
    return swing_twist_limits
