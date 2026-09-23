""" Build the kinematic tree and the coordinate index maps"""
from dataclasses import dataclass

from bolt.loader.converters import joint_helper
from bolt.loader.converters.converted_objects import GROUND
from bolt.loader.converters.converted_objects import GROUND_PARENT
from bolt.loader.converters.kinematic_tree import KinematicTree
from bolt.loader.converters.python_util import apply_map_to_list
from bolt.loader.converters.python_util import exclusive_scan
from bolt.loader.converters.python_util import flatten_nested_list
from bolt.loader.converters.python_util import string_list_to_ordering
from bolt.loader.model_parser import ParsedModel


@dataclass
class ModelTopology:
    bodies: list  # in fk order
    joints: list
    joint_ordering: dict[str, int]  # joint name -> mobilizer id
    body_ordering: dict[str, int]  # body name -> body id
    body_parent_id: list[int]
    body_tree: list[list[int]]  # body ids at each level of the tree
    body_children: list[int]  # flattened child body ids of each body
    body_children_num: list[int]
    body_children_adr: list[int]
    qpos_ordering: dict[str, int]  # coordinate name -> global qpos id
    dof_ordering: dict[str, int]  # coordinate name -> global dof id
    relative_dof_ordering: dict[str, int]  # coordinate name -> dof id relative to its joint
    mob_qpos_adr: list[int]
    mob_dof_adr: list[int]


def build_topology(parsed: ParsedModel, render_kinematic_tree: bool) -> ModelTopology:
    # Create a lookup from body name -> body data. Needed for joint->body lookup
    body_name_to_body = {body.name: body for body in parsed.bodies}

    # Build the kinematic tree, storing the joint that connects each node to its parent
    tree = KinematicTree(root_body=body_name_to_body[GROUND], root_joint=parsed.joints[0])
    for joint in parsed.joints:
        if joint.parent_body_name != GROUND_PARENT:
            parent_body = body_name_to_body[joint.parent_body_name]
            child_body = body_name_to_body[joint.child_body_name]
            tree.add_edge(parent_body, child_body, joint)

    tree.verify()
    if render_kinematic_tree:
        tree.render()  # graphviz is such a great tool

    # Using the kinematic tree, compute a forward ordering
    tree_ordering = tree.forward_ordering()
    ordered_bodies = [node.body for node in tree_ordering]
    ordered_joints = [node.joint for node in tree_ordering]
    ordered_bodies_names = [body.name for body in ordered_bodies]

    # Body name -> body id
    body_ordering = string_list_to_ordering(ordered_bodies_names)

    # Get all the children of each body
    body_children = [node.get_children_no_roots() for node in tree_ordering]
    body_children_names = [[node.body.name for node in children] for children in body_children]
    body_children_indices = [apply_map_to_list(children, body_ordering) for children in body_children_names]
    body_children_num = [len(children) for children in body_children_indices]

    # Starting address of joint's coordinates/speeds
    mob_qpos_adr, mob_dof_adr = joint_helper.compute_qpos_dof_adr(ordered_joints)

    return ModelTopology(
        bodies=ordered_bodies,
        joints=ordered_joints,
        joint_ordering=joint_helper.compute_joint_name_ordering(ordered_joints),
        body_ordering=body_ordering,
        body_parent_id=[joint_helper.get_joint_parent_id(joint, ordered_bodies_names) for joint in ordered_joints],
        # The "body-level" array (contains list of all bodies at level i)
        body_tree=[apply_map_to_list(level, body_ordering) for level in tree.create_body_tree()],
        body_children=flatten_nested_list(body_children_indices),
        body_children_num=body_children_num,
        body_children_adr=exclusive_scan(body_children_num),
        # The *global* ordering lookup for each coordinate in qpos, dof
        qpos_ordering=joint_helper.get_global_qpos_ordering_lookup(ordered_joints),
        dof_ordering=joint_helper.get_global_dof_ordering_lookup(ordered_joints),
        # Ordering lookup for coordinates relative to each joint's starting address
        relative_dof_ordering=joint_helper.get_relative_dof_ordering_lookup(ordered_joints),
        mob_qpos_adr=mob_qpos_adr,
        mob_dof_adr=mob_dof_adr,
    )
