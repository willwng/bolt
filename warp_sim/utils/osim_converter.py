from .osim_objs import Model


def num_bodies(model: Model) -> int:
    return len(model.body_set.bodies)


def num_dofs(model: Model) -> int:
    dof_count = 0
    for joint in model.joint_set.joints.values():
        dof_count += len(joint.coordinates)
    return dof_count


def num_pos_dofs(model: Model) -> int:
    pos_dof_count = 0
    for joint in model.joint_set.joints.values():
        if "ground" in joint.socket_parent_frame:  # root body (free)
            pos_dof_count += 7  # 3 for position, 4 for orientation (quaternion)
        else:
            pos_dof_count += len(joint.coordinates)
    return pos_dof_count


def num_muscles(model: Model) -> int:
    return len(model.force_set.muscles)


def num_colliders(model: Model) -> int:
    return len(model.contact_geometry_set.contact_spheres)


def num_sites(model: Model) -> int:
    num_sites = 0
    for muscle in model.force_set.muscles.values():
        num_sites += len(muscle.geometry_path.path_point_set.path_points)
    return num_sites


def get_default_positions(model: Model) -> list[float]:
    positions = []
    for joint in model.joint_set.joints.values():
        if "ground" in joint.socket_parent_frame:  # root body (free)
            # Default position (0,0,0) and orientation (1,0,0,0)
            positions.extend([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            continue
        for coord in joint.coordinates:
            positions.append(coord.default_value)
    return positions


def body_masses(model: Model) -> list[float]:
    masses = []
    for body in model.body_set.bodies.values():
        masses.append(body.mass)
    return masses


def get_body_inertias(model: Model) -> list[list[float]]:
    inertias = []
    for body in model.body_set.bodies.values():
        inertia = body.inertia
        # should be diagonal
        assert inertia.xy == 0.0
        assert inertia.xz == 0.0
        assert inertia.yz == 0.0
        inertias.append([inertia.xx, inertia.yy, inertia.zz])
    return inertias


def get_local_body_com_pos(model: Model) -> list[list[float]]:
    com_positions = []
    for body in model.body_set.bodies.values():
        com = body.mass_center
        com_positions.append([com.x, com.y, com.z])
    return com_positions


def get_local_body_rot(model: Model) -> list[list[float]]:
    local_rots = []
    for body in model.body_set.bodies.values():
        local_rots.append([[1.0, 0.0, 0.0, 0.0], ])
    return local_rots


def get_body_num_colliders(model: Model) -> list[int]:
    body_name_to_index = {body.name: idx for idx, body in enumerate(model.body_set.bodies.values())}
    collider_counts = [0] * len(body_name_to_index)
    for collider in model.contact_geometry_set.contact_spheres.values():
        if "bodyset" in collider.socket_frame:
            parent_body_name = collider.socket_frame.split("/bodyset/")[1]
        else:
            parent_body_name = collider.socket_frame
        if parent_body_name in body_name_to_index:
            body_idx = body_name_to_index[parent_body_name]
            collider_counts[body_idx] += 1
    return collider_counts


def get_joint_frame(joint, frame_name: str):
    for frame in joint.frames:
        if frame.name == frame_name:
            return frame
    return None


def remove_prefix(name: str) -> str:
    if "bodyset/" in name:
        return name.split("bodyset/")[1]
    return name


def get_body_parent_ids(model: Model) -> list[int]:
    body_name_to_index = {body.name: idx for idx, body in enumerate(model.body_set.bodies.values())}
    parent_ids = [-1] * len(body_name_to_index)  # -1 for root bodies
    for joint in model.joint_set.joints.values():
        child_frame_name = joint.socket_child_frame
        parent_frame_name = joint.socket_parent_frame

        child_frame = get_joint_frame(joint, child_frame_name)
        parent_frame = get_joint_frame(joint, parent_frame_name)

        child_name = remove_prefix(child_frame.socket_parent)
        parent_name = remove_prefix(parent_frame.socket_parent)
        child_idx = body_name_to_index[child_name]

        if parent_name in body_name_to_index:
            parent_idx = body_name_to_index[parent_name]
            parent_ids[child_idx] = parent_idx
        else:
            parent_ids[child_idx] = -1

    return parent_ids


def get_num_joints(model: Model) -> tuple[int, int]:
    num_conventional, num_custom = 0, 0
    for joint in model.joint_set.joints.values():
        if joint.__class__.__name__ != "CustomJoint":
            num_conventional += 1
        else:
            num_custom += 1
    return num_conventional, num_custom


def get_joint_types(model: Model) -> list[str]:
    from .._src.types import JointType
    joint_types = []
    for joint in model.joint_set.joints.values():
        if "ground" in joint.socket_parent_frame:
            joint_types.append(JointType.FREE)
            continue
        if joint.__class__.__name__ == "PinJoint":
            joint_types.append(JointType.HINGE)
        elif joint.__class__.__name__ == "UniversalJoint":
            joint_types.append(JointType.UNIVERSAL)
        elif joint.__class__.__name__ == "CustomJoint":
            joint_types.append(JointType.CUSTOM)
        else:
            print(f"Warning: Unrecognized joint type {joint.__class__.__name__}")
            quit()

    return joint_types


def get_joint_num_qdofs(model: Model) -> list[int]:
    joint_num_qdofs = []
    for joint in model.joint_set.joints.values():
        if "ground" in joint.socket_parent_frame:
            joint_num_qdofs.append(7)  # free joint
            continue
        joint_num_qdofs.append(len(joint.coordinates))
    return joint_num_qdofs


def get_joint_num_vdofs(model: Model) -> list[int]:
    joint_num_dofs = []
    for joint in model.joint_set.joints.values():
        joint_num_dofs.append(len(joint.coordinates))
    return joint_num_dofs


def get_joint_rel_pos(model: Model, parent: bool) -> list[list[float]]:
    rel_parent_positions = []
    for joint in model.joint_set.joints.values():
        if parent:
            frame = get_joint_frame(joint, joint.socket_parent_frame)
        else:
            frame = get_joint_frame(joint, joint.socket_child_frame)
        pos = frame.translation
        rel_parent_positions.append([pos.x, pos.y, pos.z])
    return rel_parent_positions


def get_joint_rel_rot(model: Model, parent: bool) -> list[list[float]]:
    from scipy.spatial.transform import Rotation as R
    rel_parent_rots = []
    for joint in model.joint_set.joints.values():
        if parent:
            frame = get_joint_frame(joint, joint.socket_parent_frame)
        else:
            frame = get_joint_frame(joint, joint.socket_child_frame)
        rot = frame.orientation
        r = R.from_euler('YXZ', [-rot.z, rot.x, rot.y], degrees=True) # note conversion from y-up to z-up
        quat = r.as_quat()
        rel_parent_rots.append([quat[3], quat[0], quat[1], quat[2]])
    return rel_parent_rots
