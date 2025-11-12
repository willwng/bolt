from collections import OrderedDict
from dataclasses import dataclass

from .osim_objs import Model, Ground, ForceSet, \
    Body, Joint, Collider


@dataclass
class FullBodyDesc:
    body: Body
    joint: Joint
    colliders: OrderedDict[str, Collider]


@dataclass
class CheckedModel:
    """
    A checked model ensures that every body has an associated joint
     (this simplifies the ordering of bodies and joints).
    """
    ground: Ground
    body_full_desc: OrderedDict[str, FullBodyDesc]
    force_set: ForceSet

    def iter_descs(self):
        for body_name, full_desc in self.body_full_desc.items():
            yield body_name, full_desc

    def iter_bodies(self):
        for body_name, full_desc in self.body_full_desc.items():
            yield body_name, full_desc.body

    def iter_joints(self):
        for body_name, full_desc in self.body_full_desc.items():
            yield body_name, full_desc.joint

    def get_body_index(self, body_name: str) -> int:
        for idx, name in enumerate(self.body_full_desc.keys()):
            if name == body_name:
                return idx
        raise ValueError(f"Body name {body_name} not found.")

    def get_root_body(self) -> Body:
        for body_name, full_desc in self.body_full_desc.items():
            joint = full_desc.joint
            if "ground" in joint.socket_parent_frame:
                return full_desc.body
        assert False, "No root body found."

    def get_body_parent_name(self, body_name: str) -> str:
        full_desc = self.body_full_desc[body_name]
        joint = full_desc.joint
        if "ground" in joint.socket_parent_frame:
            return "ground"
        parent_frame = None
        for frame in joint.frames:
            if frame.name == joint.socket_parent_frame:
                parent_frame = frame
                break
        assert parent_frame is not None
        parent_body_name = remove_prefix(parent_frame.socket_parent)
        return parent_body_name

    def get_body_parent_idx(self, body_idx: int) -> int:
        full_desc = list(self.body_full_desc.values())[body_idx]
        joint = full_desc.joint
        if "ground" in joint.socket_parent_frame:
            return -1
        parent_frame = None
        for frame in joint.frames:
            if frame.name == joint.socket_parent_frame:
                parent_frame = frame
                break
        assert parent_frame is not None
        parent_body_name = remove_prefix(parent_frame.socket_parent)
        return self.get_body_index(parent_body_name)


def convert_y_up_z_up(model: CheckedModel):
    from scipy.spatial.transform import Rotation as R
    return model  # Placeholder for actual conversion logic


def remove_prefix(name: str) -> str:
    if "bodyset/" in name:
        return name.split("bodyset/")[1]
    return name


def to_checked_model(model: Model) -> CheckedModel:
    body_full_desc = OrderedDict()
    for body in model.body_set.bodies.values():
        body_full_desc[body.name] = None  # to be filled later

    # All colliders for each body
    body_name_to_colliders = {}
    for collider in model.contact_geometry_set.contact_spheres.values():
        parent_body_name = remove_prefix(collider.socket_frame)
        if parent_body_name not in body_name_to_colliders:
            body_name_to_colliders[parent_body_name] = OrderedDict()
        body_name_to_colliders[parent_body_name][collider.name] = collider

    # Find the joint that connects each body to its parent
    body_name_to_joint = {}
    for joint in model.joint_set.joints.values():
        child_frame_name = joint.socket_child_frame
        child_frame = None
        for frame in joint.frames:
            if frame.name == child_frame_name:
                child_frame = frame
                break
        child_body_name = remove_prefix(child_frame.socket_parent)
        body_name_to_joint[child_body_name] = joint

    # Now fill in the body_full_desc
    for body_name in body_full_desc.keys():
        body = model.body_set.bodies[body_name]
        joint = body_name_to_joint.get(body_name, None)
        colliders = body_name_to_colliders.get(body_name, OrderedDict())
        body_full_desc[body_name] = FullBodyDesc(
            body=body,
            joint=joint,
            colliders=colliders
        )

    return CheckedModel(
        ground=model.ground,
        body_full_desc=body_full_desc,
        force_set=model.force_set
    )


def num_bodies(model: CheckedModel) -> int:
    return len(model.body_full_desc)


def get_joint_num_dofs(model: CheckedModel, vel_dofs: bool) -> list[int]:
    joint_num_dofs = []
    for _, desc in model.iter_descs():
        joint = desc.joint
        if "ground" in joint.socket_parent_frame:
            joint_num_dofs.append(6 if vel_dofs else 7)
            continue
        joint_num_dofs.append(len(joint.coordinates))
    return joint_num_dofs


def num_muscles(model: CheckedModel) -> int:
    return len(model.force_set.muscles)


def num_colliders(model: CheckedModel) -> int:
    num_colliders = 0
    for _, desc in model.iter_descs():
        num_colliders += len(desc.colliders)
    return num_colliders


def num_sites(model: Model) -> int:
    num_sites = 0
    for muscle in model.force_set.muscles.values():
        num_sites += len(muscle.geometry_path.path_point_set.path_points)
    return num_sites


def get_default_positions(model: CheckedModel) -> list[float]:
    positions = []
    for _, desc in model.iter_descs():
        joint = desc.joint
        if "ground" in joint.socket_parent_frame:  # root body (free)
            # Default position (0,0,0) and orientation (1,0,0,0)
            positions.extend([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            continue
        for coord in joint.coordinates:
            positions.append(coord.default_value)
    return positions


def body_masses(model: CheckedModel) -> list[float]:
    masses = []
    for _, desc in model.iter_descs():
        body = desc.body
        masses.append(body.mass)
    return masses


def get_body_inertias(model: CheckedModel) -> list[list[float]]:
    inertias = []
    for _, desc in model.iter_descs():
        body = desc.body
        inertia = body.inertia
        # should be diagonal
        assert inertia.xy == 0.0
        assert inertia.xz == 0.0
        assert inertia.yz == 0.0
        inertias.append([inertia.xx, inertia.yy, inertia.zz])
    return inertias


def get_local_body_com_pos(model: CheckedModel) -> list[list[float]]:
    com_positions = []
    for _, desc in model.iter_descs():
        body = desc.body
        com = body.mass_center
        com_positions.append([com.x, com.y, com.z])
    return com_positions


def get_local_body_rot(model: CheckedModel) -> list[list[float]]:
    local_rots = []
    for _, desc in model.iter_descs():
        local_rots.append([1.0, 0.0, 0.0, 0.0])
    return local_rots


def get_body_num_colliders(model: CheckedModel) -> list[int]:
    num_body_colliders = []
    for _, desc in model.iter_descs():
        num_body_colliders.append(len(desc.colliders))
    return num_body_colliders


def get_joint_frame(joint, frame_name: str):
    for frame in joint.frames:
        if frame.name == frame_name:
            return frame
    return None


def get_body_parent_ids(model: CheckedModel) -> list[int]:
    parent_ids = []
    for _, body in model.iter_bodies():
        parent_name = model.get_body_parent_name(body.name)
        if parent_name == "ground":
            parent_ids.append(-1)
            continue
        parent_idx = model.get_body_index(parent_name)
        parent_ids.append(parent_idx)
    return parent_ids


def get_num_joints(model: CheckedModel) -> tuple[int, int]:
    num_conventional, num_custom = 0, 0
    for _, joint in model.iter_joints():
        if joint.__class__.__name__ != "CustomJoint":
            num_conventional += 1
        else:
            num_custom += 1
    return num_conventional, num_custom


def get_joint_types(model: CheckedModel) -> list[str]:
    from .._src.types import JointType
    joint_types = []
    for _, joint in model.iter_joints():
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
            print(
                f"Warning: Unrecognized joint type {joint.__class__.__name__}")
            quit()

    return joint_types


def get_joint_rel_pos(model: CheckedModel, parent: bool) -> list[list[float]]:
    rel_parent_positions = []
    for _, joint in model.iter_joints():
        if parent:
            frame = get_joint_frame(joint, joint.socket_parent_frame)
        else:
            frame = get_joint_frame(joint, joint.socket_child_frame)
        pos = frame.translation
        rel_parent_positions.append([pos.x, pos.y, pos.z])
    return rel_parent_positions


def get_joint_rel_rot(model: CheckedModel, parent: bool) -> list[list[float]]:
    from scipy.spatial.transform import Rotation as R
    rel_parent_rots = []
    for _, joint in model.iter_joints():
        if parent:
            frame = get_joint_frame(joint, joint.socket_parent_frame)
        else:
            frame = get_joint_frame(joint, joint.socket_child_frame)
        rot = frame.orientation
        r = R.from_euler('YXZ', [-rot.z, rot.x, rot.y],
                         degrees=True)  # note conversion from y-up to z-up
        quat = r.as_quat()
        rel_parent_rots.append([quat[3], quat[0], quat[1], quat[2]])
    return rel_parent_rots


def get_collision_geom_types(model: CheckedModel):
    from .._src.types import GeomType

    geom_types = []
    for _, desc in model.iter_descs():
        for collider in desc.colliders.values():
            if collider.__class__.__name__ == "ContactSphere":
                geom_types.append(GeomType.SPHERE)
            else:
                print(
                    f"Warning: Unrecognized collider type {collider.__class__.__name__}")
                assert False

    return geom_types


def get_collider_body_ids(model: CheckedModel) -> list[int]:
    collider_body_ids = []
    for _, desc in model.iter_descs():
        for collider in desc.colliders.values():
            parent_body_name = remove_prefix(collider.socket_frame)
            body_idx = model.get_body_index(parent_body_name)
            collider_body_ids.append(body_idx)
    return collider_body_ids


def get_collider_size(model: CheckedModel) -> list[list[float]]:
    collider_sizes = []
    for _, desc in model.iter_descs():
        for collider in desc.colliders.values():
            collider_sizes.append(collider.size())
    return collider_sizes


def get_collider_pos(model: CheckedModel) -> list[list[float]]:
    geom_positions = []
    for _, desc in model.iter_descs():
        for collider in desc.colliders.values():
            loc = collider.location
            geom_positions.append([loc.x, loc.y, loc.z])
    return geom_positions


def get_collider_rot(model: CheckedModel) -> list[list[float]]:
    from scipy.spatial.transform import Rotation as R
    geom_rotations = []
    for _, desc in model.iter_descs():
        for collider in desc.colliders.values():
            rot = collider.orientation
            r = R.from_euler('YXZ', [-rot.z, rot.x, rot.y],
                             degrees=True)  # note conversion from y-up to z-up
            quat = r.as_quat()
            geom_rotations.append([quat[3], quat[0], quat[1], quat[2]])
    return geom_rotations


def get_site_body_ids(model: CheckedModel) -> list[int]:
    site_body_ids = []
    for muscle in model.force_set.muscles.values():
        for path_point in muscle.geometry_path.path_point_set.path_points.values():
            parent_body_name = remove_prefix(path_point.socket_parent_frame)
            body_idx = model.get_body_index(parent_body_name)
            site_body_ids.append(body_idx)
    return site_body_ids


def get_site_pos(model: CheckedModel) -> list[list[float]]:
    site_positions = []
    for muscle in model.force_set.muscles.values():
        for path_point in muscle.geometry_path.path_point_set.path_points.values():
            loc = path_point.location
            site_positions.append([loc.x, loc.y, loc.z])
    return site_positions


def get_muscle_num_pts(model: CheckedModel) -> list[int]:
    muscle_pts_counts = []
    for muscle in model.force_set.muscles.values():
        muscle_pts_counts.append(
            len(muscle.geometry_path.path_point_set.path_points))
    return muscle_pts_counts


def get_dof_body_ids(model: CheckedModel) -> list[int]:
    dof_body_ids = []
    for _, joint in model.iter_joints():
        child_frame = get_joint_frame(joint, joint.socket_child_frame)
        child_name = remove_prefix(child_frame.socket_parent)
        body_idx = model.get_body_index(child_name)
        for _ in joint.coordinates:
            dof_body_ids.append(body_idx)
    return dof_body_ids


def create_body_tree(model: CheckedModel) -> list[tuple[int, ...]]:
    body_to_level = {}
    # starting with root
    root_body = model.get_root_body()
    body_to_level[root_body.name] = 0

    # Should be a forward pass
    for _, desc in model.iter_descs():
        body_name = desc.body.name
        if body_name in body_to_level:
            continue

        joint = desc.joint
        child_frame = get_joint_frame(joint, joint.socket_child_frame)
        child_body_name = remove_prefix(child_frame.socket_parent)
        parent_frame = get_joint_frame(
            joint, joint.socket_parent_frame)
        parent_body_name = remove_prefix(parent_frame.socket_parent)
        parent_level = body_to_level[parent_body_name]
        body_to_level[child_body_name] = parent_level + 1
    max_level = max(body_to_level.values())
    body_tree = [tuple() for _ in range(max_level + 1)]
    for body_name, level in body_to_level.items():
        body_idx = model.get_body_index(body_name)
        body_tree[level] += (body_idx,)
    return body_tree


def compute_expanded_parent(
        model: CheckedModel,
        jnt_dof_adr: list[int]
) -> list[int]:
    nv = sum(get_joint_num_dofs(model, vel_dofs=True))
    nb = num_bodies(model)

    # "lambda" function in Featherstone's book
    def lp(body_idx):
        parent_id = model.get_body_parent_idx(body_idx - 1)
        return parent_id + 1

    # Initialize (0, nv]
    expanded_parent = list(range(nv))

    for i in range(1, nb):
        expanded_parent[jnt_dof_adr[i - 1]] = jnt_dof_adr[lp(i)]
    expanded_parent = [p - 1 for p in expanded_parent]  # to zero-based
    return expanded_parent
