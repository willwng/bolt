from collections import OrderedDict
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from .converted_objs import *
from .osim_objs import (Model, ForceSet, Body, Joint, Collider, FunctionType,
                        Vector3, Quat, Inertia, AttachedGeometry,
                        DummyJoint, Muscle, _VOID_NAME)


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

    def iter_muscles(self):
        for muscle_id, muscle in enumerate(self.force_set.muscles.values()):
            yield muscle_id, muscle

    def iter_actuators(self):
        for actuator_id, actuator in enumerate(self.force_set.actuators.values()):
            yield actuator_id, actuator

    def iter_path_points(self):
        for muscle_id, muscle in self.iter_muscles():
            geom_path = muscle.geometry_path
            for path_point in geom_path.path_point_set.path_points.values():
                yield muscle_id, path_point

    def iter_colliders(self):
        for body_name, full_desc in self.body_full_desc.items():
            for collider_name, collider in full_desc.colliders.items():
                yield (body_name, collider_name), collider

    def iter_visuals(self):
        for body_name, desc in self.iter_descs():
            for mesh in desc.body.attached_geometry.meshes:
                yield body_name, mesh

    def iter_cst_joints(self):
        for _, jnt in self.iter_joints():
            if "ground" in jnt.socket_parent_frame:
                continue
            if jnt.__class__.__name__ == "CustomJoint":
                yield jnt

    def iter_transform_axes(self):
        # Transform axes for custom joints first
        for jnt in self.iter_cst_joints():
            spt_txfm = jnt.spatial_transform
            transform_axes = spt_txfm.transform_axes
            for axis in transform_axes:
                yield axis

    def iter_fns(self):
        for axis in self.iter_transform_axes():
            yield axis.function

    def get_body_index(self, body_name: str) -> int:
        if body_name == _VOID_NAME:
            return 0
        for idx, name in enumerate(self.body_full_desc.keys()):
            if name == body_name:
                return idx
        raise ValueError(f"Body name {body_name} not found.")

    def get_world_body(self) -> Body:
        return self.body_full_desc["ground"].body

    def is_world(self, body_name: str) -> bool:
        return body_name == self.body_full_desc["ground"].body.name

    def get_body_parent_name(self, body_name: str) -> Optional[str]:
        if self.is_world(body_name):
            return _VOID_NAME

        full_desc = self.body_full_desc[body_name]
        joint = full_desc.joint
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
        parent_frame = None
        for frame in joint.frames:
            if frame.name == joint.socket_parent_frame:
                parent_frame = frame
                break
        assert parent_frame is not None
        parent_body_name = remove_prefix(parent_frame.socket_parent)
        return self.get_body_index(parent_body_name)

    # don't use this for root dofs
    def lookup_dof_idx(self, coord_name: str, pos: bool) -> int:
        dof_idx = 0
        for _, joint in self.iter_joints():
            for coord in joint.coordinates:
                if coord.name == coord_name:
                    return dof_idx + 1 if pos else dof_idx
                dof_idx += 1
        raise ValueError(f"Coordinate name {coord_name} not found.")


def convert_y_up_z_up(model: CheckedModel):
    rot_convert = R.from_euler("x", 90, degrees=True)
    rot_mat = rot_convert.as_matrix()

    def convert_vec(v: Vector3) -> Vector3:
        vec = rot_mat @ [v.x, v.y, v.z]
        return Vector3(x=vec[0], y=vec[1], z=vec[2])

    def convert_quat(q: Quat) -> Quat:
        r = R.from_quat([q.x, q.y, q.z, q.w])
        r_converted = rot_convert * r * rot_convert.inv()
        q_converted = r_converted.as_quat()
        return Quat(w=q_converted[3], x=q_converted[0],
                    y=q_converted[1], z=q_converted[2])

    def convert_rel_quat(q: Quat) -> Quat:
        r = R.from_quat([q.x, q.y, q.z, q.w])
        r_converted = rot_convert * r
        q_converted = r_converted.as_quat()
        return Quat(w=q_converted[3], x=q_converted[0],
                    y=q_converted[1], z=q_converted[2])

    for _, desc in model.iter_descs():
        # Body mass properties
        body = desc.body
        body.mass_center = convert_vec(body.mass_center)
        inertia = body.inertia
        # Rotate inertia tensor
        inertia_mat = [[inertia.xx, inertia.xy, inertia.xz],
                       [inertia.xy, inertia.yy, inertia.yz],
                       [inertia.xz, inertia.yz, inertia.zz]]
        rotated_inertia = rot_mat @ inertia_mat @ rot_mat.T
        body.inertia.xx = rotated_inertia[0][0]
        body.inertia.yy = rotated_inertia[1][1]
        body.inertia.zz = rotated_inertia[2][2]
        body.inertia.xy = rotated_inertia[0][1]
        body.inertia.xz = rotated_inertia[0][2]
        body.inertia.yz = rotated_inertia[1][2]

        # Joint connections
        joint = desc.joint
        for frame in joint.frames:
            frame.translation = convert_vec(frame.translation)
            frame.orientation = convert_quat(frame.orientation)

        # Attached geom
        for collider in desc.colliders.values():
            collider.location = convert_vec(collider.location)
            collider.orientation = convert_rel_quat(collider.orientation)

    # Custom joints spatial transform axes
    for axis in model.iter_transform_axes():
        axis.axis = convert_vec(axis.axis)

    # All muscle points
    for muscle in model.force_set.muscles.values():
        geom_path = muscle.geometry_path
        for path_point in geom_path.path_point_set.path_points.values():
            path_point.location = convert_vec(path_point.location)
    return model


def remove_prefix(name: str) -> str:
    # get after the last "/"
    if "/" not in name:
        return name
    return name.split("/")[-1]


def to_checked_model(model: Model) -> CheckedModel:
    body_full_desc = OrderedDict()

    # All colliders for each body
    body_name_to_colliders = {}
    for collider in model.contact_geometry_set.contact_geom.values():
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

    # Create a body for the ground
    ground_body = Body(name="ground",
                       attached_geometry=AttachedGeometry(meshes=[]),
                       mass=0.0,
                       mass_center=Vector3(0.0, 0.0, 0.0),
                       inertia=Inertia(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    body_full_desc["ground"] = FullBodyDesc(
        body=ground_body,
        joint=DummyJoint(),
        colliders=body_name_to_colliders.get("ground", OrderedDict())
    )

    # Todo: make sure this is in forward kinematic order
    body_ordering = list(model.body_set.bodies.keys())

    # Now fill in the body_full_desc
    for body_name in body_ordering:
        body = model.body_set.bodies[body_name]
        joint = body_name_to_joint.get(body_name, None)
        colliders = body_name_to_colliders.get(body_name, OrderedDict())
        body_full_desc[body_name] = FullBodyDesc(
            body=body,
            joint=joint,
            colliders=colliders
        )

    return CheckedModel(
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
        joint_num_dofs.append(joint.num_dofs() if vel_dofs else
                              joint.num_pos_dofs())
    return joint_num_dofs


def num_muscles(model: CheckedModel) -> int:
    return len(model.force_set.muscles)


def num_actuators(model: CheckedModel) -> int:
    return len(model.force_set.actuators)


def num_colliders(model: CheckedModel) -> int:
    num_colliders = 0
    for _, desc in model.iter_descs():
        num_colliders += len(desc.colliders)
    return num_colliders


def num_visuals(model: CheckedModel) -> int:
    num_visuals = 0
    for _, desc in model.iter_descs():
        num_visuals += len(desc.body.attached_geometry.meshes)
    return num_visuals


def get_site_data(model: CheckedModel) -> SiteData:
    """
    Returns number of sites, and number of conditional sites
    """
    site_data = SiteData()
    for i, (muscle_id, path_point) in enumerate(model.iter_path_points()):
        # Body id
        parent_body_name = remove_prefix(path_point.socket_parent_frame)
        body_idx = model.get_body_index(parent_body_name)
        site_data.body_id.append(body_idx)

        # Position
        loc = path_point.location
        site_data.pos.append([loc.x, loc.y, loc.z])

        # Check conditional
        if path_point.is_conditional():
            coordinate = path_point.get_coordinate()
            cond_range = path_point.get_range()
            qadr = model.lookup_dof_idx(remove_prefix(coordinate), True)

            site_data.conditional_ids.append(i)
            site_data.conditional_qadr.append(qadr)
            site_data.conditional_range.append([cond_range.x, cond_range.y])
            site_data.nsite_cond += 1

        site_data.nsite += 1

    return site_data


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
        assert abs(inertia.xy) <= 1e-12
        assert abs(inertia.xz) <= 1e-12
        assert abs(inertia.yz) <= 1e-12
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


def get_frame_from_joint(joint, frame_name: str):
    for frame in joint.frames:
        if frame.name == frame_name:
            return frame
    return None


def get_body_parent_ids(model: CheckedModel) -> list[int]:
    parent_ids = []
    for _, body in model.iter_bodies():
        parent_name = model.get_body_parent_name(body.name)
        parent_idx = model.get_body_index(parent_name)
        parent_ids.append(parent_idx)
    return parent_ids


def get_joint_types(model: CheckedModel) -> list[types.JointType]:
    joint_types = []
    for _, joint in model.iter_joints():
        if joint.connects_to_ground():
            joint_types.append(types.JointType.FREE)
            continue

        class_name = joint.__class__.__name__
        if class_name == "PinJoint":
            joint_types.append(types.JointType.PIN)
        elif class_name == "UniversalJoint":
            joint_types.append(types.JointType.UNIVERSAL)
        elif class_name == "BallJoint":
            joint_types.append(types.JointType.BALL)
        elif class_name == "CustomJoint":
            joint_types.append(types.JointType.CUSTOM)
        elif class_name == "DummyJoint":
            joint_types.append(types.JointType.DUMMY)
        else:
            print(
                f"Warning: Unrecognized joint type {joint.__class__.__name__}")
            assert False
    return joint_types


def get_joint_rel_pos(
        model: CheckedModel,
        get_parent_rel: bool
) -> list[list[float]]:
    rel_parent_positions = []
    for _, joint in model.iter_joints():
        if get_parent_rel:
            frame = get_frame_from_joint(joint, joint.socket_parent_frame)
        else:
            frame = get_frame_from_joint(joint, joint.socket_child_frame)
        pos = frame.translation
        rel_parent_positions.append([pos.x, pos.y, pos.z])
    return rel_parent_positions


def get_joint_rel_rot(model: CheckedModel, parent: bool) -> list[list[float]]:
    rel_parent_rots = []
    for _, joint in model.iter_joints():
        if parent:
            frame = get_frame_from_joint(joint, joint.socket_parent_frame)
        else:
            frame = get_frame_from_joint(joint, joint.socket_child_frame)
        rot = frame.orientation if parent else frame.orientation.inv()
        rel_parent_rots.append([rot.w, rot.x, rot.y, rot.z])
    return rel_parent_rots


def get_collider_data(model: CheckedModel) -> ColliderData:
    collider_data = ColliderData()

    for _, collider in model.iter_colliders():
        class_name = collider.__class__.__name__
        if class_name == "ContactSphere":
            geom_type = types.GeomType.SPHERE
        elif class_name == "ContactCapsule":
            geom_type = types.GeomType.CAPSULE
        elif class_name == "ContactHalfSpace":
            geom_type = types.GeomType.PLANE
        else:
            assert False, f"Unrecognized collider type {class_name}"

        parent_body_name = remove_prefix(collider.socket_frame)
        body_id = model.get_body_index(parent_body_name)
        size = collider.size()
        loc, rot = collider.location, collider.orientation
        pos = [loc.x, loc.y, loc.z]
        rot = [rot.w, rot.x, rot.y, rot.z]
        # MuJoCo: sliding, torsional, rolling friction
        # Hunt-Crossley: static, dynamic, viscous
        friction = [0.95, 0.3, 0.3]  # default friction values
        aabb = collider.get_aabb()
        rbound = collider.get_rbound()

        collider_data.type.append(geom_type)
        collider_data.body_id.append(body_id)
        collider_data.size.append(size)
        collider_data.pos.append(pos)
        collider_data.rot.append(rot)
        collider_data.friction.append(friction)
        collider_data.aabb.append(aabb)
        collider_data.rbound.append(rbound)

    return collider_data


def get_visual_data(model: CheckedModel) -> VisualData:
    visual_data = VisualData()

    for body_name, mesh in model.iter_visuals():
        body_id = model.get_body_index(body_name)
        size = mesh.scale_factors
        mesh_file = mesh.mesh_file

        visual_data.body_id.append(body_id)
        visual_data.pos.append([0.0, 0.0, 0.0])
        visual_data.rot.append([1.0, 0.0, 0.0, 0.0])
        visual_data.scale.append([size.x, size.y, size.z])
        visual_data.file.append(mesh_file)

    return visual_data


def get_muscle_num_pts(model: CheckedModel) -> list[int]:
    muscle_pts_counts = []
    for _, muscle in model.iter_muscles():
        muscle_pts_counts.append(
            len(muscle.geometry_path.path_point_set.path_points))
    return muscle_pts_counts


def get_dof_body_ids(model: CheckedModel) -> list[int]:
    dof_body_ids = []
    for _, joint in model.iter_joints():
        if len(joint.coordinates) == 0:  # no dofs
            continue
        # get child body (the body the dofs belong to)
        child_frame = get_frame_from_joint(joint, joint.socket_child_frame)
        child_name = remove_prefix(child_frame.socket_parent)
        body_idx = model.get_body_index(child_name)
        for _ in joint.coordinates:
            dof_body_ids.append(body_idx)
    return dof_body_ids


def create_body_tree(model: CheckedModel) -> list[tuple[int, ...]]:
    body_to_level = {}
    # starting with root
    root_body = model.get_world_body()
    body_to_level[root_body.name] = 0

    # Should be a forward pass: todo make sure these are in fk order
    for _, desc in model.iter_descs():
        body_name = desc.body.name
        if body_name in body_to_level:
            continue

        joint = desc.joint
        child_frame = get_frame_from_joint(joint, joint.socket_child_frame)
        child_body_name = remove_prefix(child_frame.socket_parent)
        parent_frame = get_frame_from_joint(
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


def get_subtree_mass(model: CheckedModel) -> list[float]:
    body_masses_list = body_masses(model)
    subtree_masses = body_masses_list.copy()

    body_tree = create_body_tree(model)

    # Traverse from leaves to root
    for level in reversed(body_tree):
        for body_idx in level:
            parent_idx = model.get_body_parent_idx(body_idx)
            if parent_idx == -1:
                continue
            subtree_masses[parent_idx] += subtree_masses[body_idx]

    return subtree_masses


def make_tiles(
        model: CheckedModel,
        expanded_parent: list[int]
) -> dict[int, list[int]]:
    nv = sum(get_joint_num_dofs(model, vel_dofs=True))
    # qM_tiles records the block diagonal structure of qM
    tile_corners = [i for i in range(nv) if expanded_parent[i] == -1]
    tiles = {}
    for i in range(len(tile_corners)):
        tile_beg = tile_corners[i]
        tile_end = nv if i == len(tile_corners) - 1 else tile_corners[i + 1]
        tiles.setdefault(tile_end - tile_beg, []).append(tile_beg)
    return tiles


def get_functions(
        model: CheckedModel
) -> tuple[list[tuple[float, float]], list[float]]:
    """
    Returns linear functions and constant functions
    [(m1, b1), (m2, b2), ...] for linear functions
    [c1, c2, ...] for constant functions
    """
    linear_fns = []
    constant_fns = []
    for fn in model.iter_fns():
        if fn.type() == FunctionType.LINEAR:
            coefficients = fn.coefficients
            m, b = coefficients.x, coefficients.y
            linear_fns.append((m, b))
        elif fn.type() == FunctionType.CONSTANT:
            c = fn.value
            constant_fns.append(c)
        else:
            print(f"Warning: Unrecognized function type {fn.type()}")
            assert False
    return linear_fns, constant_fns


def get_txfm_fns(
        model: CheckedModel
) -> tuple[list[int], list[int], list[int], list[int], list[list[float]]]:
    from .._src.types import CustomFnType

    txfm_axes = []
    txfm_qpos_adr = []
    txfm_dof_adr = []
    fn_types = []
    fn_addresses = []

    const_idx, linear_idx = 0, 0
    for axis in model.iter_transform_axes():
        # function type
        fn = axis.function
        if fn.type() == FunctionType.LINEAR:
            fn_types.append(CustomFnType.LINEAR)
            fn_addresses.append(linear_idx)
            linear_idx += 1
            requires_dof = True
        elif fn.type() == FunctionType.CONSTANT:
            fn_types.append(CustomFnType.CONSTANT)
            fn_addresses.append(const_idx)
            const_idx += 1
            requires_dof = False
        else:
            print(f"Warning: Unrecognized function type {fn.type()}")
            assert False

        # coordinates
        coordinates = axis.coordinates
        if coordinates is None or not requires_dof:
            txfm_qpos_adr.append(-1)
            txfm_dof_adr.append(-1)
        else:
            qpos_adr = model.lookup_dof_idx(coordinates, True)
            dof_adr = model.lookup_dof_idx(coordinates, False)
            txfm_qpos_adr.append(qpos_adr)
            txfm_dof_adr.append(dof_adr)

        # axes
        txfm_axes.append([axis.axis.x, axis.axis.y, axis.axis.z])
    return fn_types, fn_addresses, txfm_qpos_adr, txfm_dof_adr, txfm_axes


def get_dof_limits(
        model: CheckedModel
) -> tuple[list[tuple[float, float]], list[int], list[int]]:
    dof_ranges = []
    dof_adr, dof_qadr = [], []
    for _, joint in model.iter_joints():
        # No joint limits for free joints. FIXME check for free joints better
        if joint.connects_to_ground():
            continue

        for coord in joint.coordinates:
            if coord.clamped:
                dof_ranges.append((coord.range.x, coord.range.y))
                dof_qadr.append(model.lookup_dof_idx(coord.name, True))
                dof_adr.append(model.lookup_dof_idx(coord.name, False))

    return dof_ranges, dof_adr, dof_qadr


def get_muscle_metadata(
        osim_model: CheckedModel,
        max_pennation_angle,
        min_norm_fiber_length,
        max_norm_fiber_length,
) -> list[types.MuscleMetadata]:
    metadata = []

    for _, muscle in osim_model.iter_muscles():
        muscle_meta = types.MuscleMetadata()
        muscle_meta.max_isometric_force = muscle.max_isometric_force
        muscle_meta.optimal_fiber_length = muscle.optimal_fiber_length
        muscle_meta.tendon_slack_length = muscle.tendon_slack_length
        muscle_meta.optimal_pennation_angle = muscle.pennation_angle_at_optimal
        muscle_meta.fiber_damping = 0.1
        muscle_meta.v_max = 12.0

        fl_range = get_muscle_fl_range(
            muscle,
            max_pennation_angle=max_pennation_angle,
            min_norm_fiber_length=min_norm_fiber_length,
            max_norm_fiber_length=max_norm_fiber_length,
        )
        muscle_meta.min_norm_fiber_length = fl_range[0]
        muscle_meta.max_norm_fiber_length = fl_range[1]
        muscle_meta.min_activation = 0.0
        muscle_meta.max_activation = 1.0
        metadata.append(muscle_meta)

    return metadata


def get_actuator_metadata(osim_model: CheckedModel) -> list[types.ActuatorMetadata]:
    metadata = []

    for _, actuator in osim_model.iter_actuators():
        am = types.ActuatorMetadata()
        am.optimal_force = actuator.optimal_force
        am.activation_time_constant = actuator.activation_time_constant
        am.coordinate = osim_model.lookup_dof_idx(actuator.coordinate, False)
        am.default_activation = actuator.default_activation

        am.min_activation = 0.0
        am.max_activation = 1.0
        metadata.append(am)

    return metadata


def get_muscle_fl_range(
        muscle: Muscle,
        max_pennation_angle,
        min_norm_fiber_length,
        max_norm_fiber_length,
) -> tuple[float, float]:
    optimal_pennation_angle = muscle.pennation_angle_at_optimal

    if max_pennation_angle > 1e-8:
        minimum_fiber_length = (np.sin(optimal_pennation_angle) / np.sin(
            max_pennation_angle))
    else:
        minimum_fiber_length = 0.01
    minimum_fiber_length = max(minimum_fiber_length, min_norm_fiber_length)
    return minimum_fiber_length, max_norm_fiber_length


def get_body_id_lookup(model: CheckedModel) -> dict[str, int]:
    body_id_lookup = {}
    for body_idx, (_, body) in enumerate(model.iter_bodies()):
        body_id_lookup[body.name] = body_idx
    return body_id_lookup


def get_muscle_id_lookup(model: CheckedModel) -> dict[str, int]:
    muscle_id_lookup = {}
    for muscle_idx, (_, muscle) in enumerate(model.iter_muscles()):
        muscle_id_lookup[muscle.name] = muscle_idx
    return muscle_id_lookup


def get_actuator_id_lookup(model: CheckedModel) -> dict[str, int]:
    actuator_id_lookup = {}
    for actuator_idx, (_, actuator) in enumerate(model.iter_actuators()):
        actuator_id_lookup[actuator.name] = actuator_idx
    return actuator_id_lookup
