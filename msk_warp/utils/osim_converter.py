import warp as wp
import numpy as np
from collections import OrderedDict
from typing import Optional

from .converted_objs import *
from .osim_objs import (Model, ForceSet, Body, Joint, Collider, FunctionType,
                        Vector3, Inertia, AttachedGeometry, WeldJoint, FreeJoint,
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
    root_free: bool

    def iter_descs(self):
        """ iterator for body descriptions """
        for body_name, full_desc in self.body_full_desc.items():
            yield body_name, full_desc

    def iter_bodies(self):
        """ iterator for bodies """
        for body_name, full_desc in self.body_full_desc.items():
            yield body_name, full_desc.body

    def iter_joints(self):
        """ iterator for joints """
        for body_name, full_desc in self.body_full_desc.items():
            yield body_name, full_desc.joint

    def iter_dof_limits(self):
        """ iterator for coordinates with limits """
        for _, joint in self.iter_joints():
            if joint.connects_to_ground():  # no limits (either free or fixed)
                continue

            for coord in joint.coordinates:
                if coord.clamped:
                    yield coord

    def iter_cst_joints(self):
        """ iterator for custom joints """
        for _, jnt in self.iter_joints():
            if jnt.__class__.__name__ == "CustomJoint":
                yield jnt

    def iter_transform_axes(self):
        """ iterator for the transform axes of custom joints """
        for jnt in self.iter_cst_joints():
            spt_txfm = jnt.spatial_transform
            transform_axes = spt_txfm.transform_axes
            for axis in transform_axes:
                yield axis

    def iter_fns(self):
        """ iterator for functions used in transform axes """
        for axis in self.iter_transform_axes():
            yield axis.function, axis.coordinates

    def iter_muscles(self):
        """ iterator for muscles """
        for muscle_id, muscle in enumerate(self.force_set.muscles.values()):
            yield muscle_id, muscle

    def iter_actuators(self):
        """ iterator for actuators """
        for actuator_id, actuator in enumerate(
                self.force_set.actuators.values()):
            yield actuator_id, actuator

    def iter_path_points(self):
        """ iterator for muscle path points """
        for muscle_id, muscle in self.iter_muscles():
            geom_path = muscle.geometry_path
            for path_point in geom_path.path_point_set.path_points.values():
                yield muscle_id, path_point

    def iter_colliders(self):
        """ iterator for colliders """
        for body_name, full_desc in self.body_full_desc.items():
            for collider_name, collider in full_desc.colliders.items():
                yield (body_name, collider_name), collider

    def iter_visuals(self):
        """ iterator for visual meshes """
        for body_name, desc in self.iter_descs():
            for mesh in desc.body.attached_geometry.meshes:
                yield body_name, mesh

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

    def lookup_dof_idx(self, coord_name: str, pos: bool) -> int:
        """ Lookup dof index, not to be used for root """
        dof_idx = 0
        for _, joint in self.iter_joints():
            joint_dof_tmp = 0
            for coord in joint.coordinates:
                if coord.name == coord_name:
                    return dof_idx + joint_dof_tmp
                joint_dof_tmp += 1

            dof_idx += joint.num_pos_dofs() if pos else joint.num_dofs()
        raise ValueError(f"Coordinate name {coord_name} not found.")


def remove_prefix(name: str) -> str:
    # get after the last "/"
    if "/" not in name:
        return name
    return name.split("/")[-1]


def to_checked_model(model: Model, root_free: bool) -> CheckedModel:
    body_full_desc = OrderedDict()

    # Before we do anything else, we should properly set the ground-root joint
    for joint in model.joint_set.joints.values():
        if joint.connects_to_ground():
            ground_root_joint = joint
            break
    else:
        assert False, "No ground-root joint found in the model."

    if root_free:
        new_ground_root_joint = FreeJoint.from_joint(ground_root_joint)
    else:
        new_ground_root_joint = WeldJoint.from_joint(ground_root_joint)
        new_ground_root_joint.coordinates = []
    model.joint_set.joints[ground_root_joint.name] = new_ground_root_joint

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
        for frame in joint.frames:
            if frame.name == child_frame_name:
                child_frame = frame
                break
        else:
            assert False, f"Child frame {child_frame_name} not found in joint {joint.name}"
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
        force_set=model.force_set,
        root_free=root_free
    )


def num_bodies(model: CheckedModel) -> int:
    return len(model.body_full_desc)


def num_functions(model: CheckedModel) -> int:
    return len(list(model.iter_fns()))


def num_muscles(model: CheckedModel) -> int:
    return len(model.force_set.muscles)


def num_actuators(model: CheckedModel) -> int:
    return len(model.force_set.actuators)


def num_colliders(model: CheckedModel) -> int:
    nc = 0
    for _, desc in model.iter_descs():
        nc += len(desc.colliders)
    return nc


def num_visuals(model: CheckedModel) -> int:
    nv = 0
    for _, desc in model.iter_descs():
        nv += len(desc.body.attached_geometry.meshes)
    return nv


def get_joint_num_dofs(model: CheckedModel, vel_dofs: bool) -> list[int]:
    joint_num_dofs = []
    for _, desc in model.iter_descs():
        joint = desc.joint
        if joint.connects_to_ground():
            if model.root_free:
                joint_num_dofs.append(6 if vel_dofs else 7)
            else:
                joint_num_dofs.append(0)
            continue
        joint_num_dofs.append(joint.num_dofs() if vel_dofs else joint.num_pos_dofs())
    return joint_num_dofs


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


def get_body_unit_inertias_OB_B(model: CheckedModel) -> list[list[float]]:
    unit_inertias = []
    for _, desc in model.iter_descs():
        body = desc.body
        mass = body.mass
        inertia = body.inertia
        if mass == 0.0:
            unit_inertias.append(wp.mat33(0.0))
            continue

        inertia_in_com = np.array(
            [
                [inertia.xx, inertia.xy, inertia.xz],
                [inertia.xy, inertia.yy, inertia.yz],
                [inertia.xz, inertia.yz, inertia.zz],
            ]
        )

        mass_center = -np.array([body.mass_center.x, body.mass_center.y, body.mass_center.z])

        # Perform shift to body frame
        mp = mass_center * mass
        mxx = mp[0] * mass_center[0]
        myy = mp[1] * mass_center[1]
        mzz = mp[2] * mass_center[2]
        nmx = -mp[0]
        nmy = -mp[1]
        point_mass = np.array([[myy + mzz, nmx * mass_center[1], nmx * mass_center[2]],
                               [nmx * mass_center[1], mxx + mzz, nmy * mass_center[2]],
                               [nmx * mass_center[2], nmy * mass_center[2], mxx + myy]])

        inertia_in_B = inertia_in_com + point_mass
        unit_inertia_in_B = inertia_in_B / mass
        unit_inertias.append(wp.mat33(
            unit_inertia_in_B[0, 0], unit_inertia_in_B[0, 1], unit_inertia_in_B[0, 2],
            unit_inertia_in_B[1, 0], unit_inertia_in_B[1, 1], unit_inertia_in_B[1, 2],
            unit_inertia_in_B[2, 0], unit_inertia_in_B[2, 1], unit_inertia_in_B[2, 2],
        ))

    return unit_inertias


def get_body_mass_center(model: CheckedModel) -> list[wp.vec3]:
    mass_centers = []
    for _, desc in model.iter_descs():
        body = desc.body
        com = body.mass_center
        mass_center = wp.vec3(com.x, com.y, com.z)
        mass_centers.append(mass_center)
    return mass_centers


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


def get_joint_types(model: CheckedModel) -> list[types.MobilizerType]:
    joint_types = []
    for _, joint in model.iter_joints():
        class_name = joint.__class__.__name__
        if class_name == "FreeJoint":
            joint_types.append(types.MobilizerType.FREE)
        elif class_name == "PinJoint":
            joint_types.append(types.MobilizerType.PIN)
        elif class_name == "UniversalJoint":
            joint_types.append(types.MobilizerType.UNIVERSAL)
        elif class_name == "GimbalJoint":
            joint_types.append(types.MobilizerType.GIMBAL)
        elif class_name == "BallJoint":
            joint_types.append(types.MobilizerType.BALL)
        elif class_name == "BeamJoint":
            joint_types.append(types.MobilizerType.BEAM)
        elif class_name == "EllipsoidJoint":
            joint_types.append(types.MobilizerType.ELLIPSOID)
        elif class_name == "CustomJoint":
            joint_types.append(types.MobilizerType.CUSTOM)
        elif class_name == "WeldJoint":
            joint_types.append(types.MobilizerType.WELD)
        elif class_name == "DummyJoint":
            joint_types.append(types.MobilizerType.DUMMY)
        else:
            assert False, f"Unrecognized joint type {joint.__class__.__name__}"
    return joint_types


def get_joint_rel_transform(
        model: CheckedModel,
        parent: bool
) -> list[wp.transform]:
    rel_transforms = []
    for _, joint in model.iter_joints():
        if parent:
            frame = get_frame_from_joint(joint, joint.socket_parent_frame)
        else:
            frame = get_frame_from_joint(joint, joint.socket_child_frame)
        pos = frame.translation
        rot = frame.orientation
        transform = wp.transform(wp.vec3(pos.x, pos.y, pos.z), wp.quat(rot.x, rot.y, rot.z, rot.w))

        if not parent:
            transform = wp.transform_inverse(transform)
        rel_transforms.append(transform)

    return rel_transforms


def get_joint_extra_info(model: CheckedModel) -> list[list[float]]:
    extra_info = []
    for _, joint in model.iter_joints():
        extra_info.append(joint.extra_info())
    return extra_info


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
        rot = [rot.x, rot.y, rot.z, rot.w]
        transform = wp.transform(wp.vec3(pos[0], pos[1], pos[2]), wp.quat(rot[0], rot[1], rot[2], rot[3]))
        # MuJoCo: sliding, torsional, rolling friction
        # Hunt-Crossley: static, dynamic, viscous
        friction = [0.9, 0.6, 0.]  # default friction values
        stiffness = 5e6 ** (2.0 / 3.0)
        dissipation = 1.0
        transition_velocity = 0.1
        priority = 0
        aabb = collider.get_aabb()
        rbound = collider.get_rbound()
        pc_filter = collider.pc_filter

        collider_data.type.append(geom_type)
        collider_data.body_id.append(body_id)
        collider_data.size.append(size)
        collider_data.transform.append(transform)
        collider_data.friction.append(friction)
        collider_data.stiffness.append(stiffness)
        collider_data.dissipation.append(dissipation)
        collider_data.transition_velocity.append(transition_velocity)
        collider_data.priority.append(priority)
        collider_data.aabb.append(aabb)
        collider_data.rbound.append(rbound)
        collider_data.pc_filter.append(pc_filter)

    return collider_data


def get_visual_data(model: CheckedModel) -> VisualData:
    visual_data = VisualData()

    for body_name, mesh in model.iter_visuals():
        body_id = model.get_body_index(body_name)
        size = mesh.scale_factors
        mesh_file = mesh.mesh_file

        # TODO: support socket frame for meshes
        visual_data.body_id.append(body_id)
        visual_data.transform.append(wp.transform_identity())
        visual_data.scale.append([size.x, size.y, size.z])
        visual_data.file.append(mesh_file)

    return visual_data


def get_muscle_names(model: CheckedModel) -> list[str]:
    muscle_names = []
    for _, muscle in model.iter_muscles():
        muscle_names.append(muscle.name)
    return muscle_names


def get_muscle_num_pts(model: CheckedModel) -> list[int]:
    muscle_pts_counts = []
    for _, muscle in model.iter_muscles():
        muscle_pts_counts.append(
            len(muscle.geometry_path.path_point_set.path_points))
    return muscle_pts_counts


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


def get_fn_data(model: CheckedModel) -> tuple[LinearFunctionData, ConstantFunctionData, PolynomialFunctionData]:
    linear_fn_data = LinearFunctionData()
    constant_fn_data = ConstantFunctionData()
    poly_fn_data = PolynomialFunctionData()

    for fn_idx, (fn, coordinates) in enumerate(model.iter_fns()):
        if fn.type() == FunctionType.LINEAR:
            coefficients = fn.coefficients
            m, b = coefficients.x, coefficients.y
            linear_fn_data.mb.append((m, b))
            linear_fn_data.fn_idx.append(fn_idx)
            linear_fn_data.qpos_adr.append(model.lookup_dof_idx(coordinates, True))
        elif fn.type() == FunctionType.CONSTANT:
            c = fn.value
            constant_fn_data.c.append(c)
            constant_fn_data.fn_idx.append(fn_idx)
        elif fn.type() == FunctionType.POLYNOMIAL:
            coefficients = fn.coefficients
            poly_fn_data.coefficients.append([coefficients[i] for i in range(len(coefficients))])
            poly_fn_data.fn_idx.append(fn_idx)
            poly_fn_data.qpos_adr.append(model.lookup_dof_idx(coordinates, True))
        else:
            assert False, f"Unsupported function type {fn.type()}"
    return linear_fn_data, constant_fn_data, poly_fn_data


def get_mob_to_cst_idx(model: CheckedModel) -> list[int]:
    mob_to_cst_idx = [-1 for _ in model.iter_joints()]
    num_cst_joints = 0
    for i, (_, jnt) in enumerate(model.iter_joints()):
        if jnt.__class__.__name__ == "CustomJoint":
            mob_to_cst_idx[i] = num_cst_joints
            num_cst_joints += 1
    return mob_to_cst_idx


def get_cst_to_mob_idx(model: CheckedModel) -> list[int]:
    cst_mob_idx = []
    for i, (_, jnt) in enumerate(model.iter_joints()):
        if jnt.__class__.__name__ == "CustomJoint":
            cst_mob_idx.append(i)
    return cst_mob_idx


def get_cst_txfm_axes(model: CheckedModel) -> list[list[int]]:
    cst_txfm_axes = []
    for jnt in model.iter_cst_joints():
        spt_txfm = jnt.spatial_transform
        transform_axes = spt_txfm.transform_axes
        txfm_axes = [axis.axis.to_list() for axis in transform_axes]
        cst_txfm_axes.append(txfm_axes)

    # hacky: if there are no custom joints, we still need to return something
    if len(cst_txfm_axes) == 0:
        cst_txfm_axes.append([0, 0, 0] * 6)
    return cst_txfm_axes


def get_cst_txfm_dof(model: CheckedModel) -> list[list[int]]:
    cst_txfm_dof = []
    for jnt in model.iter_cst_joints():
        # The joint starts at this dof address
        joint_coords = [joint_coord.name for joint_coord in jnt.coordinates]
        joint_coord_adr = min([model.lookup_dof_idx(joint_coord, False) for joint_coord in joint_coords])

        # Compute the relative dof addresses of the transform axes
        spt_txfm = jnt.spatial_transform
        transform_axes = spt_txfm.transform_axes
        txfm_coords = [axis.coordinates for axis in transform_axes]
        txfm_dof_adr = [model.lookup_dof_idx(txfm_coord, False) for txfm_coord in txfm_coords]
        txfm_dof_adr = [adr - joint_coord_adr for adr in txfm_dof_adr]

        cst_txfm_dof.append(txfm_dof_adr)

    # hacky: if there are no custom joints, we still need to return something
    if len(cst_txfm_dof) == 0:
        cst_txfm_dof.append([0] * 6)
    return cst_txfm_dof


def get_dof_limits(
        model: CheckedModel
) -> tuple[list[tuple[float, float]], list[int], list[int]]:
    dof_ranges = []
    dof_adr, dof_qadr = [], []
    for coord in model.iter_dof_limits():
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

        muscle_meta.activation_time_const = 0.015
        muscle_meta.deactivation_time_const = 0.060
        muscle_meta.activation_dynamics_smoothing = 0.1

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
        # Reasonable defaults for specific tension, density, slow twitch ratio
        muscle_meta.specific_tension = 0.5e6
        muscle_meta.density = 1059.7
        muscle_meta.slow_twitch_ratio = 0.5

        metadata.append(muscle_meta)

    return metadata


def get_actuator_metadata(osim_model: CheckedModel) -> list[
    types.ActuatorMetadata]:
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

    # if max_pennation_angle > 1e-8:
    #     minimum_fiber_length = (np.sin(optimal_pennation_angle) / np.sin(
    #         max_pennation_angle))
    # else:
    #     minimum_fiber_length = 0.01
    # minimum_fiber_length = max(minimum_fiber_length, min_norm_fiber_length)
    minimum_fiber_length = min_norm_fiber_length
    return minimum_fiber_length, max_norm_fiber_length


def get_body_id_lookup(model: CheckedModel) -> dict[str, int]:
    body_id_lookup = {}
    for body_idx, (_, body) in enumerate(model.iter_bodies()):
        body_id_lookup[body.name] = body_idx
    return body_id_lookup


def get_dof_id_lookup(model: CheckedModel) -> dict[str, tuple[int, int]]:
    dof_id_lookup = {}
    for _, joint in model.iter_joints():
        for coord in joint.coordinates:
            qpos_idx = model.lookup_dof_idx(coord.name, True)
            dof_idx = model.lookup_dof_idx(coord.name, False)
            # Ignore root if free
            if model.root_free and qpos_idx <= 6:
                continue
            dof_id_lookup[coord.name] = (qpos_idx, dof_idx)

    # Manually add pelvis dofs
    if model.root_free:
        dof_id_lookup["pelvis_rot_x"] = (0, 0)
        dof_id_lookup["pelvis_rot_y"] = (1, 1)
        dof_id_lookup["pelvis_rot_z"] = (2, 2)
        dof_id_lookup["pelvis_rot_w"] = (3, -1)
        dof_id_lookup["pelvis_tx"] = (4, 3)
        dof_id_lookup["pelvis_ty"] = (5, 4)
        dof_id_lookup["pelvis_tz"] = (6, 5)
    return dof_id_lookup


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


def get_limit_id_lookup(model: CheckedModel) -> dict[str, int]:
    limit_id_lookup = {}
    limit_idx = 0
    for coord in model.iter_dof_limits():
        limit_id_lookup[coord.name] = limit_idx
        limit_idx += 1
    return limit_id_lookup


def get_collider_id_lookup(model: CheckedModel) -> dict[str, int]:
    collider_id_lookup = {}
    collider_idx = 0
    for (body_name, collider_name), _ in model.iter_colliders():
        collider_id_lookup[collider_name] = collider_idx
        collider_idx += 1
    return collider_id_lookup
