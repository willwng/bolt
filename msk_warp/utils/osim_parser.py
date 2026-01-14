import xml.etree.ElementTree as ElementTree
from .osim_objs import *


def to_vector2(text: str) -> Vector2:
    v2 = tuple(map(float, text.split()))
    assert (len(v2) == 2)
    return Vector2(x=v2[0], y=v2[1])


def to_vector3(text: str) -> Vector3:
    v3 = tuple(map(float, text.split()))
    assert (len(v3) == 3)
    return Vector3(x=v3[0], y=v3[1], z=v3[2])


def to_vector6(text: str) -> Vector6:
    v6 = tuple(map(float, text.split()))
    assert (len(v6) == 6)
    return Vector6(v0=v6[0], v1=v6[1], v2=v6[2], v3=v6[3], v4=v6[4], v5=v6[5])


def parse_ground(ground) -> Ground:
    name = ground.attrib["name"]
    return Ground(name=name)


def parse_function_from_parent(parent) -> Function:
    for child in parent:
        if child.tag == "LinearFunction":
            coefficients = to_vector2(child.find("coefficients").text)
            return LinearFunction(coefficients=coefficients)
        elif child.tag == "Constant":
            value = float(child.find("value").text)
            return ConstantFunction(value=value)
        elif child.tag == "SimmSpline":
            x_text = child.find("x").text
            y_text = child.find("y").text
            x_values = list(map(float, x_text.split()))
            y_values = list(map(float, y_text.split()))
            return SimmSplineFunction(x=x_values, y=y_values)
        elif child.tag == "MultiplierFunction":
            inner_function = parse_function_from_parent(child.find("function"))
            scale = float(child.find("scale").text)
            inner_function.scale(scale)
            return inner_function
    raise ValueError("No function found")


def parse_transform_axis(transform_axis) -> TransformAxis:
    name = transform_axis.attrib["name"]
    coordinates = transform_axis.find("coordinates").text
    axis = to_vector3(transform_axis.find("axis").text)
    function = parse_function_from_parent(transform_axis)
    return TransformAxis(
        name=name,
        coordinates=coordinates,
        axis=axis,
        function=function
    )


def parse_spatial_transform(spatial_transform) -> SpatialTransform:
    transform_axes = []
    spatial_transform_objects = spatial_transform.findall("TransformAxis")
    for ta in spatial_transform_objects:
        ta_obj = parse_transform_axis(ta)
        transform_axes.append(ta_obj)

    return SpatialTransform(transform_axes=transform_axes)


def parse_physical_offset_frames(frames_element) -> list[PhysicalOffsetFrame]:
    frames = []
    frame_elements = frames_element.findall("PhysicalOffsetFrame")
    for frame in frame_elements:
        name = frame.attrib["name"]
        socket_parent = frame.find("socket_parent").text
        translation = to_vector3(frame.find("translation").text)
        orientation = to_vector3(frame.find("orientation").text)
        frame_obj = PhysicalOffsetFrame(
            name=name,
            socket_parent=socket_parent,
            translation=translation,
            orientation=Quat.from_fixed_angles(orientation),
        )
        frames.append(frame_obj)
    return frames


def parse_coordinate_set(coordinate_set_element) -> list[Coordinate]:
    coordinates = []
    coordinate_elements = coordinate_set_element.findall("Coordinate")
    for coord in coordinate_elements:
        name = coord.attrib["name"]
        default_value = float(coord.find("default_value").text)
        default_speed_value = float(coord.find("default_speed_value").text)
        range_text = coord.find("range").text
        range_values = to_vector2(range_text)
        clamped = coord.find("clamped").text.lower() == "true"
        locked = coord.find("locked").text.lower() == "true"
        coordinate_obj = Coordinate(
            name=name,
            default_value=default_value,
            default_speed_value=default_speed_value,
            range=range_values,
            clamped=clamped,
            locked=locked
        )
        coordinates.append(coordinate_obj)
    return coordinates


def parse_joint_base(joint) -> Joint:
    name = joint.attrib["name"]
    socket_parent_frame = joint.find("socket_parent_frame").text
    socket_child_frame = joint.find("socket_child_frame").text
    coordinate_set_element = joint.find("coordinates")
    coordinate_set = parse_coordinate_set(coordinate_set_element)

    frames_element = joint.find("frames")
    frames = parse_physical_offset_frames(frames_element)
    return Joint(
        name=name,
        socket_parent_frame=socket_parent_frame,
        socket_child_frame=socket_child_frame,
        coordinates=coordinate_set,
        frames=frames
    )


def parse_custom_joint(joint) -> Joint:
    joint_base = parse_joint_base(joint)

    # SpatialTransform describes how body moves wrst parent
    spatial_transform_element = joint.find("SpatialTransform")
    spatial_transform = parse_spatial_transform(spatial_transform_element)

    return CustomJoint.from_joint(joint_base, spatial_transform)


def parse_pin_joint(joint) -> PinJoint:
    joint_base = parse_joint_base(joint)
    return PinJoint.from_joint(joint_base)


def parse_ball_joint(joint) -> BallJoint:
    joint_base = parse_joint_base(joint)
    return BallJoint.from_joint(joint_base)


def parse_universal_joint(joint) -> UniversalJoint:
    joint_base = parse_joint_base(joint)
    return UniversalJoint.from_joint(joint_base)


def parse_mesh(mesh) -> Mesh:
    mesh_file = mesh.find("mesh_file").text
    scale_text = mesh.find("scale_factors").text
    return Mesh(
        mesh_file=mesh_file,
        scale_factors=to_vector3(scale_text)
    )


def parse_attached_geometry(attached_geometry) -> AttachedGeometry:
    # parse <Mesh> Children
    meshes = []
    mesh_elements = attached_geometry.findall("Mesh")
    for mesh in mesh_elements:
        mesh_obj = parse_mesh(mesh)
        meshes.append(mesh_obj)
    return AttachedGeometry(meshes=meshes)


def parse_body_set(body_set) -> BodySet:
    bodies = OrderedDict()
    body_set_objects = body_set.find("objects")
    body_set_bodies = body_set_objects.findall("Body")
    for body in body_set_bodies:
        body_name = body.attrib["name"]

        # AttachedGeometry
        attached_geometry_element = body.find("attached_geometry")
        attached_geometry = parse_attached_geometry(attached_geometry_element)

        # All mass attributes
        body_mass = body.find("mass").text
        body_mass_center = body.find("mass_center")
        inertia = body.find("inertia")
        # Remove whitespace and split
        inertia = inertia.text.strip()
        inertia_values = inertia.split()

        # Create Body object
        body_obj = Body(
            name=body_name,
            attached_geometry=attached_geometry,
            mass=float(body_mass),
            mass_center=to_vector3(body_mass_center.text),
            inertia=Inertia(
                xx=float(inertia_values[0]),
                yy=float(inertia_values[1]),
                zz=float(inertia_values[2]),
                xy=float(inertia_values[3]),
                xz=float(inertia_values[4]),
                yz=float(inertia_values[5]),
            ),
        )
        bodies[body_name] = body_obj
    return BodySet(bodies=bodies)


def parse_joint_set(joint_set) -> JointSet:
    joints = OrderedDict()
    joint_set_objects = joint_set.find("objects")
    for joint in joint_set_objects:
        if joint.tag == "CustomJoint":
            joint_obj = parse_custom_joint(joint)
        elif joint.tag == "BallJoint":
            joint_obj = parse_ball_joint(joint)
        elif joint.tag == "PinJoint":
            joint_obj = parse_pin_joint(joint)
        elif joint.tag == "UniversalJoint":
            joint_obj = parse_universal_joint(joint)
        else:
            print("Undefined joint type:", joint.tag)
            continue
        joints[joint_obj.name] = joint_obj
    return JointSet(joints=joints)


def parse_path_point(path_point) -> PathPoint:
    name = path_point.attrib["name"]
    socket_parent_frame = path_point.find("socket_parent_frame").text
    location = to_vector3(path_point.find("location").text)
    return PathPoint(
        name=name,
        socket_parent_frame=socket_parent_frame,
        location=location)


def parse_conditional_path_point(
        conditional_path_point) -> ConditionalPathPoint:
    name = conditional_path_point.attrib["name"]
    socket_parent_frame = conditional_path_point.find(
        "socket_parent_frame").text
    location = to_vector3(conditional_path_point.find("location").text)
    range_text = conditional_path_point.find("range").text
    range_values = to_vector2(range_text)
    socket_coordinate = conditional_path_point.find("socket_coordinate").text
    return ConditionalPathPoint(
        name=name,
        socket_parent_frame=socket_parent_frame,
        location=location,
        range=range_values,
        socket_coordinate=socket_coordinate
    )


def parse_moving_path_point(
        moving_path_point) -> MovingPathPoint:
    name = moving_path_point.attrib["name"]
    socket_parent_frame = moving_path_point.find(
        "socket_parent_frame").text
    location = to_vector3(moving_path_point.find("location").text)
    return MovingPathPoint(
        name=name,
        socket_parent_frame=socket_parent_frame,
        location=location,
        socket_x_coordinate=None,  # TODO
        socket_y_coordinate=None,  # TODO
        socket_z_coordinate=None,  # TODO
        x_location=None,  # TODO
        y_location=None,  # TODO
        z_location=None  # TODO
    )


def parse_path_point_set(path_point_set) -> PathPointSet:
    path_points = OrderedDict()
    path_point_set_objects = path_point_set.find("objects")
    for child in path_point_set_objects:
        if child.tag == "PathPoint":
            pp_obj = parse_path_point(child)
            path_points[pp_obj.name] = pp_obj
        elif child.tag == "ConditionalPathPoint":
            cpp_obj = parse_conditional_path_point(child)
            path_points[cpp_obj.name] = cpp_obj
        elif child.tag == "MovingPathPoint":
            mpp_obj = parse_moving_path_point(child)
            path_points[mpp_obj.name] = mpp_obj
    return PathPointSet(path_points=path_points)


def parse_geometry_path(geometry_path) -> GeometryPath:
    path_point_set_element = geometry_path.find("PathPointSet")
    path_point_set = parse_path_point_set(path_point_set_element)
    return GeometryPath(path_point_set=path_point_set)


def parse_muscle(muscle) -> Muscle:
    name = muscle.attrib["name"]
    # GeometryPath
    geometry_path_element = muscle.find("GeometryPath")
    geometry_path = parse_geometry_path(geometry_path_element)

    # Muscle properties
    max_isometric_force = float(muscle.find("max_isometric_force").text)
    optimal_fiber_length = float(muscle.find("optimal_fiber_length").text)
    tendon_slack_length = float(muscle.find("tendon_slack_length").text)
    pennation_angle_at_optimal = float(
        muscle.find("pennation_angle_at_optimal").text)
    return Muscle(
        name=name,
        geometry_path=geometry_path,
        max_isometric_force=max_isometric_force,
        optimal_fiber_length=optimal_fiber_length,
        tendon_slack_length=tendon_slack_length,
        pennation_angle_at_optimal=pennation_angle_at_optimal
    )


def parse_actuator(actuator) -> Actuator:
    name = actuator.attrib["name"]
    optimal_force = float(actuator.find("optimal_force").text)
    coordinate = actuator.find("coordinate").text
    activation_time_constant = float(actuator.find("activation_time_constant").text)
    default_activation = float(actuator.find("default_activation").text)
    return Actuator(
        name=name,
        optimal_force=optimal_force,
        coordinate=coordinate,
        activation_time_constant=activation_time_constant,
        default_activation=default_activation
    )


def parse_force_set(force_set) -> ForceSet:
    muscles = OrderedDict()
    force_set_objects = force_set.find("objects")
    force_set_muscles = [m for m in force_set_objects if "Muscle" in m.tag]
    for muscle in force_set_muscles:
        muscle_obj = parse_muscle(muscle)
        muscles[muscle_obj.name] = muscle_obj

    actuators = OrderedDict()
    force_set_actuators = force_set_objects.findall(
        "ActivationCoordinateActuator")
    for actuator in force_set_actuators:
        actuator_obj = parse_actuator(actuator)
        actuators[actuator_obj.name] = actuator_obj

    return ForceSet(muscles=muscles, actuators=actuators)


def parse_base_collider(collider) -> tuple[str, str]:
    name = collider.attrib["name"]
    socket_frame = collider.find("socket_frame").text
    location = to_vector3(collider.find("location").text)
    orientation = to_vector3(collider.find("orientation").text)

    pc_filter_element = collider.find("filter")
    pc_filter = True
    if pc_filter_element is not None:
        pc_filter_text = pc_filter_element.text
        pc_filter = pc_filter_text.lower() == "true"
    return name, socket_frame, location, orientation, pc_filter


def parse_half_space(contact_half_space) -> ContactHalfSpace:
    name, socket_frame, location, orientation, pc_filter = parse_base_collider(contact_half_space)
    return ContactHalfSpace(
        name=name,
        socket_frame=socket_frame,
        location=location,
        orientation=Quat.from_fixed_angles(orientation),
        pc_filter=pc_filter,
    )


def parse_contact_sphere(contact_sphere) -> ContactSphere:
    name, socket_frame, location, orientation, pc_filter = parse_base_collider(contact_sphere)
    radius = float(contact_sphere.find("radius").text)
    return ContactSphere(
        name=name,
        socket_frame=socket_frame,
        location=location,
        orientation=Quat.from_fixed_angles(orientation),
        radius=radius,
        pc_filter=pc_filter,
    )


def parse_contact_capsule(contact_capsule) -> ContactCapsule:
    name, socket_frame, location, orientation, pc_filter = parse_base_collider(contact_capsule)
    radius = float(contact_capsule.find("radius").text)
    length = float(contact_capsule.find("length").text)
    return ContactCapsule(
        name=name,
        socket_frame=socket_frame,
        location=location,
        orientation=Quat.from_fixed_angles(orientation),
        radius=radius,
        half_length=length,
        pc_filter=pc_filter,
    )


def parse_contact_geometry_set(contact_geometry_set) -> ContactGeometrySet:
    contact_geom = OrderedDict()
    contact_geometry_set_objects = contact_geometry_set.find("objects")

    if contact_geometry_set_objects is not None:
        # Half-space (plane)
        contact_half_space_elements = contact_geometry_set_objects.findall(
            "ContactHalfSpace")
        for chs in contact_half_space_elements:
            chs_obj = parse_half_space(chs)
            contact_geom[chs_obj.name] = chs_obj

        # Spheres
        contact_sphere_elements = contact_geometry_set_objects.findall(
            "ContactSphere")
        for cs in contact_sphere_elements:
            cs_obj = parse_contact_sphere(cs)
            contact_geom[cs_obj.name] = cs_obj

        # Capsules
        contact_capsule_elements = contact_geometry_set_objects.findall(
            "ContactCapsule")
        for cc in contact_capsule_elements:
            cc_obj = parse_contact_capsule(cc)
            contact_geom[cc_obj.name] = cc_obj

    return ContactGeometrySet(
        contact_geom=contact_geom
    )


def parse_osim_file(file_path: str) -> Model:
    """
    Model ->
        BodySet ->
            Bodies
        Constraints
        Forces
        Controllers
    """
    tree = ElementTree.parse(file_path)
    root = tree.getroot()
    # Everything is under <Model>
    model_element = root.find("Model")

    # Ground
    ground_element = model_element.find("Ground")
    ground = parse_ground(ground_element)

    # BodySet
    body_set_element = model_element.find("BodySet")
    body_set = parse_body_set(body_set_element)

    # JointSet
    joint_set_element = model_element.find("JointSet")
    joint_set = parse_joint_set(joint_set_element)

    # ForceSet
    force_set_element = model_element.find("ForceSet")
    force_set = parse_force_set(force_set_element)

    # ContactGeometrySet
    contact_geometry_set_element = model_element.find("ContactGeometrySet")
    contact_geometry_set = parse_contact_geometry_set(
        contact_geometry_set_element)

    # The remaining are unsupported
    constraint_set = model_element.find("ConstraintSet")
    component_set = model_element.find("ComponentSet")
    controller_set = model_element.find("ControllerSet")

    # Create Model object
    model = Model(
        ground=ground,
        body_set=body_set,
        joint_set=joint_set,
        force_set=force_set,
        contact_geometry_set=contact_geometry_set
    )
    return model


if __name__ == "__main__":
    # Example usage
    osim_file_path = "Scaled_FullBody_HamnerModel_Muscle_withContact.osim"
    parse_osim_file(osim_file_path)
