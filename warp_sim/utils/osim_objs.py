from dataclasses import dataclass
from collections import OrderedDict


@dataclass
class Vector2:
    x: float
    y: float


@dataclass
class Vector3:
    x: float
    y: float
    z: float


@dataclass
class Vector6:
    v0: float
    v1: float
    v2: float
    v3: float
    v4: float
    v5: float


@dataclass
class Inertia:
    xx: float
    yy: float
    zz: float
    xy: float
    xz: float
    yz: float


@dataclass
class Ground:
    name: str


@dataclass
class Mesh:
    mesh_file: str
    scale_factors: Vector3


@dataclass
class AttachedGeometry:
    meshes: list[Mesh]


@dataclass
class Coordinate:
    name: str
    default_value: float
    default_speed_value: float
    range: Vector2
    clamped: bool
    locked: bool


@dataclass
class Function:
    def scale(self, factor: float):
        pass


@dataclass
class LinearFunction(Function):
    coefficients: Vector2

    def scale(self, factor: float):
        self.coefficients.x *= factor
        self.coefficients.y *= factor


@dataclass
class ConstantFunction(Function):
    value: float

    def scale(self, factor: float):
        self.value *= factor


@dataclass
class SimmSplineFunction(Function):
    x: list[float]
    y: list[float]

    def scale(self, factor: float):
        self.y = [y * factor for y in self.y]


@dataclass
class TransformAxis:
    name: str
    coordinates: str
    axis: Vector3
    function: Function


@dataclass
class SpatialTransform:
    transform_axes: list[TransformAxis]


@dataclass
class PhysicalOffsetFrame:
    name: str
    socket_parent: str
    translation: Vector3
    orientation: Vector3


@dataclass
class Joint:
    name: str
    socket_parent_frame: str
    socket_child_frame: str

    coordinates: list[Coordinate]
    frames: list[PhysicalOffsetFrame]

    def num_dofs(self) -> int:
        raise NotImplementedError

    def num_pos_dofs(self) -> int:
        raise NotImplementedError


@dataclass
class PinJoint(Joint):
    @classmethod
    def from_joint(cls, joint: Joint) -> "PinJoint":
        assert len(
            joint.coordinates) == 1, "PinJoint must have exactly one coordinate"
        return cls(
            name=joint.name,
            socket_parent_frame=joint.socket_parent_frame,
            socket_child_frame=joint.socket_child_frame,
            coordinates=joint.coordinates,
            frames=joint.frames,
        )

    def num_dofs(self) -> int:
        return 1

    def num_pos_dofs(self) -> int:
        return 1


@dataclass
class UniversalJoint(Joint):
    @classmethod
    def from_joint(cls, joint: Joint) -> "UniversalJoint":
        assert len(
            joint.coordinates) == 2, "UniversalJoint must have exactly two coordinates"
        return cls(
            name=joint.name,
            socket_parent_frame=joint.socket_parent_frame,
            socket_child_frame=joint.socket_child_frame,
            coordinates=joint.coordinates,
            frames=joint.frames,
        )

    def num_dofs(self) -> int:
        return 2

    def num_pos_dofs(self) -> int:
        return 2


@dataclass
class CustomJoint(Joint):
    spatial_transform: SpatialTransform

    @classmethod
    def from_joint(
            cls,
            joint: Joint,
            spatial_transform: SpatialTransform
    ) -> "CustomJoint":
        return cls(
            name=joint.name,
            socket_parent_frame=joint.socket_parent_frame,
            socket_child_frame=joint.socket_child_frame,
            coordinates=joint.coordinates,
            frames=joint.frames,
            spatial_transform=spatial_transform,
        )

    def num_dofs(self) -> int:
        return len(self.coordinates)

    def num_pos_dofs(self) -> int:
        return len(self.coordinates)


@dataclass
class JointSet:
    joints: OrderedDict[str, Joint]


@dataclass
class Body:
    name: str

    attached_geometry: AttachedGeometry

    mass: float
    mass_center: Vector3
    inertia: Inertia


@dataclass
class BodySet:
    bodies: OrderedDict[str, Body]


@dataclass
class Collider:
    name: str
    socket_frame: str
    location: Vector3
    orientation: Vector3

    def size(self) -> list[float]:
        raise NotImplementedError

    def get_aabb(self) -> list[float]:
        """ concat(center, size). note center is in the geom frame """
        raise NotImplementedError

    def get_rbound(self) -> float:
        """ return the radius bound of the collider """
        raise NotImplementedError


@dataclass
class ContactSphere(Collider):
    radius: float

    def size(self) -> list[float]:
        return [self.radius, self.radius, self.radius]

    def get_aabb(self) -> list[float]:
        center = [0.0, 0.0, 0.0]
        size = [2 * self.radius, 2 * self.radius, 2 * self.radius]
        return center + size

    def get_rbound(self) -> float:
        return self.radius


@dataclass
class ContactGeometrySet:
    contact_spheres: OrderedDict[str, Collider]


@dataclass
class PathPoint:
    name: str
    socket_parent_frame: str
    location: Vector3


@dataclass
class ConditionalPathPoint(PathPoint):
    socket_coordinate: str
    range: Vector2


@dataclass
class MovingPathPoint(PathPoint):
    socket_x_coordinate: str
    socket_y_coordinate: str
    socket_z_coordinate: str
    x_location: Function
    y_location: Function
    z_location: Function


@dataclass
class PathPointSet:
    path_points: OrderedDict[str, PathPoint]


@dataclass
class GeometryPath:
    path_point_set: PathPointSet


@dataclass
class Muscle:
    name: str
    geometry_path: GeometryPath
    max_isometric_force: float
    optimal_fiber_length: float
    tendon_slack_length: float
    pennation_angle_at_optimal: float


@dataclass
class Actuator:
    name: str
    optimal_force: float
    coordinate: str


@dataclass
class ForceSet:
    muscles: OrderedDict[str, Muscle]
    actuators: OrderedDict[str, Actuator]


@dataclass
class Model:
    ground: Ground
    body_set: BodySet
    joint_set: JointSet
    force_set: ForceSet
    contact_geometry_set: ContactGeometrySet
