from dataclasses import dataclass
from collections import OrderedDict

from enum import Enum
import math


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
class Quat:
    w: float
    x: float
    y: float
    z: float

    @staticmethod
    def from_angle_axis(angle: float, axis: Vector3) -> "Quat":
        cos_half = math.cos(angle / 2)
        sin_half = math.sin(angle / 2)
        return Quat(
            w=cos_half,
            x=axis.x * sin_half,
            y=axis.y * sin_half,
            z=axis.z * sin_half,
        )

    @staticmethod
    def mul(q1: "Quat", q2: "Quat") -> "Quat":
        return Quat(
            w=q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
            x=q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
            y=q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
            z=q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        )

    @staticmethod
    def from_fixed_angles(r: Vector3) -> "Quat":
        qx = Quat.from_angle_axis(r.x, Vector3(1, 0, 0))
        qy = Quat.from_angle_axis(r.y, Vector3(0, 1, 0))
        qz = Quat.from_angle_axis(r.z, Vector3(0, 0, 1))
        return Quat.mul(qx, Quat.mul(qy, qz))

    def normalize(self) -> "Quat":
        norm = math.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)
        if norm > 0:
            self.w /= norm
            self.x /= norm
            self.y /= norm
            self.z /= norm
        return self

    def inv(self) -> "Quat":
        return Quat(w=self.w, x=-self.x, y=-self.y, z=-self.z)

    def to_list(self) -> list[float]:
        self.normalize()
        return [float(self.w), float(self.x), float(self.y), float(self.z)]


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


class FunctionType(Enum):
    LINEAR = "linear"
    CONSTANT = "constant"
    SIMM_SPLINE = "simm_spline"


@dataclass
class Function:
    def scale(self, factor: float):
        pass

    def type(self) -> FunctionType:
        raise NotImplementedError


@dataclass
class LinearFunction(Function):
    coefficients: Vector2

    def scale(self, factor: float):
        self.coefficients.x *= factor
        self.coefficients.y *= factor

    def type(self) -> FunctionType:
        return FunctionType.LINEAR


@dataclass
class ConstantFunction(Function):
    value: float

    def scale(self, factor: float):
        self.value *= factor

    def type(self) -> FunctionType:
        return FunctionType.CONSTANT


@dataclass
class SimmSplineFunction(Function):
    x: list[float]
    y: list[float]

    def scale(self, factor: float):
        self.y = [y * factor for y in self.y]

    def type(self) -> FunctionType:
        return FunctionType.SIMM_SPLINE


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
    orientation: Quat


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

    def connects_to_ground(self) -> bool:
        return "ground" in self.socket_parent_frame


_VOID_NAME = "__VOID__"


@dataclass
class DummyJoint(Joint):  # used for ground

    def __init__(self):
        super().__init__(
            name="ground_joint",
            socket_parent_frame=_VOID_NAME,
            socket_child_frame="ground",
            coordinates=[],
            frames=[
                PhysicalOffsetFrame(
                    name=_VOID_NAME,
                    socket_parent=_VOID_NAME,
                    translation=Vector3(0.0, 0.0, 0.0),
                    orientation=Quat(1.0, 0.0, 0.0, 0.0),
                ),
                PhysicalOffsetFrame(
                    name="ground",
                    socket_parent="ground",
                    translation=Vector3(0.0, 0.0, 0.0),
                    orientation=Quat(1.0, 0.0, 0.0, 0.0),
                ),
            ],
        )

    def num_dofs(self) -> int:
        return 0

    def num_pos_dofs(self) -> int:
        return 0


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
class BallJoint(Joint):
    @classmethod
    def from_joint(cls, joint: Joint) -> "BallJoint":
        return cls(
            name=joint.name,
            socket_parent_frame=joint.socket_parent_frame,
            socket_child_frame=joint.socket_child_frame,
            coordinates=joint.coordinates,
            frames=joint.frames,
        )

    def num_dofs(self) -> int:
        return 3

    def num_pos_dofs(self) -> int:
        return 4


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
class WeldJoint(Joint):
    @classmethod
    def from_joint(cls, joint: Joint) -> "WeldJoint":
        return cls(
            name=joint.name,
            socket_parent_frame=joint.socket_parent_frame,
            socket_child_frame=joint.socket_child_frame,
            coordinates=joint.coordinates,
            frames=joint.frames,
        )

    def num_dofs(self) -> int:
        return 0

    def num_pos_dofs(self) -> int:
        return 0


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
    orientation: Quat
    pc_filter: bool

    def size(self) -> list[float]:
        raise NotImplementedError

    def get_aabb(self) -> list[float]:
        """ concat(center, size). note center is in the geom frame """
        raise NotImplementedError

    def get_rbound(self) -> float:
        """ return the radius bound of the collider """
        raise NotImplementedError


@dataclass
class ContactHalfSpace(Collider):
    def size(self) -> list[float]:
        return [0.0, 0.05, 0.0]

    def get_aabb(self) -> list[float]:
        center = [0.0, -5e9, 0.0]
        size = [1e10, 5e9, 1e10]
        return center + size

    def get_rbound(self) -> float:
        return 0.0


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
class ContactCapsule(Collider):
    radius: float
    half_length: float

    def size(self) -> list[float]:
        return [self.radius, self.half_length, self.half_length]

    def get_aabb(self) -> list[float]:
        center = [0.0, 0.0, 0.0]
        length = self.half_length * 2
        size = [2 * self.radius, length + 2 * self.radius, 2 * self.radius]
        return center + size

    def get_rbound(self) -> float:
        return math.sqrt(self.half_length ** 2 + self.radius ** 2)


@dataclass
class ContactGeometrySet:
    contact_geom: OrderedDict[str, Collider]


@dataclass
class PathPoint:
    name: str
    socket_parent_frame: str
    location: Vector3

    def is_conditional(self) -> bool:
        return False

    def get_range(self) -> Vector2:
        raise ValueError

    def get_coordinate(self) -> str:
        raise ValueError


@dataclass
class ConditionalPathPoint(PathPoint):
    socket_coordinate: str
    range: Vector2

    def is_conditional(self) -> bool:
        return True

    def get_range(self) -> Vector2:
        return self.range

    def get_coordinate(self) -> str:
        return self.socket_coordinate


@dataclass
class MovingPathPoint(PathPoint):
    # can't handle these yet
    socket_x_coordinate: str
    socket_y_coordinate: str
    socket_z_coordinate: str
    x_location: Function
    y_location: Function
    z_location: Function

    def is_conditional(self) -> bool:
        return False


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
    activation_time_constant: float
    default_activation: float
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
