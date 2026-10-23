from dataclasses import dataclass
from dataclasses import field
from typing import Annotated

import warp as wp
from bolt.types import GeomType, MobilizerType

GROUND_PARENT = "N/A"
GROUND = "ground"
NO_DOF = "__NO_DOF"  # for transform axes that don't depend on a DOF, use this as a placeholder
PADDED_DOF = "__PADDED_DOF"  # for coordinates that are not actually part of the function-based path
AABB = tuple[wp.vec3, wp.vec3]


class NameMarker:
    """
    Marks a field that refers to a model component (body, joint, coordinate, ...)
    Useful for when there are more than one models in the world
    """
    pass


Name = Annotated[str, NameMarker()]


@dataclass
class BodyData:
    name: Name
    mass: float
    mass_center: wp.vec3
    unit_inertia_OB_B: wp.mat33


@dataclass
class JointData:
    name: Name
    parent_body_name: Name
    child_body_name: Name

    mob_type: MobilizerType
    coordinates: list[Name]
    num_coordinates: int  # != len(coordinates) if quaternion is used in mobilizer
    num_speeds: int

    # Transform from parent body to joint frame, mobilizer frame to child frame
    transform_PF: wp.transform
    transform_MB: wp.transform

    extra_info: wp.vec3


@dataclass
class VisualData:
    name: Name
    body_name: Name
    mesh_file: str

    transform: wp.transform
    scale_factors: tuple[float, float, float]


@dataclass
class SiteData:
    name: Name
    body_name: Name

    offset: wp.vec3

    # ConditionalPathPoint: the site is only part of the muscle path while this coordinate is inside cond_range
    cond_coord: Name | None = None
    cond_range: tuple[float, float] = (0.0, 0.0)

    # MovingPathPoint: the location in frame F is (fx(qx), fy(qy), fz(qz)), and offset is unused
    moving_fns: tuple["FunctionData", "FunctionData", "FunctionData"] | None = None
    moving_coords: tuple[Name, Name, Name] | None = None
    moving_X_BF: wp.transform = field(default_factory=wp.transform_identity)


@dataclass
class UserGeomData:
    name: Name
    body_name: Name
    geom_type: GeomType
    transform: wp.transform
    size: wp.vec3

    # Defaults
    friction: wp.vec3 = field(default_factory=lambda: wp.vec3(0.8, 0.8, 0.0))
    stiffness: float = (5e6 ** (2 / 3))
    dissipation: float = 1.0
    transition_velocity: float = 0.1
    priority: int = 0


@dataclass
class GeomData:
    name: Name
    body_name: Name

    geom_type: GeomType
    transform: wp.transform
    size: wp.vec3
    aabb: AABB
    rbound: float

    friction: wp.vec3
    stiffness: float
    dissipation: float
    transition_velocity: float
    priority: int


@dataclass
class FunctionData:
    def scale(self, scale: float) -> "FunctionData":
        raise NotImplementedError


@dataclass
class LinearFunctionData(FunctionData):
    slope: float
    intercept: float

    def scale(self, scale: float) -> "LinearFunctionData":
        return LinearFunctionData(slope=self.slope * scale, intercept=self.intercept * scale)


@dataclass
class ConstantFunctionData(FunctionData):
    value: float

    def scale(self, scale: float) -> "ConstantFunctionData":
        return ConstantFunctionData(value=self.value * scale)


@dataclass
class PolynomialFunctionData(FunctionData):
    coefficients: list[float]

    def scale(self, scale: float) -> "PolynomialFunctionData":
        return PolynomialFunctionData(coefficients=[c * scale for c in self.coefficients])


@dataclass
class SimmSplineData(FunctionData):
    x: list[float]
    y: list[float]

    def scale(self, scale: float) -> "SimmSplineData":
        return SimmSplineData(x=self.x, y=[v * scale for v in self.y])


@dataclass
class TransformAxisData:
    coordinate: Name
    axis: wp.vec3
    function: FunctionData


@dataclass
class SpatialTransformData:
    joint_name: Name

    rotation_1: TransformAxisData
    rotation_2: TransformAxisData
    rotation_3: TransformAxisData
    translation_1: TransformAxisData
    translation_2: TransformAxisData
    translation_3: TransformAxisData


@dataclass
class SpringGeneralizedForceData:
    name: Name
    coordinate: Name
    stiffness: float
    viscosity: float


@dataclass
class CoordinateLimitForceData:
    name: Name
    coordinate: Name
    stiffness: wp.vec2
    range: wp.vec2
    damping: float
    transition: float


@dataclass
class SwingTwistLimitData:
    name: Name
    joint: Name
    stiffness: float
    damping: float

    twist_limits: wp.vec2
    swing1_limits: wp.vec2
    swing2_limits: wp.vec2


@dataclass
class ActivationCoordinateActuatorData:
    name: Name

    coordinate: Name
    optimal_force: float
    activation_time_constant: float
    default_activation: float


@dataclass
class StatefulContactForce:
    name: Name
    use_exp_force: bool  # True: Exponential force, False: Smoothed Hunt-Crossley

    contact_plane_transform: wp.transform
    exponential_shape_parameters: wp.vec3
    normal_viscosity: float
    max_normal_force: float
    friction_elasticity: float
    friction_viscosity: float
    settle_velocity: float
    initial_mu_static: float
    initial_mu_kinetic: float

    station: SiteData

    # Radius of the sphere used for sphere-vs-plane contact
    # margin=0 recovers the original point-vs-plane behavior.
    margin: float


@dataclass
class MuscleData:
    name: Name

    ignore_tendon_compliance: bool

    min_control: float
    max_control: float

    max_isometric_force: float
    optimal_fiber_length: float
    tendon_slack_length: float
    pennation_angle_at_optimal: float
    fiber_damping: float

    # Fiber passive force length curve
    strain_at_zero_force: float
    strain_at_one_norm_force: float
    stiffness_at_low_force: float
    stiffness_at_one_norm_force: float
    curviness: float

    path_points: list[SiteData]


@dataclass
class MuscleFunctionPathData:
    name: Name
    coordinates: list[Name]
    coefficients: list[float]
    dimension: int
    order: int


# Place-holder for muscles that don't use function-based paths
USE_POINT_PATH = MuscleFunctionPathData(
    name="use_point_path",
    coordinates=[],
    coefficients=[],
    dimension=0,
    order=0,
)

GROUND_BODY = BodyData(
    name=GROUND,
    mass=0.0,
    mass_center=wp.vec3(),
    unit_inertia_OB_B=wp.mat33(0.0)
)
GROUND_JOINT = JointData(
    name="ground_joint",
    parent_body_name=GROUND_PARENT,
    child_body_name=GROUND,

    mob_type=MobilizerType.WORLD,
    coordinates=[],
    num_coordinates=0,
    num_speeds=0,

    transform_PF=wp.transform_identity(),
    transform_MB=wp.transform_identity(),
    extra_info=wp.vec3(),
)
GROUND_COLLIDER = GeomData(
    name="ground_collider",
    body_name=GROUND,

    geom_type=GeomType.PLANE,
    transform=wp.transform_identity(),
    size=wp.vec3(0.0, 0.05, 0.0),
    aabb=(wp.vec3(0.0, -5e9, 0.0), wp.vec3(1e10, 5e9, 1e10)),
    rbound=0.0,

    friction=wp.vec3(0.95, 0.6, 0.0),
    stiffness=(1e6 ** (2 / 3)),
    dissipation=1.0,
    transition_velocity=0.1,
    priority=0
)
