from dataclasses import dataclass
from dataclasses import fields

import warp as wp
from msk_warp import MobilizerType, GeomType, PolyInts

GROUND_PARENT = "N/A"
GROUND = "ground"
NO_DOF = "__NO_DOF"  # for transform axes that don't depend on a DOF, use this as a placeholder
PADDED_DOF = "__PADDED_DOF"  # for coordinates that are not actually part of the function-based path
AABB = tuple[wp.vec3, wp.vec3]


@dataclass
class BodyData:
    name: str
    mass: float
    mass_center: wp.vec3
    unit_inertia_OB_B: wp.mat33


@dataclass
class JointData:
    name: str
    parent_body_name: str
    child_body_name: str

    mob_type: MobilizerType
    coordinates: list[str]
    num_coordinates: int  # != len(coordinates) if quaternion is used in mobilizer
    num_speeds: int

    # Transform from parent body to joint frame, mobilizer frame to child frame
    transform_PF: wp.transform
    transform_MB: wp.transform

    extra_info: wp.vec3


@dataclass
class VisualData:
    name: str
    body_name: str
    mesh_file: str

    transform: wp.transform
    scale_factors: tuple[float, float, float]


@dataclass
class SiteData:
    name: str
    body_name: str

    offset: wp.vec3


@dataclass
class GeomData:
    name: str
    body_name: str

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
    pass


@dataclass
class LinearFunctionData(FunctionData):
    slope: float
    intercept: float


@dataclass
class ConstantFunctionData(FunctionData):
    value: float


@dataclass
class PolynomialFunctionData(FunctionData):
    coefficients: list[int]


@dataclass
class TransformAxisData:
    coordinate: str
    axis: wp.vec3
    function: FunctionData


@dataclass
class SpatialTransformData:
    joint_name: str

    rotation_1: TransformAxisData
    rotation_2: TransformAxisData
    rotation_3: TransformAxisData
    translation_1: TransformAxisData
    translation_2: TransformAxisData
    translation_3: TransformAxisData


@dataclass
class SpringGeneralizedForceData:
    name: str
    coordinate: str
    stiffness: float
    viscosity: float


@dataclass
class CoordinateLimitForceData:
    name: str
    coordinate: str
    stiffness: wp.vec2
    range: wp.vec2
    damping: float
    transition: float


@dataclass
class SwingTwistLimitData:
    name: str
    joint: str
    stiffness: float
    damping: float

    twist_limits: wp.vec2
    swing1_limits: wp.vec2
    swing2_limits: wp.vec2


@dataclass
class ActivationCoordinateActuatorData:
    name: str

    coordinate: str
    optimal_force: float
    activation_time_constant: float
    default_activation: float


@dataclass
class ExponentialContactForce:
    name: str

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


@dataclass
class MuscleData:
    name: str

    ignore_tendon_compliance: bool

    min_control: float
    max_control: float

    max_isometric_force: float
    optimal_fiber_length: float
    tendon_slack_length: float
    pennation_angle_at_optimal: float
    fiber_damping: float

    path_points: list[SiteData]


@dataclass
class MuscleFunctionPathData:
    name: str
    coordinates: list[str]
    coefficients: list[float]
    exponents: list[PolyInts]
    dimension: int
    order: int


# Place-holder for muscles that don't use function-based paths
USE_POINT_PATH = MuscleFunctionPathData(
    name="use_point_path",
    coordinates=[],
    coefficients=[],
    exponents=[],
    dimension=0,
    order=0
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


def dataclass_list_transpose(data_list: list[dataclass], cls: type) -> dict[str, list]:
    """
    Convert a list of dataclass instances into a dict where each attribute becomes a list of that attribute.
    Though handy, I prefer not to use this since it results in code that doesn't update with refactoring property names
    """
    result = {f.name: [] for f in fields(cls)}

    for obj in data_list:
        for f in fields(cls):
            result[f.name].append(getattr(obj, f.name))

    return result
