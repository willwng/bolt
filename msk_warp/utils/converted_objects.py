from dataclasses import dataclass
from dataclasses import fields

import warp as wp
from msk_warp import MobilizerType, GeomType

GROUND_PARENT = "N/A"
GROUND = "ground"
NO_DOF = "__NO_DOF"  # for transform axes that don't depend on a DOF, use this as a placeholder


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
    aabb: wp.vec3
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
class CoordinateLinearSpringData:
    name: str
    coordinate: str
    default_stiffness: float
    rest_length: float  # coordinate value at which the spring produces no force


@dataclass
class CoordinateLinearDamperData:
    name: str
    coordinate: str
    damping: float


@dataclass
class CoordinateLinearStopData:
    name: str
    coordinate: str
    stiffness_damping: wp.vec2
    range: wp.vec2


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


def dataclass_list_transpose(data_list: list[dataclass], cls: type) -> dict[str, list]:
    """
    Convert a list of dataclass instances into a dict where each attribute becomes a list of that attribute.
    """
    result = {f.name: [] for f in fields(cls)}

    for obj in data_list:
        for f in fields(cls):
            result[f.name].append(getattr(obj, f.name))

    return result
