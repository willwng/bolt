from dataclasses import dataclass
from dataclasses import fields

import warp as wp
from msk_warp import MobilizerType

GROUND_PARENT = "N/A"
GROUND = "ground"


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
    num_coordinates: int
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
    scale_factors: wp.vec3


@dataclass
class ColliderData:
    name: str
    body_name: str
    transform: wp.transform


@dataclass
class Function:
    pass


@dataclass
class LinearFunctionData(Function):
    mb: tuple[int, int]
    qpos_adr: int


@dataclass
class ConstantFunctionData(Function):
    c: float


@dataclass
class PolynomialFunctionData(Function):
    coefficients: list[int]
    qpos_adr: list[int]


@dataclass
class TransformAxisData:
    coordinates: list[str]
    axis: wp.vec3
    function: Function


@dataclass
class SpatialTransformData:
    joint_name: str

    rotation_1: TransformAxisData
    rotation_2: TransformAxisData
    rotation_3: TransformAxisData
    translation_1: TransformAxisData
    translation_2: TransformAxisData
    translation_3: TransformAxisData


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
    if not data_list:
        raise ValueError("Input list is empty")

    result = {f.name: [] for f in fields(cls)}

    for obj in data_list:
        for f in fields(cls):
            result[f.name].append(getattr(obj, f.name))

    return result
