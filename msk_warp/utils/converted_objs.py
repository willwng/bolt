from dataclasses import dataclass

from msk_warp._src import types


@dataclass
class LinearFunctionData:
    mb: list[tuple[int, int]]
    fn_idx: list[int]
    qpos_adr: list[int]

    def __init__(self):
        self.mb = []
        self.fn_idx = []
        self.qpos_adr = []

    def count(self) -> int:
        return len(self.mb)


@dataclass
class ConstantFunctionData:
    c: list[int]
    fn_idx: list[int]

    def __init__(self):
        self.c = []
        self.fn_idx: list[int] = []

    def count(self) -> int:
        return len(self.c)


@dataclass
class PolynomialFunctionData:
    coefficients: list[list[int]]
    fn_idx: list[int]
    qpos_adr: list[int]

    def __init__(self):
        self.coefficients = []
        self.fn_idx = []
        self.qpos_adr = []

    def count(self) -> int:
        return len(self.coefficients)


@dataclass
class ColliderData:
    type: list[types.GeomType]
    body_id: list[int]
    size: list[list[float]]
    transform: list[list[float]]
    friction: list[list[float]]
    stiffness: list[float]
    dissipation: list[float]
    transition_velocity: list[float]
    priority: list[int]
    aabb: list[list[float]]
    rbound: list[float]
    pc_filter: list[bool]  # If true, maintains child-parent contact filter

    # default constructor
    def __init__(self):
        self.type = []
        self.body_id = []
        self.size = []
        self.transform = []
        self.friction = []
        self.stiffness = []
        self.dissipation = []
        self.transition_velocity = []
        self.priority = []
        self.aabb = []
        self.rbound = []
        self.pc_filter = []


@dataclass
class SiteData:
    nsite: int
    nsite_cond: int
    body_id: list[int]
    pos: list[list[float]]

    conditional_ids: list[int]
    conditional_qadr: list[int]
    conditional_range: list[list[float]]

    # default constructor
    def __init__(self):
        self.nsite = 0
        self.nsite_cond = 0
        self.body_id = []
        self.pos = []
        self.conditional_ids = []
        self.conditional_qadr = []
        self.conditional_range = []


@dataclass
class VisualData:
    body_id: list[int]
    transform: list[list[float]]
    scale: list[list[float]]
    file: list[str]

    def __init__(self):
        self.body_id = []
        self.transform = []
        self.scale = []
        self.file = []
