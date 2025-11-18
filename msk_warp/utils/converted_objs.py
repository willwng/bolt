from dataclasses import dataclass

from msk_warp._src import types


@dataclass
class ColliderData:
    type: list[types.GeomType]
    body_id: list[int]
    size: list[list[float]]
    pos: list[list[float]]
    rot: list[list[float]]
    friction: list[list[float]]
    aabb: list[list[float]]
    rbound: list[float]

    # default constructor
    def __init__(self):
        self.type = []
        self.body_id = []
        self.size = []
        self.pos = []
        self.rot = []
        self.friction = []
        self.aabb = []
        self.rbound = []


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
