from dataclasses import dataclass

from warp_sim._src import types


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
