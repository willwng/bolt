import os
from pathlib import Path


__all__ = ["get_geometry_dir", "get_visual_path"]


def get_geometry_dir() -> str:
    bolt_path = Path(__file__).resolve().parent.parent
    geometry_path = os.path.join(bolt_path, "data", "geometry")
    return geometry_path


def get_visual_path(visual: str) -> str:
    visual_path = os.path.join(get_geometry_dir(), visual)
    return visual_path

