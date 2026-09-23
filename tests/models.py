""" Locates the OpenSim models used for validation """

from __future__ import annotations

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "data", "models")


def model_path(name: str) -> str:
    return os.path.join(MODELS_DIR, f"{name}.osim")


# Models exercised by the kinematics/dynamics tests
MODEL_NAMES = ["example_model", "example_model2"]

# Model + fitted function-based paths
FN_PATH_MODEL = "example_model"
FN_PATH_FILE = os.path.join(MODELS_DIR, "example_model_fn.xml")
