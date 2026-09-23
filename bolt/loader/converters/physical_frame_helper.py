import warp as wp

from .osim_types import OSimType
from .property_helper import extract_vec3, extract_quat


def wp_transform_from_osim_transform(transform: OSimType.Transform) -> wp.transform:
    """ Converts an OpenSim Transform object to a wp.transform object """
    translation = transform.p()
    orientation = transform.R().convertRotationToQuaternion()

    translation_vec = extract_vec3(translation)
    orientation_quat = extract_quat(orientation)
    return wp.transform(translation_vec, orientation_quat)


def extract_frame_transform_from_base_frame(frame: OSimType.Frame) -> wp.transform:
    """ Extract the transform of a frame relative to its parent body frame (base frame) as a wp.transform object """
    T_parent_to_frame = frame.findTransformInBaseFrame()
    return wp_transform_from_osim_transform(T_parent_to_frame)


def get_body_name_of_frame(frame: OSimType.Frame) -> str:
    """ Get the name of the body that a frame is attached to """
    base_frame = frame.findBaseFrame().getName()
    return base_frame
