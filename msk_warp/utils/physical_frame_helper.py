import opensim as osim
import warp as wp

from .osim_types import OSimType
from .property_helper import extract_vec3, extract_quat


def _get_frame_offset_transform(frame: OSimType.Frame) -> OSimType:
    """ Get the offset transform of a frame relative to its parent body frame """
    offset = OSimType.PhysicalOffsetFrame.safeDownCast(frame)
    if offset is not None:
        T_parent_to_frame = offset.getOffsetTransform()
    else:  # Frame coincides with the body frame
        T_parent_to_frame = osim.Transform()
    return T_parent_to_frame


def _extract_translation_orientation_from_transform(
        T_parent_to_frame: OSimType.Transform
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """ Returns the position, quaternion of an OpenSim Transform object """
    translation = T_parent_to_frame.p()
    orientation = T_parent_to_frame.R().convertRotationToQuaternion()

    translation_vec = extract_vec3(translation)
    orientation_quat = extract_quat(orientation)
    return translation_vec, orientation_quat


def transform_from_osim_transform(transform: OSimType.Transform) -> wp.transform:
    """ Converts an OpenSim Transform object to a wp.transform object """
    translation_vec, orientation_quat = _extract_translation_orientation_from_transform(transform)
    return wp.transform(translation_vec, orientation_quat)


def extract_transform_from_frame(frame: OSimType.Frame) -> wp.transform:
    """ Extract the transform of a frame relative to its parent body frame as a wp.transform object """
    T_parent_to_frame = _get_frame_offset_transform(frame)
    return transform_from_osim_transform(T_parent_to_frame)


def get_body_name_of_frame(frame: OSimType.Frame) -> str:
    """ Get the name of the body that a frame is attached to """
    base_frame = frame.findBaseFrame().getName()
    return base_frame
