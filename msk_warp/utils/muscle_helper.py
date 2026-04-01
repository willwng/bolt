import opensim as osim
import warp as wp

from msk_warp import MuscleMetadata, MIN_NORM_FIBER_LENGTH, MAX_NORM_FIBER_LENGTH
from msk_warp.utils.converted_objects import MuscleData, SiteData
from msk_warp.utils.osim_types import OSimType
from msk_warp.utils.physical_frame_helper import extract_frame_transform_from_base_frame, get_body_name_of_frame
from msk_warp.utils.property_helper import extract_vec3


def convert_path_point(point: OSimType.PathPoint) -> SiteData:
    """ Converts an OpenSim PathPoint to a SiteData """
    parent_frame = point.getParentFrame()
    body_name = get_body_name_of_frame(parent_frame)

    # Compute the offset from the body frame
    frame_transform = extract_frame_transform_from_base_frame(parent_frame)
    location = wp.vec3(extract_vec3(point.get_location()))
    offset = wp.transform_point(frame_transform, location)

    return SiteData(
        name=point.getName(),
        body_name=body_name,
        offset=offset
    )


def collect_scholz_path_points(muscle_path: OSimType.ScholzPath) -> list[SiteData]:
    """ Collects the path points of a muscle and converts them to SiteData """
    num_path_points = muscle_path.getNumPathPoints()
    path_points = []
    for i in range(num_path_points):
        point = muscle_path.getPathPoint(i)
        path_points.append(convert_path_point(point))
    return path_points


def collect_geometry_path_points(muscle_path: OSimType.GeometryPath) -> list[SiteData]:
    """ Collects the path points of a muscle and converts them to SiteData """
    path_point_set = muscle_path.getPathPointSet()
    num_path_points = path_point_set.getSize()
    path_points = []
    for i in range(num_path_points):
        point = path_point_set.get(i)
        point = OSimType.PathPoint.safeDownCast(point)
        path_points.append(convert_path_point(point))
    return path_points


def get_muscles(model: OSimType.Model) -> list[osim.Muscle]:
    """ Returns the list of Muscles in the model """
    force_set = model.getForceSet()
    muscles = filter(lambda f: "Muscle" in f.getConcreteClassName(), force_set)
    return list(muscles)


def convert_path(muscle_path: OSimType.Path) -> list[SiteData]:
    """ Converts an OpenSim Path to a list of SiteData """
    if muscle_path.getConcreteClassName() == "Scholz2015GeometryPath":
        muscle_path = OSimType.ScholzPath.safeDownCast(muscle_path)
        return collect_scholz_path_points(muscle_path)
    elif muscle_path.getConcreteClassName() == "GeometryPath":
        muscle_path = OSimType.GeometryPath.safeDownCast(muscle_path)
        return collect_geometry_path_points(muscle_path)
    elif muscle_path.getConcreteClassName() == "FunctionBasedPath":
        muscle_path = OSimType.FunctionBasedPath.safeDownCast(muscle_path)
        raise ValueError(f"Use the inputted polynomial path. TODO: support this better")
    else:
        raise ValueError(f"Unsupported muscle path type: {muscle_path.getConcreteClassName()}")


def convert_muscles(model: OSimType.Model) -> list[MuscleData]:
    """ Returns the all the converted Muscles in the model """
    muscle_data = []
    muscles = get_muscles(model)
    for muscle in muscles:
        if muscle.getConcreteClassName() == "Millard2012EquilibriumMuscle":
            muscle = OSimType.MillardMuscle.safeDownCast(muscle)
            muscle_name = muscle.getName()
            fiber_damping = muscle.get_fiber_damping()
        elif muscle.getConcreteClassName() == "Thelen2003Muscle":
            muscle = OSimType.ThelenMuscle.safeDownCast(muscle)
            muscle_name = muscle.getName()
            fiber_damping = 0.01
        else:
            raise ValueError(f"Unsupported muscle type: {muscle.getConcreteClassName()} for muscle {muscle.getName()}")

        muscle_data.append(
            MuscleData(
                name=muscle_name,

                ignore_tendon_compliance=muscle.get_ignore_tendon_compliance(),

                min_control=muscle.get_min_control(),
                max_control=muscle.get_max_control(),

                max_isometric_force=muscle.get_max_isometric_force(),
                optimal_fiber_length=muscle.get_optimal_fiber_length(),
                tendon_slack_length=muscle.get_tendon_slack_length(),
                pennation_angle_at_optimal=muscle.get_pennation_angle_at_optimal(),
                fiber_damping=fiber_damping,

                path_points=convert_path(muscle.getPath())
            )
        )

    return muscle_data


def flatten_sites(muscles: list[MuscleData]) -> list[SiteData]:
    """ Flattens the path points of all the muscles into a single list of SiteData """
    sites = []
    for muscle in muscles:
        sites.extend(muscle.path_points)
    return sites


def get_muscle_pts_num(muscles: list[MuscleData]) -> list[int]:
    """ Returns the number of path points for each muscle """
    return [len(muscle.path_points) for muscle in muscles]


def create_muscle_metadata(
        muscles: list[MuscleData],
        muscle_with_fn_path: set[str]
) -> list[MuscleMetadata]:
    muscle_metadata = []
    for muscle in muscles:
        muscle_meta = MuscleMetadata()
        # Whether muscle uses function-based path
        muscle_meta.fn_based_path = muscle.name in muscle_with_fn_path
        # Muscle properties
        muscle_meta.ignore_tendon_compliance = muscle.ignore_tendon_compliance
        muscle_meta.max_isometric_force = muscle.max_isometric_force
        muscle_meta.optimal_fiber_length = muscle.optimal_fiber_length
        muscle_meta.tendon_slack_length = muscle.tendon_slack_length
        muscle_meta.optimal_pennation_angle = muscle.pennation_angle_at_optimal
        muscle_meta.fiber_damping = muscle.fiber_damping
        muscle_meta.min_activation = muscle.min_control
        muscle_meta.max_activation = muscle.max_control
        # Defaults, can be user-modified later
        muscle_meta.v_max = 10.0
        muscle_meta.activation_time_const = 0.010
        muscle_meta.deactivation_time_const = 0.040
        muscle_meta.activation_dynamics_smoothing = 10.0
        muscle_meta.min_norm_fiber_length = MIN_NORM_FIBER_LENGTH
        muscle_meta.max_norm_fiber_length = MAX_NORM_FIBER_LENGTH
        muscle_meta.specific_tension = 0.5e6
        muscle_meta.density = 1059.7
        muscle_meta.slow_twitch_ratio = 0.5
        muscle_metadata.append(muscle_meta)
    return muscle_metadata


def get_muscle_ordering(muscles: list[MuscleData]) -> dict[str, int]:
    """ Returns a mapping from muscle name to its index in the list """
    return {muscle.name: i for i, muscle in enumerate(muscles)}
