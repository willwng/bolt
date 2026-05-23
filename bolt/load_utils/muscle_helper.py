import opensim as osim
import warp as wp

from bolt import MAX_NORM_FIBER_LENGTH, ContractionType, MuscleMetadata, BOLT_SIG_REAL, \
    MILLARD_MIN_NORM_ACTIVE_FIBER_LENGTH, MIN_NORM_FIBER_LENGTH
from bolt.load_utils.converted_objects import MuscleData, SiteData
from bolt.load_utils.osim_types import OSimType
from bolt.load_utils.physical_frame_helper import extract_frame_transform_from_base_frame, get_body_name_of_frame
from bolt.load_utils.property_helper import extract_vec3


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
        if path_point := OSimType.PathPoint.safeDownCast(point):
            path_points.append(convert_path_point(path_point))
        elif cond_point := OSimType.ConditionalPathPoint.safeDownCast(point):
            path_points.append(convert_path_point(cond_point))
        # elif mov_point := OSimType.MovingPathPoint.safeDownCast(point):
        #     path_points.append(convert_path_point(mov_point))
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


def get_passive_fiber_force_curve(muscle: OSimType.Muscle) -> tuple[float, float, float, float, float]:
    if muscle.getConcreteClassName() == "Millard2012EquilibriumMuscle":
        muscle = OSimType.MillardMuscle.safeDownCast(muscle)
        force_length_curve = muscle.getFiberForceLengthCurve()

        strain_at_zero_force = force_length_curve.get_strain_at_zero_force()
        strain_at_one_norm_force = force_length_curve.get_strain_at_one_norm_force()
        stiffness_at_low_force = force_length_curve.get_stiffness_at_low_force()
        stiffness_at_one_norm_force = force_length_curve.get_stiffness_at_one_norm_force()
        curviness = force_length_curve.get_curviness()
        return (strain_at_zero_force, strain_at_one_norm_force, stiffness_at_low_force,
                stiffness_at_one_norm_force, curviness)
    elif muscle.getConcreteClassName() == "Thelen2003Muscle":
        muscle = OSimType.ThelenMuscle.safeDownCast(muscle)
        strain_at_zero_force = 0.0
        strain_at_one_norm_force = muscle.get_FmaxMuscleStrain()  # TODO
        stiffness_at_low_force = 0.2
        stiffness_at_one_norm_force = 2.857142857142857
        curviness = 0.75
        return (strain_at_zero_force, strain_at_one_norm_force, stiffness_at_low_force,
                stiffness_at_one_norm_force, curviness)
    raise ValueError(f"Unsupported muscle type: {muscle.getConcreteClassName()} for muscle {muscle.getName()}")


def convert_muscles(model: OSimType.Model) -> list[MuscleData]:
    """ Returns the all the converted Muscles in the model """
    muscle_data = []
    muscles = get_muscles(model)
    for muscle in muscles:
        muscle_name = muscle.getName()

        if muscle.getConcreteClassName() == "Millard2012EquilibriumMuscle":
            muscle = OSimType.MillardMuscle.safeDownCast(muscle)
            fiber_damping = muscle.get_fiber_damping()
        elif muscle.getConcreteClassName() == "Thelen2003Muscle":
            muscle = OSimType.ThelenMuscle.safeDownCast(muscle)
            fiber_damping = 0.01
        else:
            raise ValueError(f"Unsupported muscle type: {muscle.getConcreteClassName()} for muscle {muscle.getName()}")

        (strain_at_zero_force, strain_at_one_norm_force, stiffness_at_low_force,
         stiffness_at_one_norm_force, curviness) = get_passive_fiber_force_curve(muscle)
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

                strain_at_zero_force=strain_at_zero_force,
                strain_at_one_norm_force=strain_at_one_norm_force,
                stiffness_at_low_force=stiffness_at_low_force,
                stiffness_at_one_norm_force=stiffness_at_one_norm_force,
                curviness=curviness,

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
) -> list[MuscleMetadata]:
    muscle_metadata = []
    for muscle in muscles:
        muscle_meta = MuscleMetadata()
        # Muscle properties
        muscle_meta.ignore_tendon_compliance = muscle.ignore_tendon_compliance
        muscle_meta.max_isometric_force = muscle.max_isometric_force
        muscle_meta.optimal_fiber_length = muscle.optimal_fiber_length
        muscle_meta.tendon_slack_length = muscle.tendon_slack_length
        muscle_meta.optimal_pennation_angle = muscle.pennation_angle_at_optimal
        muscle_meta.fiber_damping = muscle.fiber_damping
        muscle_meta.min_activation = muscle.min_control
        muscle_meta.max_activation = muscle.max_control
        # Fiber passive length curve
        muscle_meta.strain_at_zero_force = muscle.strain_at_zero_force
        muscle_meta.strain_at_one_norm_force = muscle.strain_at_one_norm_force
        muscle_meta.stiffness_at_low_force = muscle.stiffness_at_low_force
        muscle_meta.stiffness_at_one_norm_force = muscle.stiffness_at_one_norm_force
        muscle_meta.curviness = muscle.curviness
        # Defaults, can be user-modified later
        muscle_meta.v_max = 10.0
        muscle_meta.activation_time_const = 0.010
        muscle_meta.deactivation_time_const = 0.040
        muscle_meta.activation_dynamics_smoothing = 10.0
        muscle_meta.specific_tension = 0.5e6
        muscle_meta.density = 1059.7
        muscle_meta.slow_twitch_ratio = 0.5
        muscle_meta.active_force_width_scale = 1.0

        # To be set during model initialization
        muscle_meta.min_norm_fiber_length = MIN_NORM_FIBER_LENGTH
        muscle_meta.max_norm_fiber_length = MAX_NORM_FIBER_LENGTH

        muscle_metadata.append(muscle_meta)
    return muscle_metadata


def get_muscle_ordering(muscles: list[MuscleData]) -> dict[str, int]:
    """ Returns a mapping from muscle name to its index in the list """
    return {muscle.name: i for i, muscle in enumerate(muscles)}


def adjust_norm_fiber_length_range(muscle: MuscleMetadata, contraction_dynamics: ContractionType):
    """
    After initializing the muscles, the user may change the contraction type, this modifies the
        minimum fiber length if necesary
    """
    # Compute pennation model's minimum fiber length
    parallelogram_height = wp.sin(muscle.optimal_pennation_angle)
    maximum_pennation_angle = wp.acos(0.1)
    maximum_sin_pennation = wp.sin(maximum_pennation_angle)
    if maximum_pennation_angle > BOLT_SIG_REAL:
        pennation_min_norm_fiber_length = parallelogram_height / maximum_sin_pennation
    else:
        pennation_min_norm_fiber_length = 0.01

    # Compute active force-length's minimum fiber length
    if contraction_dynamics == ContractionType.MILLARD:
        active_curve_min_norm_fiber_length = MILLARD_MIN_NORM_ACTIVE_FIBER_LENGTH
    elif contraction_dynamics == ContractionType.MUJOCO:
        active_curve_min_norm_fiber_length = 0.0  # no such thing as a min fiber length
    else:
        active_curve_min_norm_fiber_length = MIN_NORM_FIBER_LENGTH

    muscle.min_norm_fiber_length = max(pennation_min_norm_fiber_length, active_curve_min_norm_fiber_length)
    return
