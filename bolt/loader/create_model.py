""" Create the Model struct. Each pack_* function sets one group of Model fields """
import warp as wp

from bolt.loader.array_util import assert_all_fields_set
from bolt.loader.array_util import make_zero
from bolt.loader.array_util import new_unset_class
from bolt.loader.array_util import to_warp_array
from bolt.loader.converters import actuator_helper
from bolt.loader.converters import body_helper
from bolt.loader.converters import coordinate_force_helper
from bolt.loader.converters import function_based_path_helper
from bolt.loader.converters import function_helper
from bolt.loader.converters import geom_helper
from bolt.loader.converters import joint_helper
from bolt.loader.converters import muscle_helper
from bolt.loader.converters import site_helper
from bolt.loader.converters import spatial_transform_helper
from bolt.loader.converters import stateful_contact_helper
from bolt.loader.converters import swing_twist_helper
from bolt.loader.converters import visual_helper
from bolt.loader.converters.converted_objects import ConstantFunctionData
from bolt.loader.converters.converted_objects import GeomData
from bolt.loader.converters.converted_objects import LinearFunctionData
from bolt.loader.converters.converted_objects import PolynomialFunctionData
from bolt.loader.converters.converted_objects import SimmSplineData
from bolt.loader.converters.python_util import apply_map_to_list
from bolt.loader.converters.python_util import create_nested_list
from bolt.loader.converters.python_util import exclusive_scan
from bolt.loader.converters.python_util import gather
from bolt.loader.model_topology import ModelTopology
from bolt.loader.model_parser import ParsedModel
from bolt.types_consts import ActivationType
from bolt.types_consts import ActuatorMetadata
from bolt.types_consts import ContractionType
from bolt.types_consts import CoordinateLimitForce
from bolt.types_consts import IntegratorType
from bolt.types_consts import MobilizerType
from bolt.types_consts import Model
from bolt.types_consts import MuscleMetadata
from bolt.types_consts import Option
from bolt.types_consts import PolyInts
from bolt.types_consts import StatefulContact
from bolt.types_consts import SwingTwistLimit
from bolt.types_consts import TileBlockDim


def create_model(
        parsed: ParsedModel,
        topology: ModelTopology,
        integrator: IntegratorType,
        requires_visuals: bool
) -> Model:
    m = new_unset_class(Model)
    pack_bodies(m, topology)
    pack_joints(m, topology)
    pack_custom_joints(m, parsed, topology)
    pack_coordinate_forces(m, parsed, topology)
    pack_geoms(m, parsed.geoms, topology.body_ordering, topology.body_parent_id)
    pack_sites_and_contacts(m, parsed, topology)
    pack_visuals(m, parsed, topology)
    pack_muscles(m, parsed, topology)
    pack_actuators(m, parsed, topology)

    # need muscle activation & fiber state and actuator activation state
    m.nz = 2 * m.nmuscle + m.nactuator
    m.opt = make_options(parsed, integrator, requires_visuals, nv=m.nv, nz=m.nz,
                         nbeam_visuals=5 if requires_visuals else 0)
    m.block_dim = TileBlockDim()

    assert_all_fields_set(m)
    return m


def pack_bodies(m: Model, topology: ModelTopology):
    bodies = topology.bodies
    m.nbody = len(bodies)
    m.body_mass = to_warp_array(body_helper.get_body_masses(bodies), dtype=float)
    m.body_mass_center = to_warp_array(body_helper.get_body_center(bodies), dtype=wp.vec3)
    m.body_unit_inertia_OB_B = to_warp_array(body_helper.get_body_unit_inertia_OB_B(bodies), dtype=wp.mat33)
    m.body_parentid = to_warp_array(topology.body_parent_id, dtype=int)
    m.body_tree = tuple([to_warp_array(level, dtype=int) for level in topology.body_tree])
    m.body_children = to_warp_array(topology.body_children, dtype=int)
    m.body_children_num = to_warp_array(topology.body_children_num, dtype=int)
    m.body_children_adr = to_warp_array(topology.body_children_adr, dtype=int)
    return


def pack_joints(m: Model, topology: ModelTopology):
    joints = topology.joints
    m.nq = sum([joint.num_coordinates for joint in joints])
    m.nv = sum([joint.num_speeds for joint in joints])

    m.mob_type = to_warp_array(joint_helper.get_mob_type(joints), dtype=int)
    m.mob_qposadr = to_warp_array(topology.mob_qpos_adr, dtype=int)
    m.mob_dofadr = to_warp_array(topology.mob_dof_adr, dtype=int)
    m.mob_dofnum = to_warp_array(joint_helper.get_mob_dofnum(joints), dtype=int)
    m.mob_X_PF = to_warp_array(joint_helper.get_mob_X_PF(joints), dtype=wp.transform)
    m.mob_X_MB = to_warp_array(joint_helper.get_mob_X_MB(joints), dtype=wp.transform)
    m.mob_extra_info = to_warp_array(joint_helper.get_mob_extra_info(joints), dtype=wp.vec3)

    # Index of mobilizer -> index of custom joint (-1 if not custom)
    mob_to_cst_idx, cst_to_mob_idx = joint_helper.compute_mobilizer_index_of_type(joints, MobilizerType.CUSTOM)
    m.njnts_cst = joint_helper.compute_num_joints_of_type(joints, MobilizerType.CUSTOM)
    m.mob_to_cst_id = to_warp_array(mob_to_cst_idx, dtype=int)
    m.cst_to_mob_id = to_warp_array(cst_to_mob_idx, dtype=int)

    # Index of mobilizer -> index of beam joint
    mob_to_beam_idx, beam_to_mob_idx = joint_helper.compute_mobilizer_index_of_type(joints, MobilizerType.BEAM)
    m.nbeams = joint_helper.compute_num_joints_of_type(joints, MobilizerType.BEAM)
    m.beam_to_mob_id = to_warp_array(beam_to_mob_idx, dtype=int)
    return


def pack_custom_joints(m: Model, parsed: ParsedModel, topology: ModelTopology):
    """ Custom joint spatial transforms and the functions of their transform axes """
    ordered_spatial_transforms = spatial_transform_helper.order_spatial_transforms(
        parsed.spatial_transforms, topology.joint_ordering)
    # Spatial transforms: flatten all the axes
    ordered_transform_axes = spatial_transform_helper.get_flattened_transform_axes(ordered_spatial_transforms)
    # Get all relative coordinate indices for each transform axis
    txfm_dofs = spatial_transform_helper.get_txfm_coordinate_names(ordered_transform_axes)
    txfm_qpos_relative_idx = apply_map_to_list(txfm_dofs, topology.relative_dof_ordering)
    txfm_qpos_global_idx = apply_map_to_list(txfm_dofs, topology.qpos_ordering)

    # We need to reshape the transform data to be (num_custom_joints, 6)
    txfm_axes = spatial_transform_helper.get_txfm_axes(ordered_transform_axes)
    cst_txfm_axes = create_nested_list(txfm_axes, num_per_sublist=6)
    cst_txfm_dof = create_nested_list(txfm_qpos_relative_idx, num_per_sublist=6)
    # If these lists are empty, we should fill them with dummy data so that the shape is correct
    if not cst_txfm_axes:
        cst_txfm_axes = [[wp.vec3()] * 6]
    if not cst_txfm_dof:
        cst_txfm_dof = [[0] * 6]
    m.cst_txfm_axes = to_warp_array(cst_txfm_axes, dtype=wp.vec3)
    m.cst_txfm_dof = to_warp_array(cst_txfm_dof, dtype=int)

    # Collect all functions in the spatial transforms
    linear_fns, linear_fns_idx = function_helper.get_functions_of_type(ordered_transform_axes, cls=LinearFunctionData)
    const_fns, const_fns_idx = function_helper.get_functions_of_type(ordered_transform_axes, cls=ConstantFunctionData)
    poly_fns, poly_fns_idx = function_helper.get_functions_of_type(ordered_transform_axes, cls=PolynomialFunctionData)
    spline_fns, spline_fns_idx = function_helper.get_functions_of_type(ordered_transform_axes, cls=SimmSplineData)
    m.nlinearfn = len(linear_fns)
    m.nconstfn = len(const_fns)
    m.npolyfn = len(poly_fns)
    m.nsplinefn = len(spline_fns)
    m.nfunctions = m.nlinearfn + m.nconstfn + m.npolyfn + m.nsplinefn

    # Function parameters
    poly_coeffs_num, poly_coeffs_adr = function_helper.get_poly_coeffs_num_adr(poly_fns)
    spline_xys_num, spline_xys_adr = function_helper.get_spline_xys_num_adr(spline_fns)
    m.linear_fn_mb = to_warp_array(function_helper.get_linear_fn_mb(linear_fns), dtype=wp.vec2)
    m.const_fn_c = to_warp_array(function_helper.get_const_fn_vals(const_fns), dtype=float)
    m.poly_fn_coeff = to_warp_array(function_helper.get_flattened_poly_coeffs(poly_fns), dtype=float)
    m.poly_fn_coeff_adr = to_warp_array(poly_coeffs_adr, dtype=int)
    m.poly_fn_coeff_num = to_warp_array(poly_coeffs_num, dtype=int)
    m.spline_fn_xy_y2s = to_warp_array(function_helper.get_spline_xy_y2s(spline_fns), dtype=wp.vec3)
    m.spline_fn_xys_adr = to_warp_array(spline_xys_adr, dtype=int)
    m.spline_fn_xys_num = to_warp_array(spline_xys_num, dtype=int)

    # Transform axis of each function
    m.linear_fn_adr = to_warp_array(linear_fns_idx, dtype=int)
    m.const_fn_adr = to_warp_array(const_fns_idx, dtype=int)
    m.poly_fn_adr = to_warp_array(poly_fns_idx, dtype=int)
    m.spline_fn_adr = to_warp_array(spline_fns_idx, dtype=int)

    # Use gather to find the global coordinate indices used for each function
    m.linear_fn_qpos_adr = to_warp_array(gather(txfm_qpos_global_idx, linear_fns_idx), dtype=int)
    m.poly_fn_qpos_adr = to_warp_array(gather(txfm_qpos_global_idx, poly_fns_idx), dtype=int)
    m.spline_fn_qpos_adr = to_warp_array(gather(txfm_qpos_global_idx, spline_fns_idx), dtype=int)
    return


def pack_coordinate_forces(m: Model, parsed: ParsedModel, topology: ModelTopology):
    """ Joint damping, springs, and limits """
    dof_stiffness, dof_damping = coordinate_force_helper.get_dof_stiffness_damping(
        parsed.spring_gen_forces, topology.dof_ordering)
    m.dof_damping = to_warp_array(dof_damping, dtype=float)
    m.dof_armature = make_zero(len(dof_damping), dtype=float)  # user-modified later
    m.dof_stiffness = to_warp_array(dof_stiffness, dtype=float)
    m.qpos_spring_rest = to_warp_array(coordinate_force_helper.get_qpos_spring_rest(topology.qpos_ordering),
                                       dtype=float)

    coordinate_limit_forces = coordinate_force_helper.create_coordinate_limit_force(
        parsed.limit_forces, topology.qpos_ordering, topology.dof_ordering)
    m.nlimitforce = len(parsed.limit_forces)
    m.coordinate_limit_force = wp.array(coordinate_limit_forces, dtype=CoordinateLimitForce)

    swing_twist_limits = swing_twist_helper.create_swing_twist_data(
        parsed.swing_twists, topology.joint_ordering, topology.mob_qpos_adr, topology.mob_dof_adr)
    m.nswingtwist = len(parsed.swing_twists)
    m.swing_twist_limit = wp.array(swing_twist_limits, dtype=SwingTwistLimit)
    return


def pack_geoms(m: Model, geoms: list[GeomData], body_ordering: dict[str, int], body_parent_id: list[int]):
    """ Collider (geom) fields, including broadphase pair registration. Also used to update colliders later """
    geom_type = geom_helper.get_geom_type(geoms)
    geom_body_id = apply_map_to_list(geom_helper.get_geom_body_name(geoms), body_ordering)
    m.ngeom = len(geoms)
    m.geom_type = to_warp_array(geom_type, dtype=int)
    m.geom_bodyid = to_warp_array(geom_body_id, dtype=int)
    m.geom_X_loc = to_warp_array(geom_helper.get_geom_transform(geoms), dtype=wp.transform)
    m.geom_size = to_warp_array(geom_helper.get_geom_size(geoms), dtype=wp.vec3)
    m.geom_friction = to_warp_array(geom_helper.get_geom_friction(geoms), dtype=wp.vec3)
    m.geom_stiffness = to_warp_array(geom_helper.get_geom_stiffness(geoms), dtype=float)
    m.geom_dissipation = to_warp_array(geom_helper.get_geom_dissipation(geoms), dtype=float)
    m.geom_transition_velocity = to_warp_array(geom_helper.get_geom_transition_velocity(geoms), dtype=float)
    m.geom_priority = to_warp_array(geom_helper.get_geom_priority(geoms), dtype=int)
    m.geom_aabb = to_warp_array(geom_helper.get_geom_aabb(geoms), dtype=wp.vec3)
    m.geom_rbound = to_warp_array(geom_helper.get_geom_rbound(geoms), dtype=float)

    # Broadphase registration
    geom_type_pair_count, nxn_geom_pair_filtered, nxn_pairid_filtered = (
        geom_helper.prepare_contacts(geom_type, geom_body_id, body_parent_id, m.ngeom))
    m.geom_pair_type_count = tuple(geom_type_pair_count)
    m.nxn_geom_pair_filtered = wp.array(nxn_geom_pair_filtered, dtype=wp.vec2i)
    m.nxn_pairid_filtered = wp.array(nxn_pairid_filtered, dtype=wp.vec2i)
    return


def pack_sites_and_contacts(m: Model, parsed: ParsedModel, topology: ModelTopology):
    """ Sites (muscle points, contact stations, markers, other stations) and the stateful contacts using them """
    sites = parsed.sites
    m.nsite = len(sites)
    m.nsite_muscle = len(parsed.sites_muscle)
    m.nsite_contact = len(parsed.sites_contact)
    m.nsite_marker = len(parsed.sites_marker)
    m.nsite_rem = len(parsed.sites_rem)
    m.site_adr_muscle = parsed.site_adr_muscle
    m.site_adr_contact = parsed.site_adr_contact
    m.site_adr_marker = parsed.site_adr_marker
    m.site_adr_rem = parsed.site_adr_rem
    m.site_bodyid = to_warp_array(
        apply_map_to_list(site_helper.get_site_body_name(sites), topology.body_ordering), dtype=int)
    m.site_offset = to_warp_array(site_helper.get_site_offset(sites), dtype=wp.vec3)

    stateful_contact_data = stateful_contact_helper.create_stateful_contact_data(
        stateful_contact_data=parsed.stl_contacts,
        site_start_stl=parsed.site_adr_contact,
        body_ordering=topology.body_ordering
    )
    m.nstlcontact = len(parsed.stl_contacts)
    m.stl_contact = wp.array(stateful_contact_data, dtype=StatefulContact)
    return


def pack_visuals(m: Model, parsed: ParsedModel, topology: ModelTopology):
    visuals = parsed.visuals
    m.nvis = len(visuals)
    m.vis_bodyid = to_warp_array(
        apply_map_to_list(visual_helper.get_vis_body_name(visuals), topology.body_ordering), dtype=int)
    m.vis_X_loc = to_warp_array(visual_helper.get_vis_transform(visuals), dtype=wp.transform)


def pack_muscles(m: Model, parsed: ParsedModel, topology: ModelTopology):
    """ Muscle metadata, point paths, and function-based paths """
    muscles, function_paths = parsed.muscles, parsed.function_paths
    muscle_data = muscle_helper.create_muscle_metadata(muscles)
    m.nmuscle = len(muscles)
    m.muscle_data = muscle_data
    m.muscle_metadata = wp.array(muscle_data, dtype=MuscleMetadata)

    # Point paths
    muscle_pts_num = muscle_helper.get_muscle_pts_num(muscles)
    # shift by number of sites before muscles
    muscle_pts_adr = [adr + parsed.site_adr_muscle for adr in exclusive_scan(muscle_pts_num)]
    m.muscle_pts_num = to_warp_array(muscle_pts_num, dtype=int)
    m.muscle_pts_adr = to_warp_array(muscle_pts_adr, dtype=int)

    # Determine the path type for each muscle
    point_paths_group, function_paths_groups = function_based_path_helper.path_type_to_muscle(function_paths)
    m.muscle_pt_group = to_warp_array(point_paths_group, dtype=int)
    m.muscle_pt_group_tuple = tuple(point_paths_group)
    m.muscle_fn_groups = tuple([to_warp_array(group, dtype=int) for group in function_paths_groups])

    # Function-based paths
    fn_path_term_start, fn_path_term_count = function_based_path_helper.compute_fn_path_term_start_and_count(
        function_paths)
    m.fn_path_term_coeffs = to_warp_array(
        function_based_path_helper.get_fn_path_term_coeffs(function_paths), dtype=float)
    m.fn_path_term_start = to_warp_array(fn_path_term_start, dtype=int)
    m.fn_path_qpos_adr = to_warp_array(
        function_based_path_helper.get_fn_term_adr(function_paths, topology.qpos_ordering), dtype=PolyInts)
    m.fn_path_dimension = to_warp_array(function_based_path_helper.get_fn_path_dimension(function_paths), dtype=int)
    m.fn_path_order = to_warp_array(function_based_path_helper.get_fn_path_order(function_paths), dtype=int)
    return


def pack_actuators(m: Model, parsed: ParsedModel, topology: ModelTopology):
    actuator_data = actuator_helper.create_actuator_metadata(parsed.activation_actuators, topology.dof_ordering)
    m.nactuator = len(parsed.activation_actuators)
    m.actuator_data = actuator_data
    m.actuator_metadata = wp.array(actuator_data, dtype=ActuatorMetadata)
    return


def make_options(
        parsed: ParsedModel,
        integrator: IntegratorType,
        requires_visuals: bool,
        nv: int,
        nz: int,
        nbeam_visuals: int
) -> Option:
    return Option(
        gravity=parsed.gravity,
        explicit_gravity=True,
        implicit_damping=True,
        visuals=requires_visuals,
        nbeam_visuals=nbeam_visuals,

        activation_type=ActivationType.MILLARD,
        contraction_type=ContractionType.DGF,
        integrator=integrator,

        use_linear_stop=False,

        safety=0.9,
        min_shrink=0.1,
        max_grow=5.0,
        hysteresis_low=0.9,
        hysteresis_high=1.2,
        min_step_size=1e-7,
        max_step_size=wp.inf,
        accuracy=0.01,
        use_inf_norm=False,
        qvel_weights=wp.full(nv, 1.0, dtype=float),
        z_weights=wp.full(nz, 1.0, dtype=float),
    )
