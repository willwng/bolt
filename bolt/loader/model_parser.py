""" Convert the OpenSim model into dataclasses """
from dataclasses import dataclass
from typing import Optional

import opensim as osim
import warp as wp

from bolt.loader.converters import actuator_helper
from bolt.loader.converters import body_helper
from bolt.loader.converters import coordinate_force_helper
from bolt.loader.converters import function_based_path_helper
from bolt.loader.converters import geom_helper
from bolt.loader.converters import joint_helper
from bolt.loader.converters import marker_helper
from bolt.loader.converters import muscle_helper
from bolt.loader.converters import site_helper
from bolt.loader.converters import spatial_transform_helper
from bolt.loader.converters import stateful_contact_helper
from bolt.loader.converters import swing_twist_helper
from bolt.loader.converters import visual_helper
from bolt.loader.converters.converted_objects import GROUND_BODY
from bolt.loader.converters.converted_objects import GROUND_COLLIDER
from bolt.loader.converters.converted_objects import GROUND_JOINT
from bolt.loader.converters.converted_objects import USE_POINT_PATH
from bolt.paths import get_geometry_dir


@dataclass
class ParsedModel:
    gravity: wp.vec3
    bodies: list
    joints: list
    geoms: list
    stl_contacts: list
    visuals: list
    spatial_transforms: list
    spring_gen_forces: list
    limit_forces: list
    swing_twists: list
    activation_actuators: list
    muscles: list
    function_paths: list
    # Sites, grouped (in this order) by what they belong to
    sites_muscle: list
    sites_contact: list
    sites_marker: list
    sites_rem: list

    @property
    def sites(self) -> list:
        return self.sites_muscle + self.sites_contact + self.sites_marker + self.sites_rem

    # Starting address of each site group in the concatenated sites
    @property
    def site_adr_muscle(self) -> int:
        return 0

    @property
    def site_adr_contact(self) -> int:
        return self.site_adr_muscle + len(self.sites_muscle)

    @property
    def site_adr_marker(self) -> int:
        return self.site_adr_contact + len(self.sites_contact)

    @property
    def site_adr_rem(self) -> int:
        return self.site_adr_marker + len(self.sites_marker)


def parse_osim_model(
        model_path: str,
        requires_visuals: bool,
        muscle_fn_path: Optional[str]
) -> ParsedModel:
    # All the mesh files for visuals should be located here
    osim.ModelVisualizer.addDirToGeometrySearchPaths(get_geometry_dir())
    # Run OpenSim parser
    model = osim.Model(model_path)
    model.initSystem()

    # Check every body has a mobilizer
    if model.getNumBodies() != model.getNumJoints():
        raise ValueError(f"Num bodies ({model.getNumBodies()}) does not match num Joints ({model.getNumJoints()})")

    # Parse bodies, joints, collision geometry, visuals, etc.
    bodies = [GROUND_BODY] + [body_helper.convert_body(body) for body in model.getBodyList()]
    joints = [GROUND_JOINT] + [joint_helper.convert_joint(joint) for joint in model.getJointList()]
    geoms = [GROUND_COLLIDER] + geom_helper.convert_geoms(model, include_body_components=False)
    stl_contacts = stateful_contact_helper.convert_stateful_contacts(model, model_path)
    visuals = visual_helper.convert_visuals(model) if requires_visuals else []
    spatial_transforms = spatial_transform_helper.convert_spatial_transforms(model)
    spring_gen_forces = coordinate_force_helper.convert_spring_generalized_force(model)
    limit_forces = coordinate_force_helper.convert_coordinate_limit_force(model)  # also a limit force
    swing_twists = swing_twist_helper.convert_swing_twist_limits(model_path)
    activation_actuators = actuator_helper.convert_activation_actuators(model)
    muscles = muscle_helper.convert_muscles(model)
    # Gather all sites
    sites_muscle = muscle_helper.flatten_sites(muscles)
    sites_contact = stateful_contact_helper.flatten_sites(stl_contacts)
    sites_marker = marker_helper.convert_markers(model)
    sites_rem = site_helper.convert_sites(model)

    # Function-based paths
    if muscle_fn_path is not None:
        function_paths = function_based_path_helper.parse_function_based_paths(model_path, muscle_fn_path)
        assert len(function_paths) == len(muscles)
    else:
        function_paths = [USE_POINT_PATH] * len(muscles)

    return ParsedModel(
        gravity=wp.vec3(model.getGravity().to_numpy()),
        bodies=bodies,
        joints=joints,
        geoms=geoms,
        stl_contacts=stl_contacts,
        visuals=visuals,
        spatial_transforms=spatial_transforms,
        spring_gen_forces=spring_gen_forces,
        limit_forces=limit_forces,
        swing_twists=swing_twists,
        activation_actuators=activation_actuators,
        muscles=muscles,
        function_paths=function_paths,
        sites_muscle=sites_muscle,
        sites_contact=sites_contact,
        sites_marker=sites_marker,
        sites_rem=sites_rem,
    )
