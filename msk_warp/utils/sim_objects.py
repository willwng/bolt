"""
Helper to map from simulation indices to strings. a little redundant
but we can't keep strings in the madrona sim
"""
from .osim_parser import parse_osim_file
import os

# Defaults
sim_obj_list = [
    "cube.obj",
    "sphere.obj",
    "plane.obj",
    "capsule.obj",
]

sim_bodies_list = []
sim_joints_list = []
sim_muscles_list = []
sim_actuators_list = []


def parse_mesh_file_path(mesh_file: str) -> str:
    # Remove .vtp and replace with .obj
    base, ext = os.path.splitext(mesh_file)
    if ext == ".vtp":
        mesh_file = base + ".obj"
    return mesh_file


def load_osim(osim_file: str):
    model_data = parse_osim_file(osim_file)
    # Bodies and visual obj files
    for (body_name, body) in model_data.body_set.bodies.items():
        sim_bodies_list.append(body_name)

        # Visual obj files
        attached_geometry = body.attached_geometry
        for mesh in attached_geometry.meshes:
            file = parse_mesh_file_path(mesh.mesh_file)
            sim_obj_list.append(file)

    # Coordinates
    for (joint_name, joint) in model_data.joint_set.joints.items():
        # Ignoring anything connected to ground
        if "ground" in joint.name:
            continue
        for coord in joint.coordinates:
            sim_joints_list.append(coord.name)
        pass

    # Muscles
    for (muscle_name, muscle) in model_data.force_set.muscles.items():
        sim_muscles_list.append(muscle_name)

    # Actuators
    for (actuator_name, actuator) in model_data.force_set.actuators.items():
        sim_actuators_list.append(actuator_name)
    return


def setup_sim_objects(osim_file: str):
    load_osim(osim_file)


def obj_id_to_file(obj_id):
    sim_obj = sim_obj_list[obj_id]
    return f"assets/geometry/obj/{sim_obj}"


def body_id_to_name(body_id: int):
    return sim_bodies_list[body_id]


def joint_id_to_name(joint_id: int, is_positional: bool):
    xyz = ['x', 'y', 'z']
    wxyz = ['w', 'x', 'y', 'z']
    if is_positional:
        if joint_id < 3:
            return f"pelvis {xyz[joint_id]}"
        elif joint_id < 7:
            return f"pelvis rot {wxyz[joint_id - 3]}"
        return sim_joints_list[joint_id - 7]
    else:
        if joint_id < 3:
            return f"pelvis {xyz[joint_id]}"
        elif joint_id < 6:
            return f"pelvis rot {xyz[joint_id - 3]}"
        return sim_joints_list[joint_id - 6]


def muscle_id_to_name(muscle_id):
    return sim_muscles_list[muscle_id]


def actuator_id_to_name(actuator_id):
    return sim_actuators_list[actuator_id]