import warp_sim._src.forward as forward
import warp_sim._src.types as types

from warp_sim.utils.osim_parser import parse_osim_file
from .utils.osim_converter import *


def create_model() -> types.Model:
    m = types.Model()
    # Initialize model parameters here
    return m


def exclusive_scan(v, mark_empty: bool):
    result = [0] * (len(v) + 1)
    for i in range(1, len(result)):
        result[i] = result[i - 1] + v[i - 1]
    # Remove the last element to return the exclusive scan
    result = result[:-1]

    if mark_empty:
        for i in range(len(v)):
            if v[i] == 0:
                result[i] = -1

    return result


def main():
    osim_model = parse_osim_file("data/osim/model.osim")
    nb = num_bodies(osim_model)

    joint_num_qdofs = get_joint_num_qdofs(osim_model)
    joint_num_vdofs = get_joint_num_vdofs(osim_model)

    nv = sum(joint_num_vdofs)
    nq = sum(joint_num_qdofs)
    nmuscle = num_muscles(osim_model)

    n_conv_jnts, n_custom_jnts = get_num_joints(osim_model)

    ngeom = num_colliders(osim_model)
    nsite = num_sites(osim_model)
    qpos0 = get_default_positions(osim_model)
    qpos_spring = [0.0] * len(qpos0)  # Placeholder for spring positions

    b_masses = body_masses(osim_model)
    inertias = get_body_inertias(osim_model)
    body_local_com = get_local_body_com_pos(osim_model)
    body_local_rot = get_local_body_rot(osim_model)
    body_num_colliders = get_body_num_colliders(osim_model)
    body_collider_offset = exclusive_scan(body_num_colliders, True)

    body_parent_ids = get_body_parent_ids(osim_model)
    joint_types = get_joint_types(osim_model)

    jnt_qpos_adr = exclusive_scan(joint_num_qdofs, False)
    jnt_dof_adr = exclusive_scan(joint_num_vdofs, False)

    jnt_rel_parent = get_joint_rel_pos(osim_model, parent=True)
    jnt_rel_child = get_joint_rel_pos(osim_model, parent=False)
    jnt_rel_parent_rot = get_joint_rel_rot(osim_model, parent=True)
    jnt_rel_child_rot = get_joint_rel_rot(osim_model, parent=False)

    print(f"Number of bodies: {nb}")
    print(f"Num dofs: {nv}")
    print(f"Num pos dofs: {nq}")
    print(f"Num muscles: {nmuscle}")
    print(f"Num colliders: {ngeom}")
    # print(f"Num sites: {nsite}")
    # print(f"Default positions: {qpos0}")
    # print(f"Spring positions: {qpos_spring}")
    # print(f"Body masses: {b_masses}")
    # print(f"Body inertias: {inertias}")
    # print(f"Body local COM positions: {body_local_com}")
    # print(f"Body local COM rotations: {body_local_rot}")
    # print(f"Body num colliders: {body_num_colliders}")
    # print(f"Body collider offsets: {body_collider_offset}")
    # print(f"Body parent IDs: {body_parent_ids}")
    # print(f"Number of conventional joints: {n_conv_jnts}")
    # print(f"Number of custom joints: {n_custom_jnts}")
    # print(f"Joint qpos addresses: {jnt_qpos_adr}")
    # print(f"Joint vpos addresses: {jnt_dof_adr}")
    # print(f"Joint types: {joint_types}")
    # print(f"Joint relative parent positions: {jnt_rel_parent}")
    # print(f"Joint relative child positions: {jnt_rel_child}")
    # print(f"Joint relative parent rotations: {jnt_rel_parent_rot}")
    # print(f"Joint relative child rotations: {jnt_rel_child_rot}")


    # needs shapes
    opt = types.Option(
        timestep=0.002,
        impratio=1.0,
        tolerance=1e-8,
        ls_tolerance=0.01,
        ccd_tolerance=1e-6,
        gravity=[0.0, 0.0, -9.81],
        solver=0,
        iterations=50,
        ls_iterations=100,
        ccd_iterations=50,
        is_sparse=False,
        ls_parallel=False,
        ls_parallel_min_step=1e-8,
        graph_conditional=True
    )

    quit()
    m = create_model()
    d = types.Data()
    forward.step(m, d)


if __name__ == "__main__":
    main()
