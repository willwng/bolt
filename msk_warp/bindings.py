import numpy as np
import warp as wp
import torch
import msk_warp

import msk_warp.model_loader as model_loader
from msk_warp import Model, Data, IntegratorType, ActivationType, MuscleMetadata
from msk_warp.model_load_result import ModelLoadResult
from msk_warp.render.renderer import Renderer, RendererType


def load_model(
        model_path: str,
        n_worlds: int,
        integrator: IntegratorType,
        requires_visuals: bool,
        polynomial_data_path: str = None,
        render_kinematic_tree: bool = False,
) -> ModelLoadResult:
    load_result = model_loader.load_model(
        model_path=model_path,
        n_worlds=n_worlds,
        integrator=integrator,
        requires_visuals=requires_visuals,
        polynomial_data_path=polynomial_data_path,
        render_kinematic_tree=render_kinematic_tree,
    )
    m, d = load_result.model, load_result.data
    reinitialize_model(m, d)
    return load_result


def reinitialize_model(m: Model, d: Data, ):
    """ Re-initialize the model (i.e., if any parameters have changed). """
    # Ensure the muscle metadata is up to date
    mm = wp.array(m.muscle_data, dtype=MuscleMetadata)
    m.muscle_metadata = mm

    # Same with actuator metadata
    am = wp.array(m.actuator_data, dtype=msk_warp.ActuatorMetadata)
    m.actuator_metadata = am

    d.world_reset.fill_(True)
    msk_warp.reset(m, d)


def create_renderer(
        load_result: ModelLoadResult,
        renderer_type: RendererType,
        draw_colliders: bool,
        draw_visuals: bool,
        draw_muscles: bool,
        draw_body_mass: bool,
        draw_beams: bool,
        draw_sites: bool,
):
    viewer = Renderer(
        m=load_result.model,
        renderer_type=renderer_type,
        draw_colliders=draw_colliders,
        draw_visuals=draw_visuals,
        draw_muscles=draw_muscles,
        draw_body_mass=draw_body_mass,
        draw_beams=draw_beams,
        draw_sites=draw_sites,
    )
    viewer.load_meshes(load_result.mesh_load_results)
    return viewer


# --- Model Fields ---
def damping(m: Model) -> torch.Tensor:
    return wp.to_torch(m.dof_damping)


def armature(m: Model) -> torch.Tensor:
    return wp.to_torch(m.dof_armature)


def stiffness(m: Model) -> torch.Tensor:
    return wp.to_torch(m.dof_stiffness)


def body_mass(m: Model) -> torch.Tensor:
    return wp.to_torch(m.body_mass)


def get_num_qpos(m: Model) -> int:
    return m.nq


def get_num_dofs(m: Model) -> int:
    return m.nv


def get_num_bodies(m: Model) -> int:
    return m.nbody


def get_num_visuals(m: Model) -> int:
    return m.nvis


def get_num_colliders(m: Model) -> int:
    return m.ngeom


def get_num_muscles(m: Model) -> int:
    return m.nmuscle


def get_num_actuators(m: Model) -> int:
    return m.nactuator


def get_num_limits(m: Model) -> int:
    return m.nlimitforce + (3 * m.nswingtwist)


def get_qpos_adr(m: Model, body_id: int) -> torch.Tensor:
    mob_qpos_adr = wp.to_torch(m.mob_qposadr)
    return mob_qpos_adr[body_id]


def get_dof_adr(m: Model, body_id: int) -> torch.Tensor:
    jnt_dof_adr = wp.to_torch(m.mob_dofadr)
    return jnt_dof_adr[body_id]


def get_qpos_num(m: Model, body_id: int) -> torch.Tensor:
    mob_qpos_num = wp.to_torch(m.mob_dofnum)
    return mob_qpos_num[body_id]


def get_dof_num(m: Model, body_id: int) -> torch.Tensor:
    jnt_dof_num = wp.to_torch(m.mob_dofnum)
    return jnt_dof_num[body_id]


def muscle_metadata(m: Model) -> list[MuscleMetadata]:
    return m.muscle_data


def gravity(m: Model) -> float:
    return m.opt.gravity


def set_implicit_damping(m: Model, enabled: bool):
    m.opt.implicit_damping = enabled


def set_use_tiled_fn_path(m: Model, enabled: bool):
    m.opt.use_tiled_fn_path = enabled


def set_drag_enabled(m: Model, enabled: bool):
    m.opt.enable_drag = enabled


def set_activation_type(m: Model, activation_type: ActivationType):
    m.opt.activation_type = activation_type


def steps_attempted(d: Data) -> torch.Tensor:
    return wp.to_torch(d.steps_attempted)


def set_integrator_accuracy(m: Model, accuracy: float):
    m.opt.accuracy = accuracy


def set_integrator_use_inf_norm(m: Model, use_inf_norm: bool):
    m.opt.use_inf_norm = use_inf_norm


def is_adaptive(integrator_type: IntegratorType) -> bool:
    return integrator_type in [
        IntegratorType.EULER_ADAPTIVE,
        IntegratorType.RK_MERSON_ADAPTIVE,
    ]


# --- Data Fields ---
def set_reset(d: Data, reset_worlds: torch.Tensor):
    d_reset_torch = wp.to_torch(d.world_reset)
    d_reset_torch[:] = reset_worlds.ravel()


def time(d: Data) -> torch.tensor:
    return wp.to_torch(d.time)


def body_transforms(d: Data) -> torch.Tensor:
    return wp.to_torch(d.mob_X_GB)


def body_com_positions(d: Data) -> torch.Tensor:
    return wp.to_torch(d.body_COM_G)


def body_subtree_com_positions(d: Data) -> torch.Tensor:
    return wp.to_torch(d.subtree_com)


def body_velocities(d: Data) -> torch.Tensor:
    return wp.to_torch(d.body_V_GB)


def body_accelerations(d: Data) -> torch.Tensor:
    return wp.to_torch(d.body_A_GB)


def body_user_forces(d: Data) -> torch.Tensor:
    return wp.to_torch(d.body_F_applied)


def joint_positions(d: Data) -> torch.Tensor:
    return wp.to_torch(d.qpos)


def joint_velocities(d: Data) -> torch.Tensor:
    return wp.to_torch(d.qvel)


def joint_accelerations(d: Data) -> torch.Tensor:
    return wp.to_torch(d.qacc)


def body_force(d: Data) -> torch.Tensor:
    return wp.to_torch(d.body_F)


def body_force_gravity(d: Data) -> torch.Tensor:
    return wp.to_torch(d.body_F_gravity)


def body_force_contact(d: Data) -> torch.Tensor:
    return wp.to_torch(d.body_F_contact)


def body_force_muscle(d: Data) -> torch.Tensor:
    return wp.to_torch(d.body_F_muscle)


def body_force_drag(d: Data) -> torch.Tensor:
    return wp.to_torch(d.body_F_drag)


def ufrc_spring(d: Data) -> torch.Tensor:
    return wp.to_torch(d.ufrc_spring)


def ufrc_damper(d: Data) -> torch.Tensor:
    return wp.to_torch(d.ufrc_damper)


def ufrc_muscle(d: Data) -> torch.Tensor:
    return wp.to_torch(d.ufrc_muscle)


def ufrc_muscle_passive(d: Data) -> torch.Tensor:
    return wp.to_torch(d.ufrc_muscle_passive)


def ufrc_actuator(d: Data) -> torch.Tensor:
    return wp.to_torch(d.ufrc_actuator)


def ufrc_limit(d: Data) -> torch.Tensor:
    return wp.to_torch(d.ufrc_limit)


# -- Muscles ---
def muscle_activations(d: Data) -> torch.Tensor:
    return wp.to_torch(d.m_act)


def muscle_activations_dot(d: Data) -> torch.Tensor:
    return wp.to_torch(d.m_act_dot)


def muscle_excitations(d: Data) -> torch.Tensor:
    return wp.to_torch(d.m_excitations)


def muscle_actuations(d: Data) -> torch.Tensor:
    return wp.to_torch(d.muscle_actuation)


def muscle_path_lengths(d: Data) -> torch.Tensor:
    return wp.to_torch(d.muscle_length)


def muscle_path_velocities(d: Data) -> torch.Tensor:
    return wp.to_torch(d.muscle_velocity)


def muscle_fiber_lengths(d: Data) -> torch.Tensor:
    return wp.to_torch(d.muscle_norm_fiber_length)


def muscle_fiber_velocities(d: Data) -> torch.Tensor:
    return wp.to_torch(d.m_state_dot)


def muscle_powers(d: Data) -> torch.Tensor:
    return wp.to_torch(d.muscle_metabolic)


def muscle_moment_arms(d: Data) -> torch.Tensor:
    return wp.to_torch(d.muscle_moment_arm)


def muscle_metadata_np(m: Model) -> np.ndarray:
    return m.muscle_metadata.numpy()


def muscle_length_info_np(d: Data) -> np.ndarray:
    return d.muscle_length_info.numpy()


def muscle_velocity_info_np(d: Data) -> np.ndarray:
    return d.muscle_velocity_info.numpy()


def site_positions(d: Data) -> torch.Tensor:
    return wp.to_torch(d.site_pos_G)


def muscle_site_adr(m: Model) -> torch.Tensor:
    return wp.to_torch(m.muscle_pts_adr)


def muscle_site_num(m: Model) -> torch.Tensor:
    return wp.to_torch(m.muscle_pts_num)


# --- Actuators ---
def actuator_activations(d: Data) -> torch.Tensor:
    return wp.to_torch(d.a_act)


def actuator_activations_dot(d: Data) -> torch.Tensor:
    return wp.to_torch(d.a_act_dot)


def actuator_excitations(d: Data) -> torch.Tensor:
    return wp.to_torch(d.a_excitations)


def actuator_metadata_np(m: Model) -> np.ndarray:
    return m.actuator_metadata.numpy()


# --- Visuals ---
def get_visual_transforms(d: Data) -> torch.Tensor:
    return wp.to_torch(d.vis_X)


def get_beam_visual_positions(d: Data) -> torch.Tensor:
    return wp.to_torch(d.vis_beam_pos)


# --- Colliders ---
def get_collider_types(m: Model) -> torch.Tensor:
    return wp.to_torch(m.geom_type)


def get_collider_sizes(m: Model) -> torch.Tensor:
    return wp.to_torch(m.geom_size)


def collider_stiffness(m: Model) -> torch.Tensor:
    return wp.to_torch(m.geom_stiffness)


def collider_dissipation(m: Model) -> torch.Tensor:
    return wp.to_torch(m.geom_dissipation)


def collider_priority(m: Model) -> torch.Tensor:
    return wp.to_torch(m.geom_priority)


def collider_friction(m: Model) -> torch.Tensor:
    return wp.to_torch(m.geom_friction)


def collider_transition_velocity(m: Model) -> torch.Tensor:
    return wp.to_torch(m.geom_transition_velocity)


def get_collider_transforms(d: Data) -> torch.Tensor:
    return wp.to_torch(d.geom_X)


def collider_forces(d: Data) -> torch.Tensor:
    return wp.to_torch(d.geom_cforce)


def collider_self_forces(d: Data) -> torch.Tensor:
    return wp.to_torch(d.geom_self_cforce)


def grf(d: Data) -> torch.Tensor:
    return wp.to_torch(d.grf)


def joint_moments(d: Data) -> torch.Tensor:
    return wp.to_torch(d.joint_moments)
