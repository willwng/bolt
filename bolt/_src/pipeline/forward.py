import warp as wp

from ..contact import collision_driver
from ..contact import hunt_crossley
from ..contact import stateful_contacts
from ..dynamics import acceleration
from ..dynamics import actuators
from ..dynamics import forces
from ..dynamics import inertia
from ..dynamics import limit_force
from ..kinematics import custom_joints
from ..kinematics import position
from ..kinematics import reset_qpos
from ..kinematics import velocity
from ..muscle import activation
from ..muscle import contraction
from ..muscle import function_path
from ..muscle import point_path
from ..types import Data
from ..types import Model
from ..warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@event_scope
def reset_forces(m: Model, d: Data):
    forces.reset_forces(m, d)


@event_scope
def realize_position(m: Model, d: Data, run_reset: bool = False):
    """ Position-dependent computations. """
    # If we requested a reset, let's make sure we're within the joint limits
    if run_reset:
        reset_qpos.fix_qpos_limits(m, d)
        reset_qpos.fix_quaternions(m, d)

    # evaluate the functions for custom joints
    custom_joints.evaluate_cst_functions(m, d)
    custom_joints.prepare_cst_joint(m, d)

    # mobilizer transforms
    position.calc_mobilizer_X_FM(m, d)
    # body world positions
    position.calc_body_transforms(m, d)
    # across-joint velocity Jacobian
    position.joint_velocity_jacobian(m, d)
    # COM and inertia matrices
    position.joint_independent_kinematics(m, d)
    # world positions for colliders, visuals, sites
    position.attachment_kinematics(m, d)
    # COM of body trees
    position.com_pos(m, d)
    # detect contacts
    collision_driver.collision(m, d)

    # finally, reset any contact state if requested
    if run_reset:
        stateful_contacts.reset_contact_state(m, d)
    return


@event_scope
def realize_velocity(m: Model, d: Data):
    """Velocity-dependent computations."""
    # compute qdot = N(q) * u
    velocity.calc_q_dot(m, d)
    # body velocities
    velocity.compute_body_velocities(m, d)
    # derivative of the velocity Jacobian
    velocity.joint_velocity_jacobian_dot(m, d)
    # body accelerations from parent to child body
    velocity.compute_parent_to_child_accelerations(m, d)
    # gyroscopic moment, coriolis acceleration
    velocity.joint_independent_kinematics_vel(m, d)
    # world velocities for sites
    velocity.attachment_kinematics_vel(m, d)
    return


@event_scope
def realize_articulated_body_inertia(m: Model, d: Data):
    inertia.initialize_articulated_body_inertia(m, d)
    inertia.accumulate_articulated_body_inertia(m, d)
    inertia.articulated_body_velocity(m, d)
    return


@event_scope
def compute_accelerations(m: Model, d: Data):
    acceleration.calc_udot(m, d)
    return


@event_scope
def realize_muscles(m: Model, d: Data, run_reset: bool):
    """ Muscle-related computations. """
    # Compute muscle path
    function_path.muscle_fn_path(m, d)
    point_path.muscle_point_path(m, d)

    # Activation dynamics, starting from a valid activation if reset
    if run_reset:
        activation.clamp_activation_on_reset(m, d)
    activation.activation_dynamics(m, d)

    # Contraction dynamics, equilibrate if reset
    if run_reset:
        contraction.equilibrate(m, d)
    contraction.contraction_dynamics(m, d)
    return


@event_scope
def realize_actuators(m: Model, d: Data, run_reset: bool):
    """ Actuator-related computations. """
    if run_reset:
        actuators.reset_to_default_activation(m, d)

    actuators.activation_dynamics(m, d)
    return


@event_scope
def realize_forces(m: Model, d: Data):
    if m.opt.explicit_gravity:
        forces.apply_gravity(m, d)
    # spring, damping
    forces.spring(m, d)
    forces.damping(m, d)
    # contacts
    hunt_crossley.contact_forces_hc(m, d)
    stateful_contacts.contact_forces(m, d)
    # limits
    limit_force.coordinate_limit_force(m, d)
    limit_force.swing_twist_limit_force(m, d)
    # muscles
    function_path.apply_muscle_force_fn(m, d)
    point_path.apply_muscle_force_pt(m, d)
    # actuators
    actuators.actuator_force(m, d)

    # convert any qfrc into ufrc
    forces.qfrc_to_ufrc(m, d, d.qfrc_muscle, d.ufrc_muscle)

    # accumulate
    forces.accumulate_forces(m, d)
    return


@event_scope
def fwd(m: Model, d: Data):
    """ Forward dynamics. No integration """
    reset_forces(m, d)
    realize_position(m, d, run_reset=False)
    realize_velocity(m, d)
    realize_articulated_body_inertia(m, d)
    realize_muscles(m, d, run_reset=False)
    realize_actuators(m, d, run_reset=False)
    realize_forces(m, d)
    compute_accelerations(m, d)
    return


@event_scope
def reset(m: Model, d: Data):
    """
    Called after resetting any of the worlds.
    Enforces joint limits, equilibrates muscles, and computes forward dynamics
    """
    reset_forces(m, d)
    realize_position(m, d, run_reset=True)
    realize_velocity(m, d)
    realize_articulated_body_inertia(m, d)
    realize_muscles(m, d, run_reset=True)
    realize_actuators(m, d, run_reset=True)
    realize_forces(m, d)
    compute_accelerations(m, d)
    d.world_reset.zero_()
