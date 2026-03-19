import warp as wp

from .types import Data
from .types import Model

wp.set_module_options({"enable_backward": False})


@wp.func
def step_function(
        x: float,
        start_time: float,
        end_time: float,
        start_value: float,
        end_value: float,
):
    """ A smooth step function that transitions from start_value to end_value between start_time and end_time. """
    if x <= start_time:
        return start_value
    elif x >= end_time:
        return end_value
    else:
        t = (x - start_time) / (end_time - start_time)
        smooth_t = t * t * (3.0 - 2.0 * t)
        return start_value + smooth_t * (end_value - start_value)


@wp.kernel
def _process_joint_limits(
        # Model in:
        lf_qpos_range: wp.array(dtype=wp.vec2),
        lf_dof_qadr: wp.array(dtype=int),
        lf_dof_adr: wp.array(dtype=int),
        lf_stiffness: wp.array(dtype=wp.vec2),
        lf_damping: wp.array(dtype=float),
        lf_transition: wp.array(dtype=float),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qpos_in: wp.array2d(dtype=float),
        qvel_in: wp.array2d(dtype=float),
        # Data out:
        qfrc_limit_out: wp.array2d(dtype=float),
):
    worldid, limitid = wp.tid()
    if integration_done_in[worldid]:
        return

    dof_range = lf_qpos_range[limitid]
    dof_qadr = lf_dof_qadr[limitid]
    dof_adr = lf_dof_adr[limitid]
    stiffness = lf_stiffness[limitid]
    damping = lf_damping[limitid]
    transition = lf_transition[limitid]

    lower_limit, upper_limit = dof_range[0], dof_range[1]
    lower_stiffness, upper_stiffness = stiffness[0], stiffness[1]

    q = qpos_in[worldid, dof_qadr]
    qdot = qvel_in[worldid, dof_adr]

    # [lower_limit, upper_limit] -> no force
    if q >= lower_limit and q <= upper_limit:
        return

    K_up = step_function(q, upper_limit, upper_limit + transition, 0.0, upper_stiffness)
    K_low = step_function(q, lower_limit - transition, lower_limit, lower_stiffness, 0.0)
    f_up = -K_up * (q - upper_limit)
    f_low = K_low * (lower_limit - q)
    f_damp = -damping * (K_up / upper_stiffness + K_low / lower_stiffness) * qdot
    force = f_up + f_low + f_damp

    wp.atomic_add(qfrc_limit_out, worldid, dof_adr, force)
    return


def coordinate_limit_force(m: Model, d: Data):
    wp.launch(
        _process_joint_limits,
        dim=(d.nworld, m.nlimitforce),
        inputs=[
            m.lf_qpos_range,
            m.lf_qpos_adr,
            m.lf_dof_adr,
            m.lf_stiffness,
            m.lf_damping,
            m.lf_transition,
            d.integration_done,
            d.qpos,
            d.qvel
        ],
        outputs=[
            d.qfrc_limit,
        ],
    )
    return
