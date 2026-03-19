import warp as wp

from .types import Data
from .types import Model

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _process_joint_limits(
        # Model in:
        stop_qpos_range: wp.array(dtype=wp.vec2),
        stop_dof_qadr: wp.array(dtype=int),
        stop_dof_adr: wp.array(dtype=int),
        stop_dof_stiffness_damping: wp.array(dtype=wp.vec2),
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

    dof_range = stop_qpos_range[limitid]
    dof_qadr = stop_dof_qadr[limitid]
    dof_adr = stop_dof_adr[limitid]
    sd = stop_dof_stiffness_damping[limitid]
    stiffness, damping = sd[0], sd[1]

    q = qpos_in[worldid, dof_qadr]
    qdot = qvel_in[worldid, dof_adr]

    if q >= dof_range[0] and q <= dof_range[1]:
        return
    elif q > dof_range[1]:
        force = wp.min(-stiffness * (q - dof_range[1]) * (1.0 + damping * qdot), 0.0)
    else:
        force = wp.max(-stiffness * (q - dof_range[0]) * (1.0 - damping * qdot), 0.0)

    # wp.printf("dof %d is %f, range is [%f, %f], force is %f\n", dof_qadr, qpos, dof_range[0], dof_range[1], force)
    wp.atomic_add(qfrc_limit_out, worldid, dof_adr, force)
    return


def linear_stop_force(m: Model, d: Data):
    wp.launch(
        _process_joint_limits,
        dim=(d.nworld, m.nlinearstop),
        inputs=[
            m.stop_qpos_range,
            m.stop_qpos_adr,
            m.stop_dof_adr,
            m.stop_dof_stiffness_damping,
            d.integration_done,
            d.qpos,
            d.qvel
        ],
        outputs=[
            d.qfrc_limit,
        ],
    )
    return
