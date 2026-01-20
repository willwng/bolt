import warp as wp

from .types import Data
from .types import Model

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _process_joint_limits(
        # Model in:
        limit_dof_range: wp.array(dtype=wp.vec2),
        limit_dof_adr: wp.array(dtype=int),
        limit_dof_qadr: wp.array(dtype=int),
        limit_dof_forces: wp.array(dtype=wp.vec2),
        # Data in:
        qpos_in: wp.array2d(dtype=float),
        qvel_in: wp.array2d(dtype=float),
        # Data out:
        qfrc_applied_out: wp.array2d(dtype=float),
        dof_limit_torque_out: wp.array2d(dtype=float),
):
    worldid, limitdofid = wp.tid()

    dof_range = limit_dof_range[limitdofid]
    dof_adr = limit_dof_adr[limitdofid]
    dof_qadr = limit_dof_qadr[limitdofid]
    limit_forces = limit_dof_forces[limitdofid]
    qpos = qpos_in[worldid, dof_qadr]
    qvel = qvel_in[worldid, dof_adr]

    # todo: don't hardcode these
    damping = 1.0
    stiffness = 500.0
    if qpos >= dof_range[0] and qpos <= dof_range[1]:
        return
    elif qpos > dof_range[1]:
        force = wp.min(-stiffness * (qpos - dof_range[1]) * (1.0 + damping * qvel), 0.0)
    else:
        force = wp.max(-stiffness * (qpos - dof_range[0]) * (1.0 - damping * qvel), 0.0)

    wp.atomic_add(qfrc_applied_out, worldid, dof_adr, force)
    wp.atomic_add(dof_limit_torque_out, worldid, limitdofid, force)


def apply_limit_forces(m: Model, d: Data):
    d.dof_lim_torque.zero_()
    wp.launch(
        _process_joint_limits,
        dim=(d.nworld, m.ndoflimit),
        inputs=[
            m.limit_dof_range,
            m.limit_dof_adr,
            m.limit_dof_qadr,
            m.limit_dof_forces,
            d.qpos,
            d.qvel
        ],
        outputs=[
            d.qfrc_applied,
            d.dof_lim_torque
        ],
    )
    return
