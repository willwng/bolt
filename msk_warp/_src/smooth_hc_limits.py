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
        limit_dof_shapes: wp.array(dtype=wp.vec2),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        qpos_in: wp.array2d(dtype=float),
        qvel_in: wp.array2d(dtype=float),
        # Data out:
        qfrc_limit_out: wp.array2d(dtype=float),
):
    worldid, limitdofid = wp.tid()
    if integration_done_in[worldid]:
        return

    dof_range = limit_dof_range[limitdofid]
    dof_adr = limit_dof_adr[limitdofid]
    dof_qadr = limit_dof_qadr[limitdofid]
    limit_forces = limit_dof_forces[limitdofid]
    shape_factors = limit_dof_shapes[limitdofid]
    qpos = qpos_in[worldid, dof_qadr]
    qvel = qvel_in[worldid, dof_adr]

    if qpos >= dof_range[0] and qpos <= dof_range[1]:
        return
    elif qpos > dof_range[1]:
        force = wp.min(-limit_forces[1] * (qpos - dof_range[1]) * (1.0 + shape_factors[1] * qvel), 0.0)
    else:
        force = wp.max(-limit_forces[0] * (qpos - dof_range[0]) * (1.0 - shape_factors[0] * qvel), 0.0)

    # wp.printf("dof %d is %f, range is [%f, %f], force is %f\n", dof_qadr, qpos, dof_range[0], dof_range[1], force)

    wp.atomic_add(qfrc_limit_out, worldid, dof_adr, force)


def limit_forces(m: Model, d: Data):
    wp.launch(
        _process_joint_limits,
        dim=(d.nworld, m.ndoflimit),
        inputs=[
            m.limit_dof_range,
            m.limit_dof_adr,
            m.limit_dof_qadr,
            m.limit_dof_forces,
            m.limit_dof_shapes,
            d.integration_done,
            d.qpos,
            d.qvel
        ],
        outputs=[
            d.qfrc_limit,
        ],
    )
    return
