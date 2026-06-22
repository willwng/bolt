import warp as wp

from . import smooth_frc
from . import smooth_mobilizer_post
from . import smooth_muscle_fn
from . import smooth_muscle_post
from . import smooth_muscle_pt
from .types import Data
from .types import Model
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@event_scope
def compute_muscle_moments(m: Model, d: Data):
    """
    Used for computing muscle moment arms (especially for point-based muscles). Applies muscle forces
    """
    # Point-path based muscles.
    body_F_tmp = wp.zeros((d.nworld, m.nbody), dtype=wp.spatial_vector)
    ufrc_tmp = wp.zeros((d.nworld, m.nv), dtype=float)
    qfrc_tmp = wp.zeros((d.nworld, m.nq), dtype=float)
    for muscle_id in m.muscle_pt_group_tuple:
        # Compute body F for a unit actuation,
        smooth_muscle_pt.apply_unit_force_one_muscle(m, d, body_F_out=body_F_tmp, muscle_id=muscle_id)
        # Then compute generalized force f_u = J^T F.
        smooth_mobilizer_post.multiply_by_jacobian_transpose(m, d, X_in=body_F_tmp, JtX_out=ufrc_tmp)
        # Map f_u (in u space) to q space (f_q = N^{-T} f_u), and then copy
        smooth_frc.ufrc_to_qfrc(m, d, ufrc_in=ufrc_tmp, qfrc_out=qfrc_tmp)
        # Store the result into muscle moment arm
        smooth_muscle_post.copy_ufrc_into_moment_arm(m, d, muscle_id=muscle_id, qfrc=qfrc_tmp)
        # Reset for next muscle
        body_F_tmp.zero_()
    return


@event_scope
def compute_net_joint_moments(m: Model, d: Data):
    """ Multiply acceleration by M to compute net joint moments """
    smooth_mobilizer_post.multiply_by_mass(m, d, d.qacc, d.joint_moments)


@event_scope
def compute_muscle_passive_forces(m: Model, d: Data):
    """ Compute the joint force applied just from the muscle passive forces """
    # Compute the joint moment from just the passive component of muscle force TODO: need to handle point-path muscles
    d.qfrc_muscle_passive.zero_()
    smooth_muscle_fn.apply_muscle_force_fn(m, d, passive_only=True)
    smooth_frc.qfrc_to_ufrc(m, d, d.qfrc_muscle_passive, d.ufrc_muscle_passive)
    return


@event_scope
def compute_muscle_force_breakdown(m: Model, d: Data):
    """ Compute the breakdown of the joint forces for each muscle """
    smooth_muscle_fn.apply_muscle_force_fn_breakdown(
        m, d, d.muscle_actuation_passive, d.qfrc_muscle_passive_breakdown
    )
    smooth_muscle_fn.apply_muscle_force_fn_breakdown(
        m, d, d.muscle_actuation_active, d.qfrc_muscle_active_breakdown
    )
