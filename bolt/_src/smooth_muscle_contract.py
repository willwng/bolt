import warp as wp

from . import pennation
from . import force_curves
from .consts import M_MAX_NORM_TENDON_FORCE
from .consts import M_MIN_NORM_TENDON_FORCE
from .consts import BOLT_SIG_REAL
from .types import Data
from .types import FiberVelocityInfo
from .types import Model
from .types import MuscleDynamicsInfo
from .types import MuscleLengthInfo
from .types import MuscleMetadata
from .types import ResidualResult
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.func
def _calc_eq_residual(
        norm_tendon_force: float,
        path_length: float,
        path_velocity: float,
        activation: float,
        mm: MuscleMetadata,
        contraction_type: int,
) -> ResidualResult:
    # Retrieve tendon length from force
    norm_tendon_length = force_curves.calc_tendon_force_inverse(norm_tendon_force, contraction_type)
    tendon_length = norm_tendon_length * mm.tendon_slack_length
    # From path and tendon length compute fiber length
    fiber_width = pennation.get_fiber_width(mm.optimal_fiber_length, mm.optimal_pennation_angle)
    fiber_length_along_tendon = path_length - tendon_length
    fiber_length = wp.sqrt(fiber_length_along_tendon ** 2.0 + fiber_width ** 2.0)
    norm_fiber_length = fiber_length / mm.optimal_fiber_length
    # Pennation angle
    cos_pennation_angle = fiber_length_along_tendon / fiber_length
    sin_pennation_angle = fiber_width / fiber_length
    pennation_angle = wp.asin(sin_pennation_angle)
    if pennation_angle > wp.acos(0.1):
        pennation_angle = wp.acos(0.1)
        cos_pennation_angle = wp.cos(pennation_angle)
    # Tendon velocity
    norm_tendon_velocity = force_curves.calc_tendon_force_inverse_derivative(norm_tendon_length, contraction_type)
    tendon_velocity = mm.tendon_slack_length * norm_tendon_velocity
    # Fiber velocity
    fiber_velocity_along_tendon = path_velocity - tendon_velocity
    fiber_velocity = fiber_velocity_along_tendon * cos_pennation_angle
    norm_fiber_velocity = (fiber_velocity / pennation.get_max_contraction_velocity_in_meters_per_second(
        mm.v_max, mm.optimal_fiber_length))
    # Residual
    active_fiber_force = force_curves.calc_active_fiber_force(
        max_isometric_force=mm.max_isometric_force,
        activation=activation,
        norm_fiber_length=norm_fiber_length,
        norm_fiber_velocity=norm_fiber_velocity,
        active_force_width_scale=mm.active_force_width_scale,
        contraction_type=contraction_type,
    )
    passive_fiber_force = force_curves.calc_passive_fiber_force(
        norm_fiber_length=norm_fiber_length,
        norm_fiber_velocity=norm_fiber_velocity,
        mm=mm,
        contraction_type=contraction_type
    )

    fiber_force = (active_fiber_force + passive_fiber_force)
    fiber_force_along_tendon = fiber_force * cos_pennation_angle
    residual = (norm_tendon_force - fiber_force_along_tendon / mm.max_isometric_force)

    return ResidualResult(
        norm_tendon_force=norm_tendon_force,
        residual=residual,
        pennation_angle=pennation_angle,
        fiber_length=fiber_length,
        norm_fiber_length=norm_fiber_length,
        tendon_length=tendon_length,
        norm_tendon_length=norm_tendon_length,
        norm_tendon_velocity=norm_tendon_velocity,
        active_fiber_force=active_fiber_force,
        fiber_velocity=fiber_velocity,
        fiber_force_along_tendon=fiber_force_along_tendon
    )


@wp.kernel
def _equilibrate(
        # Model:
        muscle_metadata: wp.array(dtype=MuscleMetadata),
        # Data in:
        world_reset_in: wp.array(dtype=bool),
        muscle_length_in: wp.array2d(dtype=float),
        muscle_velocity_in: wp.array2d(dtype=float),
        act_in: wp.array2d(dtype=float),
        # In:
        contraction_type: int,
        # Data out:
        mstate_out: wp.array2d(dtype=float)
):
    worldid, muscle_id = wp.tid()

    # Only equilibrate if the world was reset
    if not world_reset_in[worldid]:
        return

    metadata = muscle_metadata[muscle_id]

    # If ignoring tendon compliance, ignore the state variable
    if metadata.ignore_tendon_compliance:
        mstate_out[worldid, muscle_id] = 0.0
        return

    # Bisection to solve for equilibrium
    lower = M_MIN_NORM_TENDON_FORCE
    upper = M_MAX_NORM_TENDON_FORCE
    mid = 0.5 * (lower + upper)
    tol = 1e-8
    max_iters = 30

    path_length = muscle_length_in[worldid, muscle_id]
    path_velocity = muscle_velocity_in[worldid, muscle_id]
    activation = act_in[worldid, muscle_id]

    res_lower = _calc_eq_residual(lower, path_length, path_velocity, activation, metadata, contraction_type)
    res_upper = _calc_eq_residual(upper, path_length, path_velocity, activation, metadata, contraction_type)
    res_mid = _calc_eq_residual(mid, path_length, path_velocity, activation, metadata, contraction_type)
    res_best = res_lower if wp.abs(res_lower.residual) < wp.abs(res_upper.residual) else res_upper

    for i in range(max_iters):
        # Converted or interval is sufficiently small
        if wp.abs(res_best.residual) < tol or 0.5 * (upper - lower) < tol:
            break
        # Update bounds
        if res_lower.residual * res_mid.residual > 0.0:
            lower = mid
            res_lower = res_mid
        else:
            upper = mid
        # New midpoint
        mid = 0.5 * (lower + upper)
        res_mid = _calc_eq_residual(mid, path_length, path_velocity, activation, metadata, contraction_type)
        # Update best
        if abs(res_mid.residual) < abs(res_best.residual):
            res_best = res_mid

    # Set state
    fiber_length = res_best.fiber_length
    norm_fiber_length = fiber_length / metadata.optimal_fiber_length

    mstate_out[worldid, muscle_id] = pennation.clamp_norm_fiber_length(
        norm_fiber_length, metadata.min_norm_fiber_length, metadata.max_norm_fiber_length)
    return


@wp.kernel
def _contraction_dynamics_fused_kernel(
        # Model:
        muscle_metadata: wp.array(dtype=MuscleMetadata),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        act_in: wp.array2d(dtype=float),
        mstate_in: wp.array2d(dtype=float),
        muscle_length_in: wp.array2d(dtype=float),
        muscle_velocity_in: wp.array2d(dtype=float),
        # In:
        rng_state: wp.array(dtype=wp.uint32),
        contraction_type: int,
        # Data out:
        muscle_length_info_out: wp.array2d(dtype=MuscleLengthInfo),
        muscle_velocity_info_out: wp.array2d(dtype=FiberVelocityInfo),
        muscle_dynamics_info_out: wp.array2d(dtype=MuscleDynamicsInfo),
):
    worldid, muscle_id = wp.tid()
    if integration_done_in[worldid]:
        return

    mm = muscle_metadata[muscle_id]
    path_length = muscle_length_in[worldid, muscle_id]
    path_velocity = muscle_velocity_in[worldid, muscle_id]
    activation = act_in[worldid, muscle_id]

    ### --- LENGTH INFO ---
    if mm.ignore_tendon_compliance:  # rigid tendon
        fiber_length = pennation.calc_fiber_length(
            muscle_length=path_length,
            optimal_fiber_length=mm.optimal_fiber_length,
            optimal_pennation_angle=mm.optimal_pennation_angle,
            tendon_slack_length=mm.tendon_slack_length,
            minimum_fiber_length=mm.min_norm_fiber_length * mm.optimal_fiber_length
        )
        fiber_length = pennation.clamp_fiber_length(
            fiber_length=fiber_length,
            optimal_fiber_length=mm.optimal_fiber_length,
            min_norm_fiber_length=mm.min_norm_fiber_length,
            max_norm_fiber_length=mm.max_norm_fiber_length
        )

        norm_fiber_length = fiber_length / mm.optimal_fiber_length

    else:  # elastic tendon, fiber length is determined by state variable, already is clamped
        norm_fiber_length = mstate_in[worldid, muscle_id]
        fiber_length = norm_fiber_length * mm.optimal_fiber_length

    min_norm_fiber_length = mm.min_norm_fiber_length
    # Pennation angle
    pennation_angle = pennation.calc_pennation_angle(
        optimal_pennation_angle=mm.optimal_pennation_angle,
        optimal_fiber_length=mm.optimal_fiber_length,
        norm_fiber_length=norm_fiber_length,
        min_norm_fiber_length=min_norm_fiber_length,
    )
    cos_pennation_angle = wp.cos(pennation_angle)
    sin_pennation_angle = wp.sin(pennation_angle)
    fiber_length_along_tendon = fiber_length * cos_pennation_angle
    # Tendon
    tendon_length = path_length - fiber_length_along_tendon
    norm_tendon_length = tendon_length / mm.tendon_slack_length
    tendon_strain = norm_tendon_length - 1.0
    # Force multipliers
    fiber_passive_force_length_multiplier = force_curves.calc_passive_fiber_force_length(
        norm_fiber_length=norm_fiber_length,
        mm=mm,
        contraction_type=contraction_type
    )
    fiber_active_force_length_multiplier = force_curves.calc_active_fiber_force_length(
        norm_fiber_length=norm_fiber_length,
        active_force_width_scale=mm.active_force_width_scale,
        contraction_type=contraction_type
    )
    tendon_force_multiplier = force_curves.calc_tendon_force_length(
        norm_tendon_length=norm_tendon_length,
        contraction_type=contraction_type
    )

    ### --- VELOCITY INFO ---
    v_max_in_ms = pennation.get_max_contraction_velocity_in_meters_per_second(
        v_max=mm.v_max,
        optimal_fiber_length=mm.optimal_fiber_length
    )
    if mm.ignore_tendon_compliance:  # Rigid tendon
        if tendon_length < mm.tendon_slack_length - BOLT_SIG_REAL:
            # Tendon is buckling, fiber velocity is zero
            norm_fiber_velocity = 0.0
            fiber_force_velocity_multiplier = 1.0
        else:
            dlce = pennation.calc_fiber_velocity(
                cos_pennation_angle=cos_pennation_angle,
                muscle_velocity=path_velocity,
                tendon_velocity=0.0
            )
            norm_fiber_velocity = dlce / v_max_in_ms
            fiber_force_velocity_multiplier, _ = force_curves.calc_active_fiber_force_velocity(
                norm_fiber_velocity=norm_fiber_velocity,
                contraction_type=contraction_type
            )
    elif mm.fiber_damping > 0.0:  # Elastic tendon with damping
        dlceN_dt, fv = force_curves.calc_damped_norm_fiber_velocity(
            f_iso=mm.max_isometric_force,
            a=activation,
            fal=fiber_active_force_length_multiplier,
            fpe=fiber_passive_force_length_multiplier,
            fse=tendon_force_multiplier,
            beta=mm.fiber_damping,
            cos_phi=cos_pennation_angle,
            contraction_type=contraction_type,
            state=rng_state
        )
        norm_fiber_velocity = dlceN_dt
        fiber_force_velocity_multiplier = fv
    else:  # Elastic tendon without damping
        fv = force_curves.calc_undamped_fiber_force_velocity_multiplier(
            a=activation,
            fal=fiber_active_force_length_multiplier,
            fp=fiber_passive_force_length_multiplier,
            fse=tendon_force_multiplier,
            cos_phi=cos_pennation_angle
        )
        norm_fiber_velocity = force_curves.calc_active_fiber_force_velocity_inverse(
            force_velocity_mult=fv,
            contraction_type=contraction_type
        )
        fiber_force_velocity_multiplier = fv

    fiber_velocity = norm_fiber_velocity * v_max_in_ms
    pennation_angular_velocity = pennation.calc_pennation_angular_velocity(
        optimal_pennation_angle=mm.optimal_pennation_angle,
        fiber_length=fiber_length,
        fiber_velocity=fiber_velocity,
        tan_pennation_angle=wp.tan(pennation_angle)
    )
    fiber_velocity_along_tendon = pennation.calc_fiber_velocity_along_tendon(
        fiber_length=fiber_length,
        fiber_velocity=fiber_velocity,
        sin_pennation_angle=sin_pennation_angle,
        cos_pennation_angle=cos_pennation_angle,
        pennation_angular_velocity=pennation_angular_velocity
    )
    tendon_velocity = pennation.calc_tendon_velocity(
        cos_pennation_angle=cos_pennation_angle,
        sin_pennation_angle=sin_pennation_angle,
        pennation_angular_velocity=pennation_angular_velocity,
        fiber_length=fiber_length,
        fiber_velocity=fiber_velocity,
        muscle_velocity=path_velocity
    )
    norm_tendon_velocity = tendon_velocity / mm.tendon_slack_length

    # Check to see whether the fiber length was clamped
    min_norm_fiber_length = mm.min_norm_fiber_length
    fiber_state_clamped = pennation.is_fiber_state_clamped(
        norm_fiber_length=norm_fiber_length,
        norm_fiber_velocity=norm_fiber_velocity,
        min_norm_fiber_length=min_norm_fiber_length
    )
    if fiber_state_clamped:
        norm_fiber_velocity = 0.0
        fiber_velocity = 0.0
        fiber_velocity_along_tendon = 0.0
        pennation_angular_velocity = 0.0
        tendon_velocity = path_velocity
        norm_tendon_velocity = tendon_velocity / mm.tendon_slack_length
        fiber_force_velocity_multiplier = 1.0  # consistent w fiber vel 0

    ### --- DYNAMICS INFO ---
    fm, aFm, p1Fm, p2Fm, pFm, fmAT = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    if not fiber_state_clamped:
        aFm = (mm.max_isometric_force * activation * fiber_active_force_length_multiplier *
               fiber_force_velocity_multiplier)
        p1Fm = (mm.max_isometric_force * fiber_passive_force_length_multiplier)
        p2Fm = (mm.max_isometric_force * mm.fiber_damping * norm_fiber_velocity)
        pFm = p1Fm + p2Fm
        # Total fiber force
        fm = aFm + pFm
        #  Every configuration except the rigid tendon chooses a fiber velocity that ensures that the fiber does not
        #  generate a compressive force. Here, we must enforce that the fiber generates only tensile forces by
        #  saturating the damping force generated by the parallel element
        if mm.ignore_tendon_compliance and fm < 0.0:
            fm = 0.0
            p2Fm = -aFm - p1Fm
            pFm = p1Fm + p2Fm
        fmAT = fm * cos_pennation_angle

    fse = fmAT / mm.max_isometric_force if mm.ignore_tendon_compliance else tendon_force_multiplier

    # Final write
    mli = muscle_length_info_out[worldid]
    fvi = muscle_velocity_info_out[worldid]
    mdi = muscle_dynamics_info_out[worldid]

    mli[muscle_id].fiber_length = fiber_length
    mli[muscle_id].pennation_angle = pennation_angle
    mli[muscle_id].cos_pennation_angle = cos_pennation_angle
    mli[muscle_id].sin_pennation_angle = sin_pennation_angle
    mli[muscle_id].norm_fiber_length = norm_fiber_length
    mli[muscle_id].fiber_length_along_tendon = fiber_length_along_tendon
    mli[muscle_id].tendon_length = tendon_length
    mli[muscle_id].norm_tendon_length = norm_tendon_length
    mli[muscle_id].tendon_strain = tendon_strain
    mli[muscle_id].fiber_passive_force_length_multiplier = fiber_passive_force_length_multiplier
    mli[muscle_id].fiber_active_force_length_multiplier = fiber_active_force_length_multiplier
    mli[muscle_id].tendon_force_multiplier = tendon_force_multiplier

    fvi[muscle_id].fiber_velocity = fiber_velocity
    fvi[muscle_id].fiber_velocity_along_tendon = fiber_velocity_along_tendon
    fvi[muscle_id].norm_fiber_velocity = norm_fiber_velocity
    fvi[muscle_id].pennation_angular_velocity = pennation_angular_velocity
    fvi[muscle_id].tendon_velocity = tendon_velocity
    fvi[muscle_id].norm_tendon_velocity = norm_tendon_velocity
    fvi[muscle_id].fiber_force_velocity_multiplier = fiber_force_velocity_multiplier
    fvi[muscle_id].fiber_damping_force_multiplier = mm.fiber_damping * norm_fiber_velocity

    mdi[muscle_id].fiber_force = fm
    mdi[muscle_id].fiber_force_along_tendon = fmAT
    mdi[muscle_id].norm_fiber_force = fm / mm.max_isometric_force
    mdi[muscle_id].active_fiber_force = aFm
    mdi[muscle_id].passive_fiber_force = pFm
    mdi[muscle_id].tendon_force = fse * mm.max_isometric_force
    mdi[muscle_id].norm_tendon_force = fse
    return


@wp.kernel
def _set_state(
        # Model:
        muscle_metadata: wp.array(dtype=MuscleMetadata),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        muscle_length_info_in: wp.array2d(dtype=MuscleLengthInfo),
        muscle_velocity_info_in: wp.array2d(dtype=FiberVelocityInfo),
        muscle_dynamics_info_in: wp.array2d(dtype=MuscleDynamicsInfo),
        # Data out:
        mstate_dot_out: wp.array2d(dtype=float),
        muscle_actuation_out: wp.array2d(dtype=float),
        muscle_passive_length_multiplier_out: wp.array2d(dtype=float),
        muscle_active_length_multiplier_out: wp.array2d(dtype=float),
        muscle_active_velocity_multiplier_out: wp.array2d(dtype=float),
        muscle_actuation_active_out: wp.array2d(dtype=float),
        muscle_actuation_passive_out: wp.array2d(dtype=float),
        muscle_norm_fiber_length_out: wp.array2d(dtype=float),
):
    worldid, muscle_id = wp.tid()
    if integration_done_in[worldid]:
        return
    mm = muscle_metadata[muscle_id]
    mli = muscle_length_info_in[worldid, muscle_id]
    fvi = muscle_velocity_info_in[worldid, muscle_id]
    mdi = muscle_dynamics_info_in[worldid, muscle_id]

    # Actuation
    muscle_actuation_out[worldid, muscle_id] = mdi.tendon_force

    # State derivative
    if mm.ignore_tendon_compliance:
        mstate_dot_out[worldid, muscle_id] = 0.0
    else:
        mstate_dot_out[worldid, muscle_id] = fvi.fiber_velocity / mm.optimal_fiber_length

    # Fiber length (for output/observation purposes, not used for dynamics)
    muscle_norm_fiber_length_out[worldid, muscle_id] = mli.norm_fiber_length

    # Remaining analytics/observations
    muscle_passive_length_multiplier_out[worldid, muscle_id] = mli.fiber_passive_force_length_multiplier
    muscle_active_length_multiplier_out[worldid, muscle_id] = mli.fiber_active_force_length_multiplier
    muscle_active_velocity_multiplier_out[worldid, muscle_id] = fvi.fiber_force_velocity_multiplier
    # muscle_actuation_passive_out[worldid, muscle_id] = mdi.passive_fiber_force
    muscle_actuation_passive_out[
        worldid, muscle_id] = mm.max_isometric_force * mli.fiber_passive_force_length_multiplier
    muscle_actuation_active_out[
        worldid, muscle_id] = mdi.active_fiber_force
    return


@event_scope
def contraction_dynamics_fused(m: Model, d: Data):
    wp.launch(
        _contraction_dynamics_fused_kernel,
        dim=(d.nworld, m.nmuscle),
        inputs=[
            m.muscle_metadata,
            d.integration_done, d.m_act, d.m_state, d.muscle_length, d.muscle_velocity,
            d.rng_state, m.opt.contraction_type
        ],
        outputs=[d.muscle_length_info, d.muscle_velocity_info, d.muscle_dynamics_info],
    )


@event_scope
def equilibrate(m: Model, d: Data):
    """ Equilibrate muscles """
    if not m.nmuscle:
        return

    # Equilibrate (bisection)
    wp.launch(
        _equilibrate,
        dim=(d.nworld, m.nmuscle),
        inputs=[
            m.muscle_metadata,
            d.world_reset, d.muscle_length, d.muscle_velocity, d.m_act,
            m.opt.contraction_type
        ],
        outputs=[d.m_state],
    )


@event_scope
def contraction_dynamics(m: Model, d: Data):
    """ Muscle dynamics """
    if not m.nmuscle:
        return

    contraction_dynamics_fused(m, d)

    # Set actuation and muscle state derivatives
    wp.launch(
        _set_state,
        dim=(d.nworld, m.nmuscle),
        inputs=[
            m.muscle_metadata,
            d.integration_done, d.muscle_length_info, d.muscle_velocity_info, d.muscle_dynamics_info
        ],
        outputs=[
            d.m_state_dot, d.muscle_actuation,
            d.muscle_passive_length_multiplier,
            d.muscle_active_length_multiplier, d.muscle_active_velocity_multiplier,
            d.muscle_actuation_active, d.muscle_actuation_passive,
            d.muscle_norm_fiber_length
        ],
    )
