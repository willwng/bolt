import warp as wp

from . import math
from .types import Data
from .types import Model
from .types import ExponentialContact
from .consts import MSK_SIG_REAL
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _reset_exp_contact_state(
        # Model:
        exp_contact: wp.array(dtype=ExponentialContact),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        site_pos_G_in: wp.array2d(dtype=wp.vec3),
        # Data out:
        exp_contact_state_out: wp.array2d(dtype=wp.vec4)
):
    worldid, conid = wp.tid()
    if integration_done_in[worldid]:
        return

    contact = exp_contact[conid]
    siteid = contact.siteid
    X_GP = contact.contact_plane_transform

    # Reset anchor point
    p_G = site_pos_G_in[worldid, siteid]
    p_P = wp.transform_point(wp.transform_inverse(X_GP), p_G)
    p_P.z = 0.0  # Project onto contact plane
    p0 = p_P
    exp_contact_state_out[worldid, conid] = wp.vec4(1.0, p0.x, p0.y, p0.z)
    return


@wp.kernel
def _process_contacts_exp(
        # Model:
        exp_contact: wp.array(dtype=ExponentialContact),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        site_pos_G_in: wp.array2d(dtype=wp.vec3),
        site_vel_G_in: wp.array2d(dtype=wp.vec3),
        mob_X_GB_in: wp.array2d(dtype=wp.transform),
        exp_contact_state_in: wp.array2d(dtype=wp.vec4),
        # Data out:
        exp_contact_state_dot_out: wp.array2d(dtype=wp.vec4),
        body_F_contact_out: wp.array2d(dtype=wp.spatial_vector),
        grf_out: wp.array(dtype=wp.vec3)
):
    worldid, conid = wp.tid()
    if integration_done_in[worldid]:
        return

    # Retrieve parameters
    contact = exp_contact[conid]
    shape_params = contact.shape_parameters
    kv_norm = contact.normal_viscosity
    max_normal_force = contact.max_normal_force
    kp_fric = contact.friction_elasticity
    kv_fric = contact.friction_viscosity
    mus = contact.initial_mu_static
    muk = contact.initial_mu_kinetic
    siteid = contact.siteid
    bodyid = contact.bodyid
    station_B = contact.station_B

    state_in = exp_contact_state_in[worldid, conid]

    # Transform of contact plane
    X_GP = contact.contact_plane_transform
    R_GP = wp.transform_get_rotation(X_GP)

    # --- Realize position ---
    # Position of station in ground
    p_G = site_pos_G_in[worldid, siteid]
    # Transform into contact plane frame
    p_P = wp.transform_point(wp.transform_inverse(X_GP), p_G)
    # Resolve into normal (z) and tangential (xy) components
    pz = p_P.z
    pxy = wp.vec3(p_P.x, p_P.y, 0.0)

    # --- Realize velocity ---
    # Velocity of station in ground
    v_G = site_vel_G_in[worldid, siteid]
    # Transform into contact plane frame
    v_P = wp.quat_rotate(wp.quat_inverse(R_GP), v_G)
    # Resolve into normal (z) and tangential (xy) components
    vz = v_P.z
    vxy = wp.vec3(v_P.x, v_P.y, 0.0)

    # --- Calculate normal force ---
    d0, d1, d2 = shape_params[0], shape_params[1], shape_params[2]
    # Elastic part
    fz_elas = d1 * wp.exp(-d2 * (pz - d0))
    # Damping Part
    fz_damp = -kv_norm * vz * fz_elas
    # Total
    fz = fz_elas + fz_damp
    # Don't allow normal force to be negative or too large
    # Make sure that any change in fz is accompanied by an adjustment
    #  in fzElas and fzDamp so that 'fz = fzElas + fzDamp' remains true.
    if fz < 0.0:
        fz = 0.0
        fz_damp = -fz_elas
    if fz > max_normal_force:
        fz = max_normal_force
        fz_elas = fz - fz_damp

    # Get the Sliding state (K) and anchor point (p0) from state
    K = state_in[0]
    p0 = wp.vec3(state_in[1], state_in[2], state_in[3])

    # --- Calculate friction force ---
    p0_last = p0
    # Compute max friction force based on the instantaneous mu
    mu = mus - K * (mus - muk)
    fxy_limit = mu * fz
    # Friction limit is too small
    if fxy_limit < MSK_SIG_REAL:
        fric_mod1_P, fric_damp_mod1_P = wp.vec3(0.0), wp.vec3(0.0)
        fric_mod2_P, fric_damp_mod2_P = wp.vec3(0.0), wp.vec3(0.0)
        fric_P, fric_damp_P, fric_elas_P = wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0)
        p0 = pxy
    # Friction limit is large enough for meaningful calculations
    else:
        # Model 1: Pure damping (when sliding = 1.0)
        fxy_limit_sqr = fxy_limit * fxy_limit
        fric_damp_mod1_P, fric_damp_mod2_P = -kv_fric * vxy, -kv_fric * vxy
        if wp.length_sq(fric_damp_mod1_P) > fxy_limit_sqr:
            fric_damp_mod1_P = fxy_limit * wp.normalize(fric_damp_mod1_P)

        # Model 2: Damped Linear Spring
        fric_elas_mod2_P = -kp_fric * (pxy - p0)
        fric_mod2_P = fric_elas_mod2_P + fric_damp_mod2_P
        fxy_mod2_sqr = wp.length_sq(fric_mod2_P)
        if fxy_mod2_sqr > fxy_limit_sqr:
            scale = fxy_limit / wp.sqrt(fxy_mod2_sqr)
            fric_elas_mod2_P = fric_elas_mod2_P * scale
            fric_damp_mod2_P = fric_damp_mod2_P * scale
            fric_mod2_P = fric_elas_mod2_P + fric_damp_mod2_P

        # Blend model 1 and 2 according to K
        fric_elas_P = fric_elas_mod2_P * (1.0 - K)
        fric_damp_P = fric_damp_mod2_P + (fric_damp_mod1_P - fric_damp_mod2_P) * K
        fric_P = fric_elas_P + fric_damp_P

        # Ensure p0 is consistent with the elastic component
        p0 = pxy + fric_elas_P / kp_fric
        p0.z = 0.0

    # Store information needed for updating state
    if fxy_limit > MSK_SIG_REAL:
        p0_delta = p0 - p0_last
    else:
        p0_delta = wp.vec3(0.0)
    exp_contact_state_dot_out[worldid, conid] = wp.vec4(wp.length(p0_delta), p0.x, p0.y, p0.z)

    # --- Calculate Force ---
    f_P = fric_P
    f_P.z = fz
    f_G = wp.quat_rotate(R_GP, f_P)

    # Apply force to body
    X_GB = mob_X_GB_in[worldid, bodyid]
    wp.atomic_add(body_F_contact_out[worldid], bodyid,
                  math.apply_force_to_body_point(X_GB, station_B, f_G))

    # Update GRF
    wp.atomic_add(grf_out, worldid, f_G)


@event_scope
def reset_exp_contact_state(m: Model, d: Data):
    wp.launch(
        _reset_exp_contact_state,
        dim=(d.nworld, m.nexpcontact),
        inputs=[m.exp_contact, d.integration_done, d.site_pos_G],
        outputs=[d.exp_contact_state]
    )
    return


@event_scope
def contact_forces(m: Model, d: Data):
    wp.launch(
        _process_contacts_exp,
        dim=(d.nworld, m.nexpcontact),
        inputs=[
            m.exp_contact,
            d.integration_done, d.site_pos_G, d.site_vel_G, d.mob_X_GB, d.exp_contact_state
        ],
        outputs=[d.exp_contact_state_dot, d.body_F_contact, d.grf]
    )
    return
