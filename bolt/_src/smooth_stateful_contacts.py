import warp as wp

from . import math
from .types import Data
from .types import Model
from .types import StatefulContact
from .consts import BOLT_SIG_REAL
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _reset_stl_contact_state(
        # Model:
        stl_contact: wp.array(dtype=StatefulContact),
        # Data in:
        world_reset_in: wp.array(dtype=bool),
        integration_done_in: wp.array(dtype=bool),
        site_pos_G_in: wp.array2d(dtype=wp.vec3),
        # Data out:
        stl_contact_state_out: wp.array2d(dtype=wp.vec3)
):
    worldid, conid = wp.tid()
    if integration_done_in[worldid]:
        return

    if world_reset_in[worldid]:
        contact = stl_contact[conid]
        siteid = contact.siteid
        X_GP = contact.contact_plane_transform

        # Position of station in ground
        p_G = site_pos_G_in[worldid, siteid]
        # Transform into contact plane frame
        p_P = wp.transform_point(wp.transform_inverse(X_GP), p_G)
        stl_contact_state_out[worldid, conid] = wp.vec3(1.0, p_P.x, p_P.y)
    return


@wp.func
def compute_normal_force(
        pz: float,
        vz: float,
        shape_params: wp.vec3,
        kv_norm: float,
        max_normal_force: float,
        use_exp_force: bool,
) -> float:
    if use_exp_force:
        d0, d1, d2 = shape_params[0], shape_params[1], shape_params[2]
        fz_elas = d1 * wp.exp(-d2 * (pz - d0))
        fz_damp = -kv_norm * vz * fz_elas
        fz = wp.clamp(fz_elas + fz_damp, 0.0, max_normal_force)
    else:
        # Fixme: this is a strange conversion, consider making a separate obj
        k, c = 0.5 * (max_normal_force ** (2.0 / 3.0)), kv_norm
        radius = shape_params[0]

        cf, bd, bv = 1e-5, 300.0, 50.0  # hard-coded constants for smoothing
        indentation, v_n = -pz, -vz
        fh_pos = (4.0 / 3.0) * k * wp.sqrt(radius * k) * wp.pow(
            wp.sqrt(indentation * indentation + cf), 3. / 2.)
        fh_smooth = fh_pos * (1.0 / 2.0 + (1.0 / 2.0) * wp.tanh(bd * indentation))
        bv = 50.0
        fhc_pos = fh_smooth * (1.0 + (3.0 / 2.0) * c * v_n)
        fhc_smooth = fhc_pos * (
                1.0 / 2.0 + (1.0 / 2.0) * wp.tanh(bv * (v_n + (2.0 / (3.0 * c)))))
        fz = fhc_smooth
    return fz


@wp.func
def compute_friction(
        p0: wp.vec2,
        pxy: wp.vec2,
        vxy: wp.vec2,
        fxy_limit: float,
        K: float,
        kp_fric: float,
        kv_fric: float,
) -> tuple[wp.vec2, wp.vec2]:
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

    # Blend model 1 and 2 according to K
    fric_elas_P = fric_elas_mod2_P * (1.0 - K)
    fric_damp_P = fric_damp_mod2_P + (fric_damp_mod1_P - fric_damp_mod2_P) * K
    fric_P = fric_elas_P + fric_damp_P

    # Ensure p0 is consistent with the elastic component
    p0 = pxy + fric_elas_P / kp_fric
    return fric_P, p0


@wp.kernel
def _process_contacts(
        # Model:
        stl_contact: wp.array(dtype=StatefulContact),
        # Data in:
        integration_done_in: wp.array(dtype=bool),
        site_pos_G_in: wp.array2d(dtype=wp.vec3),
        site_vel_G_in: wp.array2d(dtype=wp.vec3),
        mob_X_GB_in: wp.array2d(dtype=wp.transform),
        stl_contact_state_in: wp.array2d(dtype=wp.vec3),
        # Data out:
        stl_contact_state_dot_out: wp.array2d(dtype=wp.vec3),
        body_F_contact_out: wp.array2d(dtype=wp.spatial_vector),
        grf_out: wp.array(dtype=wp.vec3)
):
    worldid, conid = wp.tid()
    if integration_done_in[worldid]:
        return

    # Retrieve parameters
    contact = stl_contact[conid]
    shape_params = contact.shape_parameters
    kv_norm = contact.normal_viscosity
    max_normal_force = contact.max_normal_force
    kp_fric = contact.friction_elasticity
    kv_fric = contact.friction_viscosity
    mus = contact.initial_mu_static
    muk = contact.initial_mu_kinetic
    margin = contact.margin
    siteid = contact.siteid
    bodyid = contact.bodyid
    station_B = contact.station_B
    use_exp_force = contact.use_exp_force

    state_in = stl_contact_state_in[worldid, conid]

    # Transform of contact plane
    X_GP = contact.contact_plane_transform
    R_GP = wp.transform_get_rotation(X_GP)

    # --- Realize position ---
    # Position of station in ground
    p_G = site_pos_G_in[worldid, siteid]
    # Transform into contact plane frame
    p_P = wp.transform_point(wp.transform_inverse(X_GP), p_G)
    # Resolve into normal (z) and tangential (xy) components
    pz = p_P.z - margin
    pxy = wp.vec2(p_P.x, p_P.y)

    # --- Realize velocity ---
    # Velocity of station in ground
    v_G = site_vel_G_in[worldid, siteid]
    # Transform into contact plane frame
    v_P = wp.quat_rotate(wp.quat_inverse(R_GP), v_G)
    # Resolve into normal (z) and tangential (xy) components
    vz = v_P.z
    vxy = wp.vec2(v_P.x, v_P.y)

    # --- Calculate normal force ---
    fz = compute_normal_force(
        pz=pz, vz=vz, shape_params=shape_params,
        kv_norm=kv_norm, max_normal_force=max_normal_force,
        use_exp_force=use_exp_force,
    )

    # Get the Sliding state (K) and anchor point (p0) from state
    K = state_in[0]
    p0 = wp.vec2(state_in[1], state_in[2])

    # --- Calculate friction force ---
    p0_last = p0
    # Compute max friction force based on the instantaneous mu
    mu = mus - K * (mus - muk)
    fxy_limit = mu * fz

    # Store information needed for updating state
    if fxy_limit < BOLT_SIG_REAL:
        fric_P = wp.vec2(0.0)
        p0 = pxy
    else:
        # Friction force, new anchor point
        fric_P, p0 = compute_friction(
            p0=p0, pxy=pxy, vxy=vxy,
            fxy_limit=fxy_limit, K=K,
            kp_fric=kp_fric, kv_fric=kv_fric,
        )
    stl_contact_state_dot_out[worldid, conid] = wp.vec3(wp.length(p0 - p0_last), p0.x, p0.y)

    # --- Calculate Force ---
    f_P = wp.vec3(fric_P.x, fric_P.y, fz)
    f_G = wp.quat_rotate(R_GP, f_P)
    # Apply force to body
    X_GB = mob_X_GB_in[worldid, bodyid]
    wp.atomic_add(body_F_contact_out[worldid], bodyid, math.apply_force_to_body_point(X_GB, station_B, f_G))
    # Update GRF
    wp.atomic_add(grf_out, worldid, f_G)


@event_scope
def reset_contact_state(m: Model, d: Data):
    if m.nstlcontact:
        wp.launch(
            _reset_stl_contact_state,
            dim=(d.nworld, m.nstlcontact),
            inputs=[m.stl_contact, d.world_reset, d.integration_done, d.site_pos_G],
            outputs=[d.stl_contact_state]
        )
    return


@event_scope
def contact_forces(m: Model, d: Data):
    if m.nstlcontact:
        wp.launch(
            _process_contacts,
            dim=(d.nworld, m.nstlcontact),
            inputs=[
                m.stl_contact,
                d.integration_done, d.site_pos_G, d.site_vel_G, d.mob_X_GB, d.stl_contact_state,
            ],
            outputs=[d.stl_contact_state_dot, d.body_F_contact, d.grf]
        )
    return
