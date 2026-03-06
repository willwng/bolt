import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.func
def force_at_point(frc: wp.vec3, offset: wp.vec3) -> wp.spatial_vector:
    torque = wp.cross(offset, frc)
    return wp.spatial_vector(torque, frc)


@wp.func
def transform_velocity(cvel: wp.spatial_vector, offset: wp.vec3) -> wp.spatial_vector:
    ang = wp.spatial_top(cvel)
    lin = wp.spatial_bottom(cvel)
    pvel_lin = lin + wp.cross(ang, offset)
    return wp.spatial_vector(ang, pvel_lin)
