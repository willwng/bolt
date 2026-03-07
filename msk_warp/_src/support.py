import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.func
def apply_force_to_body_point(X_GB: wp.transform, point_in_b: wp.vec3, force_in_G: wp.vec3) -> wp.spatial_vector:
    R_GB = wp.transform_get_rotation(X_GB)
    trq = wp.cross(wp.quat_rotate(R_GB, point_in_b), force_in_G)
    return wp.spatial_vector(trq, force_in_G)


@wp.func
def find_station_at_ground_point(X_GB: wp.transform, location_in_G: wp.vec3) -> wp.vec3:
    return wp.transform_point(wp.transform_inverse(X_GB), location_in_G)


@wp.func
def express_vector_in_ground_frame(X_GB: wp.transform, vec_in_B: wp.vec3) -> wp.vec3:
    """
    Re-express a vector expressed in this body B's frame into the same vector in
    G, by applying only a rotation.
    """
    R_GB = wp.transform_get_rotation(X_GB)
    return wp.quat_rotate(R_GB, vec_in_B)


@wp.func
def find_station_velocity_in_ground(X_GB: wp.transform, V_GB: wp.spatial_vector, station_on_B: wp.vec3) -> wp.vec3:
    w, v = wp.spatial_top(V_GB), wp.spatial_bottom(V_GB)  # in G
    r = express_vector_in_ground_frame(X_GB, station_on_B)
    return v + wp.cross(w, r)
