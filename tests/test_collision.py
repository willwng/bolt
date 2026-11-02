""" Narrowphase contact geometry """

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import bolt
from bolt._src.contact import collision_driver
from bolt._src.contact.collision_primitive_core import (box_box, capsule_box, mat23f, mat83f, plane_box, sphere_box,
                                                       vec8f)
from bolt.types import GeomType
from models import MODEL_NAMES, model_path


@wp.kernel
def _plane_box_kernel(normal: wp.vec3, plane_pos: wp.vec3, box_pos: wp.vec3, box_rot: wp.mat33, box_size: wp.vec3,
                      dist_out: wp.array(dtype=vec8f), pos_out: wp.array(dtype=mat83f)):
    dist, pos, _ = plane_box(normal, plane_pos, box_pos, box_rot, box_size)
    dist_out[0] = dist
    pos_out[0] = pos


@wp.kernel
def _sphere_box_kernel(sphere_pos: wp.vec3, radius: float, box_pos: wp.vec3, box_rot: wp.mat33, box_size: wp.vec3,
                       dist_out: wp.array(dtype=float), pos_out: wp.array(dtype=wp.vec3),
                       normal_out: wp.array(dtype=wp.vec3)):
    dist, pos, normal = sphere_box(sphere_pos, radius, box_pos, box_rot, box_size)
    dist_out[0] = dist
    pos_out[0] = pos
    normal_out[0] = normal


@wp.kernel
def _capsule_box_kernel(cap_pos: wp.vec3, cap_axis: wp.vec3, radius: float, half_length: float, box_pos: wp.vec3,
                        box_rot: wp.mat33, box_size: wp.vec3, dist_out: wp.array(dtype=wp.vec2),
                        pos_out: wp.array(dtype=mat23f), normal_out: wp.array(dtype=mat23f)):
    dist, pos, normal = capsule_box(cap_pos, cap_axis, radius, half_length, box_pos, box_rot, box_size)
    dist_out[0] = dist
    pos_out[0] = pos
    normal_out[0] = normal


@wp.kernel
def _box_box_kernel(pos1: wp.vec3, rot1: wp.mat33, size1: wp.vec3, pos2: wp.vec3, rot2: wp.mat33, size2: wp.vec3,
                    dist_out: wp.array(dtype=vec8f), pos_out: wp.array(dtype=mat83f),
                    normal_out: wp.array(dtype=mat83f)):
    dist, pos, normal = box_box(pos1, rot1, size1, pos2, rot2, size2, 0.0)
    dist_out[0] = dist
    pos_out[0] = pos
    normal_out[0] = normal


def _random_rotation(rng) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


@pytest.mark.parametrize("seed", range(5))
def test_plane_box_corners(seed):
    """ Each corner's distance to the plane, and a contact point halfway between the corner and the plane """
    rng = np.random.default_rng(seed)
    normal = rng.normal(size=3)
    normal /= np.linalg.norm(normal)
    plane_pos, box_pos = rng.normal(size=3), rng.normal(size=3)
    rot, size = _random_rotation(rng), rng.uniform(0.01, 0.2, size=3)

    dist, pos = wp.zeros(1, dtype=vec8f), wp.zeros(1, dtype=mat83f)
    wp.launch(_plane_box_kernel, dim=1, inputs=[wp.vec3(normal), wp.vec3(plane_pos), wp.vec3(box_pos),
                                                 wp.mat33(rot), wp.vec3(size)], outputs=[dist, pos])

    signs = np.array([[1 if i & b else -1 for b in (1, 2, 4)] for i in range(8)])
    corners = box_pos + (signs * size) @ rot.T
    expected_dist = (corners - plane_pos) @ normal
    np.testing.assert_allclose(dist.numpy()[0], expected_dist, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(pos.numpy()[0], corners - 0.5 * np.outer(expected_dist, normal), rtol=1e-5, atol=1e-6)


def test_plane_box_contacts_in_model():
    """ A box on the foot touches the ground at exactly its corners below the ground plane """
    lr = bolt.load_model(model_path=model_path(MODEL_NAMES[0]), n_worlds=1, integrator=bolt.IntegratorType.EULER_FIXED,
                         requires_visuals=False)
    m, d = lr.model, lr.data
    size = (0.1, 0.1, 0.1)  # large enough to reach the ground from the heel at the default pose
    lr.colliders.append(bolt.convert_user_collider(bolt.UserGeomData(
        name="user_box", body_name="calcn_r", geom_type=GeomType.BOX,
        transform=wp.transform_identity(), size=wp.vec3(size))))
    bolt.update_colliders(lr)
    box = lr.collider_id_lookup["user_box"]

    d.world_reset.fill_(True)
    bolt.reset(m, d)
    bolt.fk(m, d)
    collision_driver.collision(m, d)

    geom_type = m.geom_type.numpy()
    geom_X = d.geom_X.numpy()[0]
    planes = np.flatnonzero(geom_type == GeomType.PLANE)
    assert len(planes) > 0, "model has no ground plane"

    n = min(int(d.nacon.numpy()[0]), d.naconmax)
    pairs = d.contact.geom.numpy()[:n]
    box_dists = d.contact.dist.numpy()[:n][np.any(pairs == box, axis=1) & np.isin(pairs, planes).any(axis=1)]

    signs = np.array([[1 if i & b else -1 for b in (1, 2, 4)] for i in range(8)])
    box_rot = np.array(wp.quat_to_matrix(wp.quat(*geom_X[box, 3:]))).reshape(3, 3)
    corners = geom_X[box, :3] + (signs * size) @ box_rot.T
    expected = []
    for p in planes:
        plane_rot = np.array(wp.quat_to_matrix(wp.quat(*geom_X[p, 3:]))).reshape(3, 3)
        depth = (corners - geom_X[p, :3]) @ plane_rot[:, 1]  # plane normal is the geom's local y
        expected.extend(depth[depth < 0])
    assert len(expected) > 0, "box does not reach the ground; the test needs a bigger box"
    np.testing.assert_allclose(np.sort(box_dists), np.sort(expected), rtol=1e-4, atol=1e-6)

    bolt.increment_next_time(m, d, 1e-3)
    bolt.step(m, d)
    assert np.isfinite(d.qpos.numpy()).all()


def _sphere_box_reference(center: np.ndarray, radius: float, size: np.ndarray):
    """ Distance, contact point and normal (sphere -> box) in the box frame """
    closest = np.clip(center, -size, size)
    gap = np.linalg.norm(closest - center)
    if gap > 0:
        normal = (closest - center) / gap
        return gap - radius, 0.5 * (closest + center + normal * radius), normal
    # center inside: push out through the nearest face
    face_dists = np.concatenate([size - center, center + size])  # +x, +y, +z, -x, -y, -z faces
    k = int(np.argmin(face_dists))
    normal = np.zeros(3)
    normal[k % 3] = -1.0 if k < 3 else 1.0
    closest = face_dists[k]
    return -closest - radius, center + normal * (radius - closest) / 2, normal


def _run_sphere_box(sphere_pos, radius, box_pos, rot, size):
    dist, pos, normal = wp.zeros(1, dtype=float), wp.zeros(1, dtype=wp.vec3), wp.zeros(1, dtype=wp.vec3)
    wp.launch(_sphere_box_kernel, dim=1, inputs=[wp.vec3(sphere_pos), float(radius), wp.vec3(box_pos), wp.mat33(rot),
                                                  wp.vec3(size)], outputs=[dist, pos, normal])
    return dist.numpy()[0], pos.numpy()[0], normal.numpy()[0]


# sphere centers in the box frame, as multiples of the half-extents: faces, edges, corners, and inside
SPHERE_BOX_CENTERS = [(1.3, 0.2, -0.1), (0.1, -1.2, 0.3), (1.2, 1.1, 0.0), (-1.1, 1.2, -1.15), (0.9, 0.1, 0.2),
                      (0.1, 0.0, -0.95), (0.1, -0.2, 0.05)]  # off-center, so the nearest face is unique


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("scaled_center", SPHERE_BOX_CENTERS)
def test_sphere_box(seed, scaled_center):
    """ Distance, contact point and normal for a sphere touching a box face, edge or corner, or with its center inside """
    rng = np.random.default_rng(seed)
    rot, size, box_pos = _random_rotation(rng), rng.uniform(0.05, 0.2, size=3), rng.normal(size=3)
    radius = 0.4 * float(size.min())
    center = np.array(scaled_center) * size
    exp_dist, exp_pos, exp_normal = _sphere_box_reference(center, radius, size)

    dist, pos, normal = _run_sphere_box(box_pos + rot @ center, radius, box_pos, rot, size)
    np.testing.assert_allclose(dist, exp_dist, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(pos, box_pos + rot @ exp_pos, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(normal, rot @ exp_normal, atol=1e-5)


def test_sphere_box_contacts_in_model():
    """ A sphere on the foot pressed 5 mm into a box fixed to the ground gives one contact, normal pointing down """
    lr = bolt.load_model(model_path=model_path(MODEL_NAMES[0]), n_worlds=1, integrator=bolt.IntegratorType.EULER_FIXED,
                         requires_visuals=False)
    m, d = lr.model, lr.data
    radius, depth, half = 0.03, 0.005, (0.2, 0.05, 0.2)
    lr.colliders.append(bolt.convert_user_collider(bolt.UserGeomData(
        name="user_sphere", body_name="calcn_r", geom_type=GeomType.SPHERE,
        transform=wp.transform_identity(), size=wp.vec3(radius, radius, radius))))
    bolt.update_colliders(lr)
    sphere = lr.collider_id_lookup["user_sphere"]
    d.world_reset.fill_(True)
    bolt.reset(m, d)
    bolt.fk(m, d)
    sphere_pos = d.geom_X.numpy()[0, sphere, :3]

    # box on the ground whose top face sits `depth` above the bottom of the sphere
    box_center = sphere_pos - np.array([0.0, radius - depth + half[1], 0.0])
    lr.colliders.append(bolt.convert_user_collider(bolt.UserGeomData(
        name="user_box", body_name="ground", geom_type=GeomType.BOX,
        transform=wp.transform(wp.vec3(box_center), wp.quat_identity()), size=wp.vec3(half))))
    bolt.update_colliders(lr)
    box = lr.collider_id_lookup["user_box"]
    d.world_reset.fill_(True)
    bolt.reset(m, d)
    bolt.fk(m, d)
    collision_driver.collision(m, d)

    n = min(int(d.nacon.numpy()[0]), d.naconmax)
    pairs = d.contact.geom.numpy()[:n]
    sel = np.flatnonzero((pairs == [sphere, box]).all(axis=1) | (pairs == [box, sphere]).all(axis=1))
    assert len(sel) == 1, f"expected one sphere-box contact, got {len(sel)}"
    c = sel[0]
    np.testing.assert_allclose(d.contact.dist.numpy()[c], -depth, atol=1e-5)
    normal = d.contact.frame.numpy()[c][0]
    expected_normal = np.array([0.0, -1.0, 0.0]) if pairs[c][0] == sphere else np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(normal, expected_normal, atol=1e-5)
    np.testing.assert_allclose(d.contact.curvature.numpy()[c], radius, rtol=1e-6)

    bolt.increment_next_time(m, d, 1e-3)
    bolt.step(m, d)
    assert np.isfinite(d.qpos.numpy()).all()


def _run_capsule_box(cap_pos, cap_axis, radius, half_length, box_pos, rot, size):
    dist, pos, normal = wp.zeros(1, dtype=wp.vec2), wp.zeros(1, dtype=mat23f), wp.zeros(1, dtype=mat23f)
    wp.launch(_capsule_box_kernel, dim=1,
              inputs=[wp.vec3(cap_pos), wp.vec3(cap_axis), float(radius), float(half_length), wp.vec3(box_pos),
                      wp.mat33(rot), wp.vec3(size)], outputs=[dist, pos, normal])
    return dist.numpy()[0], pos.numpy()[0], normal.numpy()[0]


def _segment_box_distance(a: np.ndarray, b: np.ndarray, size: np.ndarray) -> float:
    """ Smallest distance from the segment a-b (box frame) to the box; convex along the segment, so ternary search """
    def dist(t):
        p = a + t * (b - a)
        return np.linalg.norm(p - np.clip(p, -size, size))
    lo, hi = 0.0, 1.0
    for _ in range(200):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if dist(m1) < dist(m2):
            hi = m2
        else:
            lo = m1
    return dist(0.5 * (lo + hi))


@pytest.mark.parametrize("seed", range(40))
def test_capsule_box_closest_point(seed):
    """ The first contact is at the true closest distance, and no contact is deeper than that """
    rng = np.random.default_rng(seed)
    rot, size, box_pos = _random_rotation(rng), rng.uniform(0.05, 0.2, size=3), rng.normal(size=3)
    radius, half_length = rng.uniform(0.02, 0.06), rng.uniform(0.05, 0.25)
    axis_l = rng.normal(size=3)
    axis_l /= np.linalg.norm(axis_l)
    # put the capsule axis just outside the box, so the capsule overlaps it but its axis does not
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    center_l = direction * (np.linalg.norm(size) + half_length)
    a, b = center_l - half_length * axis_l, center_l + half_length * axis_l
    gap = _segment_box_distance(a, b, size)
    center_l -= direction * (gap - 0.5 * radius)  # pull it in until the capsule is half a radius deep
    a, b = center_l - half_length * axis_l, center_l + half_length * axis_l
    gap = _segment_box_distance(a, b, size)
    if gap < 1e-4:
        pytest.skip("axis touches the box; capsule_box only reports the face push-out there")

    dist, pos, normal = _run_capsule_box(box_pos + rot @ center_l, rot @ axis_l, radius, half_length, box_pos, rot, size)
    np.testing.assert_allclose(dist[0], gap - radius, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(normal[0]), 1.0, atol=1e-5)
    if np.isfinite(dist[1]):
        assert dist[1] >= gap - radius - 1e-5, "second contact is deeper than the closest point"
        np.testing.assert_allclose(np.linalg.norm(normal[1]), 1.0, atol=1e-5)


@pytest.mark.parametrize("seed", range(3))
def test_capsule_box_lying_on_face(seed):
    """ A capsule lying flat on a box face, parallel to an edge, touches at both ends with the same depth """
    rng = np.random.default_rng(seed)
    rot, box_pos = _random_rotation(rng), rng.normal(size=3)
    size = np.array([0.3, 0.05, 0.2])
    radius, half_length, depth = 0.03, 0.1, 0.005
    center_l = np.array([0.02, size[1] + radius - depth, -0.03])  # on the +y face, well inside its edges
    axis_l = np.array([1.0, 0.0, 0.0])

    dist, pos, normal = _run_capsule_box(box_pos + rot @ center_l, rot @ axis_l, radius, half_length, box_pos, rot, size)
    np.testing.assert_allclose(dist, [-depth, -depth], atol=1e-5)
    for i in range(2):
        np.testing.assert_allclose(normal[i], rot @ np.array([0.0, -1.0, 0.0]), atol=1e-5)  # from capsule into box
    ends = sorted((rot.T @ (pos[i] - box_pos))[0] for i in range(2))
    np.testing.assert_allclose(ends, [center_l[0] - half_length, center_l[0] + half_length], atol=1e-5)


def test_capsule_box_contacts_in_model():
    """ A capsule on the foot overlapping a box fixed to the ground: the deepest contact is the true closest distance """
    lr = bolt.load_model(model_path=model_path(MODEL_NAMES[0]), n_worlds=1, integrator=bolt.IntegratorType.EULER_FIXED,
                         requires_visuals=False)
    m, d = lr.model, lr.data
    radius, half_length, half = 0.03, 0.06, np.array([0.2, 0.05, 0.2])
    lr.colliders.append(bolt.convert_user_collider(bolt.UserGeomData(
        name="user_capsule", body_name="calcn_r", geom_type=GeomType.CAPSULE,
        transform=wp.transform_identity(), size=wp.vec3(radius, half_length, radius))))
    bolt.update_colliders(lr)
    cap = lr.collider_id_lookup["user_capsule"]
    d.world_reset.fill_(True)
    bolt.reset(m, d)
    bolt.fk(m, d)
    X = d.geom_X.numpy()[0, cap]
    cap_axis = np.array(wp.quat_to_matrix(wp.quat(*X[3:]))).reshape(3, 3)[:, 2]  # the narrowphase's capsule axis
    ends = X[:3] + np.outer([-half_length, half_length], cap_axis)

    # box on the ground whose top face sits 5 mm above the lowest point of the capsule
    box_center = np.array([X[0], ends[:, 1].min() - radius + 0.005 - half[1], X[2]])
    lr.colliders.append(bolt.convert_user_collider(bolt.UserGeomData(
        name="user_box", body_name="ground", geom_type=GeomType.BOX,
        transform=wp.transform(wp.vec3(box_center), wp.quat_identity()), size=wp.vec3(half))))
    bolt.update_colliders(lr)
    box = lr.collider_id_lookup["user_box"]
    d.world_reset.fill_(True)
    bolt.reset(m, d)
    bolt.fk(m, d)
    collision_driver.collision(m, d)

    n = min(int(d.nacon.numpy()[0]), d.naconmax)
    pairs = d.contact.geom.numpy()[:n]
    sel = np.flatnonzero((pairs == [cap, box]).all(axis=1) | (pairs == [box, cap]).all(axis=1))
    assert 1 <= len(sel) <= 2, f"expected one or two capsule-box contacts, got {len(sel)}"
    expected = _segment_box_distance(ends[0] - box_center, ends[1] - box_center, half) - radius
    np.testing.assert_allclose(d.contact.dist.numpy()[sel].min(), expected, atol=1e-5)
    np.testing.assert_allclose(d.contact.curvature.numpy()[sel], radius, rtol=1e-6)

    bolt.increment_next_time(m, d, 1e-3)
    bolt.step(m, d)
    assert np.isfinite(d.qpos.numpy()).all()


def _run_box_box(pos1, rot1, size1, pos2, rot2, size2):
    """ Distances, positions and normals of the populated contacts """
    dist, pos, normal = wp.zeros(1, dtype=vec8f), wp.zeros(1, dtype=mat83f), wp.zeros(1, dtype=mat83f)
    wp.launch(_box_box_kernel, dim=1, inputs=[wp.vec3(pos1), wp.mat33(rot1), wp.vec3(size1), wp.vec3(pos2),
                                               wp.mat33(rot2), wp.vec3(size2)], outputs=[dist, pos, normal])
    dist, pos, normal = dist.numpy()[0], pos.numpy()[0], normal.numpy()[0]
    keep = np.isfinite(dist) & (dist < 0)  # what write_contact keeps
    return dist[keep], pos[keep], normal[keep]


def _box_corners(pos, rot, size) -> np.ndarray:
    signs = np.array([[1 if i & b else -1 for b in (1, 2, 4)] for i in range(8)])
    return pos + (signs * size) @ rot.T


def _sat_overlap(pos1, rot1, size1, pos2, rot2, size2) -> float:
    """ Smallest overlap of the two boxes over the 15 separating axes; negative if they are apart """
    axes = [rot1[:, i] for i in range(3)] + [rot2[:, i] for i in range(3)]
    axes += [np.cross(rot1[:, i], rot2[:, j]) for i in range(3) for j in range(3)]
    c1, c2 = _box_corners(pos1, rot1, size1), _box_corners(pos2, rot2, size2)
    best = np.inf
    for a in axes:
        if np.linalg.norm(a) < 1e-6:
            continue
        a = a / np.linalg.norm(a)
        p1, p2 = c1 @ a, c2 @ a
        best = min(best, min(p1.max() - p2.min(), p2.max() - p1.min()))
    return best


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("yaw", [0.0, 45.0])
def test_box_box_stacked(seed, yaw):
    """ A smaller box resting on a bigger one touches at its four bottom corners, all at the same depth """
    rng = np.random.default_rng(seed)
    rot, pos1 = _random_rotation(rng), rng.normal(size=3)
    size1, size2, depth = np.array([0.3, 0.05, 0.25]), np.array([0.08, 0.04, 0.06]), 0.004
    c, sn = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    rot2_l = np.array([[c, 0.0, sn], [0.0, 1.0, 0.0], [-sn, 0.0, c]])  # yaw about box 1's y
    center2_l = np.array([0.03, size1[1] + size2[1] - depth, -0.02])
    rot2, pos2 = rot @ rot2_l, pos1 + rot @ center2_l

    dist, pos, normal = _run_box_box(pos1, rot, size1, pos2, rot2, size2)
    assert len(dist) == 4, f"expected 4 contacts, got {len(dist)}"
    np.testing.assert_allclose(dist, -depth, atol=1e-5)
    np.testing.assert_allclose(normal, np.tile(rot[:, 1], (4, 1)), atol=1e-5)  # box 1's top face normal, toward box 2
    bottom = [c for c in _box_corners(center2_l, rot2_l, size2) if c[1] < center2_l[1]]
    expected = np.array(bottom) + np.array([0.0, 0.5 * depth, 0.0])  # halfway between the corner and the face
    got = (pos - pos1) @ rot  # back into box 1's frame
    np.testing.assert_allclose(sorted(map(tuple, got.round(5))), sorted(map(tuple, expected.round(5))), atol=1e-5)


def test_box_box_corner_down():
    """ A box balanced on one corner gives one contact at that corner's depth """
    rot1, pos1, size1 = np.eye(3), np.zeros(3), np.array([0.3, 0.05, 0.3])
    size2, depth = np.array([0.05, 0.05, 0.05]), 0.003
    # rotate so the body diagonal (-1, -1, -1) points straight down
    v = np.array([-1.0, -1.0, -1.0]) / np.sqrt(3)
    target = np.array([0.0, -1.0, 0.0])
    k = np.cross(v, target)
    k /= np.linalg.norm(k)
    angle = np.arccos(np.dot(v, target))
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    rot2 = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    pos2 = np.array([0.0, size1[1] + np.linalg.norm(size2) - depth, 0.0])

    dist, pos, normal = _run_box_box(pos1, rot1, size1, pos2, rot2, size2)
    assert len(dist) == 1, f"expected 1 contact, got {len(dist)}"
    np.testing.assert_allclose(dist[0], -depth, atol=1e-5)
    np.testing.assert_allclose(normal[0], [0.0, 1.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(pos[0], [0.0, size1[1] - 0.5 * depth, 0.0], atol=1e-5)


@pytest.mark.parametrize("seed", range(60))
def test_box_box_matches_separating_axes(seed):
    """ No contacts when the boxes are apart; when they overlap, the deepest contact is the smallest overlap """
    rng = np.random.default_rng(seed)
    rot1, rot2 = _random_rotation(rng), _random_rotation(rng)
    size1, size2 = rng.uniform(0.03, 0.2, size=3), rng.uniform(0.03, 0.2, size=3)
    pos1 = rng.normal(size=3)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    # slide box 2 along a random direction until it overlaps box 1 by a few millimeters, or sits just apart
    target = rng.choice([-0.003, 0.002, 0.005, 0.01])
    lo, hi = 0.0, float(np.linalg.norm(size1) + np.linalg.norm(size2))
    for _ in range(100):  # overlap shrinks as box 2 moves out
        mid = 0.5 * (lo + hi)
        if _sat_overlap(pos1, rot1, size1, pos1 + mid * direction, rot2, size2) > target:
            lo = mid
        else:
            hi = mid
    pos2 = pos1 + lo * direction
    overlap = _sat_overlap(pos1, rot1, size1, pos2, rot2, size2)

    dist, pos, normal = _run_box_box(pos1, rot1, size1, pos2, rot2, size2)
    if overlap < 0:
        assert len(dist) == 0, f"boxes are {-overlap:.4f} apart but got {len(dist)} contacts"
        return
    assert len(dist) >= 1, f"boxes overlap by {overlap:.4f} but got no contacts"
    np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(dist.min(), -overlap, atol=1e-5)
    # the normal points from box 1 to box 2
    assert np.all(normal @ (pos2 - pos1) > 0)


def test_box_box_contacts_in_model():
    """ A box on the foot overlapping a box fixed to the ground: the deepest contact is the boxes' smallest overlap """
    lr = bolt.load_model(model_path=model_path(MODEL_NAMES[0]), n_worlds=1, integrator=bolt.IntegratorType.EULER_FIXED,
                         requires_visuals=False)
    m, d = lr.model, lr.data
    foot_half, ground_half = np.array([0.08, 0.03, 0.04]), np.array([0.3, 0.05, 0.3])
    lr.colliders.append(bolt.convert_user_collider(bolt.UserGeomData(
        name="foot_box", body_name="calcn_r", geom_type=GeomType.BOX,
        transform=wp.transform_identity(), size=wp.vec3(foot_half))))
    bolt.update_colliders(lr)
    foot = lr.collider_id_lookup["foot_box"]
    d.world_reset.fill_(True)
    bolt.reset(m, d)
    bolt.fk(m, d)
    X = d.geom_X.numpy()[0, foot]
    foot_rot = np.array(wp.quat_to_matrix(wp.quat(*X[3:]))).reshape(3, 3)
    lowest = _box_corners(X[:3], foot_rot, foot_half)[:, 1].min()

    # box on the ground whose top face sits 5 mm above the foot box's lowest corner
    ground_center = np.array([X[0], lowest + 0.005 - ground_half[1], X[2]])
    lr.colliders.append(bolt.convert_user_collider(bolt.UserGeomData(
        name="ground_box", body_name="ground", geom_type=GeomType.BOX,
        transform=wp.transform(wp.vec3(ground_center), wp.quat_identity()), size=wp.vec3(ground_half))))
    bolt.update_colliders(lr)
    ground = lr.collider_id_lookup["ground_box"]
    d.world_reset.fill_(True)
    bolt.reset(m, d)
    bolt.fk(m, d)
    collision_driver.collision(m, d)

    n = min(int(d.nacon.numpy()[0]), d.naconmax)
    pairs = d.contact.geom.numpy()[:n]
    sel = np.flatnonzero((pairs == [foot, ground]).all(axis=1) | (pairs == [ground, foot]).all(axis=1))
    assert len(sel) >= 1, "expected box-box contacts"
    overlap = _sat_overlap(X[:3], foot_rot, foot_half, ground_center, np.eye(3), ground_half)
    np.testing.assert_allclose(d.contact.dist.numpy()[sel].min(), -overlap, atol=1e-5)

    bolt.increment_next_time(m, d, 1e-3)
    bolt.step(m, d)
    assert np.isfinite(d.qpos.numpy()).all()
