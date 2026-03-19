import numpy as np
import warp as wp


def create_ellipsoid_mesh(rx, ry, rz, rings=64, sectors=64):
    """Generates vertices and indices for an ellipsoid."""
    vertices = []
    indices = []

    # Generate vertices (Y-up axis)
    for r in range(rings + 1):
        phi = r * np.pi / rings  # 0 to pi
        for s in range(sectors):
            theta = s * 2 * np.pi / sectors  # 0 to 2pi

            x = rx * np.sin(phi) * np.cos(theta)
            y = ry * np.cos(phi)
            z = rz * np.sin(phi) * np.sin(theta)

            vertices.append([x, y, z])

        # Generate triangle indices
        for r in range(rings):
            for s in range(sectors):
                next_s = (s + 1) % sectors

                p0 = r * sectors + s
                p1 = r * sectors + next_s
                p2 = (r + 1) * sectors + s
                p3 = (r + 1) * sectors + next_s

                # Triangle 1 (Reversed to [p0, p1, p2] for CCW winding)
                if r != 0:
                    indices.extend([p0, p1, p2])

                # Triangle 2 (Reversed to [p2, p1, p3] for CCW winding)
                if r != rings - 1:
                    indices.extend([p2, p1, p3])

    return wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3),
        indices=wp.array(indices, dtype=int),
    )
