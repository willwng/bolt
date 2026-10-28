import numpy as np


def create_torus_mesh(torus_radius, tube_radius, sectors=64, rings=24) -> tuple[np.ndarray, np.ndarray]:
    """ Vertices and (CCW, outward facing) triangle indices of a torus around the z-axis """
    u = np.arange(sectors) * 2.0 * np.pi / sectors  # around the z-axis
    v = np.arange(rings) * 2.0 * np.pi / rings  # around the tube
    u, v = np.meshgrid(u, v, indexing="ij")
    rho = torus_radius + tube_radius * np.cos(v)
    vertices = np.stack([rho * np.cos(u), rho * np.sin(u), tube_radius * np.sin(v)], axis=-1).reshape(-1, 3)

    i, j = np.meshgrid(np.arange(sectors), np.arange(rings), indexing="ij")
    p0 = i * rings + j
    p1 = ((i + 1) % sectors) * rings + j
    p2 = i * rings + (j + 1) % rings
    p3 = ((i + 1) % sectors) * rings + (j + 1) % rings
    indices = np.stack([p0, p1, p2, p2, p1, p3], axis=-1).reshape(-1)
    return vertices.astype(np.float32), indices.astype(np.int32)
