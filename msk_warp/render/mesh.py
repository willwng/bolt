import os
import numpy as np
import warp as wp
import pyvista as pv


def load_mesh(mesh_file: str):
    file_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "data",
        "geometry",
        mesh_file
    )

     # Load mesh
    if file_path.endswith(".vtp"):
        mesh = pv.read(file_path)
    else:
        raise ValueError("Unsupported file type: expected .vtp")

    mesh = mesh.triangulate()
    geom_points = np.array(mesh.points, dtype=np.float32)
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    geom_face_vertex_indices = faces.astype(np.int32).flatten()

    geom_mesh = wp.Mesh(
        points=wp.array(geom_points, dtype=wp.vec3),
        indices=wp.array(geom_face_vertex_indices, dtype=int),
    )

    return geom_mesh
