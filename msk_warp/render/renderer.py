from enum import Enum

import numpy as np
import warp as wp
import warp.render
from scipy.spatial.transform import Rotation as R

from msk_warp._src import types
from .mesh import load_mesh


class RendererType(Enum):
    NONE = 0
    OPENGL = 1
    TILED = 2
    USD = 3


class Renderer:
    def __init__(
            self,
            renderer_type: RendererType,
            draw_colliders: bool,
            draw_visuals: bool,
            draw_muscles: bool,
    ):
        if renderer_type == RendererType.OPENGL:
            self.renderer = wp.render.OpenGLRenderer(
                title="warp-sim",
                vsync=False,
                up_axis='Y',
                screen_width=2000,
                screen_height=1200,
                camera_pos=(5.0, 1.5, 5.0),
                camera_front=(-1.0, 0.0, -1.0),
            )
        elif renderer_type == RendererType.TILED:
            self.renderer = wp.render.OpenGLRenderer(
                title="warp-sim",
                vsync=False,
                up_axis='Y',
                screen_width=2000,
                screen_height=1200,
                camera_pos=(5.0, 1.5, 5.0),
                camera_front=(-1.0, 0.0, -1.0),
            )
        elif renderer_type == RendererType.USD:
            self.renderer = wp.render.UsdRenderer(
                stage="msk_warp.usd",
                up_axis='Y',
                scaling=100.0,
            )
        elif renderer_type == RendererType.NONE:
            self.renderer = None
        else:
            raise ValueError(f"Unsupported viewer type: {renderer_type}")

        self.rot_convert = R.from_euler("z", -90, degrees=True)

        self.viewer_type = renderer_type
        self.worlds = [0]
        self.meshes = []
        self.mesh_scales = []

        self.draw_colliders = draw_colliders
        self.draw_visuals = draw_visuals
        self.draw_muscles = draw_muscles

        # Default colors
        self.colors = {
            "mesh": (0.82, 0.78, 0.74),
            "sphere": (0.7, 0.5, 0.5),
            "capsule": (0.5, 0.5, 0.5),
            "site_active": (0.2, 0.2, 0.7),
            "site_inactive": (0.3, 0.3, 0.3),
        }

    def load_meshes(self, mesh_loads: list[types.MeshLoadResult]):
        for mesh_load in mesh_loads:
            geom_mesh = load_mesh(mesh_load.file)
            self.meshes.append(geom_mesh)
            self.mesh_scales.append(mesh_load.scale)
        return

    def setup_tiled_renderer(
            self,
            m: types.Model,
            worlds: list[int]
    ):
        assert self.viewer_type == RendererType.TILED
        num_tiles = len(worlds)
        number_instances_per_world = m.ngeom
        instance_ids = []
        for i in range(num_tiles):
            world_instances = list(range(
                i * number_instances_per_world,
                (i + 1) * number_instances_per_world))
            instance_ids.append(world_instances)
        self.renderer.setup_tiled_rendering(instances=instance_ids)
        self.worlds = worlds

    def fix_capsule_rot(self, quat) -> tuple:
        rot_input = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        rot_result = self.rot_convert * rot_input
        quat_result = rot_result.as_quat()
        return quat_result[3], quat_result[0], quat_result[1], quat_result[2]

    @staticmethod
    def activation_to_color(act: float) -> tuple:
        # Map activation [0, 1] to color from blue to red
        return act, 0.0, 1.0 - act

    def render(self, m: types.Model, d: types.Data):
        def render_body(wid: int = 0):
            # Ground
            self.renderer.render_ground()

            # Colliders
            if self.draw_colliders:
                geom_xpos = d.geom_xpos.numpy()[wid]
                geom_xquat = d.geom_xquat.numpy()[wid]
                geom_types = m.geom_type.numpy()
                geom_sizes = m.geom_size.numpy()

                for i in range(m.ngeom):
                    pos, rot = geom_xpos[i], geom_xquat[i]
                    rot = (rot[1], rot[2], rot[3], rot[0])  # xyzw to wxyz

                    offset = wid * m.ngeom
                    if geom_types[i] == types.GeomType.SPHERE:
                        self.renderer.render_sphere(
                            f"sphere_{i + offset}",
                            pos,
                            rot,
                            color=self.colors["sphere"],
                            radius=float(geom_sizes[i][0]),
                        )
                    elif geom_types[i] == types.GeomType.CAPSULE:
                        # The capsule renderer is broken for z up axis in OpenGL
                        if self.viewer_type == RendererType.OPENGL or \
                                self.viewer_type == RendererType.TILED:
                            rot = self.fix_capsule_rot(rot)
                            up_axis = 1
                        else:
                            up_axis = 2
                        self.renderer.render_capsule(
                            f"capsule_{i + offset}",
                            pos,
                            rot,
                            radius=geom_sizes[i][0],
                            half_height=geom_sizes[i][1],
                            up_axis=up_axis,
                            color=self.colors["capsule"],
                        )

            # Visuals
            if self.draw_visuals:
                vis_xpos = d.vis_xpos.numpy()[wid]
                vis_xquat = d.vis_xquat.numpy()[wid]
                for i in range(m.nvis):
                    mesh = self.meshes[i]
                    scale = self.mesh_scales[i]
                    pos, rot = vis_xpos[i], vis_xquat[i]
                    rot = (rot[1], rot[2], rot[3], rot[0])
                    self.renderer.render_mesh(
                        name=f"visual_{i}",
                        points=mesh.points.numpy(),
                        indices=mesh.indices.numpy(),
                        pos=pos,
                        rot=rot,
                        scale=scale,
                        colors=self.colors["mesh"],
                    )

            # Muscles
            if self.draw_muscles:
                num_muscles = m.nmuscle
                muscle_pts_adr = m.muscle_pts_adr.numpy()
                muscle_sites_active = d.muscle_active_sites.numpy()[0]
                muscle_pts_active_num = d.muscle_num_active.numpy()[0]
                site_xpos = d.site_xpos.numpy()[wid]
                site_active = d.site_active.numpy()[wid]
                muscle_activations = d.act.numpy()[wid]

                ind_active = np.where(site_active == 1)[0]
                ind_inactive = np.where(site_active == 0)[0]

                # Render all muscle sites (gray if inactive)
                self.renderer.render_points(
                    "muscle_points_active",
                    site_xpos[ind_active],
                    radius=0.005,
                    colors=self.colors["site_active"]
                )
                self.renderer.render_points(
                    "muscle_points_inactive",
                    site_xpos[ind_inactive],
                    radius=0.005,
                    colors=self.colors["site_inactive"]
                )

                for i in range(num_muscles):
                    if not self.draw_muscles:
                        break
                    # Gather all active site indices for this muscle
                    start_idx = muscle_pts_adr[i]
                    end_idx = start_idx + muscle_pts_active_num[i]
                    pt_inds = muscle_sites_active[start_idx:end_idx]

                    # Get the positions of these sites
                    pts = site_xpos[pt_inds]
                    self.renderer.render_line_strip(
                        f"muscle_{i}",
                        pts,
                        color=self.activation_to_color(muscle_activations[i]),
                        radius=0.005,
                    )

        # Render based on viewer type
        if self.viewer_type == RendererType.OPENGL:
            time = self.renderer.clock_time
            self.renderer.begin_frame(time)
            render_body()
            self.renderer.end_frame()
        elif self.viewer_type == RendererType.TILED:
            time = self.renderer.clock_time
            self.renderer.begin_frame(time)
            for world_id in self.worlds:
                render_body(world_id)
            self.renderer.end_frame()
        elif self.viewer_type == RendererType.USD:
            sim_time = d.time.numpy()[0]
            with wp.ScopedTimer("render"):
                self.renderer.begin_frame(sim_time)
                render_body()
                self.renderer.end_frame()
        elif self.viewer_type == RendererType.NONE:
            return

    def close(self):
        if self.viewer_type == RendererType.OPENGL or \
                self.viewer_type == RendererType.TILED:
            self.renderer.clear()
        elif self.viewer_type == RendererType.USD:
            self.renderer.save()
        elif self.viewer_type == RendererType.NONE:
            return
