from enum import Enum

import numpy as np
import warp as wp
import warp.render
from scipy.spatial.transform import Rotation as R

from msk_warp._src import types
from .mesh import load_mesh
from .ellipsoid import create_ellipsoid_mesh


class RendererType(Enum):
    NONE = 0
    OPENGL = 1
    TILED = 2
    USD = 3


class Renderer:
    def __init__(
            self,
            m: types.Model,
            renderer_type: RendererType,
            draw_colliders: bool,
            draw_visuals: bool,
            draw_muscles: bool,
            draw_body_mass: bool,
            draw_beams: bool,
            draw_sites: bool,
    ):
        if renderer_type == RendererType.OPENGL:
            self.renderer = wp.render.OpenGLRenderer(
                title="msk-warp",
                vsync=False,
                up_axis='Y',
                screen_width=1000,
                screen_height=800,
                camera_pos=(5.0, 1.5, 5.0),
                camera_front=(-1.0, 0.0, -1.0),
            )
        elif renderer_type == RendererType.TILED:
            self.renderer = wp.render.OpenGLRenderer(
                title="msk-warp",
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

        self.rot_convert = R.from_euler("x", -90, degrees=True)

        self.viewer_type = renderer_type
        self.worlds = [0]
        self.meshes = []
        self.mesh_scales = []

        self.draw_colliders = draw_colliders
        self.draw_visuals = draw_visuals
        self.draw_muscles = draw_muscles
        self.draw_body_mass = draw_body_mass
        self.draw_beams = draw_beams
        self.draw_sites = draw_sites

        # model-specific: doesn't change during each step
        self.geom_types = m.geom_type.numpy()
        self.geom_sizes = m.geom_size.numpy()
        self.joint_types = m.mob_type.numpy()
        self.muscle_data = m.muscle_data
        self.muscle_pts_adr = m.muscle_pts_adr.numpy()
        self.muscle_num_pts = m.muscle_pts_num.numpy()
        self.mob_qposadr = m.mob_qposadr.numpy()
        self.joint_extra = m.mob_extra_info.numpy()
        self.joint_parent_id = m.body_parentid.numpy()
        self.beam_radius = 0.01

        # Default colors
        self.colors = {
            "mesh": (0.82, 0.78, 0.74),
            "sphere": (0.7, 0.5, 0.5),
            "capsule": (0.7, 0.5, 0.5),
            "ellipsoid": (0.7, 0.5, 0.5),
            "beam": (0.82, 0.78, 0.74),
        }

        number_instances_per_world = 0
        if self.draw_visuals:
            number_instances_per_world += m.nvis
        if self.draw_colliders:
            number_instances_per_world += m.ngeom
        if self.draw_muscles:
            number_instances_per_world += m.nmuscle
        if self.draw_body_mass:
            number_instances_per_world += m.nbody
        if self.draw_beams:
            number_instances_per_world += m.nbeams
        if self.draw_sites:
            number_instances_per_world += m.nsite

        # Required for rendering ellipsoids since they aren't built in to the renderer
        self.ellipsoid_mesh = create_ellipsoid_mesh(1.0, 1.0, 1.0)

        self.num_instances_per_world = number_instances_per_world

    def load_meshes(self, mesh_loads: list[types.MeshLoadResult]):
        for mesh_load in mesh_loads:
            geom_mesh = load_mesh(mesh_load.file)
            self.meshes.append(geom_mesh)
            self.mesh_scales.append(mesh_load.scale)
        return

    def setup_tiled_renderer(
            self,
            worlds: list[int]
    ):
        assert self.viewer_type == RendererType.TILED
        num_tiles = len(worlds)
        instance_ids = []
        for i in range(num_tiles):
            world_instances = list(range(
                i * self.num_instances_per_world,
                (i + 1) * self.num_instances_per_world))
            instance_ids.append(world_instances)

        self.renderer.setup_tiled_rendering(instances=instance_ids)
        self.worlds = worlds

    def fix_capsule_rot(self, quat) -> tuple:
        rot_input = R.from_quat(quat)
        rot_result = rot_input * self.rot_convert
        return rot_result.as_quat(scalar_first=False)

    @staticmethod
    def activation_to_color(act: float) -> tuple:
        # Map activation [0, 1] to color from blue to red
        return act, 0.0, 1.0 - act

    def render(self, m: types.Model, d: types.Data):
        def render_body(wid: int = 0):
            obj_id = wid * self.num_instances_per_world

            # Ground
            self.renderer.render_ground()

            # Sites
            if self.draw_sites:
                site_pos = d.site_pos_G.numpy()[wid]
                for i in range(m.nsite):
                    self.renderer.render_sphere(
                        f"site_{obj_id}",
                        site_pos[i],
                        (0.0, 0.0, 0.0, 1.0),
                        color=(1.0, 0.0, 1.0),
                        radius=0.01,
                    )
                    obj_id += 1

            # Colliders
            if self.draw_colliders:
                geom_X = d.geom_X.numpy()[wid]
                ellipsoid_idx = 0

                for i in range(m.ngeom):
                    pos, rot = geom_X[i, :3], geom_X[i, 3:]

                    if self.geom_types[i] == types.GeomType.SPHERE:
                        self.renderer.render_sphere(
                            f"sphere_{obj_id}",
                            pos,
                            rot,
                            color=self.colors["sphere"],
                            radius=float(self.geom_sizes[i][0]),
                        )
                    elif self.geom_types[i] == types.GeomType.CAPSULE:
                        # I think the capsule renderer may be broken for z-up axis in OpenGL
                        # this line might fix it.
                        rot = self.fix_capsule_rot(rot)
                        up_axis = 1
                        self.renderer.render_capsule(
                            f"capsule_{obj_id}",
                            pos,
                            rot,
                            radius=self.geom_sizes[i][0],
                            half_height=self.geom_sizes[i][1],
                            up_axis=up_axis,
                            color=self.colors["capsule"],
                        )

                    elif self.geom_types[i] == types.GeomType.ELLIPSOID:
                        rx, ry, rz = self.geom_sizes[i]
                        self.renderer.render_mesh(
                            name=f"ellipsoid_{obj_id}",
                            points=self.ellipsoid_mesh.points.numpy(),
                            indices=self.ellipsoid_mesh.indices.numpy(),
                            pos=pos,
                            rot=rot,
                            scale=(rx, ry, rz),
                            colors=self.colors["ellipsoid"],
                        )
                    obj_id += 1

            # Visuals
            if self.draw_visuals:
                vis_X = d.vis_X.numpy()[wid]
                for i in range(m.nvis):
                    mesh = self.meshes[i]
                    scale = self.mesh_scales[i]
                    pos, rot = vis_X[i, :3], vis_X[i, 3:]
                    self.renderer.render_mesh(
                        name=f"visual_{obj_id}",
                        points=mesh.points.numpy(),
                        indices=mesh.indices.numpy(),
                        pos=pos,
                        rot=rot,
                        scale=scale,
                        colors=self.colors["mesh"],
                    )
                    obj_id += 1

            # Muscles
            if self.draw_muscles:
                num_muscles = m.nmuscle
                site_xpos = d.site_pos_G.numpy()[wid]
                muscle_activations = d.m_act.numpy()[wid]

                for i in range(num_muscles):
                    # Muscle radius
                    mm = self.muscle_data[i]
                    radius = np.sqrt(mm.max_isometric_force) / 8000.0

                    # Gather all active site indices for this muscle
                    start_idx = self.muscle_pts_adr[i]
                    end_idx = start_idx + self.muscle_num_pts[i]
                    pt_inds = range(start_idx, end_idx)
                    # Line segment connecting active points
                    pts_xloc = site_xpos[pt_inds]
                    color = self.activation_to_color(muscle_activations[i])
                    self.renderer.render_line_strip(
                        f"muscle_{obj_id}",
                        pts_xloc,
                        color=color,
                        radius=radius,
                    )
                    obj_id += 1

            if self.draw_body_mass:
                body_com = d.body_COM_G.numpy()[wid]
                for i in range(m.nbody):
                    self.renderer.render_sphere(
                        f"mass_{obj_id}",
                        body_com[i],
                        (0.0, 0.0, 0.0, 1.0),
                        color=(1.0, 1.0, 0.0),
                        radius=0.02,
                    )
                    obj_id += 1

            if self.draw_beams:
                for i in range(m.nbeams):
                    beam_points = d.vis_beam_pos.numpy()[wid, i]
                    self.renderer.render_line_strip(
                        f"beam_{obj_id}",
                        beam_points,
                        color=self.colors["beam"],
                        radius=self.beam_radius,
                    )
                    obj_id += 1

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