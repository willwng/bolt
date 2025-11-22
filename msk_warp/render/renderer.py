import warp as wp
import warp.render

from enum import Enum
from scipy.spatial.transform import Rotation as R

from msk_warp._src import types


class ViewerType(Enum):
    NONE = 0
    OPENGL = 1
    USD = 2


class Viewer:
    def __init__(self, viewer_type: ViewerType):
        if viewer_type == ViewerType.OPENGL:
            self.renderer = wp.render.OpenGLRenderer(
                title="warp-sim",
                vsync=False,
                up_axis='Y',
                screen_width=2000,
                screen_height=1200,
                camera_pos=(0.0, 2.0, 8.0),
            )
        elif viewer_type == ViewerType.USD:
            self.renderer = wp.render.UsdRenderer(
                stage="msk_warp.usd",
                up_axis='Y',
                scaling=100.0,
            )
        elif viewer_type == ViewerType.NONE:
            self.renderer = None
        else:
            raise ValueError(f"Unsupported viewer type: {viewer_type}")
        self.rot_convert = R.from_euler("z", -90, degrees=True)

        self.viewer_type = viewer_type

    def fix_capsule_rot(self, quat) -> tuple:
        rot_input = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        rot_result = self.rot_convert * rot_input
        quat_result = rot_result.as_quat()
        return quat_result[3], quat_result[0], quat_result[1], quat_result[2]

    def render(self, m: types.Model, d: types.Data):

        def render_body():
            # self.renderer.render_ground(size=1)

            geom_xpos = d.geom_xpos.numpy()[0]
            geom_xquat = d.geom_xquat.numpy()[0]
            geom_types = m.geom_type.numpy()
            geom_sizes = m.geom_size.numpy()

            sphere_color = (0.7, 0.5, 0.5)
            capsule_color = (0.5, 0.5, 0.5)
            for i in range(m.ngeom):
                pos, rot = geom_xpos[i], geom_xquat[i]
                rot = (rot[1], rot[2], rot[3], rot[0])  # xyzw to wxyz

                if geom_types[i] == types.GeomType.SPHERE:
                    self.renderer.render_sphere(
                        f"sphere_{i}",
                        pos,
                        rot,
                        color=sphere_color,
                        radius=float(geom_sizes[i][0]),
                    )
                elif geom_types[i] == types.GeomType.CAPSULE:
                    # The capsule renderer is broken for z up axis in OpenGL
                    if self.viewer_type == ViewerType.OPENGL:
                        rot = self.fix_capsule_rot(rot)
                        up_axis = 1
                    else:
                        up_axis = 2
                    self.renderer.render_capsule(
                        f"capsule_{i}",
                        pos,
                        rot,
                        radius=geom_sizes[i][0],
                        half_height=geom_sizes[i][1],
                        up_axis=up_axis,
                        color=capsule_color,
                    )
                elif geom_types[i] == types.GeomType.PLANE:
                    self.renderer.render_box(
                        f"plane_{i}",
                        pos,
                        rot,
                        extents=(5.0, 0.001, 5.0),
                    )

            # render muscles
            num_muscles = m.nmuscle
            muscle_pts_adr = m.muscle_pts_adr.numpy()
            muscle_pts_num = m.muscle_pts_num.numpy()
            site_xpos = d.site_xpos.numpy()[0]
            muscle_activations = d.act.numpy()[0]

            def activation_to_color(act: float) -> tuple:
                # Map activation [0, 1] to color from blue to red
                return act, 0.0, 1.0 - act

            self.renderer.render_points(
                "muscle_points",
                site_xpos,
                radius=0.005,
                colors=(0.2, 0.2, 0.7)
            )

            for i in range(num_muscles):
                # Gather all active site indices for this muscle
                start_idx = muscle_pts_adr[i]
                end_idx = start_idx + muscle_pts_num[i]
                pts = site_xpos[start_idx:end_idx]

                muscle_color = activation_to_color(muscle_activations[i])
                self.renderer.render_line_strip(
                    f"muscle_{i}",
                    pts,
                    color=muscle_color,
                    radius=0.005,
                )

        if self.viewer_type == ViewerType.OPENGL:
            time = self.renderer.clock_time
            self.renderer.begin_frame(time)
            render_body()
            self.renderer.end_frame()
        elif self.viewer_type == ViewerType.USD:
            sim_time = d.time.numpy()[0]
            with wp.ScopedTimer("render"):
                self.renderer.begin_frame(sim_time)
                render_body()
                self.renderer.end_frame()
        elif self.viewer_type == ViewerType.NONE:
            return

    def close(self):
        if self.viewer_type == ViewerType.OPENGL:
            self.renderer.clear()
        elif self.viewer_type == ViewerType.USD:
            self.renderer.save()
        elif self.viewer_type == ViewerType.NONE:
            return
