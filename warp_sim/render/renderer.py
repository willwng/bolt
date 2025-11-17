import warp as wp
import warp.render

from enum import Enum

from warp_sim._src import types


class ViewerType(Enum):
    NONE = 0
    OPENGL = 1  # this has bugs rendering rotated capsule with z up axis
    USD = 2


class Viewer:
    def __init__(self, viewer_type: ViewerType):
        if viewer_type == ViewerType.OPENGL:
            self.renderer = wp.render.OpenGLRenderer(
                title="warp-sim",
                vsync=False,
                up_axis='Z',
                screen_width=2000,
                screen_height=1200,
                camera_pos=(0.0, 2.0, 8.0),
            )
        elif viewer_type == ViewerType.USD:
            self.renderer = wp.render.UsdRenderer(
                stage="warp_sim.usd",
                up_axis='Z',
                scaling=100.0,
            )
        elif viewer_type == ViewerType.NONE:
            self.renderer = None
        else:
            raise ValueError(f"Unsupported viewer type: {viewer_type}")

        self.viewer_type = viewer_type

    def render(self, m: types.Model, d: types.Data):

        def render_body():
            # self.renderer.render_ground(size=1)

            geom_xpos = d.geom_xpos.numpy()[0]
            geom_xquat = d.geom_xquat.numpy()[0]
            geom_types = m.geom_type.numpy()
            geom_sizes = m.geom_size.numpy()
            for i in range(m.ngeom):
                pos, rot = geom_xpos[i], geom_xquat[i]
                rot = (rot[1], rot[2], rot[3], rot[0])  # xyzw to wxyz

                if geom_types[i] == types.GeomType.SPHERE:
                    self.renderer.render_sphere(
                        f"sphere_{i}",
                        pos,
                        rot,
                        color=(0.7, 0.2, 0.2),
                        radius=float(geom_sizes[i][0]),
                    )
                elif geom_types[i] == types.GeomType.CAPSULE:
                    self.renderer.render_capsule(
                        f"capsule_{i}",
                        pos,
                        rot,
                        radius=geom_sizes[i][0],
                        half_height=geom_sizes[i][1],
                        up_axis=2,
                    )
                elif geom_types[i] == types.GeomType.PLANE:
                    self.renderer.render_box(
                        f"plane_{i}",
                        pos,
                        rot,
                        extents=(5.0, 5.0, 0.01)
                    )

            # render muscles
            num_muscles = m.nmuscle
            muscle_pts_num = m.muscle_pts_num.numpy()
            muscle_pts_adr = m.muscle_pts_adr.numpy()
            site_xpos = d.site_xpos.numpy()[0]
            muscle_color = (0.2, 0.2, 0.7)
            self.renderer.render_points(
                "muscle_points",
                site_xpos,
                radius=0.005,
                colors=muscle_color,
            )
            for i in range(num_muscles):
                start_idx = muscle_pts_adr[i]
                end_idx = start_idx + muscle_pts_num[i]
                pts = site_xpos[start_idx:end_idx]
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
