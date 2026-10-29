import numpy as np
import warp as wp
import newton
from newton.viewer import ViewerGL

from bolt import types
from bolt._src.muscle import wrap_path
from bolt._src.muscle.point_path import site_is_active
from .mesh import load_mesh
from .renderer import RendererType
from .torus import create_torus_mesh


@wp.kernel
def gather_transforms(
        src: wp.array2d(dtype=wp.transform),
        ids: wp.array(dtype=int),
        worlds: wp.array(dtype=int),
        world_offsets: wp.array(dtype=wp.vec3),
        out: wp.array2d(dtype=wp.transform),
):
    """ out[k, w] = src[worlds[w], ids[k]] shifted by the world's grid offset """
    k, w = wp.tid()
    X = src[worlds[w], ids[k]]
    out[k, w] = wp.transform(wp.transform_get_translation(X) + world_offsets[w], wp.transform_get_rotation(X))


@wp.kernel
def gather_points(
        src: wp.array2d(dtype=wp.vec3),
        worlds: wp.array(dtype=int),
        world_offsets: wp.array(dtype=wp.vec3),
        out: wp.array2d(dtype=wp.vec3),
):
    w, i = wp.tid()
    out[w, i] = src[worlds[w], i] + world_offsets[w]


@wp.kernel
def gather_wrap_transforms(
        mob_X_GB: wp.array2d(dtype=wp.transform),
        wrap_bodyid: wp.array(dtype=int),
        wrap_X_BS: wp.array(dtype=wp.transform),
        ids: wp.array(dtype=int),
        worlds: wp.array(dtype=int),
        world_offsets: wp.array(dtype=wp.vec3),
        out: wp.array2d(dtype=wp.transform),
):
    """ out[k, w] = surface frame of wrapping obstacle ids[k] in world worlds[w], shifted by its grid offset """
    k, w = wp.tid()
    o = ids[k]
    X = wp.transform_multiply(mob_X_GB[worlds[w], wrap_bodyid[o]], wrap_X_BS[o])
    out[k, w] = wp.transform(wp.transform_get_translation(X) + world_offsets[w], wp.transform_get_rotation(X))


@wp.func
def segment_capsule(a: wp.vec3, b: wp.vec3, radius: float, xform_out: wp.array(dtype=wp.transform),
                    scale_out: wp.array(dtype=wp.vec3), idx: int):
    """ Capsule spanning a -> b (Newton capsules run along local +Z) """
    d = b - a
    length = wp.length(d)
    rot = wp.quat_identity()
    if length > 1.0e-8:
        rot = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), d / length)
    xform_out[idx] = wp.transform(0.5 * (a + b), rot)
    scale_out[idx] = wp.vec3(radius, radius, 0.5 * length)


@wp.kernel
def muscle_segments(
        site_pos_G: wp.array2d(dtype=wp.vec3),
        qpos: wp.array2d(dtype=float),
        m_act: wp.array2d(dtype=float),
        site_cond_qposadr: wp.array(dtype=int),
        site_cond_range: wp.array(dtype=wp.vec2),
        seg_site: wp.array(dtype=int),
        seg_site_end: wp.array(dtype=int),
        seg_muscle: wp.array(dtype=int),
        muscle_radius: wp.array(dtype=float),
        worlds: wp.array(dtype=int),
        world_offsets: wp.array(dtype=wp.vec3),
        xforms: wp.array(dtype=wp.transform),
        scales: wp.array(dtype=wp.vec3),
        colors: wp.array(dtype=wp.vec3),
):
    """ One segment per path point: from that point to the next active point of the same muscle """
    w, j = wp.tid()
    wid = worlds[w]
    idx = w * seg_site.shape[0] + j
    muscle = seg_muscle[j]
    a = seg_site[j]
    q = qpos[wid]

    # Inactive starts and path ends collapse to a zero-size capsule
    b = int(-1)
    if site_is_active(a, q, site_cond_qposadr, site_cond_range):
        for s in range(a + 1, seg_site_end[j]):
            if b < 0 and site_is_active(s, q, site_cond_qposadr, site_cond_range):
                b = s

    act = m_act[wid, muscle]
    colors[idx] = wp.vec3(act, 0.0, 1.0 - act)
    if b < 0:
        xforms[idx] = wp.transform_identity()
        scales[idx] = wp.vec3(0.0)
        return

    offset = world_offsets[w]
    segment_capsule(site_pos_G[wid, a] + offset, site_pos_G[wid, b] + offset, muscle_radius[muscle],
                    xforms, scales, idx)


@wp.kernel
def wrap_muscle_segments(
        points: wp.array2d(dtype=wp.vec3),
        m_act: wp.array2d(dtype=float),
        seg_point: wp.array(dtype=int),
        seg_muscle: wp.array(dtype=int),
        muscle_radius: wp.array(dtype=float),
        worlds: wp.array(dtype=int),
        world_offsets: wp.array(dtype=wp.vec3),
        xforms: wp.array(dtype=wp.transform),
        scales: wp.array(dtype=wp.vec3),
        colors: wp.array(dtype=wp.vec3),
):
    """ One segment per pair of consecutive points sampled along a wrapping path (see wrap_path.sample_wrap_paths) """
    w, j = wp.tid()
    idx = w * seg_point.shape[0] + j
    muscle = seg_muscle[j]
    act = m_act[worlds[w], muscle]
    colors[idx] = wp.vec3(act, 0.0, 1.0 - act)
    offset = world_offsets[w]
    segment_capsule(points[w, seg_point[j]] + offset, points[w, seg_point[j] + 1] + offset, muscle_radius[muscle],
                    xforms, scales, idx)


@wp.kernel
def beam_segments(
        vis_beam_pos: wp.array3d(dtype=wp.vec3),
        radius: float,
        worlds: wp.array(dtype=int),
        world_offsets: wp.array(dtype=wp.vec3),
        xforms: wp.array(dtype=wp.transform),
        scales: wp.array(dtype=wp.vec3),
):
    w, beam, i = wp.tid()
    wid = worlds[w]
    nseg = vis_beam_pos.shape[2] - 1
    idx = (w * vis_beam_pos.shape[1] + beam) * nseg + i
    offset = world_offsets[w]
    segment_capsule(vis_beam_pos[wid, beam, i] + offset, vis_beam_pos[wid, beam, i + 1] + offset, radius,
                    xforms, scales, idx)


class NewtonRenderer:
    """
    Renders Bolt with Newton's ViewerGL. Every mesh/primitive is registered once and drawn as an
    instance batch; each frame only launches warp kernels that write instance transforms from
    Bolt's device arrays, so nothing is copied to the host.
    """

    def __init__(
            self,
            m: types.Model,
            draw_colliders: bool,
            draw_visuals: bool,
            draw_muscles: bool,
            draw_body_mass: bool,
            draw_beams: bool,
            draw_sites: bool,
            draw_wrap_geometries: bool = False,
            world_spacing: float = 2.0,
            width: int = 1200,
            height: int = 1000,
            headless: bool = False,
            invert_mouse: bool = True,
    ):
        self.viewer = ViewerGL(width=width, height=height, vsync=False, headless=headless)
        self.viewer.renderer.set_title("bolt")
        self.device = self.viewer.device

        # Bolt is Y-up; with no Newton model attached the viewer defaults to Z-up
        fb_w, fb_h = self.viewer.renderer.window.get_framebuffer_size()
        self.viewer.camera = type(self.viewer.camera)(width=fb_w, height=fb_h, up_axis="Y")
        self.viewer.set_camera(wp.vec3(3.0, 1.2, 3.0), pitch=-5.0, yaw=-135.0)
        self._setup_lighting()
        if invert_mouse and self.viewer.gui is not None:
            self._invert_mouse_rotation()

        self.draw_colliders = draw_colliders
        self.draw_visuals = draw_visuals
        self.draw_muscles = draw_muscles
        self.draw_body_mass = draw_body_mass
        self.draw_beams = draw_beams
        self.draw_sites = draw_sites
        self.draw_wrap_geometries = draw_wrap_geometries
        self.world_spacing = world_spacing
        self.beam_radius = 0.01

        self.colors = {
            "mesh": (0.82, 0.78, 0.74),
            "geom": (0.7, 0.5, 0.5),
            "beam": (0.82, 0.78, 0.74),
            "muscle_site": (1.0, 0.0, 1.0),
            "exp_contact_site": (0.7, 0.5, 0.5),
            "marker_site": (0.0, 0.0, 1.0),
            "site": (0.5, 0.5, 0.5),
            "mass": (1.0, 1.0, 0.0),
            "ground": (0.55, 0.56, 0.58),
            "wrap": (0.95, 0.78, 0.35),
        }
        # Per-instance (roughness, metallic, checker, texture)
        self.materials = {
            "mesh": (0.6, 0.0, 0.0, 0.0),
            "geom": (0.7, 0.0, 0.0, 0.0),
            "muscle": (0.35, 0.0, 0.0, 0.0),
            "ground": (0.9, 0.0, 1.0, 0.0),
            "wrap": (0.5, 0.0, 0.0, 0.0),
        }
        self.wrap_opacity = 0.6

        self.viewer_type = RendererType.NEWTON
        self.m = m
        self.mesh_loads = []
        self._log_ground()
        self.set_worlds([0])

    # --- Setup ---

    def load_meshes(self, mesh_loads: list[types.MeshLoadResult]):
        self.mesh_loads = mesh_loads
        for i, mesh_load in enumerate(mesh_loads):
            geom_mesh = load_mesh(mesh_load.file)
            self.viewer.log_mesh(
                f"/bolt/geometry/visual_{i}",
                wp.array(geom_mesh.points.numpy(), dtype=wp.vec3, device=self.device),
                wp.array(geom_mesh.indices.numpy(), dtype=wp.int32, device=self.device),
                color=self.colors["mesh"],
                hidden=True,
            )
        self._allocate_batches()

    def setup_tiled_renderer(self, worlds: list[int]):
        """ Kept for API parity with Renderer: all listed worlds are laid out on a grid in one scene """
        self.set_worlds(worlds)

    def set_worlds(self, worlds: list[int]):
        n = len(worlds)
        ncols = int(np.ceil(np.sqrt(n)))
        grid = np.arange(n)
        offsets = np.zeros((n, 3), dtype=np.float32)
        offsets[:, 0] = (grid % ncols - (ncols - 1) / 2) * self.world_spacing
        offsets[:, 2] = (grid // ncols - (int(np.ceil(n / ncols)) - 1) / 2) * self.world_spacing

        self.worlds = worlds
        self.nw = n
        self.worlds_wp = wp.array(worlds, dtype=int, device=self.device)
        self.world_offsets = wp.array(offsets, dtype=wp.vec3, device=self.device)
        # The shadow map is an ortho box centered at the origin; grow it to cover the world grid
        self.viewer.renderer.shadow_extents = max(5.0, 0.5 * ncols * self.world_spacing + 1.5)
        self._allocate_batches()

    def _setup_lighting(self):
        r = self.viewer.renderer
        # Newton's Y-up sun default lights the model from behind the default camera, leaving the
        # visible side ambient-only. Use a high key light from the camera's front-right instead.
        sun = np.array((0.5, 1.0, 0.8))
        r._sun_direction = sun / np.linalg.norm(sun)
        # The spotlight cone darkens everything outside a ~45 deg cone, which hides outer worlds
        r.spotlight_enabled = False
        r.shadow_radius = 2.0
        r.exposure = 1.1
        # Neutral studio backdrop (sky_lower doubles as the distance-fog color)
        r.sky_upper = (0.45, 0.52, 0.62)
        r.sky_lower = (0.74, 0.75, 0.78)
        # Cool fill from above, warm bounce from the floor
        r.ambient_sky = (0.78, 0.82, 0.9)
        r.ambient_ground = (0.5, 0.45, 0.4)

    def _invert_mouse_rotation(self):
        """ Flip horizontal left-drag look and middle-drag orbit so the scene follows the mouse; pan/dolly are unchanged """
        import pyglet
        mouse, key = pyglet.window.mouse, pyglet.window.key
        gui = self.viewer.gui
        handle_mouse_drag = gui.handle_mouse_drag

        def inverted(x, y, dx, dy, buttons, to_framebuffer_coords, modifiers=0):
            middle_orbit = buttons & mouse.MIDDLE and not modifiers & (key.MOD_CTRL | key.MOD_SHIFT)
            if buttons & mouse.LEFT or middle_orbit:
                dx = -dx
            handle_mouse_drag(x, y, dx, dy, buttons, to_framebuffer_coords, modifiers)

        gui.handle_mouse_drag = inverted

    def _log_ground(self):
        # Newton planes lie in local XY; rotate so the normal points along +Y
        rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * wp.pi)
        self.viewer.log_shapes(
            "/bolt/ground",
            newton.GeoType.PLANE,
            (50.0, 50.0),
            wp.array([wp.transform(wp.vec3(0.0), rot)], dtype=wp.transform, device=self.device),
            colors=wp.array([self.colors["ground"]], dtype=wp.vec3, device=self.device),
            materials=wp.array([self.materials["ground"]], dtype=wp.vec4, device=self.device),
        )

    def _full(self, n: int, value, dtype):
        return wp.array(np.tile(np.asarray(value, dtype=np.float32), (n, 1)), dtype=dtype, device=self.device)

    def _allocate_batches(self):
        """ (Re)allocate the per-frame instance buffers for the current model and world list """
        m, nw, dev = self.m, self.nw, self.device
        geom_types = m.geom_type.numpy()
        geom_sizes = m.geom_size.numpy()

        # Colliders: spheres and ellipsoids share a unit sphere scaled per instance; capsules
        # are already Z-aligned in Bolt, matching Newton's capsule convention.
        self.geom_batches = []
        if self.draw_colliders:
            sphere_mesh = "/bolt/geometry/unit_sphere"
            if sphere_mesh not in self.viewer.objects:
                self.viewer.log_geo(sphere_mesh, newton.GeoType.SPHERE, (1.0,), 0.0, True, hidden=True)
            for gtype, name in ((types.GeomType.SPHERE, "spheres"), (types.GeomType.ELLIPSOID, "ellipsoids"),
                                (types.GeomType.CAPSULE, "capsules")):
                ids = np.flatnonzero(geom_types == gtype)
                if len(ids) == 0:
                    continue
                sizes = geom_sizes[ids]
                if gtype == types.GeomType.SPHERE:
                    scale = np.repeat(sizes[:, :1], 3, axis=1)
                elif gtype == types.GeomType.ELLIPSOID:
                    scale = sizes
                else:
                    scale = np.stack([sizes[:, 0], sizes[:, 0], sizes[:, 1]], axis=1)
                self.geom_batches.append(dict(
                    name=f"/bolt/{name}",
                    capsule=gtype == types.GeomType.CAPSULE,
                    ids=wp.array(ids, dtype=int, device=dev),
                    xforms=wp.empty((len(ids), nw), dtype=wp.transform, device=dev),
                    scales=wp.array(np.repeat(scale, nw, axis=0), dtype=wp.vec3, device=dev),
                    colors=self._full(len(ids) * nw, self.colors["geom"], wp.vec3),
                    materials=self._full(len(ids) * nw, self.materials["geom"], wp.vec4),
                ))

        # Wrapping obstacles: unit spheres scaled to spheres and ellipsoids, unit cylinders, and a mesh per torus
        self.wrap_batches = []
        if self.draw_wrap_geometries and m.nwrap > 0:
            wrap_types, wrap_sizes = m.wrap_type.numpy(), m.wrap_size.numpy()
            for mesh, geo in (("/bolt/geometry/unit_sphere", newton.GeoType.SPHERE),
                              ("/bolt/geometry/unit_cylinder", newton.GeoType.CYLINDER)):
                if mesh not in self.viewer.objects:
                    self.viewer.log_geo(mesh, geo, (1.0, 1.0) if geo == newton.GeoType.CYLINDER else (1.0,), 0.0, True,
                                        hidden=True)
            groups = []
            for wtype, mesh in ((types.WrapType.SPHERE, "/bolt/geometry/unit_sphere"),
                                (types.WrapType.ELLIPSOID, "/bolt/geometry/unit_sphere"),
                                (types.WrapType.CYLINDER, "/bolt/geometry/unit_cylinder")):
                ids = np.flatnonzero(wrap_types == wtype)
                if len(ids):
                    groups.append((f"/bolt/wrap_{wtype.name.lower()}", mesh, ids))
            for o in np.flatnonzero(wrap_types == types.WrapType.TORUS):
                mesh = f"/bolt/geometry/wrap_torus_{o}"
                points, indices = create_torus_mesh(*wrap_sizes[o][:2])
                self.viewer.log_mesh(mesh, wp.array(points, dtype=wp.vec3, device=dev),
                                     wp.array(indices, dtype=wp.int32, device=dev), hidden=True)
                groups.append((f"/bolt/wrap_torus_{o}", mesh, np.array([o])))
            cylinder_scale = np.array([1.0, 1.0, wrap_path.WRAP_RENDER_CYLINDER_HALF_LENGTH])
            for name, mesh, ids in groups:
                size = wrap_sizes[ids]
                scale = {types.WrapType.SPHERE: size[:, [0, 0, 0]],
                         types.WrapType.ELLIPSOID: size,
                         types.WrapType.CYLINDER: size[:, [0, 0, 0]] * cylinder_scale,
                         types.WrapType.TORUS: np.ones_like(size)}[types.WrapType(wrap_types[ids[0]])]
                self.wrap_batches.append(dict(
                    name=name, mesh=mesh,
                    ids=wp.array(ids, dtype=int, device=dev),
                    xforms=wp.empty((len(ids), nw), dtype=wp.transform, device=dev),
                    scales=wp.array(np.repeat(scale, nw, axis=0), dtype=wp.vec3, device=dev),
                    colors=self._full(len(ids) * nw, self.colors["wrap"], wp.vec3),
                    materials=self._full(len(ids) * nw, self.materials["wrap"], wp.vec4),
                    opacities=wp.full(len(ids) * nw, self.wrap_opacity, dtype=wp.float32, device=dev),
                ))

        # Visuals: one instance batch per mesh, one instance per world
        if self.draw_visuals and self.mesh_loads:
            self.vis_ids = wp.array(np.arange(m.nvis), dtype=int, device=dev)
            self.vis_xforms = wp.empty((m.nvis, nw), dtype=wp.transform, device=dev)
            self.vis_scales = [self._full(nw, ml.scale, wp.vec3) for ml in self.mesh_loads]
            self.vis_colors = self._full(nw, self.colors["mesh"], wp.vec3)
            self.vis_materials = self._full(nw, self.materials["mesh"], wp.vec4)

        # Sites and body COMs are drawn as point spheres with static colors and radii
        if self.draw_sites:
            colors = np.tile(np.asarray(self.colors["site"], np.float32), (m.nsite, 1))
            radii = np.full(m.nsite, 0.01, np.float32)
            for (lo, hi), key, r in (
                    ((m.site_adr_muscle, m.site_adr_muscle + m.nsite_muscle), "muscle_site", 0.005),
                    ((m.site_adr_contact, m.site_adr_contact + m.nsite_contact), "exp_contact_site", 0.01),
                    ((m.site_adr_marker, m.site_adr_marker + m.nsite_marker), "marker_site", 0.02)):
                colors[lo:hi] = self.colors[key]
                radii[lo:hi] = r
            self.site_pts = wp.empty((nw, m.nsite), dtype=wp.vec3, device=dev)
            self.site_colors = wp.array(np.tile(colors, (nw, 1)), dtype=wp.vec3, device=dev)
            self.site_radii = wp.array(np.tile(radii, nw), dtype=float, device=dev)

        if self.draw_body_mass:
            self.mass_pts = wp.empty((nw, m.nbody), dtype=wp.vec3, device=dev)
            self.mass_colors = self._full(nw * m.nbody, self.colors["mass"], wp.vec3)

        # Muscles: one capsule per path point (see muscle_segments), and per point sampled along wrapping paths
        self.nseg_muscle = self.nseg_wrap = 0
        if self.draw_muscles and m.nmuscle > 0:
            adr = m.muscle_pts_adr.numpy()
            num = m.muscle_pts_num.numpy()
            seg_site, seg_end, seg_muscle = [], [], []
            for i in range(m.nmuscle):
                if i in m.muscle_wrap_group_tuple:
                    continue
                for s in range(adr[i], adr[i] + num[i] - 1):
                    seg_site.append(s)
                    seg_end.append(adr[i] + num[i])
                    seg_muscle.append(i)
            nseg = self.nseg_muscle = len(seg_site)
            radius = [np.sqrt(mm.max_isometric_force) / 8000.0 for mm in m.muscle_data]
            self.seg_site = wp.array(seg_site, dtype=int, device=dev)
            self.seg_end = wp.array(seg_end, dtype=int, device=dev)
            self.seg_muscle = wp.array(seg_muscle, dtype=int, device=dev)
            self.muscle_radius = wp.array(radius, dtype=float, device=dev)
            self.muscle_xforms = wp.empty(nw * nseg, dtype=wp.transform, device=dev)
            self.muscle_scales = wp.empty(nw * nseg, dtype=wp.vec3, device=dev)
            self.muscle_colors = wp.empty(nw * nseg, dtype=wp.vec3, device=dev)
            self.muscle_materials = self._full(nw * nseg, self.materials["muscle"], wp.vec4)

            wrap_adr, wrap_num = wrap_path.wrap_render_layout(m)
            wrap_seg_point = [p for a, n in zip(wrap_adr, wrap_num) for p in range(a, a + n - 1)]
            wrap_seg_muscle = [i for i, n in zip(m.muscle_wrap_group_tuple, wrap_num) for _ in range(n - 1)]
            nseg = self.nseg_wrap = len(wrap_seg_point)
            if nseg:
                self.wrap_render_adr = wp.array(wrap_adr, dtype=int, device=dev)
                self.wrap_points = wp.empty((nw, sum(wrap_num)), dtype=wp.vec3, device=dev)
                self.wrap_seg_point = wp.array(wrap_seg_point, dtype=int, device=dev)
                self.wrap_seg_muscle = wp.array(wrap_seg_muscle, dtype=int, device=dev)
                self.wrap_xforms = wp.empty(nw * nseg, dtype=wp.transform, device=dev)
                self.wrap_scales = wp.empty(nw * nseg, dtype=wp.vec3, device=dev)
                self.wrap_colors = wp.empty(nw * nseg, dtype=wp.vec3, device=dev)
                self.wrap_materials = self._full(nw * nseg, self.materials["muscle"], wp.vec4)

        if self.draw_beams and m.nbeams > 0:
            self.beam_shape = None  # sized from vis_beam_pos on first render
            self.beam_xforms = None

        # Static colors/materials are uploaded through host memory, so only send them on the first frame
        self._static_uploaded = False

    # --- Per frame ---

    def render(self, m: types.Model, d: types.Data):
        v, dev, nw = self.viewer, self.device, self.nw
        first = not self._static_uploaded

        def static(arr):
            return arr if first else None
        v.begin_frame(float(d.time.numpy()[0]))

        for batch in self.geom_batches:
            wp.launch(gather_transforms, dim=batch["xforms"].shape,
                      inputs=[d.geom_X, batch["ids"], self.worlds_wp, self.world_offsets],
                      outputs=[batch["xforms"]], device=dev)
            log = v.log_capsules if batch["capsule"] else v.log_instances
            log(batch["name"], "/bolt/geometry/unit_sphere", batch["xforms"].flatten(), batch["scales"],
                static(batch["colors"]), static(batch["materials"]))

        for batch in self.wrap_batches:
            wp.launch(gather_wrap_transforms, dim=batch["xforms"].shape,
                      inputs=[d.mob_X_GB, m.wrap_bodyid, m.wrap_X_BS, batch["ids"], self.worlds_wp, self.world_offsets],
                      outputs=[batch["xforms"]], device=dev)
            v.log_instances(batch["name"], batch["mesh"], batch["xforms"].flatten(), batch["scales"],
                            static(batch["colors"]), static(batch["materials"]), opacities=static(batch["opacities"]))

        if self.draw_visuals and self.mesh_loads:
            wp.launch(gather_transforms, dim=self.vis_xforms.shape,
                      inputs=[d.vis_X, self.vis_ids, self.worlds_wp, self.world_offsets],
                      outputs=[self.vis_xforms], device=dev)
            for i in range(m.nvis):
                v.log_instances(f"/bolt/visual_{i}", f"/bolt/geometry/visual_{i}", self.vis_xforms[i],
                                self.vis_scales[i], static(self.vis_colors), static(self.vis_materials))

        if self.draw_sites:
            wp.launch(gather_points, dim=self.site_pts.shape,
                      inputs=[d.site_pos_G, self.worlds_wp, self.world_offsets],
                      outputs=[self.site_pts], device=dev)
            v.log_points("/bolt/sites", self.site_pts.flatten(), self.site_radii, static(self.site_colors))

        if self.draw_body_mass:
            wp.launch(gather_points, dim=self.mass_pts.shape,
                      inputs=[d.body_COM_G, self.worlds_wp, self.world_offsets],
                      outputs=[self.mass_pts], device=dev)
            v.log_points("/bolt/body_mass", self.mass_pts.flatten(), 0.02, static(self.mass_colors))

        if self.nseg_muscle:
            wp.launch(muscle_segments, dim=(nw, self.seg_site.shape[0]),
                      inputs=[d.site_pos_G, d.qpos, d.m_act, m.site_cond_qposadr, m.site_cond_range,
                              self.seg_site, self.seg_end, self.seg_muscle, self.muscle_radius,
                              self.worlds_wp, self.world_offsets],
                      outputs=[self.muscle_xforms, self.muscle_scales, self.muscle_colors], device=dev)
            v.log_capsules("/bolt/muscles", "", self.muscle_xforms, self.muscle_scales, self.muscle_colors,
                           static(self.muscle_materials))

        if self.nseg_wrap:
            wrap_path.sample_wrap_paths(m, d, self.wrap_render_adr, self.worlds_wp, self.wrap_points)
            wp.launch(wrap_muscle_segments, dim=(nw, self.nseg_wrap),
                      inputs=[self.wrap_points, d.m_act, self.wrap_seg_point, self.wrap_seg_muscle,
                              self.muscle_radius, self.worlds_wp, self.world_offsets],
                      outputs=[self.wrap_xforms, self.wrap_scales, self.wrap_colors], device=dev)
            v.log_capsules("/bolt/wrapped_muscles", "", self.wrap_xforms, self.wrap_scales, self.wrap_colors,
                           static(self.wrap_materials))

        if self.draw_beams and m.nbeams > 0:
            nbeam_vis = d.vis_beam_pos.shape[2]
            if self.beam_xforms is None:
                n = nw * m.nbeams * (nbeam_vis - 1)
                self.beam_xforms = wp.empty(n, dtype=wp.transform, device=dev)
                self.beam_scales = wp.empty(n, dtype=wp.vec3, device=dev)
                self.beam_colors = self._full(n, self.colors["beam"], wp.vec3)
                self.beam_materials = self._full(n, self.materials["mesh"], wp.vec4)
            wp.launch(beam_segments, dim=(nw, m.nbeams, nbeam_vis - 1),
                      inputs=[d.vis_beam_pos, self.beam_radius, self.worlds_wp, self.world_offsets],
                      outputs=[self.beam_xforms, self.beam_scales], device=dev)
            v.log_capsules("/bolt/beams", "", self.beam_xforms, self.beam_scales, static(self.beam_colors),
                           static(self.beam_materials))

        self._static_uploaded = True
        v.end_frame()

    def is_running(self) -> bool:
        return self.viewer.is_running()

    def should_step(self) -> bool:
        """ False while paused (space), except once per single-step request (period) """
        return self.viewer.should_step()

    def close(self):
        self.viewer.close()
