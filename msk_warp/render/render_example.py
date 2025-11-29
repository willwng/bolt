import numpy as np

import warp as wp
import warp.render
from warp.render.imgui_manager import ImGuiManager


class ExampleImGuiManager(ImGuiManager):
    """An example ImGui manager that displays a few float values."""

    def __init__(self, renderer, window_pos=(10, 10), window_size=(300, 400)):
        super().__init__(renderer)
        if not self.is_available:
            return

        # UI properties
        self.window_pos = window_pos
        self.window_size = window_size

    def draw_ui(self):
        # set window position and size once
        self.imgui.set_next_window_size(self.window_size[0],
                                        self.window_size[1], self.imgui.ONCE)
        self.imgui.set_next_window_position(self.window_pos[0],
                                            self.window_pos[1], self.imgui.ONCE)


class Example:
    def __init__(self, num_tiles=4,
                 use_imgui=True):
        if num_tiles < 1:
            raise ValueError("num_tiles must be greater than or equal to 1.")

        self.renderer = wp.render.OpenGLRenderer(vsync=False)
        self.use_imgui = use_imgui

        if self.use_imgui:
            self.imgui_manager = ExampleImGuiManager(self.renderer)
            if self.imgui_manager.is_available:
                self.renderer.render_2d_callbacks.append(
                    self.imgui_manager.render_frame)
            else:
                self.use_imgui = False

        instance_ids = []

        positions = None
        sizes = None

        # set up instances to hide one of the capsules in each tile
        for i in range(num_tiles):
            instances = [j for j in np.arange(13) if j != i + 2]
            instance_ids.append(instances)
        self.renderer.setup_tiled_rendering(instance_ids,
                                            tile_positions=positions,
                                            tile_sizes=sizes)

        self.renderer.render_ground()

    def render(self):
        time = self.renderer.clock_time
        self.renderer.begin_frame(time)
        for i in range(10):
            self.renderer.render_capsule(
                f"capsule_{i}",
                [i - 5.0, np.sin(time + i * 0.2), -3.0],
                [0.0, 0.0, 0.0, 1.0],
                radius=0.5,
                half_height=0.8,
            )
        self.renderer.render_cylinder(
            "cylinder",
            [3.2, 1.0, np.sin(time + 0.5)],
            np.array(wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0),
                                             wp.sin(time + 0.5))),
            radius=0.5,
            half_height=0.8,
        )
        self.renderer.render_cone(
            "cone",
            [-1.2, 1.0, 0.0],
            np.array(wp.quat_from_axis_angle(wp.vec3(0.707, 0.707, 0.0), time)),
            radius=0.5,
            half_height=0.8,
        )
        self.renderer.end_frame()

    def clear(self):
        if self.use_imgui:
            self.imgui_manager.shutdown()
        self.renderer.clear()


if __name__ == "__main__":
    import argparse
    import distutils.util

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None,
                        help="Override the default Warp device.")
    parser.add_argument("--num_tiles", type=int, default=4,
                        help="Number of viewports to render in a single frame.")
    parser.add_argument(
        "--show_plot",
        type=lambda x: bool(distutils.util.strtobool(x.strip())),
        default=True,
        help="Display the pixels in an additional matplotlib figure.",
    )
    parser.add_argument("--render_mode", type=str, choices=("depth", "rgb"),
                        default="depth", help="")
    parser.add_argument(
        "--split_up_tiles",
        type=lambda x: bool(distutils.util.strtobool(x.strip())),
        default=True,
        help="Whether to split tiles into subplots when --show_plot is True.",
    )
    parser.add_argument(
        "--use_imgui",
        type=lambda x: bool(distutils.util.strtobool(x.strip())),
        default=True,
        help="Enable or disable the ImGui window.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            num_tiles=args.num_tiles,
            use_imgui=args.use_imgui,
        )

        channels = 1 if args.render_mode == "depth" else 3

        while example.renderer.is_running():
            example.render()
        example.clear()
