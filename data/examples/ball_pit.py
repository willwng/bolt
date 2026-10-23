"""
Drops a jittered stack of balls (ball_pit.yaml) onto the floor.

    python examples/ball_pit.py
    python examples/ball_pit.py --nworld 16 --renderer newton
"""
import argparse
import math
import os

import numpy as np
import warp as wp

import bolt

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--scene", type=str, default=os.path.join(os.path.dirname(__file__), "ball_pit.yaml"))
arg_parser.add_argument("--nworld", type=int, default=1)
arg_parser.add_argument("--nstep", type=int, default=100000)
arg_parser.add_argument("--renderer", type=str, default="opengl", choices=["opengl", "tiled", "newton", "none"])
args = arg_parser.parse_args()

BALL_RADIUS = 0.11
SPACING = 2.6 * BALL_RADIUS  # between ball centers in the stack, so the jitter never overlaps them


def main():
    load_result = bolt.load_scene(args.scene, n_worlds=args.nworld, integrator=bolt.IntegratorType.EULER_ADAPTIVE,
                                  requires_visuals=args.renderer != "none")
    m, d = load_result.model, load_result.data
    m.opt.accuracy = 0.1
    balls = sorted({name.split("/")[0] for name in load_result.body_id_lookup if "/" in name})

    # Stack the balls in a square column, jittered so the stack topples, and different in every world
    rng = np.random.default_rng(0)
    side = math.ceil(math.sqrt(len(balls) / 4))  # roughly 4x as tall as wide
    qpos = d.qpos.numpy()
    for i, ball in enumerate(balls):
        layer, cell = divmod(i, side * side)
        row, col = divmod(cell, side)
        pos = np.array([col * SPACING, BALL_RADIUS + 0.2 + layer * SPACING, row * SPACING])
        pos[[0, 2]] -= 0.5 * (side - 1) * SPACING
        pos = pos + rng.uniform(-0.03, 0.03, size=(args.nworld, 3))
        for axis, p in zip("xyz", pos.T):
            qpos[:, load_result.qpos_id_lookup[f"{ball}/ball_t{axis}"]] = p
        qpos[:, load_result.qpos_id_lookup[f"{ball}/ball_quat_w"]] = 1.0
    d.qpos.assign(qpos)
    d.world_reset.fill_(True)
    bolt.reset(m, d)

    dt = 1.0 / 100.0
    use_cuda_graphs = wp.get_device().is_cuda
    if use_cuda_graphs:
        with wp.ScopedCapture() as capture:
            bolt.step(m, d)

    def step():
        bolt.increment_next_time(m, d, dt)
        if use_cuda_graphs:
            wp.capture_launch(capture.graph)
        else:
            bolt.step(m, d)

    viewer = bolt.create_renderer(
        load_result=load_result,
        renderer_type=bolt.RendererType[args.renderer.upper()],
        draw_visuals=True,
        draw_colliders=False,
        draw_muscles=False,
        draw_body_mass=False,
        draw_beams=False,
        draw_sites=False,
    )
    if viewer.viewer_type in (bolt.RendererType.TILED, bolt.RendererType.NEWTON):
        viewer.setup_tiled_renderer(list(range(args.nworld)))

    i = 0
    while i < args.nstep and viewer.is_running():
        if viewer.should_step():
            step()
            i += 1
        viewer.render(m, d)
    viewer.close()


if __name__ == "__main__":
    main()
