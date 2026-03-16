import argparse

import warp as wp
import numpy as np
import torch

import msk_warp
import msk_warp._src.forward as forward
import msk_warp._src.step as step
from msk_warp.benchmark.benchmark import benchmark
from msk_warp.render.renderer import RendererType

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--recompile", action="store_true")
arg_parser.add_argument("--debug", action="store_true")
arg_parser.add_argument("--benchmark", action="store_true")
arg_parser.add_argument("--tree", action="store_true")
arg_parser.add_argument("--nworld", type=int, default=1)
arg_parser.add_argument("--nstep", type=int, default=1000)
args = arg_parser.parse_args()


def _print_trace(trace, indent, steps):
    if indent == 0:
        print("\nEvent trace:\n")
    for k, v in trace.items():
        times, sub_trace = v
        if len(times) == 1:
            print("  " * indent + f"{k}: {1e6 * times[0] / steps:.2f}")
        else:
            print("  " * indent + f"{k}: [ ", end="")
            for i in range(len(times)):
                print(f"{1e6 * times[i] / steps:.2f}", end="")
                print(", " if i < len(times) - 1 else " ", end="")
            print("]")
        _print_trace(sub_trace, indent + 1, steps)


def main():
    if args.recompile:
        wp.clear_kernel_cache()
    if args.debug:
        wp.config.mode = "debug"
        # wp.config.verbose = True
        # wp.config.verbose_warnings = True

    # model_path = "data/osim/model_motor_arms_no_hand_full_contact.osim"
    # model_path = "data/osim/all_upper.osim"
    # model_path = "data/osim/upper_no_arms.osim"
    # model_path = "data/osim/everything.osim"
    # model_path = "data/osim/gimbal_custom.osim"
    # model_path = "data/osim/example_gait3d_pin.osim"
    # model_path = "data/osim/example_gait3d_gimbal.osim"
    # model_path = "data/osim/sphere.osim"
    model_path = "data/osim/athlete.osim"
    load_result = msk_warp.load_model(model_path, args.nworld,
                                      integrator=msk_warp.IntegratorType.EULER_ADAPTIVE,
                                      polynomial_data_path="data/muscle_poly_info.json",
                                      render_kinematic_tree=args.tree, )
    m, d = load_result.model, load_result.data
    m.opt.contact_type = msk_warp.ContactType.HUNT_CROSSLEY
    m.opt.limit_type = msk_warp.LimitType.HUNT_CROSSLEY
    m.opt.use_inf_norm = False
    m.opt.accuracy = 1.0

    # quit()

    def qpos_id(name):
        return load_result.qpos_id_lookup[name]

    def dof_id(name):
        return load_result.dof_id_lookup[name]

    qpos = wp.to_torch(d.qpos)
    qvel = wp.to_torch(d.qvel)
    # qpos[:, qpos_id("pelvis_ty")] = 1.05

    dt = 1.0 / 300.0
    # dt = 1.0 / 10000.0
    cuda_graphs = wp.get_device().is_cuda
    if not args.benchmark:
        viewer = msk_warp.create_renderer(
            load_result=load_result,
            renderer_type=RendererType.OPENGL,
            draw_visuals=True,
            draw_colliders=False,
            draw_muscles=True,
            draw_body_mass=False,
            draw_beams=True
        )
        if viewer.viewer_type == RendererType.TILED:
            viewer.setup_tiled_renderer(list(range(args.nworld)))

        if cuda_graphs:
            with wp.ScopedCapture() as capture:
                step.step(m, d)
            graph = capture.graph

        for i in range(args.nstep):
            step.increment_next_time(m, d, dt)
            if cuda_graphs:
                wp.capture_launch(graph)
            else:
                step.step(m, d)
                # forward.fwd(m, d)
            # if i % steps_per_render == 0:
            #     viewer.render(m, d)
            viewer.render(m, d)
        viewer.close()

    else:
        def benchmark_fn(m: msk_warp.Model, d: msk_warp.Data, dt: float):
            step.increment_next_time(m, d, dt)
            step.step(m, d)

        n_worlds = args.nworld
        n_steps = args.nstep
        res = benchmark(fn=benchmark_fn,
                        m=m, d=d, dt=dt, nstep=n_steps,
                        event_trace=True, measure_alloc=True)
        jit_time, run_time, trace, nacon, nsuccess = res
        steps = n_worlds * n_steps

        print(f"""
        Summary for {n_worlds} parallel rollouts

        Total JIT time: {jit_time:.2f} s
        Total simulation time: {run_time:.2f} s
        Total steps per second: {steps / run_time:,.0f}
        Total realtime factor: {steps * dt / run_time:,.2f} x
        Total time per step: {1e9 * run_time / steps:.2f} ns
        Total converged worlds: {nsuccess} / {d.nworld}""")

        _print_trace(trace, 0, steps)


if __name__ == "__main__":
    main()
