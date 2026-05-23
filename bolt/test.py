import argparse

import warp as wp

import bolt
from bolt.benchmark.benchmark import benchmark

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--model", type=str, required=True)
arg_parser.add_argument("--muscle-functions", type=str)
arg_parser.add_argument("--nworld", type=int, default=1)
arg_parser.add_argument("--nstep", type=int, default=1000)
arg_parser.add_argument("--recompile", action="store_true")
arg_parser.add_argument("--debug", action="store_true")
arg_parser.add_argument("--benchmark", action="store_true")
arg_parser.add_argument("--tree", action="store_true")
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

    # Load the OpenSim model and muscle function paths
    load_result = bolt.load_model(
        model_path=args.model,
        n_worlds=args.nworld,
        integrator=bolt.IntegratorType.EULER_ADAPTIVE,
        requires_visuals=True,
        muscle_fn_path=args.muscle_functions,  # may be None
        render_kinematic_tree=args.tree,
    )
    m, d = load_result.model, load_result.data
    m.opt.use_inf_norm = False
    m.opt.accuracy = 1.0

    # Set initial joint positions here
    def qpos_id(name):
        return load_result.qpos_id_lookup[name]

    qpos = wp.to_torch(d.qpos)
    if load_result.root_free:
        qpos[:, qpos_id("pelvis_ty")] = 1.05

    # Reset sim
    d.world_reset.fill_(True)
    bolt.reset(m, d)

    dt = 1.0 / 100.0
    cuda_graphs = wp.get_device().is_cuda
    if not args.benchmark:
        viewer = bolt.create_renderer(
            load_result=load_result,
            renderer_type=bolt.RendererType.OPENGL,
            draw_visuals=True,
            draw_colliders=False,
            draw_muscles=True,
            draw_body_mass=False,
            draw_beams=True,
            draw_sites=False,
        )
        if viewer.viewer_type == bolt.RendererType.TILED:
            viewer.setup_tiled_renderer(list(range(args.nworld)))

        # Step graph capture
        if cuda_graphs:
            with wp.ScopedCapture() as capture:
                bolt.step(m, d)
            graph = capture.graph

        # Simulation loop
        for i in range(args.nstep):
            bolt.increment_next_time(m, d, dt)
            if cuda_graphs:
                wp.capture_launch(graph)
            else:
                bolt.step(m, d)

            viewer.render(m, d)
        viewer.close()

    else:
        def benchmark_fn(m: bolt.Model, d: bolt.Data, dt: float):
            bolt.increment_next_time(m, d, dt)
            bolt.step(m, d)

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
