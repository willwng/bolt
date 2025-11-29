import argparse

import warp as wp

import msk_warp
import msk_warp._src.forward as forward
from msk_warp.benchmark.benchmark import benchmark
from msk_warp.render.renderer import Viewer, ViewerType
from msk_warp.utils.osim_converter import *

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--recompile", action="store_true")
arg_parser.add_argument("--debug", action="store_true")
arg_parser.add_argument("--benchmark", action="store_true")
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


def exclusive_scan(v, mark_empty: bool):
    result = [0] * (len(v) + 1)
    for i in range(1, len(result)):
        result[i] = result[i - 1] + v[i - 1]
    # Remove the last element to return the exclusive scan
    result = result[:-1]

    if mark_empty:
        for i in range(len(v)):
            if v[i] == 0:
                result[i] = -1

    return result


def to_warp_array(lst, dtype):
    arr = np.array(lst)
    # remove 2nd dimension if it exists
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.squeeze(axis=1)
    return wp.from_numpy(arr, dtype=dtype)


def make_zero(shape, dtype):
    return wp.zeros(shape, dtype=dtype)


def make_full(val, shape, dtype):
    return wp.full(shape, val, dtype=dtype)


def main():
    model = "data/osim/model.osim"
    load_result = msk_warp.load_model(model, args.nworld)
    m, d = load_result.model, load_result.data

    if args.recompile:
        wp.clear_kernel_cache()
    if args.debug:
        wp.config.mode = "debug"

    dt = 1.0 / 500.0
    dt_sim = 1.0 / 500.0
    if not args.benchmark:
        bla = []
        bla2 = []
        viewer = Viewer(viewer_type=ViewerType.OPENGL)
        viewer.load_meshes(load_result.visuals)
        if viewer.viewer_type == ViewerType.TILED:
            viewer.setup_tiled_renderer(m, list(range(min(args.nworld, 4))))


        for i in range(args.nstep):
            forward.step_to(m, d, dt, dt_sim)
            viewer.render(m, d)
            bla.append(d.time.numpy()[0])
            bla2.append(-d.grf.numpy()[0, 1] / (75.0 * 9.81))

        # Draw step size history
        import matplotlib.pyplot as plt
        plt.plot(bla, bla2)
        plt.xlabel("Time (s)")
        plt.ylabel("GRF (N)")
        plt.show()

        viewer.close()

    else:
        n_worlds = args.nworld
        n_steps = args.nstep
        res = benchmark(fn=forward.step_to, m=m, d=d,
                        dt=dt, dt_sim=dt_sim, nstep=n_steps,
                        event_trace=True, measure_alloc=True,
                        measure_solver_niter=True)
        jit_time, run_time, trace, nacon, nefc, solver_niter, nsuccess = res
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
