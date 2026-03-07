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

    # model_path = "data/osim/model_motor_arms_no_hand_full_contact.osim"
    # model_path = "data/osim/upper_spine.osim"
    model_path = "data/osim/example_gait3d_pin.osim"
    # model_path = "data/osim/sphere.osim"
    load_result = msk_warp.load_model(model_path, args.nworld,
                                      integrator=msk_warp.types.IntegratorType.EULER_FIXED,
                                      polynomial_data_path="data/muscle_poly_info.json",
                                      root_free=True)
    m, d = load_result.model, load_result.data
    m.opt.contact_type = msk_warp.types.ContactType.HUNT_CROSSLEY
    m.opt.limit_type = msk_warp.types.LimitType.HUNT_CROSSLEY
    m.opt.use_inf_norm = False
    m.opt.accuracy = 1.0

    # quit()

    def qpos_id(name):
        return load_result.dof_id_lookup[name][0]

    def dof_id(name):
        return load_result.dof_id_lookup[name][1]

    # qpos = wp.to_torch(d.qpos)
    # qpos[:, qpos_id("pelvis_ty")] = 1.05
    # qpos[:, qpos_id("lumbar_bending")] = -np.pi / 6.0
    #
    # qpos[:, qpos_id("hip_flexion_l")] = 15.0 * np.pi / 180.0
    # qpos[:, qpos_id("knee_angle_l")] = -60.0 * np.pi / 180.0
    # qpos[:, qpos_id("ankle_angle_l")] = 20.0 * np.pi / 180.0
    #
    # qpos[:, qpos_id("hip_flexion_r")] = 15.0 * np.pi / 180.0
    # qpos[:, qpos_id("knee_angle_r")] = -60.0 * np.pi / 180.0
    # qpos[:, qpos_id("ankle_angle_r")] = 20.0 * np.pi / 180.0
    #
    # qvel = wp.to_torch(d.qvel)
    # qvel[:, dof_id("lumbar_bending")] = 2.2
    #
    # qvel[:, dof_id("hip_flexion_l")] = 10.0
    # qvel[:, dof_id("knee_angle_l")] = -5.1
    # qvel[:, dof_id("ankle_angle_l")] = 3.3
    #
    # qvel[:, dof_id("hip_flexion_r")] = 10.0
    # qvel[:, dof_id("knee_angle_r")] = -5.1
    # qvel[:, dof_id("ankle_angle_r")] = 3.3
    forward.fwd(m, d)

    body_X = wp.to_torch(d.mob_X_GB)[0]
    body_COM = wp.to_torch(d.body_COM_G)[0]
    body_V_FM = wp.to_torch(d.body_V_FM)[0]
    body_V_PB_G = wp.to_torch(d.body_V_PB_G)[0]
    body_VD_PB_G = wp.to_torch(d.body_VD_PB_G)[0]
    body_V_GB = wp.to_torch(d.body_V_GB)[0]
    body_gyro = wp.to_torch(d.body_gyro_force)[0]
    mob_coriolis = wp.to_torch(d.mob_coriolis_acc)[0]
    tot_coriolis = wp.to_torch(d.body_total_coriolis_acc)[0]
    tot_centrifugal = wp.to_torch(d.body_total_centrifugal_force)[0]
    art_centrifugal = wp.to_torch(d.body_articulated_centrifugal_force)[0]
    phi = wp.to_torch(d.mob_phi)[0]
    Mk_G = d.body_Mk_G.numpy()[0]
    id_to_body = {v: k for k, v in load_result.body_id_lookup.items()}
    for i, v in enumerate(body_X):
        p, r = v[:3], v[3:]
        # print(i, id_to_body[i])
        # print("mass", Mk_G[i]["m"])
        # print("art centrifugal", art_centrifugal[i])
        # print("total coriolis", tot_coriolis[i])
        # print("total centrifugal", tot_centrifugal[i])
        #
        # print()
    # quit()

    dt = 1.0 / 1000.0
    # dt = 1.0 / 10000.0
    cuda_graphs = wp.get_device().is_cuda
    if not args.benchmark:
        viewer = msk_warp.create_renderer(
            load_result=load_result,
            renderer_type=RendererType.OPENGL,
            draw_visuals=True,
            draw_colliders=True,
            draw_muscles=True,
            draw_body_mass=True,
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
