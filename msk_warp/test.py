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
    # model_path = "data/osim/athlete3.osim"
    # model_path = "data/osim/athlete_fitted_paths.osim"
    model_path = "data/osim/athlete10.osim"
    # model_path = "data/osim/h2190.osim"
    # model_path = "data/osim/athlete_upper_right.osim"
    # model_path = "data/osim/athlete_upper.osim"
    # model_path = "data/osim/athlete_upper_right_only.osim"
    # model_path = "data/osim/athlete_ball.osim"
    # model_path = "data/osim/athlete_notball.osim"
    # model_path = "data/osim/Scaled_FullBody_HamnerModel_Muscle_withContact.osim"
    # polynomial_data_path = "data/function_paths/athlete_lower_body_model_FunctionBasedPathSet.xml"
    # polynomial_data_path = None
    polynomial_data_path = "data/function_paths/athlete10paths.xml"
    # polynomial_data_path = "data/function_paths/scaled_model_function.xml"
    # model_path = "data/osim/athlete2.osim"
    # model_path = "data/osim/simple.osim"
    load_result = msk_warp.load_model(
        model_path,
        n_worlds=args.nworld,
        integrator=msk_warp.IntegratorType.EULER_ADAPTIVE,
        requires_visuals=True,
        polynomial_data_path=polynomial_data_path,
        render_kinematic_tree=args.tree,
    )

    m, d = load_result.model, load_result.data
    qvel = wp.to_torch(d.qvel)
    m.opt.use_inf_norm = False
    m.opt.accuracy = 1.0

    # quit()

    def qpos_id(name):
        return load_result.qpos_id_lookup[name]

    def dof_id(name):
        return load_result.dof_id_lookup[name]

    def muscle_id(name):
        return load_result.muscle_id_lookup[name]

    qpos = wp.to_torch(d.qpos)
    ufrc = wp.to_torch(d.ufrc_total)
    qvel = wp.to_torch(d.qvel)
    if load_result.root_free:
        qpos[:, qpos_id("pelvis_ty")] = 1.05
    # qpos[:, qpos_id("lumbar_extension")] = -0.2
    # qpos[:, qpos_id("thorax_extension")] = -0.2
    # qpos[:, qpos_id("cervical_extension")] = -0.2
    # qpos[:, qpos_id("pro_sup_r")] = np.pi / 2
    # qpos[:, qpos_id("humerus_r_quat_w")] = -0.50
    # qpos[:, qpos_id("shoulder_r_rot_x")] = 0.50
    # qpos[:, qpos_id("shoulder_r_rot_y")] = -0.50
    # qpos[:, qpos_id("shoulder_r_rot_z")] = 0.50

    # qpos[:, qpos_id("humerus_l_quat_w")] = -0.50
    # qpos[:, qpos_id("shoulder_l_rot_x")] = -0.50
    # qpos[:, qpos_id("shoulder_l_rot_y")] = 0.50
    # qpos[:, qpos_id("shoulder_l_rot_z")] = 0.50
    # qpos[:, qpos_id("scapula_elevation_r")] = -0.80

    d.world_reset.fill_(True)
    forward.reset(m, d)

    # a_excitations = msk_warp.actuator_excitations(d)
    # a_excitations[:] = 0.0

    dt = 1.0 / 50.0
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
            draw_beams=True,
            draw_sites=False,
        )
        if viewer.viewer_type == RendererType.TILED:
            viewer.setup_tiled_renderer(list(range(args.nworld)))

        if cuda_graphs:
            with wp.ScopedCapture() as capture:
                step.step(m, d)
            graph = capture.graph

        # moment_arms = wp.to_torch(d.muscle_moment_arm)[0]
        # excitations = wp.to_torch(d.m_excitations)
        # excitations.zero_()
        # excitations[:, muscle_id("biceps_brevis_r")] = 1.0
        # excitations[:, muscle_id("biceps_long_r")] = 1.0
        # all_moment_arms = []
        for i in range(args.nstep):
            step.increment_next_time(m, d, dt)
            if cuda_graphs:
                wp.capture_launch(graph)
            else:
                step.step(m, d)
                # forward.compute_muscle_moments(m, d)
                # all_moment_arms.append(moment_arms.clone())
                # quit()

                # qvel_diff = wp.to_torch(d.qvel_diff)
                # for dof, dofid in load_result.dof_id_lookup.items():
                #     print(f"{dof}: {qvel_diff[0, dofid].item()}")

            # if i % steps_per_render == 0:
            #     viewer.render(m, d)
            viewer.render(m, d)
        viewer.close()

        # all_moment_arms = torch.stack(all_moment_arms, dim=0)
        #
        # muscle_id_to_name = {v: k for k, v in load_result.muscle_id_lookup.items()}
        # dof_id_to_name = {v: k for k, v in load_result.dof_id_lookup.items()}
        # import matplotlib.pyplot as plt
        # for muscleid in range(m.nmuscle):
        #     metadata = m.muscle_data[muscleid]
        #     muscle_name = muscle_id_to_name[muscleid]
        #     muscle_moment_arm = all_moment_arms[:, muscleid]
        #     # Get the indices of the DOFs that have non-negligible moment arms for this muscle across all time steps
        #     nonzero_dof_indices = torch.where(muscle_moment_arm.abs().max(dim=0).values > 1e-7)[0]
        #     if len(nonzero_dof_indices) > 0:
        #         plt.figure(figsize=(10, 6))
        #         for dofid in nonzero_dof_indices:
        #             dof_name = dof_id_to_name[int(dofid)]
        #             plt.plot(all_moment_arms[:, muscleid, dofid].cpu().numpy(), label=dof_name)
        #         plt.title(f"{muscle_name} (fn-based path: {metadata.fn_based_path})")
        #         plt.xlabel("Time step")
        #         plt.ylabel("Moment arm")
        #         plt.legend()

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
