import torch
import warp as wp
import bolt

M = "data/models/"
scene = bolt.Scene(models=[
    bolt.SceneModel("subject", M + "example_model.osim"),
    bolt.SceneModel("ball", M + "ball.osim"),
])
lr = bolt.load_scene(scene, n_worlds=1, integrator=bolt.IntegratorType.EULER_ADAPTIVE,
                     requires_visuals=True)
m, d = lr.model, lr.data
m.opt.accuracy = 0.1

# Initial poses (free roots: qpos is the world pose; quaternion w defaults to 1 after reset)
q = wp.to_torch(d.qpos)
q[:, lr.qpos_id_lookup["subject/pelvis_ty"]] = 1.05
q[:, lr.qpos_id_lookup["ball/ball_tx"]] = 0.0    # in front of the subject
q[:, lr.qpos_id_lookup["ball/ball_ty"]] = 3.0    # dropped from 1 m
d.world_reset.fill_(True)
bolt.reset(m, d)

viewer = bolt.create_renderer(
    load_result=lr,
    renderer_type=bolt.RendererType.NEWTON,      # or NEWTON / TILED
    draw_visuals=True, draw_colliders=False, draw_muscles=True,
    draw_body_mass=False, draw_beams=True, draw_sites=False,
)
if viewer.viewer_type in (bolt.RendererType.TILED, bolt.RendererType.NEWTON):
    viewer.setup_tiled_renderer(list(range(d.nworld)))

with wp.ScopedCapture() as capture:   # CUDA graph; on CPU just call bolt.step each loop
    bolt.step(m, d)

dt = 0.01
while viewer.is_running():
    if viewer.should_step():
        bolt.increment_next_time(m, d, dt)
        wp.capture_launch(capture.graph)
    viewer.render(m, d)
viewer.close()