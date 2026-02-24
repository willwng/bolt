MSK Warp - GPU-accelerated Musculoskeletal Simulations
============================
GPU-accelerated physics simulations for articulated rigid bodies with muscle actuators, designed for many-world parallel simulations. Inspired by [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) and [OpenSim](https://github.com/opensim-org/opensim-core).
<div float="center">
  <img src="assets/screenshot.png" />
</div>

**Current features include:**
- Articulated body motion computations performed in generalized coordinates, including [OpenSim CustomJoint](https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1CustomJoint.html) logic.
- Stateful elastic tendon dynamics and muscle activation dynamics based on [De Groote et al.](https://pubmed.ncbi.nlm.nih.gov/27001399/).
- Geometry-based and polynomial/function-based muscle paths.
- Force-based Hunt-Crossley contacts and constraint-based MuJoCo contacts.
- Exponential force-based joint limits [Anderson and Pandy](https://pubmed.ncbi.nlm.nih.gov/11264828/), [Hunt-Crossley joint limits](https://simtk.org/api_docs/simbody/api_docs33/Simbody/html/classSimTK_1_1Force_1_1MobilityLinearStop.html), and constraint-based MuJoCo limits.
- Explicit Euler, RK4, adaptive Euler, adaptive Runge-Kutta-Merson (in-progress) integrators.

We also include a basic [OpenGL renderer](msk_warp/render) (not tuned for performance) for debugging.


## Setup
```bash
pip install warp-lang
pip install -e .
```


## Example
The following command will launch a simple renderer with a full-body muscle model.
```bash
python -m msk_warp.test
```
Command line:
- `--recompile` - forces recompilation of the warp kernels
- `--debug`     - enables debug mode
- `--benchmark` - (GPU only) tests simulator speed

