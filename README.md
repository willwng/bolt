MSK Warp - GPU-accelerated Musculoskeletal Simulations
============================
GPU-accelerated physics simulations for articulated rigid bodies with muscle actuators, designed for many-world parallel simulations. Inspired by [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) and [OpenSim](https://github.com/opensim-org/opensim-core).
<div float="center">
  <img src="assets/screenshot.png" />
</div>

**Current features include:**
- Articulated body motion computations performed in generalized coordinates, including [OpenSim CustomJoint](https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1CustomJoint.html) logic.
- Stateful elastic tendon dynamics (based on [Millard et al.](https://doi.org/10.1115/1.4023390)) and muscle activation dynamics with force-curves based on [De Groote et al.](https://pubmed.ncbi.nlm.nih.gov/27001399/).
  - Includes option to use rigid tendons per muscle
  - Activation dynamics includes Degroote et al. and Millard et al. formulations.
- Geometry-based and polynomial/function-based muscle paths.
- Force-based [Hunt-Crossley contacts](https://simtk.org/api_docs/molmodel/api_docs22/Simbody/html/classSimTK_1_1HuntCrossleyForce.html) and ExponentialContactForces.
- Force-based joint limits:
Exponential joint limits based on [Anderson and Pandy](https://pubmed.ncbi.nlm.nih.gov/11264828/), 
[Hunt-Crossley joint limits](https://simtk.org/api_docs/simbody/api_docs33/Simbody/html/classSimTK_1_1Force_1_1MobilityLinearStop.html), and
[CoordinateLimitForce](https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1CoordinateLimitForce.html).
- Symplectic Euler, Midpoint Euler, RK4, adaptive symplectic Euler, adaptive midpoint Euler, and adaptive Runge-Kutta-Merson integrators.

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
- `--nsteps`    - number of simulation steps to run
- `--nworlds`   - number of parallel simulations
- `--recompile` - forces recompilation of the warp kernels
- `--debug`     - enables debug mode
- `--benchmark` - (GPU only) tests simulator speed

