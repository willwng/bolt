<h2>
    <img src="assets/logo.svg" width="500">
    <br>GPU-accelerated Musculoskeletal Simulator
</h2>

[Bolt](https://bolt-simulator.github.io) is a GPU-accelerated, high-fidelity musculoskeletal simulator designed for massively parallel (1k+) environments 
and predictive simulation at hundreds-to-thousands times real-time speed.
Inspired by [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) and [OpenSim](https://github.com/opensim-org/opensim-core).
<div float="center">
  <img src="assets/screenshot.png" alt="bolt screenshot"/>
</div>

## Current features:
- **Articulated body motion computations performed in generalized coordinates, including [OpenSim CustomJoint](https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1CustomJoint.html) logic**
  - Implements Featherstone's Articulated Body Algortithm for O(n) forward dynamics
- **Compliant/elastic tendon dynamics**
  - Implements [Millard et al.](https://doi.org/10.1115/1.4023390) (spline-based) or [De Groote et al.](https://pubmed.ncbi.nlm.nih.gov/27001399/) (polynomial) characteristic curves
  - Option to assume rigid tendons per each muscle
- **Muscle excitation/activation dynamics**
  - Supports [Millard et al.](https://doi.org/10.1115/1.4023390) and [De Groote et al.](https://pubmed.ncbi.nlm.nih.gov/27001399/) activation dynamics
- **Force-based constraints**
  - Contacts: [Hunt-Crossley](https://simtk.org/api_docs/molmodel/api_docs22/Simbody/html/classSimTK_1_1HuntCrossleyForce.html) and [ExponentialContactForces](https://github.com/opensim-org/opensim-core/blob/main/OpenSim/Simulation/Model/ExponentialContactForce.h)
    - Bolt also features its own `StatefulHalfspaceContact` which uses the Hunt-Crossley normal force model and 
    the fast friction model of ExponentialContactForces
  - Joint limits: [MobilityLinearStop](https://simtk.org/api_docs/simbody/api_docs33/Simbody/html/classSimTK_1_1Force_1_1MobilityLinearStop.html) and [CoordinateLimitForce](https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1CoordinateLimitForce.html)
- **Geometry-based and polynomial/function-based muscle paths**
  - Optimized kernels for polynomial evaluations within function-based paths
- **Several integrators, including error-controlled adaptive integrators**
  - Symplectic Euler and RK4 
  - Adaptive symplectic Euler and Runge-Kutta-Merson

We also include a basic [OpenGL renderer](bolt/render) (not tuned for performance) for debugging.

## Limitations/Unsupported Features
- Bolt only supports open-loop kinematic trees. Closed-loop kinematic trees 
(e.g., via [CoordinateCouplerConstraint](https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1CoordinateCouplerConstraint.html)) are
not planned to be supported in the near future.
- Bolt currently has very limited support for muscle wrapping.
- The above can be worked around by using [function-based muscle paths](https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1FunctionBasedPath.html), 
which can be fitted to OpenSim models with wrapping surfaces. 
Bolt accelerates the evaluation of these function-based paths.

## Installation
To install Bolt, you'll need to install OpenSim first (required for model parsing). 
Setting up a conda environment first is recommended.
### 1. Conda Environment Setup
```bash
cd bolt
conda create -n ENV_NAME python=3.11
conda activate ENV_NAME
conda install opensim-org::opensim
```

### 2. Install Requirements + Bolt
```bash
pip install -r requirements.txt
pip install -e .
```


## Example
The following command will launch a simple renderer with a full-body muscle model.
```bash
python -m bolt.test --model data/models/example_model.osim --nstep 1000 --nworld 1
```
Command line:
- `--model`     - path of OpenSim model to load
- `--muscle-functions` - path of muscle's fitted function-based paths
- `--nsteps`    - number of simulation steps to run
- `--nworlds`   - number of parallel simulations
- `--tree`      - whether to create a Graphviz of the model's kinematic tree
- `--recompile` - forces recompilation of the warp kernels
- `--debug`     - enables debug mode
- `--benchmark` - (GPU only) tests simulator speed

**Note: The first time running the simulator will take a while to 
compile the warp kernels (especially the function-based muscle paths code).
Subsequent runs will use cached kernels and are much faster.**
