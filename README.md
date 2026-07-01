Bolt: GPU-accelerated Musculoskeletal Simulator
============================
GPU-accelerated physics simulations for articulated rigid bodies with muscle actuators, designed for many-world parallel simulations. Inspired by [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) and [OpenSim](https://github.com/opensim-org/opensim-core).
<div float="center">
  <img src="assets/screenshot.png"  alt="bolt screenshot"/>
</div>

**Current features include:**
- Articulated body motion computations performed in generalized coordinates, including [OpenSim CustomJoint](https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1CustomJoint.html) logic.
  - Implements Featherstone's Articulated Body Algortithm for computing forward dynamics in O(n) time.
- Stateful elastic tendon dynamics and muscle activation dynamics
  - Force-curves based on either [Millard et al.](https://doi.org/10.1115/1.4023390) or [De Groote et al.](https://pubmed.ncbi.nlm.nih.gov/27001399/)
  - Includes option to use rigid tendons per muscle
  - Activation dynamics includes Degroote et al. and Millard et al. formulations.
- Geometry-based and polynomial/function-based muscle paths.
- Force-based [Hunt-Crossley contacts](https://simtk.org/api_docs/molmodel/api_docs22/Simbody/html/classSimTK_1_1HuntCrossleyForce.html) and ExponentialContactForces.
- Force-based joint limits:
Exponential joint limits based on [Anderson and Pandy](https://pubmed.ncbi.nlm.nih.gov/11264828/), 
[Hunt-Crossley joint limits](https://simtk.org/api_docs/simbody/api_docs33/Simbody/html/classSimTK_1_1Force_1_1MobilityLinearStop.html), and
[CoordinateLimitForce](https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1CoordinateLimitForce.html).
- Symplectic Euler, RK4, adaptive symplectic Euler, and adaptive Runge-Kutta-Merson integrators.
  - Adaptive integrators are error-controlled according to a user-specified accuracy setting

We also include a basic [OpenGL renderer](bolt/render) (not tuned for performance) for debugging.


## Installation
To install Bolt, you'll need to install OpenSim first. We recommend setting up a conda environment first
### 1. Conda Environment Setup
```bash
cd bolt
conda create -n ENV_NAME python=3.11
conda activate ENV_NAME
```

### 2. Install OpenSim
#### 2.1. Create the `config.yaml` file
Create a file named `config.yaml` in the root directory of the repository with the field `python_root_dir`, 
which is a full path to a Python installation directory.

Here is an example:
```
echo "python_root_dir: '/opt/anaconda3/envs/ENV_NAME'" > config.yaml
```
#### 2.2. Build OpenSim
Run the following command from the root directory to build OpenSim and install it into your conda environment.
```
python install_opensim.py
```

On Linux, you may have to add `dependencies/opensim/opensim_dependencies_install/simbody/lib` to your `LD_LIBRARY_PATH`

### 3. Install Bolt
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
compile the warp kernels (especially the polynomial muscle paths).
Subsequent runs will use cached kernels and are much faster.**