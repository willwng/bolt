MSK Warp - GPU-accelerated Musculoskeletal Simulations
============================
GPU-accelerated physics simulations for articulated rigid bodies with muscle actuators, designed for many-world parallel simulations. Inspired by [mujoco_warp](https://github.com/google-deepmind/mujoco_warp) and [OpenSim](https://github.com/opensim-org/opensim-core).

**Features include:**
- Articulated body motion computations performed in generalized coordinates, including [OpenSim CustomJoint](https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1CustomJoint.html) logic.
- Elastic tendon dynamics and muscle activation dynamics based on [De Groote et al.](https://pubmed.ncbi.nlm.nih.gov/27001399/).
- Geometry-based and polynomial/function-based muscle paths.
- Force-based Hunt-Crossley contacts and constraint-based MuJoCo contacts.
- Exponential force-based joint limits [Anderson and Pandy](https://pubmed.ncbi.nlm.nih.gov/11264828/).
- Explicit Euler, RK4, and adaptive Euler (in-progress) integrators

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

