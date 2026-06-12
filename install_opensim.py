import os
import sys
import subprocess
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Build OpenSim
python_root_dir = config['python_root_dir']
subprocess.run(['bash', 'install_opensim.sh', python_root_dir], check=True, cwd='dependencies')

# Install the OpenSim Python package in the current environment.
package = os.path.join('dependencies', 'opensim', 'opensim_core_install', 'sdk', 'Python','.')
subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])