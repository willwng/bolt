#!/bin/bash

# Exit when an error happens instead of continue.
set -e

# Default values for flags.
DEBUG_TYPE="Release"
NUM_JOBS=24
MOCO="off"
ORG="nickbianco"
COMMIT="athlete_model"
GENERATOR="Unix Makefiles"
PYTHON_ROOT_DIR=$1

# Set the working directory.
WORKING_DIR="$(pwd)/opensim"
if [ -d "$WORKING_DIR" ]; then
    sudo rm -r $WORKING_DIR
fi
mkdir -p $WORKING_DIR

# Get opensim-core.
git clone --branch $COMMIT https://github.com/$ORG/opensim-core.git $WORKING_DIR/opensim-core
cd $WORKING_DIR/opensim-core

# Build opensim-core dependencies.
mkdir -p $WORKING_DIR/opensim-core/dependencies/build
cd $WORKING_DIR/opensim-core/dependencies/build
cmake $WORKING_DIR/opensim-core/dependencies -DCMAKE_INSTALL_PREFIX=$WORKING_DIR/opensim_dependencies_install/ -DSUPERBUILD_ezc3d=on -DOPENSIM_WITH_CASADI=$MOCO -DBUILD_PYTHON_WRAPPING=on -DPython3_ROOT_DIR=$PYTHON_ROOT_DIR
cmake . -LAH
cmake --build . --config $DEBUG_TYPE -j$NUM_JOBS


# Build and install opensim-core.
mkdir -p $WORKING_DIR/opensim-core/build
cd $WORKING_DIR/opensim-core/build
cmake $WORKING_DIR/opensim-core -G"$GENERATOR" -DOPENSIM_DEPENDENCIES_DIR=$WORKING_DIR/opensim_dependencies_install/ -DOPENSIM_C3D_PARSER=ezc3d -DBUILD_TESTING=off -DCMAKE_INSTALL_PREFIX=$WORKING_DIR/opensim_core_install -DOPENSIM_INSTALL_UNIX_FHS=off -DOPENSIM_WITH_CASADI=$MOCO -DBUILD_PYTHON_WRAPPING=on -DPython3_ROOT_DIR=$PYTHON_ROOT_DIR
cmake --build . --config $DEBUG_TYPE -j$NUM_JOBS
cmake --install .
