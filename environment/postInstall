#!/usr/bin/env bash
set -e

CONDA_ENV_NAME="bigstitcher_env"

# Ensure conda is available
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

git clone -b refactor-exaspim-utils https://github.com/AllenNeuralDynamics/aind-exaSPIM-pipeline-utils.git
#git clone -b feat-phase-correlation https://github.com/AllenNeuralDynamics/aind-exaSPIM-pipeline-utils.git

pip install -e ./aind-exaSPIM-pipeline-utils
ImageJ --headless --update add-update-site BigStitcher https://sites.imagej.net/BigStitcher/
ImageJ --headless --update add-update-site AllenNeuralDynamics https://sites.imagej.net/AllenNeuralDynamics 
ImageJ --headless --update update jars/bigdataviewer-omezarr-0.2.4.jar 
ImageJ --headless --update update plugins/Big_Stitcher-1.2.12.jar
ImageJ --headless --update update

# cp -r /code/FIJI.app /FIJI.app