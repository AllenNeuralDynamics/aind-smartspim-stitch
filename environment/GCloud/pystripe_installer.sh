#!/usr/bin/env bash

# Pystripe installer
conda create -n pystripe python=3.6 -y
conda activate pystripe

# pip version >= 20.2
pip install --no-input https://github.com/chunglabmit/pystripe/archive/master.zip mpi4py