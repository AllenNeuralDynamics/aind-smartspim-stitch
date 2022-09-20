#!/usr/bin/env bash

# From the run file
# chmod +x terastitcher_installer.sh

sudo apt-get update

# Giving execution permission to java setup
sudo chmod +x ./terastitcher_java_setup.sh

# Executing installation of java setup
./terastitcher_java_setup.sh

# Updating paths in current terminal
source ~/.bash_profile

tar -xzf TeraStitcher-portable-1.11.10-with-BF-Linux.tar.gz
echo "export PATH=$PATH:$PWD/TeraStitcher-portable-1.11.10-with-BF-Linux" >> ~/.bash_profile

sudo chmod +x ./terastitcher_openmpi_setup.sh

./terastitcher_openmpi_setup.sh

# Creating hostfile
echo "localhost slots=70" > $PWD/hostfile

# Moving all used tar.gz to terastitcher_installers 
mkdir terastitcher_installers
mv zulu*.tar.gz openmpi*tar.gz apache*.tar.gz terastitcher_installers/

# Autopep parastitcher python code
pip install --no-input --upgrade autopep8
autopep8 -i $PWD/TeraStitcher-portable-1.11.10-with-BF-Linux/pyscripts/*.py