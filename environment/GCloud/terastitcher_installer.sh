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

# Installing gcc compiler for openmpi
sudo apt-get install -y build-essential

# Installing make
sudo apt-get install -y make

echo "- Starting terastitcher installation with MPI"

sudo chmod +x ./terastitcher_openmpi_setup.sh

./terastitcher_openmpi_setup.sh

# Creating hostfile
echo "localhost slots=70" > $PWD/hostfile

python -V
status=$?
python_cmd=1

# Installing pystripe and mpi4py
if ! (exit $status)
then
    python_version=2
    sudo apt install -y python3-pip
    python3 -m pip install --no-input https://github.com/chunglabmit/pystripe/archive/master.zip \
    git+https://github.com/carshadi/aicsimageio.git@feature/zarrwriter-multiscales \
    git+https://github.com/AllenInstitute/argschema.git \
    mpi4py
else
    sudo apt install -y python-pip
    python -m pip install --no-input https://github.com/chunglabmit/pystripe/archive/master.zip \
    git+https://github.com/carshadi/aicsimageio.git@feature/zarrwriter-multiscales \
    git+https://github.com/AllenInstitute/argschema.git \
    mpi4py
fi

# If an error exists, it is mostly because mpi4py needs a linux library
status=$?

if ! (exit $status)
then
    echo "- mpi4py could not be installed."
    echo "- Trying installing some libraries..."

    sudo apt-get install -y libopenmpi-dev

    if (( $python_version == 1 ))
    then
        python -m pip install mpi4py
    else
        python3 -m pip install mpi4py
    fi
fi

# Moving all used tar.gz to terastitcher_installers 
mkdir terastitcher_installers
mv zulu*.tar.gz openmpi*tar.gz apache*.tar.gz terastitcher_installers/

# Autopep parastitcher python code
pip install --no-input --upgrade autopep8
autopep8 -i $PWD/TeraStitcher-portable-1.11.10-with-BF-Linux/pyscripts/*.py