#!/usr/bin/env bash

wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.4.tar.gz

tar -xzf openmpi-4.1.4.tar.gz 

cd openmpi-4.1.4

./configure --prefix=/usr/local/openmpi

make all

sudo make install

# Creating hostfile
echo "localhost slots=70" > $PWD/hostfile