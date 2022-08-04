#!/usr/bin/env bash

tar -xzf /home/openmpi-4.1.4.tar.gz 

cd /home/openmpi-4.1.4

./configure --prefix=/usr/local/openmpi

make all

sudo make install 