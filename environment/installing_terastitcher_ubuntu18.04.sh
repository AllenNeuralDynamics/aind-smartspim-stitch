# From the run file
# chmod +x installing_terastitcher.sh

# Installing gcc compiler for openmpi
# apt-get install -y build-essential

# Installing make
# apt-get install -y make

echo "- Starting terastitcher installation with MPI"

chmod +x /home/terastitcher_openmpi_setup.sh

./home/terastitcher_openmpi_setup.sh

python_cmd=1
python -V
status=$?

if ! (exit $status)
then
    python_version=2
    apt install -y python3-pip
    python3 -m pip install mpi4py
else
    apt install -y python-pip
    python -m pip install mpi4py
fi

status=$?

if ! (exit $status)
then
    echo "- mpi4py could not be installed."
    echo "- Trying installing libopenmpi first..."

    apt-get install -y libopenmpi-dev

    if (( $python_version == 1 ))
    then
        python -m pip install mpi4py
    else
        python3 -m pip install mpi4py
    fi
fi