#!/bin/bash

# -------------------------------------------------------------
# This script executes terastitcher pipeline and outputs files in tiff 2DSeries 
#
# The parameters are:
# (str): path where the unstitched data is located.
# (str): path where the output of terastitcher will be located.
#
# Example:
# $ .terastitcher_parallel.sh path/to/input/data path/to/output/data
#
# The xmls files will be saved in the output_path/xmls.
#
# -------------------------------------------------------------

if [ -z "$1" ]
    then
        echo "Plase, introduce a source path."
        exit 1
fi

if [ -d "$2" ]; then
    echo "'$2' found and it is not necessary to create it..."
else
    mkdir $2
    mkdir $2/xmls
    echo "Created directory in path: '$2'"
fi

echo "- Starting import step..."

# mount gsc bucket gcsfuse --implicit-dirs --rename-dir-limit=100 --disable-http2 --max-conns-per-host=100 aind-transfer-service-test /home/jupyter/external_bucket
# unmount gsc folder fusermount -u /home/shared/local_folder/

# Important things in the import step
# reference system and voxel size per axis
time terastitcher --import --volin=$1 --projout="$2"/xmls/xml_import.xml --ref1=X --ref2=Y --ref3=Z --vxl1=1.8 --vxl2=1.8 --vxl3=2 --sparse_data

# Important things in the align step
# 1. We can go to a memory allocation error, then we need to decrease the amount of workers or adjust the --subvoldim parameter
# 2. For the subvoldim parameter, if you increase it, you use less memory.
# 3. the parameter in MPI --use-hwthread-cpus parameter is used to automatically estimate the number of hardware thread in 
# each core
# 4. If the command goes into a not enough slot error, it could be that the default configuration for available slots is 1.
# This could be change creating a hostfile with the following 'localhost slots=70'
# 5. In order that the computation does not go into tile level, the following formula helps you estimate the best parameters for this command:
# ceil(D, subvoldim) > 2 (num_proc - 1)
# For example, if I want to use subvoldim=100 then the best value for processes would be 20.
# 42 > 38
time mpirun --hostfile mpi_hostfile --use-hwthread-cpus --allow-run-as-root -np 20 python ~/parastitcher.py -2 --projin="$2"/xmls/xml_import.xml --projout="$2"/xmls/xml_displcomp_par2.xml --subvoldim=100

echo "- Starting projection step..."

# The projection step does not require a lot of computational resources
time terastitcher --displproj --projin="$2"/xmls/xml_displcomp_par2.xml --projout="$2"/xmls/xml_displproj.xml

echo "- Starting threshold step..."

# The threshold step does not require a lot of computational resources
time terastitcher --displthres --projin="$2"/xmls/xml_displproj.xml --projout="$2"/xmls/xml_displthres.xml --threshold=0.7

echo "- Starting placing step..."

# The placing step does not require a lot of computational resources
time terastitcher --placetiles --projin="$2"/xmls/xml_displthres.xml --projout="$2"/xmls/xml_merging.xml

echo "- Starting merging step..."

# Important things in the merging step
# 1. Just as in the align step, the following parameters control the partitioning in the threads and the end image 
# slice size: --slicewidth, --sliceheight, --slicedepth
# 2. To estimate the required memory for the computation, we can, before execution, add --info parameter and check 
# at the .txt file with the info
time mpirun --hostfile mpi_hostfile --use-hwthread-cpus -np 20 --allow-run-as-root python ~/parastitcher.py -6 --projin="$2"/xmls/xml_merging.xml --volout="$2" --slicewidth=20000 --sliceheight=20000 > "$2"/xmls/step6par.txt