# aind-smartspim-stitch
---

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![Programming Languages](https://img.shields.io/github/languages/count/AllenNeuralDynamics/aind-smartspim-stitch)](https://github.com/AllenNeuralDynamics/aind-smartspim-stitch)
![Code Style](https://img.shields.io/badge/code%20style-black-black)

Destriping, stitching and OMEZarr conversion module for lightsheet microscopic images using  [Pystripe](https://github.com/chunglabmit/pystripe), [TeraStitcher](https://github.com/abria/TeraStitcher) and [Aicsimageio](https://github.com/AllenCellModeling/aicsimageio). This module currently supports Windows and Linux OS, it can also be configured to work in a VM in GCloud using GSCFUSE but it's mainly intended to work with Code Ocean and AWS.

## Installation on GCP VertexAI
You need to create a User-Managed Notebook with `Python 3`. If you want to add GPU processing, please select `Python 3 (CUDA Toolkit 11.0)` or with a superior available Toolkit. We recommend a GPU P100 device or Titan V(olta) if available.

Prior to the installation, download the lastest version of TeraStitcher with support for BioFormats from its website. As today, the most recent version this is the most recent version of [TeraStitcher with BF](https://unicampus365-my.sharepoint.com/:u:/g/personal/g_iannello_unicampus_it/EeOijoWwL9xBheoGjty4KU4BlsV_x1iJk7hJTc38OFwWsg?e=Kjbfr4).

Create a new conda environment and execute the following commands:

```
$ conda create -n stitching -y python=3.8
$ conda activate stitching
$ conda install -y ipykernel
```

Then, open a terminal and clone this repository in the virtual machine and place the dowloaded `.tar.gz` file in the following path: `aind-smartspim-stitch/environment/GCloud/`. Now, execute the installer (this will take a couple of minutes):

```
$ cd aind-smartspim-stitch/environment/GCloud/
$ chmod +x terastitcher_installer.sh
$ ./terastitcher_installer.sh
```

You should be able to run terastitcher and pystripe with MPI. To verify the installation close the current terminal and execute the following commands in a new one:

```
$ terastitcher -h
```

```
$ pystripe -h
```

```
$ mpirun -V
```

## Local installation
Follow the steps provided [here](https://github.com/abria/TeraStitcher/wiki/Download-and-install) to install TeraStitcher for your platform. Please, install the command-line version with BioFormats support [Link](https://github.com/abria/TeraStitcher/wiki/Binary-packages#terastitcher-portable-with-support-for-bioformats-command-line-version).

After TeraStitcher is correctly installed in your platform, create a new conda environment and execute the following commands:

```
$ conda create -n stitch -y python=3.8
$ conda activate stitch
```

Then, clone the repository and install the packages using pyproject.toml file.

```
$ git clone https://github.com/AllenNeuralDynamics/aind-smartspim-stitch
$ cd aind-smartspim-stitch
$ pip install -e .
```

You can also use the **Dockerfile** placed inside the `environment` folder for installation purposes.

Afterwards, set --pyscripts_path parameter to the directory where Parastitcher and paraconverter python scripts are. You can do this in the command line or directly in the code. It should be located in `YOUR_DIRECTORY/TeraStitcher-portable-1.11.10-with-BF-Linux/pyscripts`. (place both scripts there).

Additionally, there are some known bugs related to the input parameters in terastitcher's python scripts. Some of them were solved in **https://github.com/camilolaiton/TeraStitcher/tree/fix/data_paths**. You might want to update **parastitcher.py** and **paraconverter.py** files with the ones provided there. It is important to mention that you might want to fix python format problems before using these scripts. You can do that by using the following command:
```
$ pip install --upgrade autopep8
$ autopep8 -i YOUR_DIRECTORY/TeraStitcher-portable-1.11.10-with-BF-Linux/pyscripts/*.py
```

## Parameters
This module uses the following principal parameters:
- --input_data: Path where the data is located. If it's located in a GCS bucket, please refer to it such as: `gs://bucket-name/dataset_name/image_folder`.
- --preprocessed_data: Path where the intermediate files will be stored. This folder is deleted if the **--clean_output** flag is set to True. If it's located in a GCS bucket, please refer to it such as: `gs://bucket-name/dataset_name/intermediate_folder_name`.
- --output_data: Output path. If it's located in a GCS bucket, please refer to it such as: `gs://bucket-name/output_path`.
- --pyscripts_path: Path where the python parallel scripts are stored (Parastitcher.py and paraconverter.py).

By default, this modules loads the parameters used in the stitching process based on the GUI software developed in lab 440. To check the default values, please execute:
```
$ python terastitcher.py --help
```

This will be helpful if you want to overwrite them. 

**Note**: It is important to mention that this module uses different tools that output processed images. For erasing streaks in EM images, we are using pystripe which outputs the same dataset folder structure with the processed images. Terastitcher does the same but with a different folder structure and finally we convert the stitched images to OME-Zarr. Therefore, **you need 3X disk space based on your original disk size**. By default, we have a flag `clean_output` which is set to True. When this flag is set to True, it deletes the intermediate images generated by this module leaving only the OME-Zarr output.

If you run this pipeline in GCP, this module will automatically load the bucket(s) using GCSFUSE. As today, there are some performance limitations when using GCSFUSE with MPI if your dataset contains many small images. The performance will improve if you have larger files containing these small images (e.g. 3D stacks of images).

### Configuration file
Since TeraStitcher has multiple steps, we created a configuration dictionary with the parameters for each step. You can add more parameters based on terastitcher documentation. You can check the default parameters in `aind-smartspim-stitch/code/params.py`. For example:

```
"import_data" : {
    // Parameters that have values
    "ref1":"H",
    "ref2":"V",
    "ref3":"D",
    "vxl1":"1.800",
    "vxl2":"1.800",
    "vxl3":"2",
    // Flag parameters
    "additional_params": [
        "sparse_data",
        "libtiff_uncompress",
        'rescan'
    ]
}
```

The align and merge steps are computationally expensive. Therefore, we added cpu parameters for executing these steps. For example:

```
"align" : {
    "cpu_params": {
        // Set estimate processes True if you are not sure what's the best number of workers for your CPU or the image depth you have
        "estimate_processes": false,
        "image_depth": 4200,
        "number_processes": 16, //16 cores
        // Hostfile for mpirun, it is not considered on Windows OS
        "hostfile": "/home/jupyter/aind-smartspim-stitch/environment/GCloud/hostfile",
        // Additional mpi params, these work only on Linux based distributions 
        "additional_params": [
            "use-hwthread-cpus",
            "allow-run-as-root"
        ]
    },
    // terastitcher's align parameters
    "subvoldim": 100
}
```

You can set the new parameters by the command line or modifying the `default_config.yaml` placed in `code/aind_smartspim_stitch`. It is worth mentioning that these parameters work well with our SmartSPIM datasets and our machine configuration (**Ubuntu 20.04, 16 cores and 128 GB RAM**).

### Execution example
In a local machine:
```
$ cd ~
$ python aind-smartspim-stitch/code/terastitcher.py --input_data path/to/dataset --preprocessed_data path/to/intermediate/data --output_data path/to/output/zarr
```

In a code ocean capsule:
```
$ cd ~
$ python main.py --input_data path/to/dataset --preprocessed_data path/to/intermediate/data --output_data path/to/output/zarr
```

In VertexAI:
```
$ cd ~
$ python main.py --input_data gs://bucket-name/dataset/images --preprocessed_data gs://bucket-name/path/to/intermediate/data --output_data gs://bucket-name/dataset_stitched
```

## Documentation
You can access the documentation for this module [here]().

## TeraStitcher Documentation
You can download TeraStitcher documentation from [here](https://unicampus365-my.sharepoint.com/:b:/g/personal/g_iannello_unicampus_it/EYT9KbapjBdGvTAD2_MdbKgB5gY_h9rlvHzqp6mUNqVhIw?e=s8GrFC)

## Pystripe Documentation
You can access Pystripe documentation from [here](https://github.com/chunglabmit/pystripe)

## Contributing

To develop the code, run
```
pip install -e .[dev]
```

### Linters and testing

There are several libraries used to run linters, check documentation, and run tests.

- Please test your changes using the **coverage** library, which will run the tests and log a coverage report:

```
coverage run -m unittest discover && coverage report
```

- Use **interrogate** to check that modules, methods, etc. have been documented thoroughly:

```
interrogate .
```

- Use **flake8** to check that code is up to standards (no unused imports, etc.):
```
flake8 .
```

- Use **black** to automatically format the code into PEP standards:
```
black .
```

- Use **isort** to automatically sort import statements:
```
isort .
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repo and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect the build system or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bug fix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests

### Documentation
To generate the rst files source files for documentation, run
```
sphinx-apidoc -o doc_template/source/ src 
```
Then to create the documentation html files, run
```
sphinx-build -b html doc_template/source/ doc_template/build/html
```
More info on sphinx installation can be found here: https://www.sphinx-doc.org/en/master/usage/installation.html