"""
Utility functions
"""

import json
import logging
import multiprocessing
import os
import platform
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import psutil
from aind_data_schema.base import AindCoreModel
from aind_data_schema.core.processing import DataProcess, PipelineProcess, Processing

# IO types
PathLike = Union[str, Path]


def get_code_ocean_cpu_limit():
    """
    Gets the Code Ocean capsule CPU limit

    Returns
    -------
    int:
        number of cores available for compute
    """
    # Checks for environmental variables
    co_cpus = os.environ.get("CO_CPUS")
    aws_batch_job_id = os.environ.get("AWS_BATCH_JOB_ID")

    if co_cpus:
        return co_cpus
    if aws_batch_job_id:
        return 1

    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fp:
            cfs_quota_us = int(fp.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
            cfs_period_us = int(fp.read())

        container_cpus = cfs_quota_us // cfs_period_us

    except FileNotFoundError as e:
        container_cpus = 0

    # For physical machine, the `cfs_quota_us` could be '-1'
    return psutil.cpu_count(logical=False) if container_cpus < 1 else container_cpus


def create_folder(dest_dir: PathLike, verbose: Optional[bool] = False) -> None:
    """
    Create new folders.

    Parameters
    ------------------------

    dest_dir: PathLike
        Path where the folder will be created if it does not exist.

    verbose: Optional[bool]
        If we want to show information about the folder status. Default False.

    Raises
    ------------------------

    OSError:
        if the folder exists.

    """

    if not (os.path.exists(dest_dir)):
        try:
            if verbose:
                print(f"Creating new directory: {dest_dir}")
            os.makedirs(dest_dir)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise


def delete_folder(dest_dir: PathLike, verbose: Optional[bool] = False) -> None:
    """
    Delete a folder path.

    Parameters
    ------------------------

    dest_dir: PathLike
        Path that will be removed.

    verbose: Optional[bool]
        If we want to show information about the folder status. Default False.

    Raises
    ------------------------

    shutil.Error:
        If the folder could not be removed.

    """
    if os.path.exists(dest_dir):
        try:
            shutil.rmtree(dest_dir)
            if verbose:
                print(f"Folder {dest_dir} was removed!")
        except shutil.Error as e:
            print(f"Folder could not be removed! Error {e}")


def execute_command_helper(
    command: str,
    print_command: bool = False,
    stdout_log_file: Optional[PathLike] = None,
) -> None:
    """
    Execute a shell command.

    Parameters
    ------------------------

    command: str
        Command that we want to execute.
    print_command: bool
        Bool that dictates if we print the command in the console.

    Raises
    ------------------------

    CalledProcessError:
        if the command could not be executed (Returned non-zero status).

    """

    if print_command:
        print(command)

    if stdout_log_file and len(str(stdout_log_file)):
        save_string_to_txt("$ " + command, stdout_log_file, "a")

    popen = subprocess.Popen(command, stdout=subprocess.PIPE, universal_newlines=True, shell=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield str(stdout_line).strip()
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def execute_command(command: str, logger: logging.Logger, verbose: Optional[bool] = False):
    """
    Execute a shell command with a given configuration.

    Parameters
    ------------------------
    command: str
        Command that we want to execute.

    logger: logging.Logger
        Logger object

    verbose: Optional[bool]
        Prints the command in the console

    Raises
    ------------------------
    CalledProcessError:
        if the command could not be executed (Returned non-zero status).

    """
    for out in execute_command_helper(command, verbose):
        if len(out):
            logger.info(out)


def check_path_instance(obj: object) -> bool:
    """
    Checks if an objects belongs to pathlib.Path subclasses.

    Parameters
    ------------------------

    obj: object
        Object that wants to be validated.

    Returns
    ------------------------

    bool:
        True if the object is an instance of Path subclass, False otherwise.
    """

    for childclass in Path.__subclasses__():
        if isinstance(obj, childclass):
            return True

    return False


def save_dict_as_json(filename: str, dictionary: dict, verbose: Optional[bool] = False) -> None:
    """
    Saves a dictionary as a json file.

    Parameters
    ------------------------

    filename: str
        Name of the json file.

    dictionary: dict
        Dictionary that will be saved as json.

    verbose: Optional[bool]
        True if you want to print the path where the file was saved.

    """

    if dictionary is None:
        dictionary = {}

    else:
        for key, value in dictionary.items():
            # Converting path to str to dump dictionary into json
            if check_path_instance(value):
                # TODO fix the \\ encode problem in dump
                dictionary[key] = str(value)

    with open(filename, "w") as json_file:
        json.dump(dictionary, json_file, indent=4)

    if verbose:
        print(f"- Json file saved: {filename}")


def read_json_as_dict(filepath: str) -> dict:
    """
    Reads a json as dictionary.

    Parameters
    ------------------------

    filepath: PathLike
        Path where the json is located.

    Returns
    ------------------------

    dict:
        Dictionary with the data the json has.

    """

    dictionary = {}

    if os.path.exists(filepath):
        with open(filepath) as json_file:
            dictionary = json.load(json_file)

    return dictionary


def helper_build_param_value_command(params: dict, equal_con: Optional[bool] = True) -> str:
    """
    Helper function to build a command based on key:value pairs.

    Parameters
    ------------------------

    params: dict
        Dictionary with key:value pairs used for building the command.

    equal_con: Optional[bool]
        Indicates if the parameter is followed by '='. Default True.

    Returns
    ------------------------

    str:
        String with the parameters.

    """
    equal = " "
    if equal_con:
        equal = "="

    parameters = ""
    for param, value in params.items():
        if type(value) in [str, float, int] or check_path_instance(value):
            parameters += f"--{param}{equal}{str(value)} "

    return parameters


def helper_additional_params_command(params: List[str]) -> str:
    """
    Helper function to build a command based on values.

    Parameters
    ------------------------

    params: list
        List with additional command values used.

    Returns
    ------------------------

    str:
        String with the parameters.

    """
    additional_params = ""
    for param in params:
        additional_params += f"--{param} "

    return additional_params


def save_string_to_txt(txt: str, filepath: PathLike, mode="w") -> None:
    """
    Saves a text in a file in the given mode.

    Parameters
    ------------------------

    txt: str
        String to be saved.

    filepath: PathLike
        Path where the file is located or will be saved.

    mode: str
        File open mode.

    """

    with open(filepath, mode) as file:
        file.write(txt + "\n")


def generate_processing(
    data_processes: List[DataProcess],
    dest_processing: str,
    processor_full_name: str,
    pipeline_version: str,
):
    """
    Generates data description for the output folder.

    Parameters
    ------------------------

    data_processes: List[dict]
        List with the processes aplied in the pipeline.

    dest_processing: PathLike
        Path where the processing file will be placed.

    processor_full_name: str
        Person in charged of running the pipeline
        for this data asset

    pipeline_version: str
        Terastitcher pipeline version

    """
    # flake8: noqa: E501
    processing_pipeline = PipelineProcess(
        data_processes=data_processes,
        processor_full_name=processor_full_name,
        pipeline_version=pipeline_version,
        pipeline_url="https://github.com/AllenNeuralDynamics/aind-smartspim-pipeline",
        note="Metadata for the stitching step, it does not include stitching compute time.",
    )

    processing = Processing(
        processing_pipeline=processing_pipeline,
        notes="This processing only contains metadata about fusion \
            and needs to be compiled with other steps at the end",
    )

    processing.write_standard_file(output_directory=dest_processing)


def check_type_helper(value: Any, val_type: type) -> bool:
    """
    Checks if a value belongs to a specific type.

    Parameters
    ------------------------

    value: Any
        variable data.

    val_type: type
        Type that we want to check.

    Returns
    ------------------------

    bool:
        True if the type is what we expect
        from the variable data, False otherwise.
    """

    if type(value) != val_type:
        return False

    return True


def generate_timestamp(time_format: str = "%Y-%m-%d_%H-%M-%S") -> str:
    """
    Generates a timestamp in string format.

    Parameters
    ------------------------
    time_format: str
        String following the conventions
        to generate the timestamp (https://strftime.org/).

    Returns
    ------------------------
    str:
        String with the actual datetime
        moment in string format.
    """
    return datetime.now().strftime(time_format)


def copy_file(input_filename: PathLike, output_filename: PathLike):
    """
    Copies a file to an output path

    Parameters
    ----------
    input_filename: PathLike
        Path where the file is located

    output_filename: PathLike
        Path where the file will be copied
    """

    try:
        shutil.copy(input_filename, output_filename)

    except shutil.SameFileError:
        raise shutil.SameFileError(f"The filename {input_filename} already exists in the output path.")

    except PermissionError:
        raise PermissionError(
            f"Not able to copy the file. Please, check the permissions in the output path."
        )


def copy_available_metadata(
    input_path: PathLike, output_path: PathLike, ignore_files: List[str]
) -> List[PathLike]:
    """
    Copies all the valid metadata from the aind-data-schema
    repository that exists in a given path.

    Parameters
    -----------
    input_path: PathLike
        Path where the metadata is located

    output_path: PathLike
        Path where we will copy the found
        metadata

    ignore_files: List[str]
        List with the filenames of the metadata
        that we need to ignore from the aind-data-schema

    Returns
    --------
    List[PathLike]
        List with the metadata files that
        were copied
    """

    # We get all the valid filenames from the aind core model
    metadata_to_find = [cls.default_filename() for cls in AindCoreModel.__subclasses__()]

    # Making sure the paths are pathlib objects
    input_path = Path(input_path)
    output_path = Path(output_path)

    found_metadata = []

    for metadata_filename in metadata_to_find:
        metadata_filename = input_path.joinpath(metadata_filename)

        if metadata_filename.exists() and metadata_filename.name not in ignore_files:
            found_metadata.append(metadata_filename)

            # Copying file to output path
            output_filename = output_path.joinpath(metadata_filename.name)
            copy_file(metadata_filename, output_filename)

    return found_metadata


def create_align_folder_structure(output_alignment_path: PathLike, channel_name: str) -> str:
    """
    Creates the stitch folder structure.

    Parameters
    -----------
    output_alignment_path: PathLike
        Path where we will generate
        the XMLs for the image
        transformations

    channel_name: str
        SmartSPIM channel name

    Returns
    -----------
    str
        Path to the metadata folder
    """

    # Creating folders if necessary
    if not output_alignment_path.exists():
        logging.info(f"Path {output_alignment_path} does not exists. We're creating one.")
        create_folder(dest_dir=output_alignment_path)

    output_alignment_path = output_alignment_path.joinpath(f"stitch_{channel_name}")
    metadata_folder = output_alignment_path.joinpath(f"metadata")

    create_folder(metadata_folder)

    return metadata_folder


def create_logger(output_log_path: PathLike) -> logging.Logger:
    """
    Creates a logger that generates
    output logs to a specific path.

    Parameters
    ------------
    output_log_path: PathLike
        Path where the log is going
        to be stored

    Returns
    -----------
    logging.Logger
        Created logger pointing to
        the file path.
    """
    CURR_DATE_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    LOGS_FILE = f"{output_log_path}/stitch_log_{CURR_DATE_TIME}.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s : %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_FILE, "a"),
        ],
        force=True,
    )

    logging.disable("DEBUG")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    return logger


def find_smartspim_channels(path: PathLike, channel_regex: str = r"Ex_([0-9]*)_Em_([0-9]*)$"):
    """
    Find image channels of a dataset using a regular expression.

    Parameters
    ------------------------

    path:PathLike
        Dataset path

    channel_regex:str
        Regular expression for filtering folders in dataset path.


    Returns
    ------------------------

    List[str]:
        List with the image channels. Empty list if
        it does not find any channels with the
        given regular expression.

    """
    return [path for path in os.listdir(path) if re.search(channel_regex, path)]


def profile_resources(
    time_points: List,
    cpu_percentages: List,
    memory_usages: List,
    monitoring_interval: int,
):
    """
    Profiles compute resources usage.

    Parameters
    ----------
    time_points: List
        List to save all the time points
        collected

    cpu_percentages: List
        List to save the cpu percentages
        during the execution

    memory_usage: List
        List to save the memory usage
        percentages during the execution

    monitoring_interval: int
        Monitoring interval in seconds
    """
    start_time = time.time()

    while True:
        current_time = time.time() - start_time
        time_points.append(current_time)

        # CPU Usage
        cpu_percent = psutil.cpu_percent(interval=monitoring_interval)
        cpu_percentages.append(cpu_percent)

        # Memory usage
        memory_info = psutil.virtual_memory()
        memory_usages.append(memory_info.percent)

        time.sleep(monitoring_interval)


def generate_resources_graphs(
    time_points: List,
    cpu_percentages: List,
    memory_usages: List,
    output_path: str,
    prefix: str,
):
    """
    Profiles compute resources usage.

    Parameters
    ----------
    time_points: List
        List to save all the time points
        collected

    cpu_percentages: List
        List to save the cpu percentages
        during the execution

    memory_usage: List
        List to save the memory usage
        percentages during the execution

    output_path: str
        Path where the image will be saved

    prefix: str
        Prefix name for the image
    """
    time_len = len(time_points)
    memory_len = len(memory_usages)
    cpu_len = len(cpu_percentages)

    min_len = min([time_len, memory_len, cpu_len])
    if not min_len:
        return

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_points[:min_len], cpu_percentages[:min_len], label="CPU Usage")
    plt.xlabel("Time (s)")
    plt.ylabel("CPU Usage (%)")
    plt.title("CPU Usage Over Time")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time_points[:min_len], memory_usages[:min_len], label="Memory Usage")
    plt.xlabel("Time (s)")
    plt.ylabel("Memory Usage (%)")
    plt.title("Memory Usage Over Time")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_path}/{prefix}_compute_resources.png", bbox_inches="tight")


def stop_child_process(process: multiprocessing.Process):
    """
    Stops a process

    Parameters
    ----------
    process: multiprocessing.Process
        Process to stop
    """
    process.terminate()
    process.join()


def get_size(bytes, suffix: str = "B") -> str:
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'

    Parameters
    ----------
    bytes: bytes
        Bytes to scale

    suffix: str
        Suffix used for the conversion
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def print_system_information(logger: logging.Logger):
    """
    Prints system information

    Parameters
    ----------
    logger: logging.Logger
        Logger object
    """
    co_memory = int(os.environ.get("CO_MEMORY"))
    # System info
    sep = "=" * 40
    logger.info(f"{sep} Code Ocean Information {sep}")
    logger.info(f"Code Ocean assigned cores: {get_code_ocean_cpu_limit()}")
    logger.info(f"Code Ocean assigned memory: {get_size(co_memory)}")
    logger.info(f"Computation ID: {os.environ.get('CO_COMPUTATION_ID')}")
    logger.info(f"Capsule ID: {os.environ.get('CO_CAPSULE_ID')}")
    logger.info(f"Is pipeline execution?: {bool(os.environ.get('AWS_BATCH_JOB_ID'))}")

    logger.info(f"{sep} System Information {sep}")
    uname = platform.uname()
    logger.info(f"System: {uname.system}")
    logger.info(f"Node Name: {uname.node}")
    logger.info(f"Release: {uname.release}")
    logger.info(f"Version: {uname.version}")
    logger.info(f"Machine: {uname.machine}")
    logger.info(f"Processor: {uname.processor}")

    # Boot info
    logger.info(f"{sep} Boot Time {sep}")
    boot_time_timestamp = psutil.boot_time()
    bt = datetime.fromtimestamp(boot_time_timestamp)
    logger.info(f"Boot Time: {bt.year}/{bt.month}/{bt.day} {bt.hour}:{bt.minute}:{bt.second}")

    # CPU info
    logger.info(f"{sep} CPU Info {sep}")
    # number of cores
    logger.info(f"Physical node cores: {psutil.cpu_count(logical=False)}")
    logger.info(f"Total node cores: {psutil.cpu_count(logical=True)}")

    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    logger.info(f"Max Frequency: {cpufreq.max:.2f}Mhz")
    logger.info(f"Min Frequency: {cpufreq.min:.2f}Mhz")
    logger.info(f"Current Frequency: {cpufreq.current:.2f}Mhz")

    # CPU usage
    logger.info("CPU Usage Per Core before processing:")
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
        logger.info(f"Core {i}: {percentage}%")
    logger.info(f"Total CPU Usage: {psutil.cpu_percent()}%")

    # Memory info
    logger.info(f"{sep} Memory Information {sep}")
    # get the memory details
    svmem = psutil.virtual_memory()
    logger.info(f"Total: {get_size(svmem.total)}")
    logger.info(f"Available: {get_size(svmem.available)}")
    logger.info(f"Used: {get_size(svmem.used)}")
    logger.info(f"Percentage: {svmem.percent}%")
    logger.info(f"{sep} Memory - SWAP {sep}")
    # get the swap memory details (if exists)
    swap = psutil.swap_memory()
    logger.info(f"Total: {get_size(swap.total)}")
    logger.info(f"Free: {get_size(swap.free)}")
    logger.info(f"Used: {get_size(swap.used)}")
    logger.info(f"Percentage: {swap.percent}%")

    # Network information
    logger.info(f"{sep} Network Information {sep}")
    # get all network interfaces (virtual and physical)
    if_addrs = psutil.net_if_addrs()
    for interface_name, interface_addresses in if_addrs.items():
        for address in interface_addresses:
            logger.info(f"=== Interface: {interface_name} ===")
            if str(address.family) == "AddressFamily.AF_INET":
                logger.info(f"  IP Address: {address.address}")
                logger.info(f"  Netmask: {address.netmask}")
                logger.info(f"  Broadcast IP: {address.broadcast}")
            elif str(address.family) == "AddressFamily.AF_PACKET":
                logger.info(f"  MAC Address: {address.address}")
                logger.info(f"  Netmask: {address.netmask}")
                logger.info(f"  Broadcast MAC: {address.broadcast}")
    # get IO statistics since boot
    net_io = psutil.net_io_counters()
    logger.info(f"Total Bytes Sent: {get_size(net_io.bytes_sent)}")
    logger.info(f"Total Bytes Received: {get_size(net_io.bytes_recv)}")


def get_data_config(
    data_folder: PathLike,
    processing_manifest_path: str = "processing_manifest.json",
    data_description_path: str = "data_description.json",
    acquisition_path: str = "acquisition.json",
) -> Tuple:
    """
    Returns the first smartspim dataset found
    in the data folder

    Parameters
    -----------
    data_folder: str
        Path to the folder that contains the data

    processing_manifest_path: str
        Path for the processing manifest

    data_description_path: str
        Path for the data description

    Returns
    -----------
    Tuple[Dict, str]
        Dict: Empty dictionary if the path does not exist,
        dictionary with the data otherwise.

        Str: Empty string if the processing manifest
        was not found
    """

    # Returning first smartspim dataset found
    # Doing this because of Code Ocean, ideally we would have
    # a single dataset in the pipeline

    processing_manifest_path = Path(f"{data_folder}/{processing_manifest_path}")
    data_description_path = Path(f"{data_folder}/{data_description_path}")

    if not processing_manifest_path.exists():
        raise ValueError(f"Please, check processing manifest path: {processing_manifest_path}")

    if not data_description_path.exists():
        raise ValueError(f"Please, check data description path: {data_description_path}")

    derivatives_dict = read_json_as_dict(str(processing_manifest_path))
    data_description_dict = read_json_as_dict(str(data_description_path))
    acquisition_dict = read_json_as_dict(f"{data_folder}/{acquisition_path}")

    smartspim_dataset = data_description_dict["name"]

    return derivatives_dict, smartspim_dataset, acquisition_dict


def set_up_pipeline_parameters(pipeline_config: dict, default_config: dict, acquisition_config: dict):
    """
    Sets up smartspim stitching parameters that come from the
    pipeline configuration

    Parameters
    -----------
    smartspim_dataset: str
        String with the smartspim dataset name

    pipeline_config: dict
        Dictionary that comes with the parameters
        for the pipeline described in the
        processing_manifest.json

    default_config: dict
        Dictionary that has all the default
        parameters to execute this capsule with
        smartspim data

    Returns
    -----------
    Dict
        Dictionary with the combined parameters
    """

    # Grabbing a tile with metadata from acquisition - we assume all dataset
    # was acquired with the same resolution
    tile_coord_transforms = acquisition_config["tiles"][0]["coordinate_transformations"]

    scale_transform = [x["scale"] for x in tile_coord_transforms if x["type"] == "scale"][0]

    x = float(scale_transform[0])
    y = float(scale_transform[1])
    z = float(scale_transform[2])

    default_config["import_data"]["vxl1"] = x
    default_config["import_data"]["vxl2"] = y
    default_config["import_data"]["vxl3"] = z

    dict_cpus = pipeline_config["stitching"].get("cpus")
    cpus = get_code_ocean_cpu_limit() if dict_cpus is None else dict_cpus

    default_config["align"]["cpu_params"]["number_processes"] = cpus
    default_config["stitching"] = pipeline_config["stitching"].copy()

    return default_config


def validate_capsule_inputs(input_elements: List[str]) -> List[str]:
    """
    Validates input elemts for a capsule in
    Code Ocean.

    Parameters
    -----------
    input_elements: List[str]
        Input elements for the capsule. This
        could be sets of files or folders.

    Returns
    -----------
    List[str]
        List of missing files
    """

    missing_inputs = []
    for required_input_element in input_elements:
        required_input_element = Path(required_input_element)

        if not required_input_element.exists():
            missing_inputs.append(str(required_input_element))

    return missing_inputs


def get_resolution(acquisition_config: dict) -> Tuple[float]:
    """
    Get the image resolution from the acquisition.json metadata

    Parameters
    ----------
    acquisition_config: dict
        Acquisition metadata

    Returns
    -------
    Tuple[float]
        Tuple with the floats for image resolution
    """

    # Grabbing a tile with metadata from acquisition - we assume all dataset
    # was acquired with the same resolution
    tile_coord_transforms = acquisition_config["tiles"][0]["coordinate_transformations"]

    scale_transform = [x["scale"] for x in tile_coord_transforms if x["type"] == "scale"][0]

    x = float(scale_transform[0])
    y = float(scale_transform[1])
    z = float(scale_transform[2])

    return x, y, z
