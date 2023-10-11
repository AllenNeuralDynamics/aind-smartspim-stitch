"""
Utility functions
"""
import json
import logging
import os
import re
import shutil
import subprocess
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from aind_data_schema import DerivedDataDescription
from aind_data_schema.base import AindCoreModel
from aind_data_schema.data_description import Institution, Modality, RawDataDescription
from aind_data_schema.processing import DataProcess, PipelineProcess, Processing

# IO types
PathLike = Union[str, Path]


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


def gscfuse_mount(bucket_name: PathLike, params: dict) -> None:
    """
    Mounts a bucket in a GCP Virtual Machine using GCSFUSE.

    Parameters
    ------------------------

    bucket_name: str
        Name of the bucket.

    params: dict
        Dictionary with the GCSFUSE params.

    """

    built_params = helper_build_param_value_command(params, equal_con=False)
    additional_params = helper_additional_params_command(params["additional_params"])

    gfuse_cmd = f"""gcsfuse {additional_params}
     {built_params} {bucket_name} {bucket_name}"""

    for out in execute_command_helper(gfuse_cmd, True):
        print(out)


def gscfuse_unmount(mount_dir: PathLike) -> None:
    """
    Unmounts a bucket in a VM's local folder.

    Parameters
    ------------------------

    bucket_name: str
        Name of the bucket.

    """

    fuser_cmd = f"fusermount -u {mount_dir}"

    for out in execute_command_helper(fuser_cmd, True):
        print(out)


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


def get_deepest_dirpath(folder: PathLike, ignore_folders: List[str] = ["metadata"]) -> PathLike:
    """
    Returns the deepest folder path in the provided folder.

    Parameters
    ------------------------

    folder: PathLike
        Path where the search will be carried out.

    ignore_folders: List[str]
        List of folders that need to be ignored

    Returns
    ------------------------

    PathLike:
        Path of the deepest directory
    """

    deepest_path = None
    deep_val = 0

    for root, dirs, files in os.walk(folder, topdown=False):
        if any(ignore_folder in root for ignore_folder in ignore_folders):
            continue

        for foldername in dirs:
            tmp_path = os.path.join(root, foldername)
            if tmp_path.count(os.path.sep) > deep_val and not any(
                ignore_folder in foldername for ignore_folder in ignore_folders
            ):
                deepest_path = tmp_path
                deep_val = tmp_path.count(os.path.sep)

    return Path(deepest_path)


def generate_data_description(
    raw_data_description_path: PathLike,
    dest_data_description: PathLike,
    process_name: str = "stitched",
) -> None:
    """
    Generates data description for the output folder.

    Parameters
    ------------------------

    raw_data_description_path: PathLike
        Path where the data description file is located.

    dest_data_description: PathLike
        Path where the new data description will be placed.

    process_name: str
        Process name of the new dataset

    """

    f = open(raw_data_description_path, "r")
    data = json.load(f)
    del data["name"]

    dt = datetime.now()
    data["schema_version"] = "0.7.1"
    data["modality"] = [Modality.SPIM]
    data["experiment_type"] = "SmartSPIM"

    institution = data["institution"]
    if isinstance(data["institution"], dict) and "abbreviation" in data["institution"]:
        institution = data["institution"]["abbreviation"]

    data["institution"] = Institution[institution]
    data["investigators"] = []
    data = RawDataDescription(**data)

    derived = DerivedDataDescription(
        input_data_name=data.name,
        process_name=process_name,
        creation_date=dt.date(),
        creation_time=dt.time(),
        institution=data.institution,
        funding_source=data.funding_source,
        modality=data.modality,
        experiment_type=data.experiment_type,
        subject_id=data.subject_id,
        investigators=data.investigators,
    )

    derived.write_standard_file(output_directory=dest_data_description)


def generate_processing(
    data_processes: List[DataProcess],
    dest_processing: PathLike,
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
        note="Metadata for fusion step",
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


def wavelength_to_hex(wavelength: int) -> int:
    """
    Converts wavelength to corresponding color hex value.

    Parameters
    ------------------------
    wavelength: int
        Integer value representing wavelength.

    Returns
    ------------------------
    int:
        Hex value color.
    """
    # Each wavelength key is the upper bound to a wavelgnth band.
    # Wavelengths range from 380-750nm.
    # Color map wavelength/hex pairs are generated by sampling along a CIE diagram arc.
    color_map = {
        460: 0x690AFE,  # Purple
        470: 0x3F2EFE,  # Blue-Purple
        480: 0x4B90FE,  # Blue
        490: 0x59D5F8,  # Blue-Green
        500: 0x5DF8D6,  # Green
        520: 0x5AFEB8,  # Green
        540: 0x58FEA1,  # Green
        560: 0x51FF1E,  # Green
        565: 0xBBFB01,  # Green-Yellow
        575: 0xE9EC02,  # Yellow
        580: 0xF5C503,  # Yellow-Orange
        590: 0xF39107,  # Orange
        600: 0xF15211,  # Orange-Red
        620: 0xF0121E,  # Red
        750: 0xF00050,
    }  # Pink

    for ub, hex_val in color_map.items():
        if wavelength < ub:  # Exclusive
            return hex_val
    return hex_val  # hex_val is set to the last color in for loop


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

    metadata_folder = output_alignment_path.joinpath(f"metadata/stitch_{channel_name}")

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
    LOGS_FILE = f"{output_log_path}/fusion_log_{CURR_DATE_TIME}.log"

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
