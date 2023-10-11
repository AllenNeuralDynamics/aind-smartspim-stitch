""" top level run script """
import os
from pathlib import Path
from typing import List, Tuple

from aind_smartspim_stitch import stitch
from aind_smartspim_stitch._shared.types import PathLike
from aind_smartspim_stitch.params import get_yaml
from aind_smartspim_stitch.utils import utils


def get_data_config(
    data_folder: PathLike,
    processing_manifest_path: str = "processing_manifest.json",
    data_description_path: str = "data_description.json",
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

    derivatives_dict = utils.read_json_as_dict(str(processing_manifest_path))
    data_description_dict = utils.read_json_as_dict(str(data_description_path))

    smartspim_dataset = data_description_dict["name"]

    return derivatives_dict, smartspim_dataset


def set_up_pipeline_parameters(pipeline_config: dict, default_config: dict):
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

    def get_resolution_from_array(resolutions: list, axis_name: str) -> float:
        """
        Gets the resolution from a list of dict.
        This is based on the processing manifest json.

        Parameters
        resolutions: List[dict]
            List with dictionaries that have
            the resolution and axis name
        """

        axis_size = None
        axis_name = axis_name.casefold()

        for resolution in resolutions:
            if axis_name == resolution["axis_name"].casefold():
                axis_size = resolution["resolution"]
                break

        return axis_size

    default_config["import_data"]["vxl1"] = get_resolution_from_array(
        resolutions=pipeline_config["stitching"]["resolution"], axis_name="x"
    )
    default_config["import_data"]["vxl2"] = get_resolution_from_array(
        resolutions=pipeline_config["stitching"]["resolution"], axis_name="y"
    )
    default_config["import_data"]["vxl3"] = get_resolution_from_array(
        resolutions=pipeline_config["stitching"]["resolution"], axis_name="z"
    )

    dict_cpus = pipeline_config["stitching"].get("cpus")
    cpus = 16 if dict_cpus is None else dict_cpus

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


def run():
    """Function to start image fusion"""

    # Absolute paths of common Code Ocean folders
    data_folder = os.path.abspath("../data")
    results_folder = os.path.abspath("../results")
    # scratch_folder = os.path.abspath("../scratch")

    # It is assumed that these files
    # will be in the data folder
    required_input_elements = [
        f"{data_folder}/processing_manifest.json",
    ]

    missing_files = validate_capsule_inputs(required_input_elements)

    if len(missing_files):
        raise ValueError(f"We miss the following files in the capsule input: {missing_files}")

    pipeline_config, smartspim_dataset_name = get_data_config(data_folder=data_folder)
    pipeline_config = pipeline_config["pipeline_processing"]

    default_config = get_yaml(
        os.path.abspath("./aind_smartspim_stitch/params/default_terastitcher_config.yaml")
    )

    smartspim_config = set_up_pipeline_parameters(
        pipeline_config=pipeline_config, default_config=default_config
    )

    smartspim_config["name"] = smartspim_dataset_name

    stitch.main(
        data_folder=data_folder, output_alignment_path=results_folder, smartspim_config=smartspim_config
    )


if __name__ == "__main__":
    run()
