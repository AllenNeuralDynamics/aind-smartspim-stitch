"""
Main function to execute dataset processing
"""
import logging
import os
from pathlib import Path
from typing import Tuple

from aind_smartspim_stitch import terastitcher
from aind_smartspim_stitch.params import get_default_config
from aind_smartspim_stitch.utils import utils

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s : %(message)s",
    datefmt="%Y-%m-%d %H:%M",
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler("test.log", "a"),
    ],
)
logging.disable("DEBUG")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def set_up_pipeline_parameters(smartspim_dataset: str, pipeline_config: dict, default_config: dict):
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

    default_config["input_data"] = "../data"
    default_config["preprocessed_data"] = f"../scratch/{smartspim_dataset}"
    default_config["output_data"] = f"../results/{smartspim_dataset}"
    default_config["metadata_folder"] = "../data"
    default_config["generate_metadata"] = True

    default_config["stitch_channel"] = pipeline_config["stitching"]["channel"]
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
    default_config["merge"]["cpu_params"]["number_processes"] = cpus

    return default_config


def copy_fused_results(output_folder: str, s3_path: str, results_folder: str):
    """
    Copies the smartspim fused results to S3

    Parameters
    -----------
    output_folder: str
        Path where the results are

    s3_path: str
        Path where the results will be
        copied in S3

    results_folder: str
        Results folder where the .txt
        will be placed
    """
    for out in utils.execute_command_helper(f"aws s3 cp --recursive {output_folder} {s3_path}"):
        logger.info(out)

    utils.save_string_to_txt(
        f"Stitched dataset saved in: {s3_path}", f"{results_folder}/output_stitching.txt",
    )


def get_data_config(
    data_folder: str,
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

    derivatives_dict = utils.read_json_as_dict(f"{data_folder}/{processing_manifest_path}")
    data_description_dict = utils.read_json_as_dict(f"{data_folder}/{data_description_path}")

    smartspim_dataset = data_description_dict["name"]

    return derivatives_dict, smartspim_dataset


def main() -> None:
    """
    Main function to execute stitching
    """

    results_folder = os.path.abspath("../results")
    data_folder = os.path.abspath("../data")

    # Dataset configuration in the processing_manifest.json
    pipeline_config, smartspim_dataset = get_data_config(data_folder=data_folder)

    renamed_folder = None

    if len(pipeline_config) and len(smartspim_dataset):

        default_config = get_default_config()

        pipeline_config = pipeline_config["pipeline_processing"]
        # Setting up incoming parameters from pipeline
        smartspim_config = set_up_pipeline_parameters(smartspim_dataset, pipeline_config, default_config)

        logger.info(f"Smartspim config: {smartspim_config}")
        output_folder = terastitcher.main(smartspim_config)
        bucket_path = "aind-open-data"

        pipeline_config["stitching"]["output_folder"] = output_folder

        logger.info(f"Fused dataset in: {output_folder}")

        # Copying output to bucket
        # co_folder = output_folder.split("/")[1]
        dataset_name = output_folder.split("/")[-1]
        s3_path = f"s3://{bucket_path}/{dataset_name}"
        copy_fused_results(output_folder, s3_path, results_folder)

        # Saving processing manifest
        pipeline_config["stitching"]["s3_path"] = s3_path
        utils.save_dict_as_json(f"{results_folder}/processing_manifest.json", pipeline_config)

        # Renaming output folder to connect the pipeline
        renamed_folder = Path(output_folder).parent.joinpath("fused")
        os.rename(output_folder, renamed_folder)

    else:
        logger.error("No SmartSPIM dataset was found.")

    return renamed_folder


if __name__ == "__main__":
    main()
