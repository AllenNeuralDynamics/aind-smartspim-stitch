"""
Run capsule for TeraStitcher-based image stitching.
"""

import os
from pathlib import Path

from aind_smartspim_stitch.algorithms import terastitcher
from aind_smartspim_stitch.params import get_yaml
from aind_smartspim_stitch.utils import utils


def run():
    """Function to start image stitching with TeraStitcher"""

    # Absolute paths of common Code Ocean folders
    data_folder = os.path.abspath("../data")
    results_folder = os.path.abspath("../results")

    required_input_elements = [
        f"{data_folder}/processing_manifest.json",
        f"{data_folder}/data_description.json",
        f"{data_folder}/acquisition.json",
    ]

    missing_files = utils.validate_capsule_inputs(required_input_elements)

    if len(missing_files):
        raise ValueError(f"We miss the following files in the capsule input: {missing_files}")

    pipeline_config, smartspim_dataset_name, acquisition_dict = utils.get_data_config(
        data_folder=data_folder,
        processing_manifest_path="processing_manifest.json",
        data_description_path="data_description.json",
        acquisition_path="acquisition.json",
    )
    pipeline_config = pipeline_config["pipeline_processing"]

    default_config = get_yaml(
        os.path.abspath("./aind_smartspim_stitch/params/default_terastitcher_config.yaml")
    )

    smartspim_config = utils.set_up_pipeline_parameters(
        pipeline_config=pipeline_config,
        default_config=default_config,
        acquisition_config=acquisition_dict,
    )

    # Set required path parameters for TeraStitcher
    smartspim_config["name"] = smartspim_dataset_name
    smartspim_config["input_data"] = str(data_folder)
    smartspim_config["output_data"] = str(results_folder)
    smartspim_config["preprocessed_data"] = str(results_folder)
    smartspim_config["metadata_folder"] = str(Path(results_folder) / "metadata")

    terastitcher.main(smartspim_config=smartspim_config)


if __name__ == "__main__":
    run()
