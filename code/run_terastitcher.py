"""
Run capsule for TeraStitcher-based image stitching.
"""

import logging
import os
import time
from pathlib import Path

from aind_smartspim_stitch import __pipeline_name__, __title__, __version__
from aind_smartspim_stitch.algorithms import terastitcher
from aind_smartspim_stitch.params import get_yaml
from aind_smartspim_stitch.utils import utils
from schlog import setup_logging

logger = logging.getLogger(__name__)


def run():
    """Function to start image stitching with TeraStitcher"""

    # Absolute paths of common Code Ocean folders
    data_folder = os.path.abspath("../data")
    results_folder = os.path.abspath("../results")

    process_name = f"{__title__}-terastitcher"
    setup_logging(
        model={
            "pipeline_name": __pipeline_name__,
            "process_name": process_name,
            "software_name": __title__,
            "software_version": __version__,
        }
    )

    start_time = time.monotonic()
    dataset_name = None

    try:
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
        dataset_name = smartspim_dataset_name
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

        logger.info(
            "TeraStitcher stitching started",
            extra={
                "event_type": "stage_start",
                "dataset_name": dataset_name,
                "input_data": smartspim_config["input_data"],
                "output_data": smartspim_config["output_data"],
            },
        )

        terastitcher.main(smartspim_config=smartspim_config)

        duration_seconds = round(time.monotonic() - start_time, 3)
        logger.info(
            "TeraStitcher stitching completed",
            extra={
                "event_type": "stage_complete",
                "dataset_name": dataset_name,
                "duration_seconds": duration_seconds,
            },
        )
    except Exception:
        duration_seconds = round(time.monotonic() - start_time, 3)
        logger.error(
            "TeraStitcher stitching failed",
            exc_info=True,
            extra={
                "event_type": "stage_failure",
                "dataset_name": dataset_name,
                "duration_seconds": duration_seconds,
            },
        )
        raise


if __name__ == "__main__":
    run()
