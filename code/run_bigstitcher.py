"""
Run capsule for BigStitcher stitching (standalone mode, local data).
"""

import logging
import os
import time
from pathlib import Path

from aind_smartspim_stitch import __pipeline_name__, __title__, __version__
from aind_smartspim_stitch.algorithms import bigstitcher
from aind_smartspim_stitch.utils import metadata_compat, utils
from log_schema import setup_logging

logger = logging.getLogger(__name__)


def run():
    """Function that runs image stitching with BigStitcher"""
    data_folder = Path(os.path.abspath("../data"))
    results_folder = Path(os.path.abspath("../results"))

    process_name = f"{__title__}-bigstitcher"
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
        logger.info(
            "BigStitcher stitching started",
            extra={
                "event_type": "stage_start",
                "data_folder": str(data_folder),
                "results_folder": str(results_folder),
            },
        )

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
        dataset_name = metadata_compat.get_raw_dataset_name(smartspim_dataset_name)

        logger.info(
            f"Processing derived asset {smartspim_dataset_name}",
            extra={
                "event_type": "dataset_resolved",
                "dataset_name": dataset_name,
                "asset_name": smartspim_dataset_name,
            },
        )

        voxel_resolution = utils.get_resolution(acquisition_dict)
        stitching_channel = pipeline_config["pipeline_processing"]["stitching"]["channel"]

        stitching_channel_path = data_folder.joinpath(f"preprocessed_data/{stitching_channel}")

        output_json_file = results_folder.joinpath(f"{smartspim_dataset_name}_tile_metadata.json")

        logger.info(
            f"Stitching channel {stitching_channel} resolved",
            extra={
                "dataset_name": dataset_name,
                "asset_name": smartspim_dataset_name,
                "channel": stitching_channel,
                "stitching_channel_path": str(stitching_channel_path),
                "output_json_file": str(output_json_file),
            },
        )

        bigstitcher.main(
            stitching_channel_path=stitching_channel_path,
            voxel_resolution=voxel_resolution,
            output_json_file=output_json_file,
            results_folder=results_folder,
            smartspim_dataset_name=smartspim_dataset_name,
            res_for_transforms=(8.0, 7.2, 7.2),
        )

        duration_seconds = round(time.monotonic() - start_time, 3)
        logger.info(
            "BigStitcher stitching completed",
            extra={
                "event_type": "stage_complete",
                "dataset_name": dataset_name,
                "duration_seconds": duration_seconds,
            },
        )
    except Exception as e:
        duration_seconds = round(time.monotonic() - start_time, 3)
        logger.error(
            "BigStitcher stitching failed",
            exc_info=True,
            extra={
                "event_type": "stage_failure",
                "error": f"{type(e).__name__}: {e}",
                "dataset_name": dataset_name,
                "duration_seconds": duration_seconds,
            },
        )
        raise


if __name__ == "__main__":
    run()
