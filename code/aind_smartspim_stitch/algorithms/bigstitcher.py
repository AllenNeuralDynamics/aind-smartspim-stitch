"""
Computes stitching transformations using BigStitcher for SmartSPIM datasets.

BigStitcher runs as a Java/Spark process invoked via bigstitcher_spark_scripts/run_classes.sh.
Phase-correlation tile alignment is performed at a downsampled pyramid level, followed by
global optimization to resolve the full tile graph.
"""

import json
import logging
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import dask.array as da
from aind_data_schema.components.identifiers import Code
from aind_data_schema.core.processing import DataProcess, ProcessName, ProcessStage
from natsort import natsorted

from .. import __maintainers__, __pipeline_name__, __pipeline_version__, __title__, __url__, __version__
from ..utils import utils
from . import bigstitcher_xml_builder

logger = logging.getLogger(__name__)


def create_tile_metadata(
    dataset_path: str, multiscale: str, cols: int, rows: int, xyz_resolution: List[float]
) -> dict:
    """
    Creates tile metadata for image stitching with BigStitcher wrapper

    Parameters
    ----------
    dataset_path: str
        Dataset path
    multiscale: str
        Multiscale used for stitching. It should be a string
        pointing to the multiscale. e.g., "0" for high resolution.
    cols: int
        Number of columns in the dataset.
    rows: int
        Number of rows in the dataset.
    xyz_resolution: List[float]
        Image resolution in xyz order.

    Returns
    -------
    Dict
        Dictionary with tile metadata useful for stitching
    """

    smartspim_to_tile_metadata = []

    # Setting origin as (0,0) tile
    origin = (cols[0], rows[0])

    for curr_col in cols:
        for curr_row in rows:
            # Zarr path -> needs to be relative
            zarr_path = dataset_path.joinpath(f"{curr_col}0_{curr_row}0.zarr")
            if not zarr_path.exists():
                zarr_path = dataset_path.joinpath(f"{curr_col}0_{curr_row}0.ome.zarr")

            # Image data
            img_data = da.from_zarr(Path(zarr_path).joinpath(multiscale))

            # um position relative to origin
            um_position = [(curr_col - origin[0]), (curr_row - origin[1]), 0]

            pixel_position = [i / j for i, j in zip(um_position, xyz_resolution)]

            smartspim_to_tile_metadata.append(
                {
                    "file": str(zarr_path),
                    "size": [img_data.shape[-1], img_data.shape[-2], img_data.shape[-3]],
                    "pixel_resolution": xyz_resolution,
                    "position": pixel_position,
                }
            )

    return smartspim_to_tile_metadata


def create_smartspim_tile_metadata(
    stitching_channel_path: str,
    xyz_resolution: List[float],
    output_json_file,
    zarr_tile_multiscale: Optional[str] = "0",
) -> str:
    """
    Create smartspim tile metadata for image stitching with BigStitcher

    Parameters
    ----------
    stitching_channel_path: str
        Path where the channel for stitching is located
    xyz_resolution: List[float]
        List of floats with the image resolution in xyz order
    output_json_file: str
        Output json file with the tile metadata
    zarr_tile_multiscale: Optional[str]
        Multiscale used for image stitching. Default: "0"

    Returns
    -------
    str:
        Path where the json was written with the tile metadata
    """

    # Getting tiles from channel
    tiles = [
        f
        for f in os.listdir(stitching_channel_path)
        if os.path.isdir(stitching_channel_path.joinpath(f))
    ]

    cols = []
    rows = []

    # SmartSPIM format is a folder in tenths of microns
    for tile in tiles:
        # adding right split for .ome just in case
        curr_col, curr_row = tile.replace(".zarr", "").rsplit(".", 1)[0].split("_")
        curr_col = int(curr_col) // 10
        curr_row = int(curr_row) // 10

        if curr_col not in cols:
            cols.append(curr_col)

        if curr_row not in rows:
            rows.append(curr_row)

    cols = natsorted(cols)
    rows = natsorted(rows)

    smartspim_to_tile_metadata = create_tile_metadata(
        dataset_path=stitching_channel_path,
        multiscale=zarr_tile_multiscale,
        cols=cols,
        rows=rows,
        xyz_resolution=xyz_resolution,
    )

    try:
        with open(output_json_file, "w") as f:
            json.dump(smartspim_to_tile_metadata, f, indent=4)

    except Exception as e:
        output_json_file = None
        logger.error(f"Error writing json: {e}")

    return output_json_file


def get_stitching_dict(specimen_id: str, dataset_xml_path: str, downsample: Optional[int] = 2) -> dict:
    """
    A function that writes a stitching dictionary that will be used for
    creating a json file that gives parameters to bigstitcher stitching run

    Parameters
    ----------
    specimen_id: str
        Specimen ID
    dataset_xml_path: str
        Path where the xml is located
    downsample: Optional[int] = 2
        Image multiscale used for stitching

    Returns
    -------
    dict
        Dictionary with the stitching parameters used for bigstitcher
    """

    stitching_dict = {
        "session_id": str(specimen_id),
        "memgb": 100,
        "parallel": utils.get_code_ocean_cpu_limit(),
        "dataset_xml": str(dataset_xml_path),
        "do_phase_correlation": True,
        "do_detection": False,
        "do_registrations": False,
        "phase_correlation_params": {
            "downsample": downsample,
            "min_correlation": 0.6,
            "max_shift_in_x": 30,
            "max_shift_in_y": 30,
            "max_shift_in_z": 30,
        },
    }
    return stitching_dict


def get_estimated_downsample(
    voxel_resolution: List[float], phase_corr_res: Tuple[float] = (8.0, 8.0, 4.0)
) -> int:
    """
    Estimate the multiscale level (power-of-two downsampling) such that
    the resolution at that level is at least the phase_corr_res in all axes.

    Parameters
    ----------
    voxel_resolution : List[float]
        Resolution of the original image at level 0 (in XYZ order).
    phase_corr_res : Tuple[float]
        Target resolution for phase correlation (in XYZ order).
        Must be >= voxel_resolution in every axis.

    Returns
    -------
    int
        Estimated downsample level (0 or higher).

    Raises
    ------
    ValueError
        If phase_corr_res is smaller than voxel_resolution in any axis.
    """

    levels = []
    for vres, cres in zip(voxel_resolution, phase_corr_res):
        if cres < vres:
            raise ValueError("phase_corr_res must be greater than or equal to voxel_resolution.")
        ratio = cres / vres
        levels.append(math.floor(math.log2(ratio)))

    return max(levels)


def get_max_shifts(
    shape: tuple, overlap: float, pyramid_level: int, min_shift: int = 10, room: int = 10
):
    """
    Calculate the maximum shifts in Z, Y, and X dimensions
    based on image shape and overlap percentage.

    Parameters
    ----------
    shape : tuple of int
        Shape of the image (Z, Y, X) at the given pyramid level.
    overlap : float
        Overlap as a fraction (e.g., 0.1 for 10%).
    pyramid_level : int
        Pyramid level from the zarr multiscale (1 = full res).
    min_shift : int
        Minimum shift allowed.
    room : int
        Extra tolerance to add.

    Returns
    -------
    tuple of int
        Maximum shift in (Z, Y, X) directions.
    """
    if not (0 <= overlap <= 1):
        raise ValueError("Overlap must be between 0 and 1.")

    # Ensure shape corresponds to this pyramid level
    level_shape = tuple(int(dim // (2 ** (pyramid_level - 1))) for dim in shape)

    shifts = []
    for dim in level_shape:
        s = int(dim * overlap)
        if s < min_shift:
            s = min_shift
        s += room
        shifts.append(s)

    return tuple(shifts)


def main(
    stitching_channel_path,
    voxel_resolution,
    output_json_file,
    results_folder,
    smartspim_dataset_name,
    res_for_transforms=(8.0, 8.0, 8.0),
    s3_path_to_data: Optional[str] = None,
):
    """
    Computes image stitching with BigStitcher using Phase Correlation.

    Parameters
    ----------
    stitching_channel_path: str
        Path where the stitching channel is located locally.
    voxel_resolution: Tuple[float]
        Voxel resolution in order XYZ.
    output_json_file: str
        Path where the json file with tile metadata will be written.
    results_folder: Path
        Results folder.
    smartspim_dataset_name: str
        SmartSPIM dataset name.
    res_for_transforms: Tuple[float]
        Target resolution used for phase correlation. Default: (8.0, 8.0, 8.0).
    s3_path_to_data: Optional[str]
        S3 path to the data used as the image loader path in the BigStitcher XML.
        When None, the local stitching_channel_path is used instead.
    """

    BIGSTITCHER_PATH = os.getenv("BIGSTITCHER_HOME")

    if BIGSTITCHER_PATH is None:
        raise ValueError("Please, set the BIGSTITCHER_HOME env value.")

    BIGSTITCHER_PATH = Path(BIGSTITCHER_PATH)
    env = os.environ.copy()

    if not BIGSTITCHER_PATH.exists():
        raise ValueError("Please, set the BIGSTITCHER_PATH env value.")

    start_time = datetime.now(timezone.utc)
    resource_monitor = utils.ResourceMonitor(interval_seconds=30.0).start()

    metadata_folder = results_folder.joinpath("metadata")
    utils.create_folder(str(metadata_folder))

    output_json = create_smartspim_tile_metadata(
        stitching_channel_path=stitching_channel_path,
        xyz_resolution=voxel_resolution,
        output_json_file=output_json_file,
        zarr_tile_multiscale="0",
    )

    output_big_stitcher_xml = None
    if output_json is not None:
        stitching_channel = stitching_channel_path.name

        # Use S3 path for the XML image loader when provided; fall back to local path
        xml_data_path = s3_path_to_data if s3_path_to_data is not None else str(stitching_channel_path)
        tree = bigstitcher_xml_builder.parse_json(output_json, xml_data_path, microns=True)

        output_big_stitcher_xml = (
            f"{results_folder}/{smartspim_dataset_name}_stitching_channel_{stitching_channel}.xml"
        )
        output_big_stitcher_resolved_xml = f"{results_folder}/bigstitcher.xml"

        bigstitcher_xml_builder.write_xml(tree, output_big_stitcher_xml)
        bigstitcher_xml_builder.write_xml(tree, output_big_stitcher_resolved_xml)

        estimated_downsample = get_estimated_downsample(
            voxel_resolution=voxel_resolution, phase_corr_res=res_for_transforms
        )
        downsampled_scale = estimated_downsample * 2 if estimated_downsample else 1

        logger.debug(f"Estimated downsample: {downsampled_scale}")
        max_shift_z, max_shift_y, max_shift_x = get_max_shifts(
            shape=(20, 1600, 2000), overlap=0.1, pyramid_level=downsampled_scale
        )

        # Go up one level from algorithms/ to aind_smartspim_stitch/ where
        # bigstitcher_spark_scripts/ lives
        curr_folder = Path(os.path.realpath(__file__)).parent.parent

        logger.debug(f"Current file path: {curr_folder}")

        # Assuming machine with 128G and 16 cores
        env.update(
            {
                "JAVA_HEAP_SIZE": "128g",
                "SPARK_THREADS": "16",
                "MAIN_CLASS": "net.preibisch.bigstitcher.spark.SparkPairwiseStitching",
            }
        )

        stitching_command = [
            "bash",
            "./bigstitcher_spark_scripts/run_classes.sh",
            "--xml",
            str(output_big_stitcher_resolved_xml),
            "--downsampling",
            f"{downsampled_scale},{downsampled_scale},{downsampled_scale}",
            "--maxShiftZ",
            str(max_shift_z),
            "--maxShiftY",
            str(max_shift_y),
            "--maxShiftX",
            str(max_shift_x),
        ]

        _ = subprocess.run(
            stitching_command,
            check=True,
            cwd=curr_folder,
            env=env,
        )

        # Updating java class to solver for global optimization
        env.update({"MAIN_CLASS": "net.preibisch.bigstitcher.spark.Solver"})

        global_opt_command = [
            "bash",
            "./bigstitcher_spark_scripts/run_classes.sh",
            "--xml",
            str(output_big_stitcher_resolved_xml),
            "--sourcePoints",
            "STITCHING",
        ]

        _ = subprocess.run(
            global_opt_command,
            check=True,
            cwd=curr_folder,
            env=env,
        )

        resource_monitor.stop()
        end_time = datetime.now(timezone.utc)

        output_big_stitcher_json = (
            f"{results_folder}/{smartspim_dataset_name}_stitch_channel_{stitching_channel}_params.json"
        )

        data_processes = []
        data_processes.append(
            DataProcess(
                process_type=ProcessName.IMAGE_TILE_ALIGNMENT,
                name=f"BigStitcher phase correlation + global optimization - {stitching_channel}",
                stage=ProcessStage.PROCESSING,
                code=Code(
                    url=__url__,
                    name=__title__,
                    version=__version__,
                ),
                experimenters=__maintainers__,
                pipeline_name=__pipeline_name__,
                start_date_time=start_time,
                end_date_time=end_time,
                output_path=str(output_big_stitcher_json),
                output_parameters={
                    "input_location": str(smartspim_dataset_name),
                    "output_file": str(output_big_stitcher_json),
                    "stitching": stitching_command,
                    "global_optimization": global_opt_command,
                    "duration_seconds": (end_time - start_time).total_seconds(),
                },
                resources=resource_monitor.to_resource_usage(
                    cpu_cores=int(utils.get_code_ocean_cpu_limit())
                ),
                notes="Running stitching and global optimization separately",
            )
        )

        utils.generate_processing(
            data_processes=data_processes,
            dest_processing=metadata_folder,
            pipeline_name=__pipeline_name__,
            pipeline_version=__pipeline_version__,
            pipeline_url="https://github.com/AllenNeuralDynamics/aind-smartspim-pipeline",
        )

    else:
        resource_monitor.stop()
        logger.error(f"An error happened while trying to write {output_json_file}")


if __name__ == "__main__":
    main()
