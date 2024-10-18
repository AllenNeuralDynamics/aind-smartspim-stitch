"""
Computes stitching transformations using
bigstitcher for SmartSPIM data structure
"""

import json
import os
from pathlib import Path
from time import time
from typing import List, Optional, Tuple

import dask.array as da
from aind_data_schema.core.processing import DataProcess, ProcessName
from natsort import natsorted

from . import smartspim_bigstitcher_utility
from .utils import utils


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


def get_data_config(
    data_folder: str,
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

    derivatives_dict = utils.read_json_as_dict(str(processing_manifest_path))
    data_description_dict = utils.read_json_as_dict(str(data_description_path))
    acquisition_dict = utils.read_json_as_dict(f"{data_folder}/{acquisition_path}")

    smartspim_dataset = data_description_dict["name"]

    return derivatives_dict, smartspim_dataset, acquisition_dict


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

            # zarr_path = Path(f"../{zarr_path.relative_to(current_script_dir)}")
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
    # overlap: Optional[float] = 0.1,
    # tile_width_px: Optional[int] = 1600,
    # tile_height_px: Optional[int] = 2000,
) -> str:
    """
    Create smartspim tile metadata for image stitching
    with BigStitcher

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
    overlap: Optional[float]
        Overlap percentage between image stacks. Default: 0.1 -> 10%
    tile_width_px: Optional[int]
        Tile image width in px. Default: 1600
    tile_height_px: Optional[int]
        Tile image height in px. Default: 20000

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
        curr_col, curr_row = tile.replace(".zarr", "").rsplit('.', 1)[0].split("_")
        curr_col = int(curr_col) // 10
        curr_row = int(curr_row) // 10

        if curr_col not in cols:
            cols.append(curr_col)

        if curr_row not in rows:
            rows.append(curr_row)

    cols = natsorted(cols)
    rows = natsorted(rows)

    # Getting image resolution
    # res_x = xyz_resolution[0]
    # res_y = xyz_resolution[1]

    # # Getting width and height in microns
    # tile_width_um = tile_width_px * res_x
    # tile_height_um = tile_height_px * res_y

    # # Jumps in microns based on overlap
    # jump_width_microns = tile_width_um * overlap
    # jump_height_microns = tile_height_um * overlap

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
        print(f"Error writing json: {e}")

    return output_json_file


def get_stitching_dict(specimen_id: str, dataset_xml_path: str, downsample: Optional[int] = 2) -> dict:
    """
    A function that writes a stitching dictioonary that will be used for
    creating a json file that gives parmaters to bigstitcher sittching run

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
        Dictionary with the stitching parameters
        used for bigstitcher
    """
    # assert pathlib.Path(dataset_xml_path).exists()

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
            "max_shift_in_x": 10,
            "max_shift_in_y": 10,
            "max_shift_in_z": 10,
        },
    }
    return stitching_dict


def get_estimated_downsample(
    voxel_resolution: List[float], phase_corr_res: Tuple[float] = (8.0, 8.0, 4.0)
) -> int:
    """
    Get the estimated multiscale based on the provided
    voxel resolution. This is used for image stitching.

    e.g., if the original resolution is (1.8. 1.8, 2.0)
    in XYZ order, and you provide (3.6, 3.6, 4.0) as
    image resolution, then the picked resolution will be
    1.

    Parameters
    ----------
    voxel_resolution: List[float]
        Image original resolution. This would be the resolution
        in the multiscale "0".
    phase_corr_res: Tuple[float]
        Approximated resolution that will be used for bigstitcher
        in the computation of the transforms. Default: (8.0, 8.0, 4.0)
    """

    downsample_versions = []
    for idx in range(len(voxel_resolution)):
        downsample_versions.append(phase_corr_res[idx] // voxel_resolution[idx])

    downsample_res = int(min(downsample_versions) - 1)
    return downsample_res


def main(
    stitching_channel_path,
    voxel_resolution,
    output_json_file,
    results_folder,
    smartspim_dataset_name,
):
    """
    Computes image stitching with BigStitcher using Phase Correlation

    Parameters
    ----------
    stitching_channel_path: str
        Path where the stitching channel is located
    voxel_resolution: Tuple[float]
        Voxel resolution in order XYZ
    output_json_file: str
        Path where the json file will be written
    results_folder: Path
        Results folder
    smartspim_dataset_name: str
        SmartSPIM dataset name
    """
    start_time = time()
    metadata_folder = results_folder.joinpath("metadata")
    utils.create_folder(str(metadata_folder))

    output_json = create_smartspim_tile_metadata(
        stitching_channel_path=stitching_channel_path,
        xyz_resolution=voxel_resolution,
        output_json_file=output_json_file,
        zarr_tile_multiscale="0",
        # overlap=0.1,
        # tile_width_px=1600,  # Default values on SmartSPIM
        # tile_height_px=2000,
    )

    output_big_stitcher_xml = None
    if output_json is not None:
        stitching_channel = stitching_channel_path.name
        tree = smartspim_bigstitcher_utility.parse_json(
            output_json, str(stitching_channel_path), microns=True
        )
        output_big_stitcher_xml = (
            f"{results_folder}/{smartspim_dataset_name}_stitching_channel_{stitching_channel}.xml"
        )

        smartspim_bigstitcher_utility.write_xml(tree, output_big_stitcher_xml)

        res_for_transforms = (4.0, 4.0, 2.0)
        estimated_downsample = get_estimated_downsample(
            voxel_resolution=voxel_resolution, phase_corr_res=res_for_transforms
        )

        # print(f"Voxel resolution: {voxel_resolution} - Estimating transforms in res: {res_for_transforms} - Scale: {estimated_downsample}")

        smartspim_stitching_params = get_stitching_dict(
            specimen_id=smartspim_dataset_name,
            dataset_xml_path=output_big_stitcher_xml,
            downsample=estimated_downsample,
        )
        end_time = time()

        output_big_stitcher_json = (
            f"{results_folder}/{smartspim_dataset_name}_stitch_channel_{stitching_channel}_params.json"
        )

        data_processes = []
        data_processes.append(
            DataProcess(
                name=ProcessName.IMAGE_TILE_ALIGNMENT,
                software_version="1.2.11",
                start_date_time=start_time,
                end_date_time=end_time,
                input_location=str(smartspim_dataset_name),
                output_location=str(output_big_stitcher_json),
                outputs={"output_file": str(output_big_stitcher_json)},
                code_url="",
                code_version="1.2.5",
                parameters=smartspim_stitching_params,
                notes="Creation of stitching parameters",
            )
        )

        utils.generate_processing(
            data_processes=data_processes,
            dest_processing=metadata_folder,
            processor_full_name="Camilo Laiton",
            pipeline_version="3.0.0",
        )

        with open(output_big_stitcher_json, "w") as f:
            json.dump(smartspim_stitching_params, f, indent=4)

        # Printing to get output on batch script
        print(output_big_stitcher_json)

    else:
        print(f"An error happened while trying to write {output_json_file}")


if __name__ == "__main__":
    main()
