import os
from pathlib import Path

from aind_smartspim_stitch import bigstitcher
from aind_smartspim_stitch.utils import utils


def run():
    """Function that runs image stitching with BigStitcher"""
    data_folder = Path(os.path.abspath("../data"))
    results_folder = Path(os.path.abspath("../results"))
    # scratch_folder = Path(os.path.abspath("../scratch"))

    # It is assumed that these files
    # will be in the data folder

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

    voxel_resolution = utils.get_resolution(acquisition_dict)
    stitching_channel = pipeline_config["pipeline_processing"]["stitching"]["channel"]

    stitching_channel_path = data_folder.joinpath(
        f"preprocessed_data/{stitching_channel}"
    )

    print("Stitching channel path: ", stitching_channel_path)

    output_json_file = results_folder.joinpath(f"{smartspim_dataset_name}_tile_metadata.json")

    # Computing image transformations with bigtstitcher
    bigstitcher.main(
        stitching_channel_path=stitching_channel_path,
        voxel_resolution=voxel_resolution,
        output_json_file=output_json_file,
        results_folder=results_folder,
        smartspim_dataset_name=smartspim_dataset_name,
    )


if __name__ == "__main__":
    run()