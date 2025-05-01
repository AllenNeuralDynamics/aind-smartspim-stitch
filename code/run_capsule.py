"""
Run the capsule
This script is used to run the capsule for stitching images using BigStitcher.
"""

from pathlib import Path

from aind_smartspim_stitch import bigstitcher
from aind_smartspim_stitch.utils import utils


def run():
    """Function that runs image stitching with BigStitcher"""
    data_folder = Path("../data/SmartSPIM_764702_2025-04-28_16-29-18")  # os.path.abspath(
    results_folder = Path("../results")  # os.path.abspath(
    # scratch_folder = Path(os.path.abspath("../scratch"))

    # It is assumed that these files
    # will be in the data folder

    required_input_elements = [
        f"{data_folder}/SPIM/derivatives/processing_manifest.json",
        f"{data_folder}/data_description.json",
        f"{data_folder}/acquisition.json",
    ]

    # it'll fail explicitly
    # TODO replace for data
    path_to_cloud_data = list(Path("/data").glob("path_to_cloud_*"))[0]

    with path_to_cloud_data.open("r", encoding="utf-8") as f:
        read_s3_path = f.readlines()

    if not len(read_s3_path):
        raise ValueError(f"No contents in {path_to_cloud_data}")

    read_s3_path = read_s3_path[0].strip()
    missing_files = utils.validate_capsule_inputs(required_input_elements)

    if len(missing_files):
        raise ValueError(f"We miss the following files in the capsule input: {missing_files}")

    pipeline_config, smartspim_dataset_name, acquisition_dict = utils.get_data_config(
        data_folder=data_folder,
        processing_manifest_path="SPIM/derivatives/processing_manifest.json",
        data_description_path="data_description.json",
        acquisition_path="acquisition.json",
    )

    voxel_resolution = utils.get_resolution(acquisition_dict)
    stitching_channel = pipeline_config["pipeline_processing"]["stitching"]["channel"]

    stitching_channel_path = data_folder.joinpath(f"SPIM/{stitching_channel}")
    s3_path_to_data = f"{read_s3_path}/{stitching_channel}"

    output_json_file = results_folder.joinpath(f"{smartspim_dataset_name}_tile_metadata.json")

    # Computing image transformations with bigtstitcher
    bigstitcher.main(
        stitching_channel_path=stitching_channel_path,
        voxel_resolution=voxel_resolution,
        output_json_file=output_json_file,
        results_folder=results_folder,
        smartspim_dataset_name=smartspim_dataset_name,
        res_for_transforms=(8.0, 8.0, 8.0),
        s3_path_to_data=s3_path_to_data,
    )


if __name__ == "__main__":
    run()
