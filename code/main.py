"""
Main function to execute dataset processing
"""
import logging
import sys

from aind_smartspim_stitch import terastitcher, utils


def main() -> None:
    """
    Main function to execute stitching
    """
    output_folder = terastitcher.main()
    bucket_path = sys.argv[-1]

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

    logger.info(f"Bucket path: {bucket_path} - Output path: {output_folder}")

    # Copying output to bucket
    dataset_name = output_folder.replace("/scratch/", "")
    s3_path = f"s3://{bucket_path}/{dataset_name}"
    for out in utils.execute_command_helper(
        f"aws s3 mv --recursive {output_folder} {s3_path}"
    ):
        logger.info(out)

    utils.save_string_to_txt(
        f"Stitched dataset saved in: {s3_path}",
        "/root/capsule/results/output_stitching.txt",
    )


if __name__ == "__main__":
    main()
