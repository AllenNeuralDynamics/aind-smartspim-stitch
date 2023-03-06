"""
Main function to execute dataset processing
"""
import logging
import sys

from aind_smartspim_stitch import terastitcher


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


if __name__ == "__main__":
    main()
