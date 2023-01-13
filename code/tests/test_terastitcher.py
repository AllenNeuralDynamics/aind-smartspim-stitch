"""
Tests for the stitching module
"""
import os
import tempfile
import unittest
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import tifffile
from terastitcher_module import terastitcher
from terastitcher_module.params import get_default_config
from utils import utils

# IO types
PathLike = Union[str, Path]


def _create_images(
    path: PathLike,
    n_tiffs: Optional[int] = 4,
    shape: Optional[Tuple[int]] = (128, 128),
):
    """
    Creates images for tests

    Parameters
    -------------
    path: PathLike
        Path where toy images are going to be stored

    n_tiffs: Optional[int]
        Number of tiffs per folder. Default: 4

    shape: Optional[Tuple[int]]
        Shape of each tiff image
    """

    for idx in range(3):
        name = f"00{idx*2}000"
        new_path = path.joinpath(name)
        utils.create_folder(new_path)

        for inner_idx in range(3):
            inner_folder = f"{name}_00{inner_idx*2}000"
            inner_path = new_path.joinpath(inner_folder)
            utils.create_folder(inner_path)

            for tiffs_idx in range(n_tiffs):
                tiff_name = f"{name}_{inner_folder}_00000{tiffs_idx}.tif"

                tiff_data = np.ones(shape, dtype=np.uint16)
                tifffile.imwrite(inner_path.joinpath(tiff_name), tiff_data)


def _create_sample_channel(
    input_path: PathLike,
    n_channels: Optional[int],
    n_tiffs: Optional[int] = 4,
    shape: Optional[Tuple[int]] = (128, 128),
):
    """
    Creates a dataset with a single/multiple channel(s)

    Parameters
    -------------
    input_path: PathLike
        Path where toy images are going to be stored

    n_channels: Optional[int]
        Number of channels in the toy dataset

    n_tiffs: Optional[int]
        Number of tiffs per folder. Default: 4

    shape: Optional[Tuple[int]]
        Shape of each tiff image
    """

    for channel_idx in range(n_channels):
        channel_name = f"CH_{channel_idx}"

        channel_path = input_path.joinpath(channel_name)
        utils.create_folder(channel_path)

        _create_images(channel_path, n_tiffs, shape)


class TestTerastitcher(unittest.TestCase):
    """
    Class for testing the methods in the Terastitcher
    module
    """

    def setUp(self):
        """
        Sets up the testing environment
        """
        self._tmp_dir = tempfile.TemporaryDirectory()
        path = Path(self._tmp_dir.name)
        self._single_channel_path = path.joinpath("single_channel")
        utils.create_folder(self._single_channel_path)

        self._multi_channel_path = path.joinpath("multi_channel")
        utils.create_folder(self._multi_channel_path)

        # Get single channel and multichannel data
        _create_sample_channel(self._single_channel_path, n_channels=1)

        _create_sample_channel(self._multi_channel_path, n_channels=3)

    def test_single_channel_stitching(self):
        """
        Tests the single channel stitching
        """
        default_config = get_default_config()
        default_config["regex_channels"] = "CH_([0-9])$"
        default_config["stitch_channel"] = 0
        default_config["verbose"] = False
        default_config["clean_output"] = False
        default_config["ome_zarr_params"]["physical_pixels"] = None

        # TODO set pyscripts as env variables in the system when
        # building to fix this hard coded?
        # flake8: noqa: E501
        default_config[
            "pyscripts_path"
        ] = "C:/Users/camilo.laiton/Documents/Project1/Terastitcher/TeraStitcher-portable-1.11.10-win64/pyscripts"

        # Paths
        input_data = str(self._single_channel_path)

        stitched_folder = terastitcher.execute_terastitcher(
            input_data=input_data,
            output_folder=input_data,
            preprocessed_data=input_data,
            config_teras=default_config,
        )

        self.assertTrue(os.path.isdir(stitched_folder))

    def test_multi_channel_stitching(self):
        """
        Tests the multichannel stitching
        """
        default_config = get_default_config()
        default_config["regex_channels"] = "CH_([0-9])$"
        default_config["stitch_channel"] = 0
        default_config["verbose"] = False
        default_config["clean_output"] = False
        default_config["ome_zarr_params"]["physical_pixels"] = None

        # TODO set pyscripts as env variables in the system when building to fix this hard coded?
        default_config[
            "pyscripts_path"
        ] = "C:/Users/camilo.laiton/Documents/Project1/Terastitcher/TeraStitcher-portable-1.11.10-win64/pyscripts"

        input_data = str(self._multi_channel_path)
        stitched_folder = terastitcher.execute_terastitcher(
            input_data=input_data,
            output_folder=input_data,
            preprocessed_data=input_data,
            config_teras=default_config,
        )

        self.assertTrue(os.path.isdir(stitched_folder))

    def tearDown(self):
        """
        Cleans the environment
        """
        self._tmp_dir.cleanup()
