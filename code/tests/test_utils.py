"""
Testing module for utility functions
"""
import os
import tempfile
import unittest
from pathlib import Path
from typing import Optional, Union

from aind_smartspim_stitch.utils import utils
from parameterized import parameterized
from tests import params

# IO types
PathLike = Union[str, Path]


class TestUtils(unittest.TestCase):
    """
    Tests the functions in the utility file
    """

    def setUp(self):
        """
        Sets up the testing environment
        """
        self._tmp_dir = tempfile.TemporaryDirectory()
        utils.save_dict_as_json(
            f"{self._tmp_dir.name}/test_json_2.json",
            {"number": 13, "text": "this is a test"},
        )

    @parameterized.expand(params.get_save_dict_as_json_params())
    def test_save_dict_as_json(self, dict_example: dict, filename: PathLike):
        """
        Tests the save_dict_as_json method

        Parameters
        ------------
        dict_example: dict
            Dictionary with the dictionaries to save as
            jsons

        filename: PathLike
            Filename to use for storing the json
        """
        path = Path(self._tmp_dir.name).joinpath(filename)
        utils.save_dict_as_json(path, dict_example)
        self.assertTrue(os.path.isfile(path))

    @parameterized.expand(params.get_read_json_as_dict_params())
    def test_read_json_as_dict(
        self, path_to_file: PathLike, expected_dict: dict
    ):
        """
        Tests the read json as dictionary method

        Parameters
        -----------
        path_to_file: PathLike
            Path where the json is stored

        expected_dict: dict
            Expected dictionary that must be returned
            after reading it
        """
        read_dict_example = utils.read_json_as_dict(
            Path(self._tmp_dir.name).joinpath(path_to_file)
        )
        self.assertDictEqual(read_dict_example, expected_dict)

    @parameterized.expand(params.get_helper_build_param_value_command_params())
    def test_helper_build_param_value_command(
        self,
        default_config: dict,
        expected_output: str,
        equal_con: Optional[bool] = True,
    ):
        """
        Tests the helper function to build parameters
        for the stitching pipeline

        Parameters
        ------------
        default_config: dict
            Configuration with the parameters that
            need to be parsed

        expected_output: str
            Expected output after parsing the dictionary

        equal_con: Optional[bool]
            If the command needs a '=' symbol or not.
            Default: True
        """
        output = utils.helper_build_param_value_command(
            default_config, equal_con
        )
        self.assertEqual(expected_output, output)
        self.assertIsInstance(output, str)

    @parameterized.expand(params.get_helper_additional_params_command_params())
    def test_helper_additional_params_command(
        self, default_config: dict, expected_output: str
    ):
        """
        Tests the helper function to build parameters
        for the stitching pipeline

        Parameters
        ------------
        default_config: dict
            Configuration with the parameters that
            need to be parsed

        expected_output: str
            Expected output after parsing the dictionary
        """
        output = utils.helper_additional_params_command(default_config)
        self.assertEqual(output, expected_output)
        self.assertIsInstance(output, str)

    def test_save_string_to_txt(self):
        """
        Tests the save_string_to_text method
        """
        string = "this is a message"
        path = Path(self._tmp_dir.name).joinpath("text_file.txt")
        utils.save_string_to_txt(string, path)
        self.assertTrue(os.path.isfile(path))

    def tearDown(self):
        """
        Tears down the testing environment
        """
        self._tmp_dir.cleanup()
