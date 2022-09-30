import unittest
from parameterized import parameterized
from utils import utils
import os
from tests import params
from typing import Optional, List, Union
from pathlib import Path
import tempfile
from terastitcher_module import terasticher

# IO types
PathLike = Union[str, Path]

# def _download_sample_dataset(url:str):

class TestTerastitcher(unittest.TestCase):
    
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        
        # Get single channel data
        utils.create(f"{self._tmp_dir.name}/single_channel_data")
        _create_single_channel_data()
        
        # Get multichannel data
        utils.create(f"{self._tmp_dir.name}/multi_channel_data")
    
    def tearDown(self):
        self._tmp_dir.cleanup()