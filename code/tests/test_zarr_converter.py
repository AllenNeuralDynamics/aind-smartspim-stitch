import unittest
from parameterized import parameterized
import os
from utils import utils
import numpy as np
import tifffile
from zarr_converter import ZarrConverter
import dask
from tests import params

def _create_tiffs_files(path_to_tiffs:str, n_tiffs=4, shape=(64, 128, 128)):
    for idx_tiffs in range(n_tiffs):
        tiff_data = np.ones(shape, dtype=np.uint16)
        tifffile.imwrite(os.path.join(path_to_tiffs, f"data_{idx_tiffs}.tif"), tiff_data)

class TestZarrConverter(unittest.TestCase):
    
    def setUp(self):
        # io folders
        path_to_tiffs = 'tmp'
        utils.create_folder(path_to_tiffs)
        _create_tiffs_files(path_to_tiffs)
        
        # Using stitching structure for reading multichannel
        utils.create_folder(path_to_tiffs + '/x/y/')
        _create_tiffs_files(path_to_tiffs + '/x/y/')
    
    def test_pad_array_n_d(self):
        # io folders
        path_to_tiffs = 'tmp'
        
        zarr_writer = ZarrConverter(
            path_to_tiffs,
            path_to_tiffs,
            {'codec': 'zstd', 'clevel': 1}
        )
        
        # Reading images and padding
        array = zarr_writer.read_channel_image(path_to_tiffs)
        pad_array = zarr_writer.pad_array_n_d(array, 5)
        re_pad_array = zarr_writer.pad_array_n_d(array, 5)
        
        # Tests
        self.assertEqual(pad_array.ndim, 5)
        with self.assertRaises(ValueError):
            zarr_writer.pad_array_n_d(array, 6)
            
        self.assertEqual(re_pad_array.ndim, 5)
    
    @parameterized.expand(params.get_compute_pyramid_params())
    def test_compute_pyramid(self, n_lvls:int, scale_axis:int):
        path_to_tiffs = 'tmp'
        zarr_writer = ZarrConverter(
            path_to_tiffs,
            path_to_tiffs,
            {'codec': 'zstd', 'clevel': 1}
        )
        array = zarr_writer.read_channel_image(path_to_tiffs)
        
        array_pyramid = zarr_writer.compute_pyramid(array, n_lvls, scale_axis)
        self.assertIsInstance(array_pyramid, list)
        self.assertEqual(len(array_pyramid), n_lvls)
        
        for multiscale in array_pyramid:
            self.assertIsInstance(multiscale, dask.array.core.Array)
            self.assertEqual(multiscale.ndim, 4)
    
    @parameterized.expand(params.get_compute_pyramid_raises_params())
    def test_compute_pyramid_raises(self, n_lvls:int, scale_axis:int):
        path_to_tiffs = 'tmp'
        zarr_writer = ZarrConverter(
            path_to_tiffs,
            path_to_tiffs,
            {'codec': 'zstd', 'clevel': 1}
        )
        array = zarr_writer.read_channel_image(path_to_tiffs)
        
        with self.assertRaises(ValueError):
            array_pyramid = zarr_writer.compute_pyramid(array, n_lvls, scale_axis)
            
    def test_convert_to_ome_zarr(self):
        path_to_tiffs = 'tmp'
        
        zarr_writer = ZarrConverter(
            path_to_tiffs,
            path_to_tiffs,
            {'codec': 'zstd', 'clevel': 1}
        )
        
        config = {
            'codec': 'zstd',
            'clevel': 1,
            'scale_factor': [2, 2, 2],
            'pyramid_levels': 5
        }
        
        zarr_writer.convert(config, image_name='test_tiffs.zarr')
        self.assertTrue(os.path.isdir('tmp/test_tiffs.zarr'))
        
    def tearDown(self):
        utils.delete_folder('tmp')
        