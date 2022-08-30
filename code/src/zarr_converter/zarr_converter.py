import os
from pathlib import Path
import logging
from distributed import Client, LocalCluster, progress
from numcodecs import blosc
import dask
from dask.array.image import imread
from dask.diagnostics import ProgressBar
import argparse
import time
from typing import List, Optional, Union, Tuple, Any
from xarray_multiscale import multiscale
from xarray_multiscale.reducers import windowed_mean
import numpy as np
from .zarr_converter_params import ZarrConvertParams, get_default_config
from argschema import ArgSchemaParser

#pip install git+https://github.com/carshadi/aicsimageio.git@feature/zarrwriter-multiscales
#pip install git+https://github.com/AllenInstitute/argschema.git
from aicsimageio.writers import OmeZarrWriter
from aicsimageio.readers.tiff_reader import TiffReader

PathLike = Union[str, Path]
ArrayLike = Union[dask.array.core.Array, np.ndarray]
blosc.use_threads = False

class ZarrConverter():
    
    def __init__(
        self, 
        input_data:PathLike, 
        output_data:PathLike, 
        blosc_config:dict,
        file_format:List[str]='tif'
    ) -> None:
        
        self.input_data = input_data
        self.output_data = output_data
        self.file_format = file_format
        
        self.writer = OmeZarrWriter(
            output_data
        )
        
        self.opts = {
            'compressor': blosc.Blosc(cname=blosc_config['codec'], clevel=blosc_config['clevel'], shuffle=blosc.SHUFFLE)
        }
        # get_blosc_codec(writer_config['codec'], writer_config['clevel'])
        
    def read_files(
        self
    ) -> dask.array.core.Array:
    
        """
            Reads image files and stores them in a dask array.
            
            file_format:str
                Accepted file format
            
            Returns
            ------------------------
            dask.array.core.Array:
                Dask array with the images. Returns None if it was not possible to read the images.
        """
    
        images = None
        
        try:
            filename_pattern = f'{self.input_data}/*.{self.file_format}*'
            images = imread(filename_pattern)
            
        except ValueError:
            raise ValueError('- No images found')
        
        return images
    
    def pad_array_5d(self, arr:ArrayLike) -> ArrayLike:
    
        """
            Pads a daks array to be in a 5D shape.
            
            Parameters
            ------------------------
            arr: ArrayLike
                Dask/numpy array that contains image data.
                
            Returns
            ------------------------
            ArrayLike:
                Padded dask/numpy array.
        """
        
        while arr.ndim < 5:
            arr = arr[np.newaxis, ...]
        return arr

    def compute_pyramid(
        self,
        data:dask.array.core.Array, 
        n_lvls:int, 
        scale_axis:Tuple[int]
    ) -> List[dask.array.core.Array]:

        """
        Computes the pyramid levels given an input full resolution image data
        
        Parameters
        ------------------------
        data: dask.array.core.Array
            Dask array of the image data
            
        n_lvls: int
            Number of downsampling levels that will be applied to the original image
        
        scale_axis: Tuple[int]
            Scaling applied to each axis
        
        Returns
        ------------------------
        List[dask.array.core.Array]:
            List with the downsampled image(s)
        """
        
        pyramid = multiscale(
            data,
            windowed_mean,  # func
            scale_axis,  # scale factors
            depth=n_lvls - 1,
            preserve_dtype=True
        )
        
        return [arr.data for arr in pyramid]

    def convert(
        self,
        writer_config:dict,
        image_name:str='zarr_multiscale'
    ) -> None:
        
        """
        Executes the OME-Zarr conversion
        
        Parameters
        ------------------------
        
        writer_config: dict
            OME-Zarr writer configuration
        
        image_name: str
        Name of the image
        
        """
        
        client = Client()
        
        image = self.read_files()
            
        scale_axis = []
        for axis in range(len(image.shape)-len(writer_config['scale_factor'])):
            scale_axis.append(1)
        
        scale_axis.extend(list(writer_config['scale_factor']))
        scale_axis = tuple(scale_axis)
        
        pyramid_data = self.compute_pyramid(
            image, 
            writer_config['pyramid_levels'],
            scale_axis
        )
        
        pyramid_data = [self.pad_array_5d(pyramid) for pyramid in pyramid_data]
        
        with ProgressBar():
            self.writer.write_multiscale(
                pyramid=pyramid_data,  # : types.ArrayLike,  # must be 5D TCZYX
                image_name=image_name,  #: str,
                physical_pixel_sizes=None,
                channel_names=None,
                channel_colors=None,
                scale_factor=scale_axis,  # : float = 2.0,
                chunks=pyramid_data[0].chunksize,#writer_config['chunks'],
                storage_options=self.opts,
            )
        
        client.close()

def main():
    default_config = get_default_config()

    mod = ArgSchemaParser(
        input_data=default_config,
        schema_type=ZarrConvertParams
    )
    
    args = mod.args
    
    logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M")
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.INFO)
    
    zarr_converter = ZarrConverter(
        input_data=args['input_data'],
        output_data=args['output_data'], 
        blosc_config={
            'codec': args['writer']['codec'], 
            'clevel': args['writer']['clevel']
        }
    )
    
    start_time = time.time()
    
    zarr_converter.convert(args['writer'])
    
    end_time = time.time()
    
    LOGGER.info(
        f"Done converting dataset. Took {end_time - start_time}s."
    )

if __name__ == "__main__":
    main()