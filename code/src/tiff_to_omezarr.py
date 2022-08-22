import os
from pathlib import Path
import numpy as np
import logging
from distributed import Client, LocalCluster, progress
from numcodecs import blosc
from typing import Optional
import dask
from dask.array.image import imread
from dask.diagnostics import ProgressBar
import argparse
import time
from typing import List, Optional, Union, Tuple, Any
import math
from xarray_multiscale import multiscale
from xarray_multiscale.reducers import windowed_mean

#pip install git+https://github.com/carshadi/aicsimageio.git@feature/zarrwriter-multiscales
#pip install git+https://github.com/AllenInstitute/argschema.git
from aicsimageio.writers import OmeZarrWriter
from aicsimageio.readers.tiff_reader import TiffReader

PathLike = Union[str, Path]

def get_blosc_codec(codec, clevel):
    return blosc.Blosc(cname=codec, clevel=clevel, shuffle=blosc.SHUFFLE)

def pad_array_5d(arr):
    
    """
        Pads a daks array to be in a 5D shape.
        
        Parameters
        ------------------------
        arr: Dask.Array
            Dask array that contains image data.
            
        Returns
        ------------------------
        Dask.Array:
            Padded dask array.
    """
    
    while arr.ndim < 5:
        arr = arr[np.newaxis, ...]
    return arr

def read_files(
        input_path:PathLike, 
        file_format:str
    ):
    
    """
        Reads image files and stores them in a dask array.
        
        Parameters
        ------------------------
        input_path: PathLike
            Path where image files are stored.
        
        file_format:str
            Accepted file format
        
        Returns
        ------------------------
        dask.array.core.Array:
            Dask array with the images. Returns None if it was not possible to read the images.
    """
    
    images = None
    
    try:
        filename_pattern = f'{input_path}/*.{file_format}'
        images = imread(filename_pattern)
        
    except ValueError:
        print(f"- No images found with .{file_format} extension")
    
    return images

def compute_pyramid(
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

def execute_tiff_omezarr_conversion(
        input_path:PathLike,
        output_path:PathLike,
        writer_config:dict,
        image_name:str='zarr_multiscale'
    ) -> None:
    
    """
    Executes the OME-Zarr conversion
    
    Parameters
    ------------------------
    input_path: PathLike
        Path where the tiff files are stored.
        
    output_path: List[str]
        Path where the OME-Zarr file will be stored.
    
    writer_config: dict
        OME-Zarr writer configuration
    
    image_name: str
       Name of the image
    
    """
    
    client = Client()
    print(client)
    
    image = read_files(input_path, 'tiff')
    
    if not isinstance(image, dask.array.core.Array):
        image = read_files(input_path, 'tif')
    
    if not isinstance(image, dask.array.core.Array):
        raise ValueError('- No images found')
        
    scale_axis = []
    for axis in range(len(image.shape)-len(writer_config['scale_factor'])):
        scale_axis.append(1)
    
    scale_axis.extend(list(writer_config['scale_factor']))
    scale_axis = tuple(scale_axis)
    
    pyramid_data = compute_pyramid(
        image, 
        writer_config['pyramid_levels'],
        scale_axis
    )
    
    pyramid_data = [pad_array_5d(pyramid) for pyramid in pyramid_data]
        
    opts = {
        'compressor': get_blosc_codec(writer_config['codec'], writer_config['clevel'])
    }
    
    writer = OmeZarrWriter(
        output_path
    )
    
    # writer.write_dask_multiscale(
    #     pyramid=pyramid_data,  # : types.ArrayLike,  # must be 5D TCZYX
    #     image_name='test',  #: str,
    #     physical_pixel_sizes=None,
    #     channel_names=None,
    #     channel_colors=None,
    #     scale_factor=writer_config['scale_factor'],  # : float = 2.0,
    #     chunks=pyramid_data[0].chunksize,#writer_config['chunks'],
    #     # storage_options=opts,
    # )
    with ProgressBar():
        writer.write_multiscale(
            pyramid=pyramid_data,  # : types.ArrayLike,  # must be 5D TCZYX
            image_name=image_name,  #: str,
            physical_pixel_sizes=None,
            channel_names=None,
            channel_colors=None,
            scale_factor=scale_axis,  # : float = 2.0,
            chunks=pyramid_data[0].chunksize,#writer_config['chunks'],
            storage_options=opts,
        )
    
    client.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="directory of images to transcode",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="output directory",
    )
    parser.add_argument("--codec", type=str, default="zstd")
    parser.add_argument("--clevel", type=int, default=1)
    parser.add_argument(
        "--chunk_size", type=float, default=64, help="chunk size in MB"
    )
    parser.add_argument(
        "--chunk_shape", type=int, nargs='+', default=None, help="5D sequence of chunk dimensions, in TCZYX order"
    )
    
    parser.add_argument(
        "--pyramid_levels", type=int, default=1, help="number of resolution levels"
    )
    
    parser.add_argument(
        "--scale_factor",
        type=tuple,
        default=(2, 2),
        help="scale factor for downsampling",
    )
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    
    if not len(args.input):
        raise ValueError("Please, provide a correct input path.")
    
    blosc.use_threads = False
    logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M")
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.INFO)

    output_path = None
    
    if not args.output or args.output == args.input:
        output_path = Path(args.input)
    else:
        output_path = Path(args.output)

    start_time = time.time()
      
    writer = {
        'codec': args.codec,
        'clevel' : args.clevel,
        'chunk_size': args.chunk_size,
        'chunk_shape': args.chunk_shape,
        'pyramid_levels': args.pyramid_levels,
        'scale_factor': args.scale_factor,
    }
    
    execute_tiff_omezarr_conversion(args.input, output_path, writer)
    
    end_time = time.time()
    
    LOGGER.info(
        f"Done converting dataset. Took {end_time - start_time}s."
    )

if __name__ == "__main__":
    main()