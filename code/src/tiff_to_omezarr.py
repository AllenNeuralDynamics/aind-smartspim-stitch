import os
from pathlib import Path
import numpy as np
import logging
from distributed import Client, LocalCluster, progress
from numcodecs import blosc
from typing import Optional
import dask
import argparse
import time
from typing import List, Optional, Union

#pip install git+https://github.com/carshadi/aicsimageio.git@feature/zarrwriter-compression
#pip install git+https://github.com/AllenInstitute/argschema.git
from aicsimageio.writers import OmeZarrWriter
from aicsimageio.readers.tiff_reader import TiffReader

PathLike = Union[str, Path]

def get_blosc_codec(codec, clevel):
    return blosc.Blosc(cname=codec, clevel=clevel, shuffle=blosc.SHUFFLE)

def get_images(
        input_dir:PathLike, 
        file_format:Optional[str]='.tiff'
    ) -> List[str]:
    
    """
        Get tiff images from an directory
        
        Parameters
        ------------------------
        input_dir: PathLike
            Path where the data is stored.
        file_format: str
            File format of the images.
            
        Raises
        ------------------------
        NotImplementedError:
            If the zarr converter does not accept a specific file format.
        
        Returns
        ------------------------
        List[str]:
            List with paths to the dataset images
    """
    
    if file_format not in ['.tif', '.tiff']:
        raise NotImplementedError("We have not yet implemented the support for other file formats")
    
    # valid_exts = DataReaderFactory().VALID_EXTENSIONS
    image_paths = []
    
    for root, _, files in os.walk(input_dir):
        for f in files:
            filepath = os.path.join(root, f)
            if not os.path.isfile(filepath):
                continue
            _, ext = os.path.splitext(filepath)
            
            if ext in file_format:
                image_paths.append(filepath)
                
    return image_paths

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

def convert_tiff_image_to_omezarr(
        image_path:PathLike,
        writer_config:dict,
        writer: OmeZarrWriter,
        compressor, 
        opts:dict
    ) -> None:
    
    """
        Convert a tiff image to a OME-Zarr format.
        
        Parameters
        ------------------------
        image_path: PathLike
            Path where the image is stored.
        
        writer_config:
            Arguments provided by user to store in OmeZarr format.
            
        writer: aicsimageio.writers.OmeZarrWriter
            OmeZarrWritter object
            
        compressor:
            Compressor object that will be used to save tiff images to OmeZarr
            
        opts: dict
            Storage options for OmeZarrWriter
    """
    
    tile_name = Path(image_path).stem
    data = TiffReader(image_path).get_image_data().astype(np.uint8)
   
    data = pad_array_5d(data)
    
    writer.write_image(
        image_data=data,  # : types.ArrayLike,  # must be 5D TCZYX
        image_name=tile_name + '.zarr',  #: str,
        physical_pixel_sizes=None,
        channel_names=None,
        channel_colors=None,
        # scale_num_levels=writer_config['n_levels'],  # : int = 1,
        # scale_factor=writer_config['scale_factor'],  # : float = 2.0,
        # chunks=chunks,
        # storage_options=opts,
    )
    
@dask.delayed
def process_batched_images(
        seq_images:List[PathLike], 
        writer_config:dict, 
        writer:OmeZarrWriter, 
        compressor, 
        opts:dict
    ) -> List[dask.delayed]:
    
    """
        Dask delayed function that process the batched images in each worker converting them to OME-Zarr.
        
        Parameters
        ------------------------
        seq_images: List[PathLike]
            List that contains batched images
        
        args:
            Arguments provided by user to store in OmeZarr format.
            
        writer: aicsimageio.writers.OmeZarrWriter
            OmeZarrWritter object
            
        compressor:
            Compressor object that will be used to save tiff images to OmeZarr
            
        opts: dict
            Storage options for OmeZarrWriter
            
        Returns
        ------------------------
        List[dask.delayed]:
            List with delayed functions to process images in parallel
    """
    
    sub_results = []
    for image in seq_images:
        sub_results.append(
            convert_tiff_image_to_omezarr(
                image, 
                writer_config, 
                writer, 
                compressor, 
                opts
            )
        )
        
    return sub_results

def batch_images(
        list_images:List[PathLike], 
        output_path:PathLike, 
        writer_config:dict, 
        image_batch:int=10
    ) -> List[List[dask.delayed]]:
    
    """
        Function that batches images to each dask worker to convert them to OME-Zarr.
        
        Parameters
        ------------------------
        list_images: List[PathLike]
            List that contains batched images
        
        output_path:
            Path where the OME-Zarr files will be written.
            
        args:
            Arguments provided by user to store in OmeZarr format.
            
        image_batch: Optional[int]
            Number of images per batch. Please, set this number depending image resolution.
            If images are not computationally expensive to process, set this parameter with
            a high value so workers do not have to come back a get more data. Otherwise, 
            set this to a lower value.
            
        Returns
        ------------------------
        List[List[dask.delayed]]:
            List with the batches of delayed functions to process images in parallel
    """
    
    batches = []
    
    writer = OmeZarrWriter(
        output_path
    )
    
    compressor = get_blosc_codec(writer_config['codec'], writer_config['clevel'])
    opts = {
        "compressor": compressor,
    }
    
    for idx in range(0, len(list_images), image_batch):
        result_batch = process_batched_images(
            list_images[idx: idx + image_batch],
            writer_config,
            writer,
            compressor,
            opts
        )
        batches.append(result_batch)
    
    return batches

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
        "--n_levels", type=int, default=1, help="number of resolution levels"
    )
    
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=2.0,
        help="scale factor for downsampling",
    )
    args = parser.parse_args()
    
    return args

def execute_tiff_omezarr_conversion(
        input_path,
        output_path,
        writer_config:dict
    ):
    
    client = Client()
    print(client)
    
    image_paths = get_images(input_path)
    print(f"Found {len(image_paths)} images to process")
    
    convert_images = batch_images(image_paths, output_path, writer_config)
    convert_images = dask.persist(convert_images)
    progress(convert_images)
    
    # Use scheduler with 'threads' when we're using it locally to avoid cost of transferring data between tasks
    # and avoid garbage collections. However, it might take more time to process the images.
    
    dask.compute(convert_images)#, scheduler='threads')#, scheduler='single-threaded')

    client.close()

    
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
        'n_levels': args.n_levels,
        'scale_factor': args.scale_factor,
    }
    
    execute_tiff_omezarr_conversion(args.input, output_path, writer)
    
    end_time = time.time()
    
    LOGGER.info(
        f"Done converting dataset. Took {end_time - start_time}s."
    )
        
if __name__ == "__main__":
    main()