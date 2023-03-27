"""
Module to convert stitched images to the OME-Zarr format
"""

import logging
import multiprocessing
import os
import time
from pathlib import Path
from typing import Dict, Hashable, List, Optional, Sequence, Tuple, Union

import dask
import numpy as np
import pims
import xarray_multiscale
from aicsimageio.types import PhysicalPixelSizes
from aicsimageio.writers import OmeZarrWriter
from argschema import ArgSchemaParser
from dask.array import concatenate, pad
from dask.array.core import Array
from dask.base import tokenize
from dask.distributed import Client, LocalCluster, performance_report
from distributed import wait
from natsort import natsorted
from numcodecs import blosc
from skimage.io import imread as sk_imread

from ..utils import utils
from .zarr_converter_params import ZarrConvertParams, get_default_config

PathLike = Union[str, Path]
ArrayLike = Union[dask.array.core.Array, np.ndarray]
blosc.use_threads = False


def add_leading_dim(data: ArrayLike) -> ArrayLike:
    """
    Adds a leading dimension

    Parameters
    ------------------------

    data: ArrayLike
        Input array that will have the
        leading dimension

    Returns
    ------------------------

    ArrayLike:
        Array with the new dimension in front.
    """
    return data[None, ...]


def pad_array_n_d(arr: ArrayLike, dim: int = 5) -> ArrayLike:
    """
    Pads a daks array to be in a 5D shape.

    Parameters
    ------------------------

    arr: ArrayLike
        Dask/numpy array that contains image data.
    dim: int
        Number of dimensions that the array will be padded

    Returns
    ------------------------
    ArrayLike:
        Padded dask/numpy array.
    """
    if dim > 5:
        raise ValueError("Padding more than 5 dimensions is not supported.")

    while arr.ndim < dim:
        arr = arr[np.newaxis, ...]
    return arr


def read_image_directory_structure(folder_dir: PathLike) -> dict:
    """
    Creates a dictionary representation of all the images
    saved by folder/col_N/row_N/images_N.[file_extention]

    Parameters
    ------------------------
    folder_dir:PathLike
        Path to the folder where the images are stored

    Returns
    ------------------------
    dict:
        Dictionary with the image representation where:
        {channel_1: ... {channel_n: {col_1: ... col_n: {row_1: ... row_n: [image_0, ..., image_n]} } } }
    """

    directory_structure = {}
    folder_dir = Path(folder_dir)

    channel_paths = natsorted(
        [
            folder_dir.joinpath(folder)
            for folder in os.listdir(folder_dir)
            if os.path.isdir(folder_dir.joinpath(folder))
        ]
    )

    for channel_idx in range(len(channel_paths)):
        directory_structure[channel_paths[channel_idx]] = {}

        cols = natsorted(os.listdir(channel_paths[channel_idx]))

        for col in cols:
            possible_col = channel_paths[channel_idx].joinpath(col)

            if os.path.isdir(possible_col):
                directory_structure[channel_paths[channel_idx]][col] = {}

                rows = natsorted(os.listdir(possible_col))

                for row in rows:
                    possible_row = channel_paths[channel_idx].joinpath(col).joinpath(row)

                    if os.path.isdir(possible_row):
                        directory_structure[channel_paths[channel_idx]][col][row] = natsorted(
                            os.listdir(possible_row)
                        )

    return directory_structure


def lazy_tiff_reader(filename: PathLike):
    """
    Creates a dask array to read an image located in a specific path.

    Parameters
    ------------------------

    filename: PathLike
        Path to the image

    Returns
    ------------------------

    dask.array.core.Array
        Array representing the image data
    """
    name = "imread-%s" % tokenize(filename, map(os.path.getmtime, filename))

    with pims.open(filename) as imgs:
        shape = (1,) + (len(imgs),) + imgs.frame_shape
        dtype = np.dtype(imgs.pixel_type)

    key = [(name,) + (0,) * len(shape)]
    value = [(add_leading_dim, (sk_imread, filename))]
    dask_arr = dict(zip(key, value))
    chunks = tuple((d,) for d in shape)

    return Array(dask_arr, name, chunks, dtype)


def fix_image_diff_dims(
    new_arr: ArrayLike, chunksize: Tuple[int], len_chunks: int, work_axis: int
) -> ArrayLike:
    """
    Fixes the array dimension to match the shape of
    the chunksize.

    Parameters
    ------------------------

    new_arr: ArrayLike
        Array to be fixed

    chunksize: Tuple[int]
        Chunksize of the original array

    len_chunks: int
        Length of the chunksize. Used as a
        parameter to avoid computing it
        multiple times

    work_axis: int
        Axis to concatenate. If the different
        axis matches this one, there is no need
        to fix the array dimension

    Returns
    ------------------------

    ArrayLike
        Array with the new dimensions
    """

    zeros_dim = []
    diff_dim = -1
    c = 0

    for chunk_idx in range(len_chunks):
        new_chunk_dim = new_arr.chunksize[chunk_idx]

        if new_chunk_dim != chunksize[chunk_idx]:
            c += 1
            diff_dim = chunk_idx

        zeros_dim.append(abs(chunksize[chunk_idx] - new_chunk_dim))

    if c > 1:
        raise ValueError("Block has two different dimensions")
    else:
        if (diff_dim - len_chunks) == work_axis:
            return new_arr

        n_pad = tuple(tuple((0, dim)) for dim in zeros_dim)
        new_arr = pad(new_arr, pad_width=n_pad, mode="constant", constant_values=0).rechunk(chunksize)

    return new_arr


def concatenate_dask_arrays(arr_1: ArrayLike, arr_2: ArrayLike, axis: int) -> ArrayLike:
    """
    Concatenates two arrays in a given
    dimension

    Parameters
    ------------------------

    arr_1: ArrayLike
        Array 1 that will be concatenated

    arr_2: ArrayLike
        Array 2 that will be concatenated

    axis: int
        Concatenation axis

    Returns
    ------------------------

    ArrayLike
        Concatenated array that contains
        arr_1 and arr_2
    """

    shape_arr_1 = arr_1.shape
    shape_arr_2 = arr_2.shape

    if shape_arr_1 != shape_arr_2:
        slices = []
        dims = len(shape_arr_1)

        for shape_dim_idx in range(dims):
            if shape_arr_1[shape_dim_idx] > shape_arr_2[shape_dim_idx] and (
                shape_dim_idx - dims != axis
            ):
                raise ValueError(
                    f"""
                    Array 1 {shape_arr_1} must have
                     a smaller shape than array 2 {shape_arr_2}
                     except for the axis dimension {shape_dim_idx}
                     {dims} {shape_dim_idx - dims} {axis}
                    """
                )

            if shape_arr_1[shape_dim_idx] != shape_arr_2[shape_dim_idx]:
                slices.append(slice(0, shape_arr_1[shape_dim_idx]))

            else:
                slices.append(slice(None))

        slices = tuple(slices)
        arr_2 = arr_2[slices]

    try:
        res = concatenate([arr_1, arr_2], axis=axis)
    except ValueError:
        raise ValueError(
            f"""
            Unable to cancat arrays - Shape 1:
             {shape_arr_1} shape 2: {shape_arr_2}
            """
        )

    return res


def read_chunked_stitched_image_per_channel(
    directory_structure: dict,
    channel_name: str,
    start_slice: int,
    end_slice: int,
) -> ArrayLike:
    """
    Creates a dask array of the whole image volume
    based on image chunks preserving the chunksize.

    Parameters
    ------------------

    directory_structure:dict
        dictionary to store paths of images with the following structure:
        {channel_1: ... {channel_n: {col_1: ... col_n: {row_1: ... row_n: [image_0, ..., image_n]} } } }

    channel_name : str
        Channel name to reconstruct the image volume

    start_slice: int
        When using multiprocessing, this is
        the start slice the worker will use for
        the array concatenation

    end_slice: int
        When using multiprocessing, this is
        the final slice the worker will use for
        the array concatenation

    Returns
    ------------------------

    ArrayLike
        Array with the image volume
    """
    concat_z_3d_blocks = concat_horizontals = horizontal = None

    # Getting col structure
    cols = list(directory_structure.values())[0]
    cols_paths = list(cols.keys())
    first = True
    # len_chunks = len(chunksize)

    for slice_pos in range(start_slice, end_slice):
        idx_col = 0
        idx_row = 0

        concat_horizontals = None

        for column_name in cols_paths:
            idx_row = 0
            horizontal = []

            for row_name in directory_structure[channel_name][column_name]:
                valid_image = True

                try:
                    slice_name = directory_structure[channel_name][column_name][row_name][slice_pos]

                    filepath = str(
                        channel_name.joinpath(column_name).joinpath(row_name).joinpath(slice_name)
                    )

                    new_arr = lazy_tiff_reader(filepath)

                except ValueError:
                    print("No valid image in ", slice_pos)
                    valid_image = False

                if valid_image:
                    horizontal.append(new_arr)

                idx_row += 1

            # Concatenating horizontally lazy images
            horizontal_concat = concatenate(horizontal, axis=-1)

            if not idx_col:
                concat_horizontals = horizontal_concat
            else:
                concat_horizontals = concatenate_dask_arrays(
                    arr_1=concat_horizontals, arr_2=horizontal_concat, axis=-2
                )

            idx_col += 1

        if first:
            concat_z_3d_blocks = concat_horizontals
            first = False

        else:
            concat_z_3d_blocks = concatenate_dask_arrays(
                arr_1=concat_z_3d_blocks, arr_2=concat_horizontals, axis=-3
            )

    return concat_z_3d_blocks, [start_slice, end_slice]


def _read_chunked_stitched_image_per_channel(args_dict: dict):
    """
    Function used to be dispatched to workers
    by using multiprocessing
    """
    return read_chunked_stitched_image_per_channel(**args_dict)


def channel_parallel_reading(
    directory_structure: dict,
    channel_idx: int,
    sample_img: ArrayLike,
    workers: Optional[int] = 0,
    chunks: Optional[int] = 1,
    ensure_parallel: Optional[bool] = True,
) -> ArrayLike:
    """
    Creates a dask array of the whole image channel volume based
    on image chunks preserving the chunksize and using
    multiprocessing.

    Parameters
    ------------------------

    directory_structure: dict
        dictionary to store paths of images with the following structure:
        {channel_1: ... {channel_n: {col_1: ... col_n: {row_1: ... row_n: [image_0, ..., image_n]} } } }

    channel_name : str
        Channel name to reconstruct the image volume

    sample_img: ArrayLike
        Image used as guide for the chunksize

    workers: Optional[int]
        Number of workers that will be used
        for reading the chunked image.
        Default value 0, it means that the
        available number of cores will be used.

    chunks: Optional[int]
        Chunksize of the 3D chunked images.

    ensure_parallel: Optional[bool]
        True if we want to read the images in
        parallel. False, otherwise.

    Returns
    ------------------------

    ArrayLike
        Array with the image channel volume
    """
    if workers == 0:
        workers = multiprocessing.cpu_count()

    cols = list(directory_structure.values())[0]
    n_images = len(list(list(cols.values())[0].values())[0])
    print(f"n_images: {n_images}")

    channel_paths = list(directory_structure.keys())
    dask_array = None

    if n_images < workers and ensure_parallel:
        workers = n_images

    print("Chunk size for parallel reading: ", sample_img.chunksize)

    if n_images < workers or not ensure_parallel:
        dask_array = read_chunked_stitched_image_per_channel(
            directory_structure=directory_structure,
            channel_name=channel_paths[channel_idx],
            start_slice=0,
            end_slice=n_images,
        )[0]
        print(f"No need for parallel reading... {dask_array}")

    else:
        images_per_worker = n_images // workers
        print(
            f"""
            Setting workers to {workers} - {images_per_worker}
             - total images: {n_images}
            """
        )

        # Getting 5 dim image TCZYX
        args = []
        start_slice = 0
        end_slice = images_per_worker

        for idx_worker in range(workers):
            arg_dict = {
                "directory_structure": directory_structure,
                "channel_name": channel_paths[channel_idx],
                "start_slice": start_slice,
                "end_slice": end_slice,
            }

            args.append(arg_dict)

            if idx_worker + 1 == workers - 1:
                start_slice = end_slice
                end_slice = n_images
            else:
                start_slice = end_slice
                end_slice += images_per_worker

        res = []
        with multiprocessing.Pool(workers) as pool:
            results = pool.imap(
                _read_chunked_stitched_image_per_channel,
                args,
                chunksize=chunks,
            )

            for pos in results:
                res.append(pos)

        for res_idx in range(len(res)):
            if not res_idx:
                dask_array = res[res_idx][0]
            else:
                dask_array = concatenate([dask_array, res[res_idx][0]], axis=-3)

            print(f"Slides: {res[res_idx][1]}")

    return dask_array


def parallel_read_chunked_stitched_multichannel_image(
    directory_structure: dict,
    sample_img: dask.array.core.Array,
    workers: Optional[int] = 0,
    ensure_parallel: Optional[bool] = True,
) -> ArrayLike:
    """
    Creates a dask array of the whole image volume based
    on image chunks preserving the chunksize and using
    multiprocessing.

    Parameters
    ------------------------

    directory_structure: dict
        dictionary to store paths of images with the following structure:
        {channel_1: ... {channel_n: {col_1: ... col_n: {row_1: ... row_n: [image_0, ..., image_n]} } } }

    sample_img: ArrayLike
        Image used as guide for the chunksize

    workers: Optional[int]
        Number of workers that will be used
        for reading the chunked image.
        Default value 0, it means that the
        available number of cores will be used.

    ensure_parallel: Optional[bool]
        True if we want to read the images in
        parallel. False, otherwise.

    Returns
    ------------------------

    ArrayLike
        Array with the image channel volume
    """

    multichannel_image = None

    channel_paths = list(directory_structure.keys())

    multichannels = []
    print(f"Channel in directory structure: {channel_paths}")

    for channel_idx in range(len(channel_paths)):
        print(f"Reading images from {channel_paths[channel_idx]}")
        start_time = time.time()
        read_chunked_channel = channel_parallel_reading(
            directory_structure,
            channel_idx,
            sample_img,
            workers=workers,
            ensure_parallel=ensure_parallel,
        )
        end_time = time.time()

        print(f"Time reading single channel image: {end_time - start_time}")

        # Padding to 4D if necessary
        read_chunked_channel = pad_array_n_d(read_chunked_channel, 4)
        multichannels.append(read_chunked_channel)

    if len(multichannels) > 1:
        multichannel_image = concatenate(multichannels, axis=0)
    else:
        multichannel_image = multichannels[0]

    return multichannel_image


def get_sample_img(directory_structure: dict) -> ArrayLike:
    """
    Gets the sample image for the dataset

    Parameters
    ---------------

    directory_structure: dict
        Whole brain volume directory structure

    Returns
    ---------------

    ArrayLike
        Array with the sample image
    """
    sample_img = None
    for channel_dir, val in directory_structure.items():
        for col_name, rows in val.items():
            for row_name, images in rows.items():
                sample_path = channel_dir.joinpath(col_name).joinpath(row_name).joinpath(images[0])

                if not isinstance(sample_img, dask.array.core.Array):
                    sample_img = lazy_tiff_reader(str(sample_path))
                else:
                    sample_img_2 = lazy_tiff_reader(str(sample_path))

                    if sample_img.chunksize != sample_img_2.chunksize:
                        print("Changes ", sample_img, sample_img_2)
                        return sample_img

    return sample_img


class ZarrConverter:
    """
    Class to convert smartspim datasets to the zarr format
    """

    def __init__(
        self,
        input_data: PathLike,
        output_data: PathLike,
        blosc_config: dict,
        channels: List[str] = None,
        physical_pixels: List[float] = None,
    ) -> None:
        """
        Class constructor

        Parameters
        ------------

        input_data: PathLike
            Path where the stitched images are stored

        output_data: PathLike
            Path where the OME-zarr file will be stored

        blosc_config: dict
            Configuration for image compression

        channels: List[str]
            List with the channel names

        physical_pixels: List[float]
            Physical pixel sizes per dimension

        """
        self.input_data = input_data
        self.output_data = output_data
        self.physical_pixels = None
        self.dask_folder = Path("/root/capsule/scratch")

        if physical_pixels:
            self.physical_pixels = PhysicalPixelSizes(
                physical_pixels[0], physical_pixels[1], physical_pixels[2]
            )

        self.writer = OmeZarrWriter(output_data)

        self.opts = {
            "compressor": blosc.Blosc(
                cname=blosc_config["codec"],
                clevel=blosc_config["clevel"],
                shuffle=blosc.SHUFFLE,
            )
        }

        self.channels: list[str] = channels
        self.channel_colors: list[int] = []

        for channel_str in self.channels:
            em_wav: int = int(channel_str.split("_")[-1])
            em_hex: int = utils.wavelength_to_hex(em_wav)
            self.channel_colors.append(em_hex)

        # get_blosc_codec(writer_config['codec'], writer_config['clevel'])

    def compute_pyramid(
        self,
        data: dask.array.core.Array,
        n_lvls: int,
        scale_axis: Tuple[int],
        chunks: Union[str, Sequence[int], Dict[Hashable, int]] = "auto",
    ) -> List[dask.array.core.Array]:
        """
        Computes the pyramid levels given an input full resolution image data

        Parameters
        ------------------------

        data: dask.array.core.Array
            Dask array of the image data

        n_lvls: int
            Number of downsampling levels
            that will be applied to the original image

        scale_axis: Tuple[int]
            Scaling applied to each axis

        chunks: Union[str, Sequence[int], Dict[Hashable, int]]
            chunksize that will be applied to the multiscales
            Default: "auto"

        Returns
        ------------------------

        List[dask.array.core.Array]:
            List with the downsampled image(s)
        """

        pyramid = xarray_multiscale.multiscale(
            data,
            xarray_multiscale.reducers.windowed_mean,  # func
            scale_axis,  # scale factors
            preserve_dtype=True,
            chunks=chunks,
        )[:n_lvls]

        return [arr.data for arr in pyramid]

    def get_pyramid_metadata(self) -> dict:
        """
        Gets pyramid metadata in OMEZarr format

        Returns
        ------------------------

        dict:
            Dictionary with the downscaling OMEZarr metadata
        """

        return {
            "metadata": {
                "description": """Downscaling implementation based on the
                 windowed mean of the original array""",
                "method": "xarray_multiscale.reducers.windowed_mean",
                "version": str(xarray_multiscale.__version__),
                "args": "[false]",
                # No extra parameters were used different
                # from the orig. array and scales
                "kwargs": {},
            }
        }

    def convert(self, writer_config: dict, image_name: str = "zarr_multiscale.zarr") -> None:
        """
        Executes the OME-Zarr conversion

        Parameters
        ------------------------

        writer_config: dict
            OME-Zarr writer configuration

        image_name: str
            Name of the image

        """
        directory_structure = read_image_directory_structure(self.input_data)
        sample_img = get_sample_img(directory_structure)

        # Reading multichannel image volume
        workers = 0
        start_time = time.time()

        image = pad_array_n_d(
            parallel_read_chunked_stitched_multichannel_image(
                directory_structure, sample_img, workers, ensure_parallel=True
            )
        )
        end_time = time.time()

        print(
            f"""
            Image: {image} {image.npartitions}
            Time: {end_time - start_time}s
            """
        )
        if not isinstance(image, dask.array.core.Array):
            raise ValueError(
                f"""
                There was an error reading
                the images from: {self.input_data}
                """
            )

        image = dask.optimize(image)[0]

        # Setting dask configuration
        dask.config.set(
            {
                "temporary-directory": self.dask_folder,
                "local_directory": self.dask_folder,
                "tcp-timeout": "300s",
                "array.chunk-size": "384MiB",
                "distributed.comm.timeouts": {
                    "connect": "300s",
                    "tcp": "300s",
                },
                "distributed.scheduler.bandwidth": 100000000,
                # "managed_in_memory",#
                "distributed.worker.memory.rebalance.measure": "optimistic",
                "distributed.worker.memory.target": False,  # 0.85,
                "distributed.worker.memory.spill": 0.92,  # False,#
                "distributed.worker.memory.pause": 0.95,  # False,#
                "distributed.worker.memory.terminate": 0.98,  # False, #
                # 'distributed.scheduler.unknown-task-duration': '15m',
                # 'distributed.scheduler.default-task-durations': '2h',
            }
        )

        n_workers = multiprocessing.cpu_count()
        threads_per_worker = 1
        # Using 1 thread since is in single machine.
        # Avoiding the use of multithreaded due to GIL

        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            processes=True,
            memory_limit="auto",
        )
        client = Client(cluster)

        n_channels = image.shape[1]

        # Getting scale axis
        scale_axis = tuple(list(writer_config["scale_factor"]))
        dask_report_file = f"{self.output_data}/dask_report.html"

        # Writing multiscale image
        with performance_report(filename=dask_report_file):
            for idx in range(n_channels):
                channel_img = image[0][idx]

                pyramid_data = self.compute_pyramid(
                    data=channel_img,
                    n_lvls=writer_config["pyramid_levels"],
                    scale_axis=scale_axis,
                    chunks=channel_img.chunksize,
                )

                # Getting 5D
                pyramid_data = [pad_array_n_d(pyramid) for pyramid in pyramid_data]
                print(f"Pyramid {self.channels[idx]}: ", pyramid_data)

                for pyramid in pyramid_data:
                    print(
                        f"""
                        Pyramid {pyramid}
                        - partitions: {pyramid.npartitions}
                        """
                    )

                image_name = self.channels[idx] + ".zarr" if self.channels else image_name
                channel_names = [self.channels[idx]] if self.channels else None
                channel_colors = [self.channel_colors[idx]] if self.channel_colors else None

                print(pyramid_data[0].chunksize)

                dask_jobs = self.writer.write_multiscale(
                    pyramid=pyramid_data,
                    image_name=image_name,
                    chunks=pyramid_data[0].chunksize,
                    physical_pixel_sizes=self.physical_pixels,
                    channel_names=channel_names,
                    channel_colors=channel_colors,
                    scale_factor=scale_axis,
                    storage_options=self.opts,
                    compute_dask=False,
                    **self.get_pyramid_metadata(),
                )

                if len(dask_jobs):
                    dask_jobs = dask.persist(*dask_jobs)
                    wait(dask_jobs)

        client.close()


def main():
    """
    Main function to convert a smartspim dataset to
    the OME-zarr format
    """
    default_config = get_default_config()

    mod = ArgSchemaParser(input_data=default_config, schema_type=ZarrConvertParams)

    args = mod.args

    logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M")
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.INFO)

    zarr_converter = ZarrConverter(
        input_data=args["input_data"],
        output_data=args["output_data"],
        blosc_config={
            "codec": args["writer"]["codec"],
            "clevel": args["writer"]["clevel"],
        },
        channels=["CH_0", "CH_1"],
        physical_pixels=[2.0, 1.8, 1.8],
    )

    start_time = time.time()

    zarr_converter.convert(args["writer"])

    end_time = time.time()

    LOGGER.info(f"Done converting dataset. Took {end_time - start_time}s.")


if __name__ == "__main__":
    main()
