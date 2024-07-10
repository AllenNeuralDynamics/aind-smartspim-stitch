"""
Module to declare the parameters for the stitching package
"""

import os
import platform
import pprint as pp
from pathlib import Path

import yaml
from argschema import ArgSchema, ArgSchemaParser
from argschema.fields import Boolean, Float, InputDir, InputFile, Int, List, Nested, Str
from argschema.schemas import DefaultSchema
from marshmallow import validate

from .._shared.types import PathLike
from ..zarr_converter.zarr_converter_params import OmeZarrParams


class InputFileBasedLinux(InputFile):
    """

    InputFileBasedOS is a :class:`argschema.fields.InputFile`
    subclass which is a path to a file location which can be
    read by the user depending if it's on Linux or not.

    """

    def _validate(self, value: str):
        """
        Validates the filesystem

        Parameters
        -------------
        value: str
            Path where the file is located
        """
        if platform.system() != "Windows":
            super()._validate(value)


class InputDirGCloud(InputDir):
    """

    InputDirGCloud is a :class:`argschema.fields.InputDir`
    subclass which is a path to a directory location which
    will be validated if the user is not on GCloud.

    """

    def _validate(self, value: str):
        """
        Validates the filesystem

        Parameters
        -------------
        value: str
            Path where the file is located
        """
        if not value.startswith("gs://"):
            super()._validate(value)
        else:
            # Validate after mounting
            pass


class ImportParameters(DefaultSchema):
    """
    Import parameters for the first stitching step
    """

    ref1 = Str(
        required=False,
        metadata={"description": "First axis of the dataset reference system"},
        dump_default="X",
    )

    ref2 = Str(
        required=False,
        metadata={"description": "Second axis of the dataset reference system"},
        dump_default="Y",
    )

    ref3 = Str(
        required=False,
        metadata={"description": "Third axis of the dataset reference system"},
        dump_default="D",
    )

    vxl1 = Float(
        required=False,
        matadata={"description": "Voxel size along first axis in microns"},
        dump_default=1.8,
    )

    vxl2 = Float(
        required=False,
        matadata={"description": "Voxel size along second axis in microns"},
        dump_default=1.8,
    )

    vxl3 = Float(
        required=False,
        matadata={"description": "Voxel size along third axis in microns"},
        dump_default=2,
    )

    mdata_bin = Str(
        required=False,
        metadata={"description": "Location of the metadata files"},
        dump_default="../scratch",
    )

    additional_params = List(Str(), required=False, cli_as_single_argument=True)


class CPUParams(DefaultSchema):
    """
    CPU parameters for multiprocessing
    """

    estimate_processes = Boolean(
        required=False,
        matadata={"description": "Estimate processes option for align or merge steps"},
        dump_default=False,
    )

    image_depth = Int(
        required=False,
        metadata={"description": "Layer thickness along Z axis"},
    )

    number_processes = Int(
        required=False,
        metadata={"description": "Degree of parallelization in gpu or cpu"},
    )

    hostfile = InputFileBasedLinux(
        required=True,
        metadata={"description": "Path to MPI hostfile, only for Linux."},
        dump_default="/home/hostfile",
    )

    additional_params = List(
        Str(),
        required=False,
        dump_default=["use-hwthread-cpus", "allow-run-as-root"],
        cli_as_single_argument=True,
    )


class AlignParameters(DefaultSchema):
    """
    Parameters to align tiles in the stitching algorithm
    """

    subvoldim = Int(
        required=False,
        metadata={"description": "Layer thickness along Z axis"},
        dump_default=100,
    )

    algorithm = Str(
        required=False,
        metadata={"description": "Algorithm used for stitching"},
        dump_default="MIPNCC",
    )

    sV = Int(
        required=False,
        metadata={"description": "Radius of search (pixels) for displacements in Y"},
        dump_default=25,
    )

    sH = Int(
        required=False,
        metadata={"description": "Radius of search (pixels) for displacements in X"},
        dump_default=25,
    )

    sD = Int(
        required=False,
        metadata={"description": "Radius of search (pixels) for displacements in Z"},
        dump_default=25,
    )

    cpu_params = Nested(CPUParams)


class ThresholdParameters(DefaultSchema):
    """
    Threshold value applied in the stitching algorithm
    """

    reliability_threshold = Float(
        required=False,
        metadata={
            "description": """
            Reliability threshold applied to the computed
             displacements to select the most reliable ones
            """
        },
        dump_default=0.7,
        validate=validate.Range(min=0, min_inclusive=False, max=1, max_inclusive=True),
    )


class MergeParameters(DefaultSchema):
    """
    Merge parameters in the stitching algorithm.
    """

    slice_extent = List(
        Int(),
        required=True,
        metadata={
            "description": """
            Supposing the output image is saved in a tiled format,
            this is an array that contains the slice size of output
            tiles in order [slicewidth, sliceheight, slicedepth]
            """
        },
        cli_as_single_argument=True,
        dump_default=[250, 250, 250],
    )

    volout_plugin = Str(
        required=False,
        metadata={
            "description": """
            Tiling images output. For 2D 'TiledXY|2Dseries',
             for 3D 'TiledXY|3Dseries'
            """
        },
        dump_default='"TiledXY|2Dseries"',
    )

    algorithm = Str(
        required=False,
        metadata={"description": "Algorithm used for blending"},
        dump_default="SINBLEND",
    )

    cpu_params = Nested(CPUParams)


class Visualization(DefaultSchema):
    """
    Parameters for generating the visualization link
    """

    ng_base_url = Str(
        required=False,
        metadata={"description": "Base url for neuroglancer web app"},
        dump_default="https://neuroglancer-demo.appspot.com",
    )

    mount_service = Str(
        required=True,
        metadata={
            "description": """
            Set to s3 if the dataset will be saved
            in a Amazon Bucket, gs for a Google Bucket
            """
        },
        dump_default="s3",
    )

    bucket_path = Str(
        required=True,
        metadata={"description": "Amazon Bucket or Google Bucket name"},
    )


class PipelineParams(ArgSchema):
    """
    Parameters for all the stitching pipeline
    """

    input_data = InputDirGCloud(
        required=True,
        metadata={"description": "Path where the data is located"},
    )

    output_data = Str(
        required=True,
        metadata={"description": "Path where the data will be saved"},
    )

    preprocessed_data = Str(
        required=True,
        metadata={
            "description": """
            Path where the preprocessed data
            will be saved (this includes terastitcher output)
            """
        },
    )

    metadata_folder = Str(
        required=True,
        metadata={
            "description": """
            Path where the metadata is located
            for a smartspim dataset
            """
        },
    )

    stitch_channel = Str(
        required=True,
        metadata={"description": "Position of the informative channel for stitching"},
        dump_default="Ex_488_Em_525",
    )

    regex_channels = Str(
        required=False,
        metadata={"description": "Path where the data will be saved"},
        dump_default="(Ex_[0-9]*_Em_[0-9]*)",
    )

    pyscripts_path = InputDir(
        required=False,
        metadata={
            "description": """
            Path to stitched parallel scripts
            (parastitcher and paraconverter must be there).
            """
        },
        dump_default="/home/TeraStitcher/src/utils/pyscripts",
    )

    # Processing params
    import_data = Nested(ImportParameters, required=True)
    align = Nested(AlignParameters, required=False)
    threshold = Nested(ThresholdParameters, required=False)
    merge = Nested(MergeParameters, required=False)
    verbose = Boolean(
        required=False,
        matadata={"description": "Set verbose for stitching."},
        dump_default=True,
    )

    # Conversion params
    ome_zarr_params = Nested(OmeZarrParams, required=False)

    clean_output = Boolean(
        required=False,
        matadata={
            "description": """
            Set True if you want to delete intermediate
            output images (e.g. pystripe, terastitcher)
            and keep only OME-Zarr images.
            Set False otherwise.
            """
        },
        dump_default=False,
    )

    visualization = Nested(Visualization, required=True)

    info = Boolean(
        required=False,
        metadata={
            "description": """
            Set True if you want to output the
            execution commands
            """
        },
        dump_default=False,
    )

    generate_metadata = Boolean(
        required=False,
        matadata={"description": "Generates AIND metadata for the processed dataset"},
        dump_default=True,
    )


def get_default_config(filename: str = "default_config.yaml") -> None:
    """
    Gets the default configuration from a YAML file

    Parameters
    --------------
    filename: str
        Path where the YAML is located

    """
    filename = Path(os.path.dirname(__file__)).joinpath(filename)

    config = None
    try:
        with open(filename, "r") as stream:
            config = yaml.safe_load(stream)
    except Exception as error:
        raise error

    return config


def get_yaml(yaml_path: PathLike):
    """
    Gets the default configuration from a YAML file

    Parameters
    --------------
    filename: str
        Path where the YAML is located

    Returns
    --------------
    dict
        Dictionary with the yaml configuration
    """

    config = None
    try:
        with open(yaml_path, "r") as stream:
            config = yaml.safe_load(stream)
    except Exception as error:
        raise error

    return config


if __name__ == "__main__":
    # this defines a default dictionary
    # that will be used if input_json is not specified
    example = get_default_config()
    mod = ArgSchemaParser(input_data=example, schema_type=PipelineParams)

    pp.pprint(mod.args)
