from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import NumpyArray, Boolean, Int, Str, Nested, List, Float, InputFile, InputDir
from argschema.schemas import DefaultSchema
from marshmallow import validate
from zarr_converter import OmeZarrParams
import pprint as pp
import platform

class InputFileBasedLinux(InputFile):
    """
    
    InputFileBasedOS is a :class:`argschema.fields.InputFile` subclass which is a path to
    a file location which can be read by the user depending if it's on Linux or not.
    
    """
    def _validate(self, value):
        if platform.system() != 'Windows':
            super()._validate(value)
            
class InputDirGCloud(InputDir):
    """
    
    InputDirGCloud is a :class:`argschema.fields.InputDir` subclass which is a path to
    a directory location which will be validated if the user is not on GCloud.
    
    """
    
    def _validate(self, value):
        if not value.startswith('gs://'):
            super()._validate(value)
        else:
            # Validate after mounting
            pass

class ImportParameters(DefaultSchema):
    
    ref1 = Str(
        required=False, 
        metadata={
            'description':'First axis of the dataset reference system'
        },
        dump_default='X'
    )
    
    ref2 = Str(
        required=False,
        metadata={
            'description':'Second axis of the dataset reference system'
        },
        dump_default='Y'
    )
    
    ref3 = Str(
        required=False, 
        metadata={
            'description':'Third axis of the dataset reference system'
        },
        dump_default='D'
    )
    
    vxl1 = Float(
        required=False,
        matadata={
            'description':'Voxel size along first axis in microns'
        },
        dump_default=1.8
    )
    
    vxl2 = Float(
        required=False,
        matadata={
            'description':'Voxel size along second axis in microns'
        },
        dump_default=1.8
    )
    
    vxl3 = Float(
        required=False,
        matadata={
            'description':'Voxel size along third axis in microns'
        },
        dump_default=2
    )
    
    additional_params = List(
        Str(), 
        required=False, 
        cli_as_single_argument=True
    )

class CPUParams(DefaultSchema):
    
    estimate_processes = Boolean(
        required=False,
        matadata={
            'description':'Estimate processes option for align or merge steps'
        },
        dump_default=False
    )
    
    image_depth = Int(
        required=False, 
        metadata={
            'description':'Layer thickness along Z axis when processing layer by layer'
        }
    )
    
    number_processes = Int(
        required=False, 
        metadata={
            'description':'Degree of parallelization in gpu or cpu'
        }
    )
    
    
    hostfile = InputFileBasedLinux(
        required=False, 
        metadata={
            'description':'Path to MPI hostfile. Only for Linux kernel machines'
        },
        dump_default='/home/jupyter/terastitcher-module/environment/GCloud/hostfile'
    )
    
    additional_params = List(
        Str(), 
        required=False,
        dump_default=[
            "use-hwthread-cpus",
            "allow-run-as-root"
        ],
        cli_as_single_argument=True
    )

class AlignParameters(DefaultSchema):
    
    subvoldim = Int(
        required=False, 
        metadata={
            'description':'Layer thickness along Z axis when processing layer by layer'
        },
        dump_default=100
    )
    
    cpu_params = Nested(CPUParams)
        
class ThresholdParameters(DefaultSchema):
    
    reliability_threshold = Float(
        required=False, 
        metadata={
            'description':'Reliability threshold applied to the computed displacements to select the most reliable ones'
        },
        dump_default=0.7,
        validate=validate.Range(
            min=0, 
            min_inclusive=False,
            max=1,
            max_inclusive=True
        )
    )

class MergeParameters(DefaultSchema):
    
    slice_extent = List(
        Int(),
        required=True,
        metadata={
            'description':'Supposing the output image is saved in a tiled format, this is an array that contains the slice size of output tiles in order [slicewidth, sliceheight, slicedepth]'
        },
        cli_as_single_argument=True,
        dump_default=[20000, 20000, 0]
    )
    
    volout_plugin = Str(
        required=False, 
        metadata={
            'description':"Tiling images output. For 2D 'TiledXY|2Dseries', for 3D 'TiledXY|3Dseries'"
        },
        dump_default="\"TiledXY|2Dseries\""
    )
    
    cpu_params = Nested(CPUParams)
   
class PystripeParams(DefaultSchema):
    # input and output are already defined in PipelineParams Class
    sigma1 = List(
        Int(),
        required=False,
        metadata={
            'description':'bandwidth of the stripe filter for the foreground for each channel'
        },
        cli_as_single_argument=True,
        dump_default=[256, 800, 800]
    )
    
    sigma2 = List(
        Int(),
        required=False,
        metadata={
            'description':'bandwidth of the stripe filter for the background for each channel'
        },
        cli_as_single_argument=True,
        dump_default=[256, 800, 800]
    )
    
    workers = Int(
        required=False, 
        metadata={
            'description':'number of cpu workers to use in batch processing'
        },
        dump_default=8
    )

class PreprocessingSteps(DefaultSchema):
    pystripe = Nested(PystripeParams, required=False)

class PipelineParams(ArgSchema):
    
    input_data = InputDirGCloud(
        required=True, 
        metadata={
            'description':'Path where the data is located'
        }
    )
    
    output_data = Str(
        required=True, 
        metadata={
            'description':"Path where the data will be saved"
        }
    )
    
    stitch_channel = Int(
        required=True, 
        metadata={
            'description':"Position of the informative channel for stitching"
        },
        dump_default=0,
        validate=validate.Range(
            min=0, 
            min_inclusive=True,
            max=3,
            max_inclusive=True
        )
    )
    
    regex_channels = Str(
        required=False, 
        metadata={
            'description':"Path where the data will be saved"
        },
        dump_default='Ex_([0-9]*)_Em_([0-9]*)$'
    )
    
    parastitcher_path = InputFile(
        required=False, 
        metadata={
            'description':'Path to parastitcher'
        },
        dump_default='/home/jupyter/terastitcher-module/environment/GCloud/TeraStitcher-portable-1.11.10-with-BF-Linux/pyscripts/Parastitcher.py'
    )
    
    paraconverter_path = InputFile(
        required=False, 
        metadata={
            'description':'Path to parastitcher'
        },
        dump_default='/home/jupyter/terastitcher-module/environment/GCloud/TeraStitcher-portable-1.11.10-with-BF-Linux/pyscripts/paraconverter.py'
    )
    
    # Processing params
    preprocessing_steps = Nested(PreprocessingSteps, required=False)
    import_data = Nested(ImportParameters, required=True)
    align = Nested(AlignParameters, required=False)
    threshold = Nested(ThresholdParameters, required=False)
    merge = Nested(MergeParameters, required=False)
    
    # Conversion params
    ome_zarr_params = Nested(OmeZarrParams, required=False)
    
    clean_output = Boolean(
        required=False,
        matadata={
            'description':'Set True if you want to delete intermediate output images (e.g. pystripe, terastitcher) and keep only OME-Zarr images. Set False otherwise.'
        },
        dump_default=False
    )

def get_default_config():
    return {
        'preprocessing_steps': {
            'pystripe': {
                "sigma1" : [256, 800, 800, 800],
                "sigma2" : [256, 800, 800, 800],
                "workers" : 16
            }
        },
        'import_data' : {
            "ref1":"X",
            "ref2":"Y",
            "ref3":"D",
            "vxl1":1.800,
            "vxl2":1.800,
            "vxl3":2,
            "additional_params": [
                "sparse_data",
                "libtiff_uncompress"
            ]
        },
        "align" : {
            "cpu_params": {
                "estimate_processes": False,
                "image_depth": 4200,
                "number_processes": 16,
                "hostfile": "/home/jupyter/terastitcher-module/environment/GCloud/hostfile",
                "additional_params": [
                    "use-hwthread-cpus",
                    "allow-run-as-root"
                ]
            },
            "subvoldim": 100
        },
        "threshold" : {
            "reliability_threshold" : 0.7
        },
        "merge" : {
            "cpu_params": {
                "estimate_processes": False,
                "image_depth": 1000,
                "number_processes": 16,
                "hostfile": "/home/jupyter/terastitcher-module/environment/GCloud/hostfile",
                "additional_params": [
                    "use-hwthread-cpus",
                    "allow-run-as-root"
                ]
            },
            "volout_plugin": "\"TiledXY|2Dseries\"",
            "slice_extent": [20000, 20000, 1]
        },
        'ome_zarr_params': {
            'codec': 'zstd',
            'clevel': 1,
            'scale_factor': [2, 2, 2],
            'pyramid_levels': 5
        }
    }

if __name__ == '__main__':

    # this defines a default dictionary that will be used if input_json is not specified
    example = get_default_config()

    mod = ArgSchemaParser(
        input_data=example,
        schema_type=PipelineParams
    )

    pp.pprint(mod.args)
    