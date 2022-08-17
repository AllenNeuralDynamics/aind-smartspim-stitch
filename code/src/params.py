from argschema import ArgSchemaParser, ArgSchema
from argschema.fields import NumpyArray, Boolean, Int, Str, Nested, List, Float, InputFile, InputDir
from argschema.schemas import DefaultSchema
from marshmallow import validate
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
    
    sparse_data = Boolean(
        required=False,
        matadata={
            'description':'If data is sparsed or not'
        },
        dump_default=True
    )
    
    additional_params = List(Str(), required=False)

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
        ]
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
    
    threshold = Float(
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
    
    slicewidth = Int(
        required=False, 
        metadata={
            'description':'Supposing the output image is saved in a tiled format, this is the width of output tiles along X axis'
        },
        dump_default=20000
    )
    
    sliceheight = Int(
        required=False, 
        metadata={
            'description':'Supposing the output image is saved in a tiled format, this is the width of output tiles along Y axis'
        },
        dump_default=20000
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
    # input and output will are already defined in MyParameters Class
    sigma1 = Int(
        required=False, 
        metadata={
            'description':'bandwidth of the stripe filter for the foreground'
        },
        dump_default=256
    )
    
    sigma2 = Int(
        required=False, 
        metadata={
            'description':'bandwidth of the stripe filter for the background'
        },
        dump_default=256
    )
    
    workers = Int(
        required=False, 
        metadata={
            'description':'number of cpu workers to use in batch processing'
        },
        dump_default=8
    )

class PreprocessingSteps(DefaultSchema):
    pystripe = Nested(PystripeParams, required=True)

class MyParameters(ArgSchema):
    
    # TODO Implement custom class for GCSCloud
    input_data = InputDir(
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
    
    parastitcher_path = InputFile(
        required=False, 
        metadata={
            'description':'Path to parastitcher'
        },
        dump_default='/home/jupyter/terastitcher-module/environment/GCloud/hostfile'
    )
    
    preprocessing_steps = Nested(PreprocessingSteps, required=True)
    import_data = Nested(ImportParameters, required=True)
    align = Nested(AlignParameters, required=True)
    threshold = Nested(ThresholdParameters, required=True)
    merge = Nested(MergeParameters, required=True)

def get_default_config():
    return {
        'preprocessing_steps': {
            'pystripe': {
                "sigma1" : 256,
                "sigma2" : 256,
                "workers" : 8
            }  
        },
        'import_data' : {
            "ref1":"X",
            "ref2":"Y",
            "ref3":"D",
            "vxl1":1.800,
            "vxl2":1.800,
            "vxl3":2,
            "sparse_data": True,
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
            "threshold" : 0.7
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
            "volout_plugin": "\"TiledXY|3Dseries\"",
            "slicewidth": 20000,
            "sliceheight": 20000
        }
    }

if __name__ == '__main__':

    # this defines a default dictionary that will be used if input_json is not specified
    example = get_default_config()

    mod = ArgSchemaParser(
        input_data=example,
        schema_type=MyParameters
    )

    pp.pprint(mod.args)
    
    # pp.pprint(mod.args['import_data'])
    # pp.pprint(mod.args['input_data'])
    # pp.pprint(mod.args['output_data'])
    