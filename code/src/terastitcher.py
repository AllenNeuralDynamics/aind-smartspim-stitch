import platform
import os
import subprocess
from typing import List, Optional, Union
from utils import utils
import math
import sys
import errno
from pathlib import Path
from glob import glob
from path_parser import PathParser
from params import PipelineParams, get_default_config
from argschema import ArgSchemaParser
from zarr_converter import ZarrConverter
import warnings
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s : %(message)s",
    datefmt="%Y-%m-%d %H:%M",
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler("test.log", "a"),
    ],
)

PathLike = Union[str, Path]

class TeraStitcher():
    
    def __init__(self, 
            input_data:PathLike, 
            output_folder:PathLike, 
            parallel:bool=True,
            parastitcher_path:Optional[PathLike]=None,
            computation:Optional[str]='cpu',
            preprocessing:Optional[dict]=None,
            verbose:Optional[bool]=False,
        ) -> None:
        
        """
        Class constructor
        
        Parameters
        ------------------------
        input_data: PathLike
            Path where the data is stored.
        output_folder: PathLike
            Path where the stitched data will be stored.
        parallel: Optional[bool]
            True if you want to run terastitcher in parallel, False otherwise.
        parastitcher_path: Optional[PathLike] 
            Path where parastitcher execution file is located.
        computation: Optional[str]
            String that indicates where will terastitcher run. Available options are: ['cpu', 'gpu']
        preprocessing: Optional[dict]:
            All the preprocessing steps prior to terastitcher's pipeline. Default None.
        verbose: Optional[bool]
            True if you want to print outputs of all executed commands.
            
        Raises
        ------------------------
        FileNotFoundError:
            If terastitcher or Parastitcher (if provided) were not found in the system.
        
        """
        
        self.__input_data = Path(input_data)
        self.__output_folder = Path(output_folder)
        self.__parallel = parallel
        self.__computation = computation
        self.__platform = platform.system()
        self.__parastitcher_path = Path(parastitcher_path)
        self.preprocessing = preprocessing
        self.__verbose = verbose
        self.__python_terminal = None
        self.metadata_path = self.__output_folder.joinpath("metadata/params")
        self.xmls_path = self.__output_folder.joinpath("metadata/xmls")
        
        # Check python
        self.__check_python()
        
        if computation not in ['cpu', 'gpu']:
            print("Setting computation to cpu")
            self.__computation = 'cpu'
            
        if computation == 'gpu':
            # Setting environment variable that terastitcher sees for cuda implementation of MIP-NCC algorithm
            #TODO check if cuda is availabe and the toolkit and drivers are correct
            os.environ['USECUDA_X_NCC'] = '1'
            
        else:
            try:
                del os.environ['USECUDA_X_NCC']
            except KeyError:
                warnings.warn("environmental variable 'USECUDA_X_NCC' could not be removed. Ignore this warning if you're using CPU")
                
        if not self.__check_installation():
            print(f"Please, check your terastitcher installation in the system {self.__platform}")
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), "terastitcher")
        
        # TODO get versions
        tools = [
            {
                'Name': 'TeraStitcher',
                'Version': '1.11.10',
                'CodeURL': 'http://abria.github.io/TeraStitcher'
            },
            {
                'Name': 'aicsimageio',
                'Version': 'feature/zarrwriter-multiscales',
                'CodeURL': 'https://github.com/carshadi/aicsimageio/tree/feature/zarrwriter-multiscales'
            }
        ]
        
        pystripe_info = {
            'Name': 'pystripe',
            'Version': '0.2.0',
            'CodeURL': 'https://github.com/chunglabmit/pystripe'
        }
        
        data_description = utils.generate_data_description(input_folder=self.__input_data, tools=tools)
        data_description_path = self.__output_folder.joinpath('data_description.json')
        
        # If parastitcher path is not found, we set computation to sequential cpu as default.
        self.__check_parastitcher()
        
        if self.preprocessing:
            self.__change_io_paths()
        
        # We create the folders for the xmls and metadata in the output directory
        utils.create_folder(self.xmls_path, self.__verbose)
        utils.create_folder(self.metadata_path, self.__verbose)
        
        # Setting stdout log file last because the folder structure depends if preprocessing steps are provided
        self.stdout_log_file = self.metadata_path.joinpath("stdout_log.txt")
        
        # Saving data description
        
        if preprocessing and 'pystripe' in preprocessing:
            tools.insert(0, pystripe_info)
            data_description_path = Path(*self.__output_folder.parts[:-1]).joinpath('data_description.json')
        
        utils.save_dict_as_json(
            data_description_path, 
            data_description
        )
        
        # Setting logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    def __change_io_paths(self) -> None:
        """
        Changes the file order to add an extra folder for pystriped and stitched data.
        It is necessary to take the striped data for terastitcher to include it in the pipeline.
        Thus, it should be saved.
        
        Parameters
        ------------------------
        None
        
        Returns
        ------------------------
        None
        
        """
        
        change_file_order = False
        
        for tool_name, params in self.preprocessing.items():
            if tool_name == 'pystripe':
                change_file_order = True
                self.preprocessing['pystripe']['input'] = Path(self.preprocessing['pystripe']['input'])
                self.preprocessing['pystripe']['output'] = Path(self.preprocessing['pystripe']['output']).joinpath("destriped")
        
        if change_file_order:
            # Organizing repo to add striping folder
            self.__input_data = self.__output_folder.joinpath("destriped")
            self.__output_folder = self.__output_folder.joinpath("stitched")
            
            # Organizing output paths
            self.metadata_path = self.__output_folder.joinpath("metadata/params")
            self.xmls_path = self.__output_folder.joinpath("metadata/xmls")
    
    def __check_installation(self, tool_name:str="terastitcher") -> bool:
        """
        Checks the installation of any tool in the system environment.
        
        Parameters
        ------------------------
        tool_name: str
            command name to check the installation. Default: 'terastitcher'
        
        Returns
        ------------------------
        bool:
            True if the command was correctly executed, False otherwise.
        
        """
        
        try:
            devnull = open(os.devnull)
            subprocess.Popen([tool_name], stdout=devnull, stderr=devnull).communicate()
        except OSError as e:
            return False
        return True
    
    def __check_python(self) -> None:
        """
        Checks python3 installation in the system.
        
        Parameters
        ------------------------
        None
        
        Raises
        ------------------------
        FileNotFoundError:
            If python was not found in the system.
        
        Returns
        ------------------------
        None
        
        """
        
        def helper_status_cmd(cmd:List[str]) -> int:
            """
            Helper function to check python terminal execution.
            
            Parameters
            ------------------------
            cmd: List[str]
                command splitted in list mode.
                
            Returns
            ------------------------
            int:
                Process exit status.
            """
            exit_status = None
            
            try:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                exit_status = proc.returncode
            
            except FileNotFoundError as err:
                exit_status = -1
            
            return exit_status
        
        found = True
        if sys.version_info.major == 3:
            
            if not helper_status_cmd(['python', '-V']):
                self.__python_terminal = 'python'
            
            elif not helper_status_cmd(['python3', '-V']):
                self.__python_terminal = 'python3'
                
            else:
                found = False
        else:
            found = False
        
        if not found:
            self.logger.info(f"Please, check your python 3 installation in the system {self.__platform}")
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), "python")
        
    def __check_parastitcher(self) -> None:
        """
        Checks parastitcher installation using a provided path.
        
        Raises
        ------------------------
        FileNotFoundError:
            If Parastitcher was not found in the system.
        
        """
        
        if self.__parastitcher_path != None:
            if not os.path.exists(self.__parastitcher_path):
                raise FileNotFoundError("Parastitcher path not found.")
            
        else:
            # Parallel false, but we might still be using gpu
            self.__parallel = False
    
    def __build_parallel_command(self, params:dict, step_name:str) -> str:
        """
        Builds a mpi command based on a provided configuration dictionary. 
        
        Parameters
        ------------------------
        params: dict
            Configuration dictionary used to build the mpi command depending on the platform.
        step_name: str
            Terastitcher runs in parallel the align and merge steps. Then, we build the command
            based on which step terastitcher is running.
        
        Returns
        ------------------------
        str:
            Command that will be executed for terastitcher.
        
        """
        
        if not self.__parallel:
            return ''
        
        cpu_params = params['cpu_params']
        
        # mpiexec for windows, mpirun for linux or macs OS
        mpi_command = 'mpiexec -n' if self.__platform == 'Windows' else 'mpirun -np'
        additional_params = ''
        hostfile = ''
        n_procs = cpu_params['number_processes']
        
        # Additional params provided in the configuration
        if len(cpu_params['additional_params']) and self.__platform != 'Windows':
            additional_params = utils.helper_additional_params_command(cpu_params['additional_params'])
        
        # Windows does not require a hostfile to work
        if self.__platform != 'Windows':
            try:
                hostfile = f"--hostfile {cpu_params['hostfile']}"
            except KeyError:
                self.logger.info('Hostfile was not found. This could lead to execution problems.')
        
        # If we want to estimate the number of processes used in any of the steps.
        if cpu_params['estimate_processes']:
            if step_name == 'align':
                n_procs = TeraStitcher.get_aprox_number_processes_align_step(
                    {
                        'image_depth': cpu_params["image_depth"], 
                        'subvoldim': params['subvoldim'], 
                        'number_processes': cpu_params['number_processes']
                    }
                )
                
                self.logger.info(f"- Changing number of processes for align step to {n_procs}")
                
            elif step_name == 'merge':
                # TODO estimate in merge step
                self.logger.info("Aproximate number of processes for the merge step is not implemented yet.")
        
        cmd = f"{mpi_command} {n_procs} {hostfile} {additional_params}"
        cmd += f"{self.__python_terminal} {self.__parastitcher_path}"
        return cmd
    
    def import_step_cmd(self, params:dict) -> str:
        """
        Builds the terastitcher's import command based on a provided configuration dictionary.
        It outputs a json file in the xmls folder of the output directory with all the parameters 
        used in this step. 
        
        Parameters
        ------------------------
        params: dict
            Configuration dictionary used to build the terastitcher's import command.
        
        Returns
        ------------------------
        str:
            Command that will be executed for terastitcher.
        
        """
        
        # TODO Check if params comes with all necessary keys so it does not raise KeyNotFound error
        volume_input = f"--volin={self.__input_data}"
        
        output_path = self.xmls_path.joinpath("xml_import.xml")
        output_folder = f"--projout={output_path}"
        
        parameters = utils.helper_build_param_value_command(params)
        
        additional_params = ''
        if len(params['additional_params']):
            additional_params = utils.helper_additional_params_command(params['additional_params'])
        
        cmd = f"terastitcher --import {volume_input} {output_folder} {parameters} {additional_params}"
        
        output_json = self.metadata_path.joinpath("import_params.json")
        utils.save_dict_as_json(f"{output_json}", params, self.__verbose)
        
        return cmd
    
    @staticmethod
    def get_aprox_number_processes_align_step(config_params:dict) -> int:
        """
        Get the estimate number of processes to partition the dataset and calculate the align step.
        Using MPI, check if the number of slots are enough for running the number of processes.
        You can automatically set --use-hwthread-cpus to automatically estimate the number of 
        hardware threads in each core and increase the allowed number of processes. There is 
        another option with -oversubscribe.
        
        Parameters:
        -----------------
        config_params: dict
            Parameters that will be used in the align step. 
            i.e. {'image_depth': 4200, 'subvoldim': 100, 'number_processes': 10}
        
        Returns:
        -----------------
        int: 
            Number of processes to be used in the align step. If it is not possible to perform
            the estimation, we return 2 processes as default (the master and slave processes).
        
        """
        
        if config_params['image_depth'] < config_params['number_processes'] or config_params['subvoldim'] > config_params['image_depth']:
            self.logger.info("Please check the parameters for aproximate number of processes in align step")
            return 2
        
        # Partitioning depth for the tiles
        partitioning_depth = math.ceil( config_params['image_depth'] / config_params['subvoldim'] )
        
        for proc in range(config_params['number_processes'], 0, -1):
            tiling_proc = 2 * ( proc - 1)
            
            if partitioning_depth > tiling_proc:
                return proc
                
        return 2
    
    def align_step_cmd(self, params:dict):
        """
        Builds the terastitcher's align command based on a provided configuration dictionary. 
        It outputs a json file in the xmls folder of the output directory with all the parameters 
        used in this step.
        
        Parameters
        ------------------------
        params: dict
            Configuration dictionary used to build the terastitcher's align command.
        
        Returns
        ------------------------
        str:
            Command that will be executed for terastitcher.
        
        """
        input_path = self.xmls_path.joinpath("xml_import.xml")
        input_xml = f"--projin={input_path}"
        
        output_path = self.xmls_path.joinpath("xml_displcomp.xml")
        output_xml = f"--projout={output_path}"
        parallel_command = ''

        if self.__parallel:
            parallel_command = self.__build_parallel_command(params, 'align')
    
        else:
            # Sequential execution or gpu execution if USECUDA_X_NCC flag is 1
            parallel_command = f"terastitcher"
        
        parameters = utils.helper_build_param_value_command(params)
        
        cmd = f"{parallel_command} --displcompute {input_xml} {output_xml} {parameters} > {self.xmls_path}/step2par.txt"
        
        output_json = self.metadata_path.joinpath("align_params.json")
        utils.save_dict_as_json(f"{output_json}", params, self.__verbose)
        
        return cmd
    
    def input_output_step_cmd(self, 
            step_name:str, 
            input_xml:str, 
            output_xml:str, 
            params:Optional[dict]=None,
        ) -> str:
        
        """
        Builds the terastitcher's input-output commands based on a provided configuration dictionary.
        These commands are: displproj for projection, displthres for threshold and placetiles for 
        placing tiles. Additionally, it outputs a json file in the xmls folder of the output directory 
        with all the parameters used in this step.
        
        Parameters
        ------------------------
        step_name: str
            Name of the step that will be executed. The names should be: 'displproj' for projection, 
            'displthres' for threshold and 'placetiles' for placing tiles step.
        input_xml: str
            The xml filename outputed from the previous command.
        output_xml: str
            The xml filename that will be used as output for this step.
        params: dict
            Configuration dictionary used to build the terastitcher's command.
            
        Returns
        ------------------------
        str:
            Command that will be executed for terastitcher.
        
        """
        
        input_path = self.xmls_path.joinpath(input_xml)
        input_xml = f"--projin={input_path}"
        
        output_path = self.xmls_path.joinpath(output_xml)
        output_xml = f"--projout={output_path}"
        
        parameters = ''
        
        if params:
            parameters = utils.helper_build_param_value_command(params)
        
        cmd = f"terastitcher --{step_name} {input_xml} {output_xml} {parameters}"
        
        output_json = self.metadata_path.joinpath(f"{step_name}_params.json")
        utils.save_dict_as_json(f"{output_json}", params, self.__verbose)
        
        return cmd
    
    def merge_step_cmd(self, params:dict):
        """
        Builds the terastitcher's merge command based on a provided configuration dictionary. 
        It outputs a json file in the xmls folder of the output directory with all the parameters 
        used in this step.
        
        Parameters
        ------------------------
        params: dict
            Configuration dictionary used to build the terastitcher's merge command.
        
        Returns
        ------------------------
        str:
            Command that will be executed for terastitcher.
        
        """
        
        input_path = self.xmls_path.joinpath("xml_merging.xml")
        input_xml = f"--projin={input_path}"
        
        output_path = f"--volout={self.__output_folder}"
        parallel_command = ''
        
        params = {
            'slicewidth': params['slice_extent'][0],
            'sliceheight': params['slice_extent'][1],
            'slicedepth': params['slice_extent'][2],
            'volout_plugin': params['volout_plugin'],
            'cpu_params': params['cpu_params']
        }
        
        if self.__parallel:
            parallel_command = self.__build_parallel_command(params, 'merge')
    
        else:
            # Sequential execution or gpu execution if USECUDA_X_NCC flag is 1
            parallel_command = f"terastitcher"
        
        parameters = utils.helper_build_param_value_command(params)
        
        cmd = f"{parallel_command} --merge {input_xml} {output_path} {parameters} > {self.xmls_path}/step6par.txt"
        
        output_json = self.metadata_path.joinpath("merge_params.json")
        utils.save_dict_as_json(f"{output_json}", params, self.__verbose)
        
        return cmd
    
    def convert_to_ome_zarr(self, config):
        
        output_json = self.metadata_path.joinpath("ome_zarr_params.json")
        utils.save_dict_as_json(output_json, config, self.__verbose)
        
        input_folder = utils.get_deepest_dirpath(self.__output_folder)
        
        converter = ZarrConverter(
            input_folder, 
            self.__output_folder.joinpath('OMEZarr'), 
            {'codec': config['codec'], 'clevel': config['clevel']}
        )
        
        converter.convert(
            config
        )
    
    def __preprocessing_tool_cmd(
            self, 
            tool_name:str, 
            params:dict, 
            equal_con:bool
        ) -> str:
        
        """
        Builds the execution command for the given tool.
        
        Parameters
        ------------------------
        params: dict
            Configuration dictionary used to build the command.
        
        tool_name: str
            Tool name to be used in the terminal for execution.
            
        equal_con: Optional[bool]
            Indicates if the parameter is followed by '='.
        
        Returns
        ------------------------
        str:
            Command that will be executed for pystripe.
        
        """
        parameters = utils.helper_build_param_value_command(params, equal_con=equal_con)
        
        output_json = self.metadata_path.joinpath(f"{tool_name}_params.json")
        utils.save_dict_as_json(f"{output_json}", params, self.__verbose)
        
        cmd = f"{tool_name} {parameters}"
        
        return cmd
    
    def __execute_preprocessing_steps(self, exec_config:dict) -> None:
        """
        Executes any preprocessing steps that are required for the pipeline.
        It is necessary to have the begining of the terminal command as key and
        the parameters as a dictionary. i.e. "pystripe": {"input" : input_data,
        "output" : output_folder,"sigma1" : 256,"sigma2" : 256,"workers" : 8}
        Command line would be: pystripe --input input_data --output output_folder 
        --sigma1 256 --sigma2 256 --workers 8
        
        Parameters
        ------------------------
        exec_config: dict
            Configuration for command line execution. Mostly for logger.
        
        """
        
        for tool_name, params in self.preprocessing.items():
            exec_config['command'] = self.__preprocessing_tool_cmd(tool_name, params, False)
            utils.execute_command(
                exec_config
            )
    
    def execute_pipeline(self, config:dict) -> None:
        """
        Executes the terastitcher's stitching pipeline that includes the following steps:
        Import, Align, Project, Threshold, Place and Merge. Please refer to the following
        link for more information: https://github.com/abria/TeraStitcher/wiki/Stitching-pipeline
        
        Parameters
        ------------------------
        config: dict
            Configuration dictionary for the stitching pipeline. It should include the configuration
            for each of the steps in the pipeline. i.e. {'import': {...}, 'align': {...}, ...}
        
        """
        
        exec_config = {
            'command': '',
            'verbose': self.__verbose,
            'stdout_log_file': self.stdout_log_file,
            'logger': self.logger,
            # Checking if stdout log file exists
            'exists_stdout': os.path.exists(self.stdout_log_file)
        }
        
        # Preprocessing steps
        if self.preprocessing:
            self.__execute_preprocessing_steps(exec_config)
        
        # Step 1
        
        exec_config['command'] = self.import_step_cmd(config['import_data'])
        self.logger.info("Import step...")
        utils.execute_command(
            exec_config
        )
        
        # Step 2
        self.logger.info("Align step...")
        exec_config['command'] = self.align_step_cmd(config['align'])
        utils.execute_command(
            exec_config
        )
        
        # Step 3
        self.logger.info("Projection step...")
        exec_config['command'] = self.input_output_step_cmd(
            'displproj', 'xml_displcomp.xml', 'xml_displproj.xml'
        )
        utils.execute_command(
            exec_config
        )
        
        # Step 4
        self.logger.info("Threshold step...")
        threshold_cnf = {'threshold': config['threshold']['reliability_threshold']}
        exec_config['command'] = self.input_output_step_cmd(
            'displthres', 'xml_displproj.xml', 'xml_displthres.xml', threshold_cnf
        )
        utils.execute_command(
            exec_config
        )
        
        # Step 5
        self.logger.info("Placing tiles step...")
        exec_config['command'] = self.input_output_step_cmd(
            'placetiles', 'xml_displthres.xml', 'xml_merging.xml'
        )
        utils.execute_command(
            exec_config
        )
        
        # Step 6
        self.logger.info("Merging step...")
        exec_config['command'] = self.merge_step_cmd(config["merge"])
        utils.execute_command(
            exec_config
        )
        
        self.logger.info("Converting to OME-Zarr...")
        self.convert_to_ome_zarr(config['ome_zarr_params'])
        
        if config['clean_output']:
            destriped_folder = Path(os.path.sep.join(list(self.__output_folder.parts)[:-1])).joinpath('destriped')
            utils.delete_folder(destriped_folder, self.__verbose)
            
            stitched_folder = Path(glob(str(self.__output_folder) + '/RES*')[0])
            utils.delete_folder(stitched_folder, self.__verbose)

def execute_terastitcher(
        input_data:PathLike, 
        output_folder:PathLike,
        config_teras:PathLike
    ) -> None:
    
    """
    Executes terastitcher with in-command parameters. It could be on-premise or in the cloud.
    If the process in being carried-out on a GCP VM (i.e. VertexAI jupyter notebook), the
    corresponding buckets will be loaded.
    
    Parameters
    ------------------------
    input_data: PathLike
        Path where the data is located.
        
    output_folder: PathLike
        Path where the data will be saved.
        
    config_teras: Dict
        Dictionary with terastitcher's configuration.
    
    """
    
    parser_result = PathParser.parse_path_gcs(
        input_data,
        output_folder
    )
    
    if len(parser_result):
        # changing paths to mounted dirs
        input_data = input_data.replace('gs://', os.getcwd()+'/')
        output_folder = output_folder.replace('gs://', os.getcwd()+'/')
        print(f"- New input folder: {input_data}")
        print(f"- New output folder: {output_folder}")
    
    try:
        config_teras['preprocessing_steps']['pystripe']['input'] = input_data
        config_teras['preprocessing_steps']['pystripe']['output'] = output_folder
    except KeyError:
        config_teras['preprocessing_steps'] = None
        
    terastitcher_tool = TeraStitcher(
        input_data=input_data,
        output_folder=output_folder,
        parallel=True,
        computation='cpu',
        parastitcher_path=config_teras["parastitcher_path"],
        verbose=True,
        preprocessing=config_teras['preprocessing_steps']
    )
    
    # Saving log command
    terastitcher_cmd = f"$ python terastitcher.py --input {input_data} --output {output_folder} --config_teras {config_teras}\n"
    utils.save_string_to_txt(terastitcher_cmd, terastitcher_tool.stdout_log_file)
    
    # Executing terastitcher
    terastitcher_tool.execute_pipeline(
        config_teras
    )
    
    if len(parser_result):
        # unmounting dirs
        if parser_result[0] == parser_result[1]:
            utils.gscfuse_unmount(parser_result[0])
        else:
            utils.gscfuse_unmount(parser_result[0])
            utils.gscfuse_unmount(parser_result[1])

def main() -> None:
    default_config = get_default_config()

    mod = ArgSchemaParser(
        input_data=default_config,
        schema_type=PipelineParams
    )
    
    args = mod.args
    
    execute_terastitcher(
        input_data=args['input_data'],
        output_folder=args['output_data'],
        config_teras=args
    )
        
if __name__ == "__main__":
    main()