import platform
import os
import subprocess
from typing import List, Optional
from utils import utils
import math
import sys
import errno

def helper_build_param_value_command(params:dict) -> str:
    """
    Helper function to build a command based on key:value pairs.
    
    Parameters
    ------------------------
    params: dict
        Dictionary with key:value pairs used for building the command.
    
    Returns
    ------------------------
    str:
        String with the parameters.
    
    """
    parameters = ''
    for (param, value) in params.items():
        if type(value) in [str, float, int]:
            parameters += f"--{param}={value} "
            
    return parameters

def helper_additional_params_command(params:list) -> str:
    """
    Helper function to build a command based on values.
    
    Parameters
    ------------------------
    params: list
        List with additional command values used.
    
    Returns
    ------------------------
    str:
        String with the parameters.
    
    """
    additional_params = ''
    for param in params:
        additional_params += f"--{param} "
        
    return additional_params

class TeraStitcher():
    
    def __init__(self, 
            input_data:str, 
            output_folder:str, 
            parallel:bool=True,
            parastitcher_path:Optional[str]=None,
            computation:Optional[str]='cpu',
            verbose:Optional[bool]=False
        ) -> None:
        
        """
        Class constructor
        
        Parameters
        ------------------------
        input_data: str
            Path where the data is stored.
        output_folder: str
            Path where the stitched data will be stored.
        parallel: Optional[bool]
            True if you want to run terastitcher in parallel, False otherwise.
        parastitcher_path: Optional[str] 
            Path where parastitcher execution file is located.
        computation: Optional[str]
            String that indicates where will terastitcher run. Available options are: ['cpu', 'gpu']
        verbose: Optional[bool]
            True if you want to print outputs of all executed commands.
        
        Raises
        ------------------------
        FileNotFoundError:
            If terastitcher or Parastitcher (if provided) were not found in the system.
        
        """
        
        self.__input_data = input_data
        self.__output_folder = output_folder
        self.__parallel = parallel
        self.__computation = computation
        self.__platform = platform.system()
        self.__parastitcher_path = parastitcher_path
        self.__verbose = verbose
        self.__python_terminal = None
        
        
        # Check python
        self.__check_python()
        
        if computation not in ['cpu', 'gpu']:
            print("Setting computation to cpu")
            self.__computation = 'cpu'
            
        if computation == 'gpu':
            # Setting environment variable that terastitcher sees for cuda implementation of MIP-NCC algorithm
            #TODO check if cuda is availabe and the toolkit and drivers are correct
            os.environ['USECUDA_X_NCC'] = '1'
        
        if not self.__check_installation():
            print(f"Please, check your terastitcher installation in the system {self.__platform}")
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), "terastitcher")
        
        # If parastitcher path is not found, we set computation to sequential gpu as default.
        self.__check_parastitcher()
        
        # We create the folders for the xmls and metadata in the output directory
        utils.create_folder(self.__output_folder + "/xmls")
        utils.create_folder(self.__output_folder + "/metadata")
    
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
        
        Raises
        ------------------------
        FileNotFoundError:
            If python was not found in the system.
        
        """
        
        def helper_status_cmd(cmd:list) -> int:
            """
            Helper function to check python terminal execution.
            
            Parameters
            ------------------------
            cmd: list
                command splitted in list mode.
                
            Returns
            ------------------------
            int:
                Process exit status.
            """
            exit_status = None
            
            try:
                proc = subprocess.run(cmd, capture_output=True)
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
            print(f"Please, check your python 3 installation in the system {self.__platform}")
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), "python")
        
    def __check_parastitcher(self) -> None:
        """
        Checks parastitcher installation using a provided path.
        
        Raises
        ------------------------
        FileNotFoundError:
            If Parastitcher was not found in the system.
        
        """
        
        if self.__parastitcher_path != None and self.__computation == 'cpu':
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
            additional_params = helper_additional_params_command(cpu_params['additional_params'])
        
        # Windows does not require a hostfile to work
        if self.__platform != 'Windows':
            try:
                hostfile = f"--hostfile {cpu_params['hostfile']}"
            except KeyError:
                print('Hostfile was not found. This could lead to execution problems.')
        
        # If we want to estimate the number of processes used in any of the steps.
        if cpu_params['estimate_processes']:
            if step_name == 'align':
                n_procs = self.__get_aprox_number_processes_align_step(
                    {
                        'image_depth': 4200, 
                        'subvoldim': params['subvoldim'], 
                        'number_processes': cpu_params['number_processes']
                    }
                )
                
            elif step_name == 'merge':
                # TODO estimate in merge step
                print("Not implemented yet")
            
            print("Using estimated number of processes: ", n_procs)
        
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
        output_folder = f"--projout={self.__output_folder}/xmls/xml_import.xml"
        
        parameters = helper_build_param_value_command(params)
        
        additional_params = ''
        if len(params['additional_params']):
            additional_params = helper_additional_params_command(params['additional_params'])
        
        cmd = f"terastitcher --import {volume_input} {output_folder} {parameters} {additional_params}"
        
        utils.save_dict_as_json(f"{self.__output_folder}/metadata/import_params.json", params, self.__verbose)
        
        return cmd
    
    def __get_aprox_number_processes_align_step(self, config_params:dict) -> int:
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
            print("Please check the parameters")
            return 2
        
        # Partitioning depth for the tiles
        partitioning_depth = math.ceil( config_params['image_depth'] / config_params['subvoldim'] )
        left = 2
        right = config_params['number_processes']

        while (True):
            mid_process = int((left + right) / 2)
            tiling_proc = 2 * ( mid_process - 1)
            
            if partitioning_depth > tiling_proc:
                return mid_process + 1
            
            else:
                right = mid_process + 1
                
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
        
        input_xml = f"--projin={self.__output_folder}/xmls/xml_import.xml"
        output_xml = f"--projout={self.__output_folder}/xmls/xml_displcomp.xml"
        parallel_command = ''

        if self.__parallel and self.__computation == 'cpu':
            parallel_command = self.__build_parallel_command(params, 'align')
    
        else:
            # Sequential execution or gpu execution if USECUDA_X_NCC flag is 1
            parallel_command = f"terastitcher"
        
        parameters = helper_build_param_value_command(params)
        
        cmd = f"{parallel_command} --displcompute {input_xml} {output_xml} {parameters} > {self.__output_folder}/xmls/step2par.txt"
        utils.save_dict_as_json(f"{self.__output_folder}/metadata/align_params.json", params, self.__verbose)
        
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
        
        input_xml = f"--projin={self.__output_folder}/xmls/{input_xml}"
        output_xml = f"--projout={self.__output_folder}/xmls/{output_xml}"
        parameters = ''
        
        if params:
            parameters = helper_build_param_value_command(params)
        
        cmd = f"terastitcher --{step_name} {input_xml} {output_xml} {parameters}"
        utils.save_dict_as_json(f"{self.__output_folder}/metadata/{step_name}_params.json", params, self.__verbose)
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
        
        # TODO Check the best number of processes using formula
        input_xml = f"--projin={self.__output_folder}/xmls/xml_merging.xml"
        output_path = f"--volout={self.__output_folder}"
        parallel_command = ''
        
        if self.__parallel and self.__computation == 'cpu':
            parallel_command = self.__build_parallel_command(params, 'merge')
    
        else:
            # Sequential execution or gpu execution if USECUDA_X_NCC flag is 1
            parallel_command = f"terastitcher"
        
        parameters = helper_build_param_value_command(params)
        
        cmd = f"{parallel_command} --merge {input_xml} {output_path} {parameters} > {self.__output_folder}/xmls/step6par.txt"
        utils.save_dict_as_json(f"{self.__output_folder}/metadata/merge_params.json", params, self.__verbose)
        
        return cmd
    
    def execute_pipeline(self, config:dict) -> None:
        """
        Executes the terastitcher's stitching pipeline that includes the following steps:
        Import, Align, Project, Threshold, Place and Merge. Please refer to the following
        link for more information: https://github.com/abria/TeraStitcher/wiki/Stitching-pipeline
        
        Parameters
        ------------------------
        params: dict
            Configuration dictionary for the stitching pipeline. It should include the configuration
            for each of the steps in the pipeline. i.e. {'import': {...}, 'align': {...}, ...}
        
        """
        
        # Step 1
        print("\n- Import step...")
        for out in utils.execute_command(
            self.import_step_cmd(config['import']), self.__verbose
        ):
            print(out)
        
        # Step 2
        print("- Align step...")
        for out in utils.execute_command(
            self.align_step_cmd(config['align']), self.__verbose
        ):
            print(out)
        
        # Step 3
        print("- Projection step...")
        for out in utils.execute_command(
            self.input_output_step_cmd(
                'displproj', 'xml_displcomp.xml', 'xml_displproj.xml'
            ), self.__verbose
        ):
            print(out)
          
        # Step 4
        print("- Threshold step...")
        for out in utils.execute_command(
            self.input_output_step_cmd(
                'displthres', 'xml_displproj.xml', 'xml_displthres.xml', config['threshold']
            ), self.__verbose
        ):
            print(out)
        
        # Step 5
        print("- Placing tiles step...")
        for out in utils.execute_command(
            self.input_output_step_cmd(
                'placetiles', 'xml_displthres.xml', 'xml_merging.xml'
            ), self.__verbose
        ):
            print(out)
        
        # Step 6
        print("- Merging step...")
        for out in utils.execute_command(
            self.merge_step_cmd(config["merge"]), self.__verbose
        ):
            print(out)

def main():
    input_data = '/home/data/Project1/Terastitcher/TestData/mouse.cerebellum.300511.sub3/tomo300511_subv3'#"C:/Users/camilo.laiton/Documents/Project1/Terastitcher/TestData/mouse.cerebellum.300511.sub3/tomo300511_subv3"#
    output_folder = "/home/data/Project1/Terastitcher/TestData/linux_test_3d"#"C:/Users/camilo.laiton/Documents/Project1/Terastitcher/TestData/test_processes"
    
    # TODO if we pass another path that exists instead of parastitcher's path, it builds the command
    #parastitcher_path_windows = 'C:/Users/camilo.laiton/Documents/Project1/Terastitcher/TeraStitcher-portable-1.11.10-win64/pyscripts/Parastitcher.py'
    parastitcher_path_linux = '/home/TeraStitcher-portable-1.11.10-with-BF-Linux/pyscripts/Parastitcher.py'
    
    terastitcher_tool = TeraStitcher(
        input_data=input_data,
        output_folder=output_folder,
        parallel=True,
        computation='cpu',
        parastitcher_path=parastitcher_path_linux,
        verbose=True
    )

    config = {
        "import" : {
            'ref1':'1',#'H',
            'ref2':'-2',#'V',
            'ref3':'3',#'D',
            'vxl1':'0.8',#'1.800',
            'vxl2':'0.8',#'1.800',
            'vxl3':'1',#'2',
            'additional_params':['sparse_data', 'libtiff_uncompress'] # 'rescan'
        },
        "align" : {
            'cpu_params' : {
                'estimate_processes': True,
                'image_depth': 1000, # This parameter should be passed if estimate_processes is True
                'number_processes': 10, # np for linux or mac
                'hostfile': '/home/data/Project1/Terastitcher/TestData/hostfile',
                'additional_params' : ['use-hwthread-cpus', 'allow-run-as-root']
            },
            'subvoldim': 100,
        },
        "threshold" : {
            "threshold" : 0.7
        },
        "merge" : {
            'cpu_params' : {
                # TODO Estimate processes for merge step
                'estimate_processes': False,
                'image_depth': 1000, # This parameter should be passed if estimate_processes is True
                'number_processes': 6, # np for linux or mac
                'hostfile': '/home/data/Project1/Terastitcher/TestData/hostfile',
                'additional_params' : ['use-hwthread-cpus', 'allow-run-as-root']
            },
            'volout_plugin' : '"TiledXY|2Dseries"', # This parameter need to be provided with "" TiledXY|3Dseries
            'slicewidth' : 20000,
            'sliceheight' : 20000,
        }
    }
    
    terastitcher_tool.execute_pipeline(config)

if __name__ == "__main__":
    main()