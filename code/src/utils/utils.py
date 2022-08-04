import os
import subprocess
import json
from pathlib import Path
from typing import Optional, List, Union
import shutil

# IO types
PathLike = Union[str, Path]

def create_folder(dest_dir:PathLike, verbose:Optional[bool]=False) -> None:
    """
    Create new folders.
    
    Parameters
    ------------------------
    dest_dir: PathLike 
        Path where the folder will be created if it does not exist.
    verbose: Optional[bool]
        If we want to show information about the folder status. Default False.
        
    Raises
    ------------------------
    OSError: 
        if the folder exists.
    
    """
    
    if not (os.path.exists(dest_dir)):
        try:
            if verbose:
                print(f"Creating new directory: {dest_dir}")
            os.makedirs(dest_dir)
        except OSError as e:
            if (e.errno != os.errno.EEXIST):
                raise

def delete_folder(dest_dir:PathLike, verbose:Optional[bool]=False) -> None:
    """
    Delete a folder path.
    Parameters
    ------------------------
    dest_dir: PathLike
        Path that will be removed.
    verbose: Optional[bool]
        If we want to show information about the folder status. Default False.
        
    Raises
    ------------------------
    shutil.Error:
        If the folder could not be removed.
    
    Returns
    ------------------------
        None
    
    """
    if (os.path.exists(dest_dir)):
        try:
            shutil.rmtree(dest_dir)
            if verbose:
                print(f"Folder {dest_dir} was removed!")
        except shutil.Error as e:
            print(f"Folder could not be removed! Error {e}")

def execute_command(command:str, print_command:bool=False) -> None:
    """
    Execute a shell command.
    
    Parameters
    ------------------------
    command: str
        Command that we want to execute.
    print_command: bool
        Bool that dictates if we print the command in the console.
        
    Raises
    ------------------------
    CalledProcessError: 
        if the command could not be executed (Returned non-zero status).
    
    """
    
    if print_command:
        print(command)
    
    popen = subprocess.Popen(command, stdout=subprocess.PIPE, universal_newlines=True, shell=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    
def check_path_instance(obj: object) -> bool:
    """
    Checks if an objects belongs to pathlib.Path subclasses.
    
    Parameters
    ------------------------
    obj: object
        Object that wants to be validated.
    
    Returns
    ------------------------
    bool:
        True if the object is an instance of Path subclass, False otherwise.
    """
    
    for childclass in Path.__subclasses__():
        if isinstance(obj, childclass):
            return True
        
    return False

def save_dict_as_json(
        filename:str, 
        dictionary:dict, 
        verbose:Optional[bool]=False
    ) -> None:
    """
    Saves a dictionary as a json file.
    
    Parameters
    ------------------------
    filename: str
        Name of the json file.
    dictionary: dict
        Dictionary that will be saved as json.
    verbose: Optional[bool]
        True if you want to print the path where the file was saved.
        
    """
    
    if dictionary == None:
        dictionary = {}
    
    else:
        for key, value in dictionary.items():
            # Converting path to str to dump dictionary into json
            if check_path_instance(value):
                # TODO fix the \\ encode problem in dump
                dictionary[key] = str(value)
    
    with open(filename, "w") as json_file:
        json.dump(dictionary, json_file, indent=4)
    
    if verbose:
        print(f"- Json file saved: {filename}")
        
def helper_build_param_value_command(params:dict, equal_con:Optional[bool]=True) -> str:
    """
    Helper function to build a command based on key:value pairs.
    
    Parameters
    ------------------------
    params: dict
        Dictionary with key:value pairs used for building the command.
    equal_con: Optional[bool]
        Indicates if the parameter is followed by '='. Default True.
    Returns
    ------------------------
    str:
        String with the parameters.
    
    """
    equal = ' '
    if equal_con:
        equal = '='
        
    parameters = ''
    for (param, value) in params.items():
        if type(value) in [str, float, int] or check_path_instance(value):
            parameters += f"--{param}{equal}{str(value)} "
            
    return parameters

def helper_additional_params_command(params:List[str]) -> str:
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