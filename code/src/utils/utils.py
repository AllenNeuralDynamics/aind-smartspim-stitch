import os
import subprocess
import json

def create_folder(dest_dir:str) -> None:

    """
    Create new folders.
    
    Parameters
    ------------------------
        - dest_dir (str): Path where the folder will be created if it does not exist.
    
    Raises
    ------------------------
        - OSError if the folder exists.
    
    Returns:
    ------------------------
        - None
    """
    
    if not (os.path.exists(dest_dir)):
        try:
            print(f"Creating new directory: {dest_dir}")
            os.makedirs(dest_dir)
        except OSError as e:
            if (e.errno != os.errno.EEXIST):
                raise
            
def execute_command(command:str) -> None:
        """
        Execute a shell command.
        
        Parameters
        ------------------------
        
            - command (str): Command that we want to execute.
        
        Returns
        ------------------------
            - None
        """
        popen = subprocess.Popen(command, stdout=subprocess.PIPE, universal_newlines=True, shell=True)
        for stdout_line in iter(popen.stdout.readline, ""):
            yield stdout_line
        popen.stdout.close()
        return_code = popen.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
        
def save_dict_as_json(filename:str, dictionary:dict, verbose:False) -> None:
    """
    
    """
    with open(filename, "w") as json_file:
        json.dump(dictionary, json_file, indent=4)
    
    if verbose:
        print(f"- Json file saved: {filename}")