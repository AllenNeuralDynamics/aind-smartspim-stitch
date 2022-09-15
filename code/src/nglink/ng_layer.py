from typing import Any, Optional, Union
from pathlib import Path
import sys

sys.path.append('../')
from utils import utils

# IO types
PathLike = Union[str, Path]

class NgLayer():
    
    def __init__(self, image_config:dict, image_type:Optional[str]='image') -> None:
        """
        Class constructor
        
        Parameters
        ------------------------
        image_config: dict
            Dictionary with the image configuration based on neuroglancer documentation.
        image_type: Optional[str]
            Image type based on neuroglancer documentation.
            
        """
        
        self.layer_state = {}
        self.image_config = image_config
        self.image_source = self.__fix_image_source(image_config['source'])
        self.image_type = image_type
        
        self.update_state(image_config)
    
    def __fix_image_source(self, source_path:PathLike) -> str:
        """
        Fixes the image source path to include the type of image neuroglancer accepts.
        
        Parameters
        ------------------------
        source_path: PathLike
            Path where the image is located.
        
        Returns
        ------------------------
        str
            Fixed path for neuroglancer json configuration.
        """
        
        source_path = str(source_path)
        
        # replacing jupyter path
        source_path = source_path.replace('/home/jupyter/', '')
        source_path = 'gs://' + source_path
        
        if source_path.endswith('.zarr'):
            source_path = "zarr://" + source_path
            
        else:
            raise NotImplementedError("This format has not been implemented yet for visualization")
        
        return source_path
    
    def set_default_values(self, image_config:dict={}, overwrite:bool=False) -> None:
        """
        Set default values for the image.
        
        Parameters
        ------------------------
        image_config: dict
            Dictionary with the image configuration. Similar to self.image_config
        
        overwrite: bool
            If the parameters already have values, with this flag they can be overwritten.
        
        """
        
        
        if overwrite:
            self.__set_image_channel(0)
            self.__set_shader_control(
                {
                    "normalized": {
                        "range": [0, 600]
                    }
                }
            )
            self.__set_visible(True)
            self.__set_str_state('name', Path(self.image_source).stem)
            self.__set_str_state('type', self.image_type)
        
        elif len(image_config):
            # Setting default image_config in json image layer
            if 'channel' not in image_config:
                # Setting channel to 0 for image
                self.__set_image_channel(0)
                
            if 'shaderControls' not in image_config:
                self.__set_shader_control(
                    {
                        "normalized": {
                            "range": [0, 600]
                        }
                    }
                )
                
            if 'visible' not in image_config:
                self.__set_visible(True)
                
            if 'name' not in image_config:
                try:
                    channel = self.layer_state['localDimensions']["c'"][0]
                
                except KeyError:
                    channel = ''
                self.__set_str_state('name', Path(self.image_source).stem + f"_{channel}")
                
            if 'type' not in image_config:
                self.__set_str_state('type', self.image_type)
    
    def update_state(self, image_config:dict) -> None:
        """
        Set default values for the image.
        
        Parameters
        ------------------------
        image_config: dict
            Dictionary with the image configuration. Similar to self.image_config
            e.g.: image_config = {
                'type': 'image', # Optional
                'source': 'image_path',
                'channel': 0, # Optional
                'name': 'image_name', # Optional
                'shader': {
                    'color': 'green',
                    'emitter': 'RGB',
                    'vec': 'vec3'
                },
                'shaderControls': { # Optional
                    "normalized": {
                        "range": [0, 600]
                    }
                }
            }
        """
        
        for param, value in image_config.items():
            if param in ['type', 'source', 'name']:
                self.__set_str_state(param, value)
                
            if param in ['visible']:
                self.__set_boolean_state(param, value)
            
            if param == 'shader':
                self.__set_shader(
                    self.__create_shader(
                        value
                    )
                )
                
            if param == 'channel':
                self.__set_image_channel(value)
                
            if param == 'shaderControls':
                self.__set_shader_control(value)
        
        self.set_default_values(image_config)
    
    def __set_str_state(self, param:str, value:str) -> None:
        
        """
        Sets a string value for a state's parameter.
        
        Parameters
        ------------------------
        param: str
            Key name for the parameter inside the layer state.
        
        value: str
            Value for the layer state parameter
            
        Raises
        ------------------------
        ValueError:
            If the provided parameter is not a string.
            
        """
        
        if param == 'source':
            value = str(self.image_source)            
        
        if not utils.check_type_helper(value, str):
            raise ValueError(f"{param} accepts only str. Received value: {value}")
            
        self.layer_state[param] = value
    
    def __set_boolean_state(self, param:str, value:bool) -> None:
        """
        Sets a boolean value for a state's parameter.
        
        Parameters
        ------------------------
        param: str
            Key name for the parameter inside the layer state.
        
        value: bool
            Value for the layer state parameter
            
        Raises
        ------------------------
        ValueError:
            If the provided parameter is not a boolean.
        """
        
        if not utils.check_type_helper(value, bool):
            raise ValueError(f"{param} accepts only bool. Received value: {value}")
            
        self.layer_state[param] = value
    
    def __create_shader(self, shader_config:dict) -> str:
        """
        Creates a configuration for the neuroglancer shader.
        
        Parameters
        ------------------------
        shader_config: dict
            Configuration of neuroglancer's shader.
        
        Returns
        ------------------------
        str
            String with the shader configuration for neuroglancer.
        """
        
        color = shader_config['color']
        emitter = shader_config['emitter']
        vec = shader_config['vec']
        
        # Add all necessary ui controls here
        ui_controls = [
            f"#uicontrol {vec} color color(default=\"{color}\")",
            "#uicontrol invlerp normalized",
        ]
        
        # color emitter
        emit_color = "void main() {\n" + f"emit{emitter}(color * normalized());" + "\n}"
        shader_string = ""
        
        for ui_control in ui_controls:
            shader_string += ui_control + '\n'
        
        shader_string += emit_color
        
        return shader_string
        
    def __set_shader(self, shader_config:str) -> None:
        """
        Sets a configuration for the neuroglancer shader.
        
        Parameters
        ------------------------
        shader_config: str
            Shader configuration for neuroglancer in string format.
        
        Raises
        ------------------------
        ValueError:
            If the provided shader_config is not a string.
        
        """
        
        # #uicontrol vec3 color color(default=\"green\")\n#uicontrol invlerp normalized\nvoid main() {\n  emitRGB(color * normalized());\n}
        if not utils.check_type_helper(shader_config, str):
            raise ValueError(f"Shader accepts only str. Received value: {value}")
        
        self.layer_state['shader'] = shader_config
        
    def __set_shader_control(self, shader_control_config:dict) -> None:
        """
        Sets a configuration for the neuroglancer shader control.
        
        Parameters
        ------------------------
        shader_control_config: dict
            Shader control configuration for neuroglancer.
        
        Raises
        ------------------------
        ValueError:
            If the provided shader_control_config is not a dictionary.
        
        """
        
        if not utils.check_type_helper(shader_control_config, dict):
            raise ValueError(f"Shader control accepts only dict. Received value: {value}")
        
        self.layer_state['shaderControls'] = shader_control_config 
    
    def __set_image_channel(self, channel:int) -> None:
        """
        Sets the image channel in case the file contains multiple channels.
        
        Parameters
        ------------------------
        channel: int
            Channel position. It will be incremented in 1 since neuroglancer channels starts in 1.
        
        Raises
        ------------------------
        ValueError:
            If the provided channel is not an integer.
        
        """
        
        if not utils.check_type_helper(channel, int):
            raise ValueError(f"Channel accepts only integer. Received value: {value}")

        self.layer_state['localDimensions'] = {
            "c'": [
                channel + 1,
                ""
            ]
        }
        
    def __set_visible(self, visible:bool) -> None:
        """
        Sets the visible parameter in neuroglancer link.
        
        Parameters
        ------------------------
        visible: bool
            Boolean that dictates if the image is visible or not.
        
        Raises
        ------------------------
        ValueError:
            If the parameter is not an boolean.
        
        """
        
        if not utils.check_type_helper(visible, bool):
            raise ValueError(f"Visible param accepts only boolean. Received value: {value}")

        self.layer_state['visible'] = visible
        
    def get_layer(self):
        return self.layer_state

if __name__ == '__main__':
    
    example_data = {
        'type': 'image', # Optional
        'source': 'image_path',
        'channel': 1, # Optional
        # 'name': 'image_name', # Optional
        'shader': {
            'color': 'red',
            'emitter': 'RGB',
            'vec': 'vec3'
        },
        'shaderControls': { # Optional
            "normalized": {
                "range": [0, 500]
            }
        },
        'visible': False # Optional
    }
    
    dict_data = NgLayer(image_config=example_data).get_layer()
    print(dict_data)