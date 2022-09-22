from typing import List, Any, Optional, Union
from .ng_layer import NgLayer
from pint import UnitRegistry
from pathlib import Path
import sys
import re
from utils import utils

# IO types
PathLike = Union[str, Path]

class NgState():
    def __init__(self, input_config:dict, output_json:PathLike, verbose:Optional[bool]=False) -> None:
        """
        Class constructor
        
        Parameters
        ------------------------
        image_config: dict
            Dictionary with the image configuration based on neuroglancer documentation.
        output_json: PathLike
            Path where the json will be written.
        
        verbose: Optional[bool]
            If true, additional information will be shown. Default False.
        """
        
        self.input_config = input_config
        self.output_json = Path(self.__fix_output_json_path(output_json))
        self.verbose = verbose
        
        # State and layers attributes
        self.__state = {}
        self.dimensions = {}
        self.__layers = []
        
        self.__prefix = 'ng_link_'
        
        # Initialize principal attributes
        self.initialize_attributes(self.input_config)
    
    def __fix_output_json_path(self, output_json:PathLike) -> str:
        
        """
        Fixes the json output path in order to have a similar structure for all links.
        
        Parameters
        ------------------------
        output_json: PathLike
            Path of the json output path.
        
        Returns
        ------------------------
        str
            String with the fixed outputh path.
        """
        output_json = Path(
            str(output_json).replace(
                '/home/jupyter/', ''
            ).replace(
                "////", "//"
            )
        )
        name = str(output_json.name)
        
        if not name.endswith('.json'):
            name += '.json'
            
        if not name.startswith('ng_link_'):
            name = 'ng_link_' + name
        
        output_json = output_json.parent.joinpath(name)
        
        return output_json
    
    def __unpack_axis(self, axis_values:dict, dest_metric:Optional[str]='meters') -> List:
        """
        Unpack axis voxel sizes converting them to meters which neuroglancer uses by default.
        
        Parameters
        ------------------------
        axis_values: dict
            Dictionary with the axis values with the following structure for an axis: 
            e.g. for Z dimension {
                "voxel_size": 2.0,
                "unit": 'microns'
            }
        
        dest_metric: Optional[str]
            Destination metric to be used in neuroglancer. Default 'meters'.
        
        Returns
        ------------------------
        List
            List with two values, the converted quantity and it's metric in neuroglancer format.
        """
        
        if dest_metric not in ['meters', 'seconds']:
            raise NotImplementedError(f"{dest_metric} has not been implemented")
        
        # Converting to desired metric
        unit_register = UnitRegistry()
        quantity = axis_values['voxel_size'] * unit_register[axis_values['unit']]    
        dest_quantity = quantity.to(dest_metric)
        
        # Neuroglancer metric
        neuroglancer_metric = None
        if dest_metric == 'meters':
            neuroglancer_metric = 'm'
        
        elif dest_metric == 'seconds':
            neuroglancer_metric = 's'
        
        return [dest_quantity.m, neuroglancer_metric]
    
    def set_dimensions(self, dimensions:dict) -> None:
        
        """
        Set dimensions with voxel sizes for the image.
        
        Parameters
        ------------------------
        dimensions: dict
            Dictionary with the axis values with the following structure for an axis: 
            e.g. for Z dimension {
                "voxel_size": 2.0,
                "unit": 'microns'
            }
            
        """
        
        if not utils.check_type_helper(dimensions, dict):
            raise ValueError(f"Dimensions accepts only dict. Received value: {dimensions}")

        regex_axis = r'([x-zX-Z])$'
        
        for axis, axis_values in dimensions.items():
            
            if re.search(regex_axis, axis):
                self.dimensions[axis] = self.__unpack_axis(axis_values)
            else:
                self.dimensions[axis] = self.__unpack_axis(axis_values, 'seconds')
                
    @property
    def layers(self) -> List[dict]:
        """
        Property getter of layers.
        
        Returns
        ------------------------
        List[dict]
            List with neuroglancer layers' configuration.
        """
        return self.__layers
    
    def set_layers(self, layers:List[dict]) -> None:
        """
        Property setter of layers.
        
        Parameters
        ------------------------
        layers: List[dict]
            List that contains a configuration for each image layer.
            
        """
        
        if not utils.check_type_helper(layers, list):
            raise ValueError(f"layers accepts only list. Received value: {layers}")

        for layer in layers:
            self.__layers.append(NgLayer(layer).get_layer())
        
    def initialize_attributes(self, input_config:dict) -> None:
        """
        Initializes the following attributes for a given image layer: dimensions, layers.
        
        Parameters
        ------------------------
        input_config: dict
            Dictionary with the configuration for each image layer
    
        """
        
        # Initializing dimension
        self.set_dimensions(input_config['dimensions'])
        
        # Initializing layers
        self.set_layers(input_config['layers'])
        
        self.__state = self.get_state()
    
    def get_state(self) -> dict:
        """
        Property getter of state.
        
        Returns
        ------------------------
        dict
            Dictionary with the actual layer state.
        """
        
        actual_state = {}
        actual_state['dimensions'] = {}

        # Getting actual state for all attributes
        for axis, value_list in self.dimensions.items():
            actual_state['dimensions'][axis] = value_list
            
        actual_state['layers'] = self.__layers
        
        return actual_state
    
    def save_state_as_json(
        self, 
        output_json:Optional[PathLike]='', 
        update_state:Optional[bool]=False
    ) -> None:
        """
        Saves a neuroglancer state as json.
        
        Parameters
        ------------------------
        output_json: Optional[PathLike]
            Path where the neuroglancer state will be written as json
        
        update_state: Optional[bool]
            Updates the neuroglancer state with dimensions and layers in case they were changed 
            using class methods.
        """
        output_json = str(output_json)
        
        if not len(output_json):
            output_json = self.output_json
        else:
            output_json = self.__fix_output_json_path(output_json)
            self.output_json = output_json
        
        if update_state:
            self.__state = self.state()
    
        utils.save_dict_as_json(output_json, self.__state, verbose=self.verbose)
    
    def get_url_link(
        self, 
        base_url:Optional[str]='https://neuroglancer-demo.appspot.com/',
        save_txt:Optional[bool]=True,
        output_txt:Optional[PathLike]=''
    ) -> str:
        """
        Creates the neuroglancer link based on where the json will be written.
        
        Parameters
        ------------------------
        base_url: Optional[str]
            Base url where neuroglancer app was deployed. Default: https://neuroglancer-demo.appspot.com/
        
        save_txt: Optional[bool]
            Saves the url link to visualize data as a .txt file in a specific path given by 
            output_txt parameter.
        
        output_txt: Optional[PathLike]
            Path where the .txt file will be written. Default: '' which then will be converted to 
            self.output_json value.
        
        Returns
        ------------------------
        str
            Neuroglancer url to visualize data.
        """
        output_txt = str(output_txt)
        
        json_path = str(self.output_json)
        json_path = 'gs://' + json_path
        
        link = f"{base_url}#!{json_path}"
        
        if save_txt:
            
            if not len(output_txt):
                output_txt = self.output_json.parent
            
            utils.save_string_to_txt(link, Path(output_txt).joinpath('ng_link.txt'))
        
        return link
    
if __name__ == '__main__':
    
    example_data = {
        'dimensions': {
            # check the order
            "z": {
                "voxel_size": 2.0,
                "unit": 'microns'
            },
            "y": {
                "voxel_size": 1.8,
                "unit": 'microns'
            },
            "x": {
                "voxel_size": 1.8,
                "unit": 'microns'
            },
            "t": {
                "voxel_size": 0.001,
                "unit": 'seconds'
            },
        },
        'layers': [
            {
                'source': 'image_path',
                'channel': 0,
                # 'name': 'image_name_0',
                'shader': {
                    'color': 'green',
                    'emitter': 'RGB',
                    'vec': 'vec3'
                },
                'shaderControls': { # Optional
                    "normalized": {
                        "range": [0, 500]
                    }
                }
            },
            {
                'source': 'image_path',
                'channel': 1,
                # 'name': 'image_name_1',
                'shader': {
                    'color': 'red',
                    'emitter': 'RGB',
                    'vec': 'vec3'
                },
                'shaderControls': { # Optional
                    "normalized": {
                        "range": [0, 500]
                    }
                }
            }
        ]
    }
    
    neuroglancer_link = NgState(example_data, "path_to_json.json")
    data = neuroglancer_link.state
    print(data)
    # neuroglancer_link.save_state_as_json('test.json')
    neuroglancer_link.save_state_as_json()
    print(neuroglancer_link.get_url_link())
    