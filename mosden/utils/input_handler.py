import json
from mosden.utils.defaults import DEFAULTS
import logging
import jsonschema
from jsonschema import validate
import os

class InputHandler:
    _default_counts = {}

    def __init__(self, input_path: str) -> None:
        """
        This class checks and parses input files.

        Parameters
        ----------
        input_path : str
            Path to the input file
        """
        self.input_path = input_path
        self.preproc_choices: dict = dict()
        self.logger = logging.getLogger(__name__)
        self.preprocessing_occurance_index = 2
        self.leaf_dict_keys: set = set((
            "reprocessing",
            "fissile_fractions"
        ))
        
        return None
    
    def read_input(self, check: bool=True, apply_defaults: bool=True) -> dict:
        """
        Read the input file and return the data as a dictionary.

        Parameters
        ----------
        check : bool, optional
            If True, checks the behaviour of the input data. Default is True.
        apply_defaults : bool, optional
            If True, applies default values to missing keys. Default is True.

        Returns
        -------
        output : dict
            Dictionary containing settings and data selections.
        """
        try:
            with open(self.input_path, 'r') as file:
                output = json.load(file)
        except TypeError or FileNotFoundError or json.decoder.JSONDecodeError:
            output = dict()
        if apply_defaults:
            output = self._apply_defaults(output, DEFAULTS)
        if check:
            self._check_behaviour(output)
        return output
    
    def _apply_defaults(self, data: dict, defaults: dict, path: str='') -> dict:
        """
        Apply default values to the input data.

        Parameters
        ----------
        data : dict
            The input data to apply defaults to.
        defaults : dict
            The default values to apply.
        path : str, optional
            The current path in the data hierarchy, by default ''

        Returns
        -------
        dict
            The input data with defaults applied.
        """
        final = {}
        for k in defaults.keys():
            full_key = f"{path}.{k}" if path else k
            if k in data:
                key_can_vary = k in self.leaf_dict_keys
                default_not_dict = not isinstance(defaults[k], dict)
                data_not_dict = not isinstance(data[k], dict)
                if key_can_vary or default_not_dict or data_not_dict:
                    final[k] = data[k]
                else:
                    final[k] = self._apply_defaults(data[k], defaults[k], full_key)
            else:
                InputHandler._default_counts.setdefault(full_key, 0)
                InputHandler._default_counts[full_key] += 1
                if InputHandler._default_counts[full_key] == self.preprocessing_occurance_index:
                    self.logger.warning(f"Using default for '{full_key}': {defaults[k]!r}", stacklevel=2)
                final[k] = defaults[k]
        return final
    
    def _check_behaviour(self, data: dict) -> None:
        """
        Check the behaviour of the input data.

        Parameters
        ----------
        data : dict
            Dictionary containing settings and data selections.
        
        Raises
        ------
        ValueError
            If the behaviour is not supported.
        """
        def _check_if_in_options(item, options):
            if item not in options:
                raise ValueError(f'Option {item} not supported in {options}')
            return None

        if data['modeling_options']['base_removal_scaling'] == 0.0:
            self.logger.warning('Base removal scaling set to 0.0, see README for more information.')
        json_schema_path = os.path.join(os.path.dirname(__file__), '../templates/input_schema.json')
        with open(json_schema_path, 'r') as f:
            schema = json.load(f)
        
        validate(data, schema=schema)

        if sum(data['data_options']['fissile_fractions'].values()) != 1.0:
            raise ValueError("Fissile fractions must sum to 1.0. Current sum: "
                             f"{sum(data['data_options']['fissile_fractions'].values())}")
        if data['modeling_options']['irrad_type'] == 'intermediate' and data['modeling_options']['concentration_handling'] != 'OMC':
            raise ValueError('Intermediate numerical integration is only available with OpenMC concentration modeling')
        if data['group_options']['initial_params']['yields']:
            if len(data['group_options']['initial_params']['yields']) != data['group_options']['num_groups']:
                raise ValueError('Initial yield guess does not match number of groups')
            if len(data['group_options']['initial_params']['half_lives']) != data['group_options']['num_groups']:
                raise ValueError('Initial half life guess does not match number of groups')
        return




    

if __name__ == "__main__":
    handler = InputHandler("../../examples/keepin_1957/input.json")
    input_data = handler.read_input()
    handler.check_behaviour(input_data)