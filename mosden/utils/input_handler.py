import json
from mosden.utils.defaults import DEFAULTS
import logging

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
        
        self.independent_fission_yields = ['omcchain']
        self.cumulative_fission_yields = ['nfy']
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

        if data['file_options']['unprocessed_data_dir'] == data['file_options']['processed_data_dir']:
            raise ValueError("Unprocessed and processed data directories cannot be the same.")
        if data['file_options']['unprocessed_data_dir'] == data['file_options']['output_dir']:
            raise ValueError("Unprocessed data directory cannot be the same as the output directory.")
        if data['modeling_options']['parent_feeding'] and not data['modeling_options']['concentration_handling'] == 'depletion':
            raise ValueError("Parent feeding option requires depletion method for concentration handling")
        if data['modeling_options']['concentration_handling'] == 'IFY':
            ify_in_yields = any(val in data['data_options']['fission_yield'] for val in self.independent_fission_yields)
            if not ify_in_yields:
                raise ValueError(f"IFY requires independent fission yield data {self.independent_fission_yields} (not in {data["data_options"]["fission_yield"]})")
        if data['modeling_options']['concentration_handling'] == 'CFY':
            cfy_in_yields = any(val in data['data_options']['fission_yield'] for val in self.cumulative_fission_yields)
            if not cfy_in_yields:
                raise ValueError(f"CFY requires cumulative fission yield data {self.cumulative_fission_yields} (not in {data["data_options"]["fission_yield"]})")
        
        possible_decay_spacings = ['linear', 'log']
        _check_if_in_options(data['data_options']['decay_time_spacing'], possible_decay_spacings)
        possible_concentration_options = ['CFY', 'IFY']
        _check_if_in_options(data['modeling_options']['concentration_handling'], possible_concentration_options)
        possible_irradiation_options = ['saturation', 'pulse']
        _check_if_in_options(data['modeling_options']['irrad_type'], possible_irradiation_options)
        possible_count_rate_options = ['data']
        _check_if_in_options(data['modeling_options']['count_rate_handling'], possible_count_rate_options)
        possible_group_method_options = ['nlls']
        _check_if_in_options(data['group_options']['method'], possible_group_method_options)
        possible_sampler_funcs = ['normal', 'uniform']
        _check_if_in_options(data['group_options']['sample_func'], possible_sampler_funcs)

        possible_reprocessing_locations = ['incore', 'excore', 'net', '']
        for item in data['modeling_options']['reprocessing_locations']:
            _check_if_in_options(item, possible_reprocessing_locations)
        possible_spatial_scaling = ['scaled', 'unscaled', 'explicit']
        _check_if_in_options(data['modeling_options']['spatial_scaling'], possible_spatial_scaling)

        if sum(data['data_options']['fissile_fractions'].values()) != 1.0:
            raise ValueError("Fissile fractions must sum to 1.0. Current sum: "
                             f"{sum(data['data_options']['fissile_fractions'].values())}")
        return




    

if __name__ == "__main__":
    handler = InputHandler("../../examples/keepin_1957/input.json")
    input_data = handler.read_input()
    handler.check_behaviour(input_data)