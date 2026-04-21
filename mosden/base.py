from mosden.utils.input_handler import InputHandler
from pathlib import Path
from mosden.utils.csv_handler import CSVHandler
import os
import logging
import json
import numpy as np
from time import time


class BaseClass:
    _INITIALIZED: bool = False

    def __init__(self, input_path: str) -> None:
        """
        This class serves as the base class for MoSDeN.

        Parameters
        ----------
        input_path : str
            Path to the input file
        """
        self.omc_data_words: list[str] = ['omcchain']
        self.endf_data_words: list[str] = ['endf']
        self.iaea_data_words: list[str] = ['iaea']
        self.jeff_data_words: list[str] = ['jeff']
        self.jendl_data_words: list[str] = ['jendl']


        self.input_path: str = input_path
        self.input_handler: InputHandler = InputHandler(input_path)
        self.input_data: dict = self.input_handler.read_input()
        self.multi_id: str = self.input_data.get('multi_id', None)

        file_options: dict = self.input_data.get('file_options', {})
        modeling_options: dict = self.input_data.get('modeling_options', {})
        data_options: dict = self.input_data.get('data_options', {})
        overwrite_options: dict = file_options.get('overwrite', {})
        group_options: dict = self.input_data.get('group_options', {})
        post_options: dict = self.input_data.get('post_options', {})


        self.log_file: str = file_options.get('log_file', 'log.log')
        self.log_level: int = file_options.get('log_level', 1)
        logger_overwrite: bool = overwrite_options.get('logger', False)

        self.logger: logging.Logger = logging.getLogger(__name__)
        if logger_overwrite and BaseClass._INITIALIZED:
            log_mode = 'w'
        else:
            log_mode = 'a'
        try:
            logging.basicConfig(filename=self.log_file,
                                level=self.log_level,
                                filemode=log_mode)
        except FileNotFoundError:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            logging.basicConfig(filename=self.log_file,
                                level=self.log_level,
                                filemode=log_mode)

        self.name: str = self.input_data['name']
        self.output_dir: str = self.input_data['file_options'].get('output_dir', '')
        if len(self.output_dir) > 1 and not self.output_dir.endswith('/'):
            self.output_dir = self.output_dir + '/'
        self.logger.debug(f'{self.name = }')

        self.energy_MeV: float = data_options.get('energy_MeV', 0.0)
        self.fissiles: dict[str, float] = data_options.get(
            'fissile_fractions', {})
        self.fissile_targets: list = list(self.fissiles.keys())

        self.debug_dnp_data: dict = data_options.get('debug_dnps', {})
        self.has_debug_dnps: bool = True
        if self.debug_dnp_data == {}:
            self.has_debug_dnps = False

        self.data_types: list[str] = [
            'fission_yield',
            'half_life',
            'cross_section',
            'emission_probability']

        self.data_dir: str = file_options['unprocessed_data_dir']
        self.lit_data_dir: str = file_options['literature_data_dir']
        self.repr_dir: str = file_options['reprocessing_data_dir']
        self.preprocess_overwrite: bool = overwrite_options.get('preprocessing', False)

        self.conc_method: str = modeling_options.get(
            'concentration_handling', 'CFY')
        self.omc = False
        if self.conc_method == 'OMC':
            self.omc = True
        self.conc_overwrite: bool = overwrite_options.get('concentrations', False)
        self.reprocessing: dict[str: float] = modeling_options.get(
            'reprocessing', {})
        self.reprocess: bool = (sum(self.reprocessing.values()) > 0)
        self.reprocess_locations: list[str] = modeling_options.get(
            'reprocessing_locations', [])
        self.t_in: float = modeling_options.get('incore_s', 0.0)
        self.t_ex: float = modeling_options.get('excore_s', 0.0)
        self.t_net: float = modeling_options.get('net_irrad_s', 0.0)
        self.t_net = self._update_t_net()
        self.irrad_type: str = modeling_options.get('irrad_type', 'saturation')
        self.spatial_scaling: dict[str: str] = modeling_options.get(
            'spatial_scaling', {})
        self.flux_scaling = self.spatial_scaling['flux']
        self.chem_scaling = self.spatial_scaling['reprocessing']
        self.base_repr_scale: float = modeling_options.get('base_removal_scaling', 0.5)
        self.temperature_K: float = data_options.get('temperature_K', 920)
        self.density_g_cc: float = data_options.get('density_g_cm3', 2.3275)
        self.openmc_settings: dict = modeling_options.get('openmc_settings', {})
        self.residual_masks: list[str] = modeling_options.get('residual_handling', ["post-irrad"])

        
        self.count_overwrite: bool = overwrite_options.get('count_rate', False)
        self.num_times: int = modeling_options['num_decay_times']
        self.decay_time: float = modeling_options['decay_time']
        self.decay_time_spacing: str = data_options['decay_time_spacing']
        self.count_method: str = self.input_data['modeling_options']['count_rate_handling']
        self.irrad_type: str = self.input_data['modeling_options']['irrad_type']
        self.seed: int = group_options.get('seed', 0)

        self.group_method: str = group_options.get('method', 'nlls')
        self.num_starts: int = group_options.get('parameter_guesses', 10)
        self.num_groups: int = group_options.get('num_groups', 6)
        self.group_overwrite: bool = overwrite_options.get('group_fitting', False)
        self.MC_samples: int = group_options.get('samples', 1)
        self.sample_func: str = group_options.get('sample_func', 'normal')
        self.initial_params: dict = group_options.get('initial_params', {'yields': [],
                                                                         "half_lives": []})
        self.energy_groups_MeV: list[float] = group_options.get('energy_groups_MeV', [0, 6.25e-7, 1e3])
        self.eV_midpoints: list[float] = self._get_midpoint_eVs(self.energy_groups_MeV)
        if len(self.energy_groups_MeV) >= 3:
            self.is_spectral_calculation = True
        else:
            self.is_spectral_calculation = False

        self.processed_data_dir: str = file_options['processed_data_dir']
        self.unprocessed_data_dir: str = file_options['unprocessed_data_dir']
        self.concentration_path: str = os.path.join(
            file_options['output_dir'], 'concentrations.csv')
        self.countrate_path: str = os.path.join(
            file_options['output_dir'], 'count_rate.csv')
        self.spectra_path: str = os.path.join(
            file_options['output_dir'], 'spectra.csv')
        self.spectra_count_path: str = os.path.join(
            file_options['output_dir'], 'spectra_counts.csv')
        self.group_path: str = os.path.join(
            file_options['output_dir'], 'group_parameters.csv')
        self.postproc_path: str = os.path.join(
            file_options['output_dir'], 'postproc.json')

        self.img_dir: str = self.output_dir + 'images/'
        self.post_overwrite: bool = overwrite_options.get('postprocessing', False)
        self.sens_subplot: bool = post_options.get('sensitivity_subplots', True)
        self.lit_data: list[str] = post_options.get('lit_data', ['keepin'])
        self.num_top = post_options.get('top_num_nuclides', {})
        self.self_relative_data: bool = post_options.get('self_relative_counts', False)
        self.num_top_yield = self.num_top.get('yield_top', 3)
        self.num_top_conc = self.num_top.get('conc_top', 3)
        self.num_over_time = self.num_top.get('conc_over_time_top', 3)
        self.nuc_colors = post_options.get('nuc_colors', {})
        self.num_stack = post_options.get('num_stacked_nuclides', 2)
        self.plot_means = post_options.get('plot_means', False)
        self.pcc_cutoff = post_options.get('pcc_cutoff', 0.2)
        self.plot_correlation = post_options.get('plot_correlation', False)

        self.post_irrad_only: bool = (len(self.residual_masks) == 1 and 'post-irrad' in self.residual_masks)
        self.no_post_irrad: bool = ('post-irrad' not in self.residual_masks and 'all' not in self.residual_masks)
        self.decay_times = self._set_decay_times()
        self.use_times = self._get_use_times()


        np.random.seed(self.seed)


        self.names: dict[str: str] = {
            'countsMC': 'countsMC',
            'groupfitMC': 'groupfitMC'
        }

        if BaseClass._INITIALIZED:
            self.load_post_data()
        else:
            BaseClass._INITIALIZED = True
        return None

    def time_track(self, starttime: float, modulename: str = '') -> None:
        self.logger.info(f'{modulename} took {round(time() - starttime, 3)}s')
        return None
    
    def _get_midpoint_eVs(self, energy_groups_MeV: list[float]) -> list[float]:
        return ([1e6 * (self.energy_groups_MeV[i] + self.energy_groups_MeV[i+1]) / 2 for i in range(len(self.energy_groups_MeV) - 1)])
     
    def _get_use_times(self, single_time_val: bool=False) -> np.ndarray[float]:
        """
        Get all the times steps over which count rate data exists

        Parameters
        ----------
        single_time_val : bool
            Whether the problem is evaluated at a single point in time

        Returns
        -------
        use_times : np.ndarray[float]
            The time values where data exists
        """
        if self.post_irrad_only:
            use_times = self.decay_times
        else:
            mask_data = self._get_times_and_rates()
            use_times = np.concatenate(([0], np.cumsum(mask_data['timesteps'])))

        if self.no_post_irrad:
            post_irrad_index = self.get_irrad_index(single_time_val)
            use_times = use_times[:post_irrad_index+1]
        return use_times

    
    def _set_cycle_times(self, residence_time: float) -> list[float]:
        """
        Returns the list of times applied in OpenMC for each residence time

        Parameters
        ----------
        residence_time : float
            In-core or ex-core residence time

        Returns
        -------
        values : list[float]
            List of times to pass to OpenMC per residence
        """
        max_value = np.min((self.openmc_settings['max_timestep'], residence_time))
        if max_value == 0.0:
            return []
        min_cycles = int(np.floor(residence_time/max_value))
        remainder = residence_time - min_cycles * max_value
        values = [max_value] * min_cycles + [remainder]
        if np.isclose(values[-1], 0.0):
            values = values[:-1]
        return values

    
    def _get_times_and_rates(self, f_in: float = 1.0) -> dict[str, list[float|int]]:
        """
        Calculates the time steps to evaluate in OpenMC, the source rates
        to use at each time step, and the chemical removal indeces where
        removal occurs.

        Parameters
        ----------
        f_in : float (optional)
            Only required if the flux is scaled. The in-core salt fraction.

        Returns
        -------
        time_rate_data : dict[str, list[float|int]]
            Keys are names for different datasets, values are the time-dependent
            data. Keys include `timesteps`, `source_rates`, `removal_indeces`,
            and `irrad_mask`
        """
        time_rate_data = dict()
        removal_indeces = list()
        timesteps = list()
        source_rates = list()
        irrad_residual_mask = list()
        current_time = 0
        index_counter = 0
        in_core = True

        incore_values = self._set_cycle_times(self.t_in)
        excore_values = self._set_cycle_times(self.t_ex)

        time_close = np.isclose(current_time, self.t_net)
        while current_time < self.t_net and not time_close:
            mask_val = 0
            if in_core:
                ts = incore_values
                region = 'incore'
                source = self.openmc_settings['source']
                in_core = False
                if 'incore' in self.residual_masks:
                    mask_val = 1
            else:
                ts = excore_values
                region = 'excore'
                source = 0
                in_core = True
                if 'excore' in self.residual_masks:
                    mask_val = 1

            for t in ts:
                current_time += t
                timesteps.append(t)
                if (region in self.reprocess_locations) or self.chem_scaling:
                    removal_indeces.append(index_counter)
                if self.flux_scaling:
                    source = self.openmc_settings['source'] * f_in
                source_rates.append(source)
                index_counter += 1
                time_close = np.isclose(current_time, self.t_net)
                if 'all' in self.residual_masks:
                    mask_val = 1
                irrad_residual_mask.append(mask_val)

        diff = sum(timesteps) - self.t_net
        timesteps[-1] = timesteps[-1] - diff

        decay_time_steps = np.diff(self.decay_times, prepend=[0.0])
        for t in decay_time_steps[1:]:
            timesteps.append(t)
            source_rates.append(0)

        time_rate_data['timesteps'] = timesteps
        time_rate_data['source_rates'] = source_rates
        time_rate_data['removal_indeces'] = removal_indeces
        time_rate_data['irrad_mask'] = irrad_residual_mask
        return time_rate_data

    
    def _set_decay_times(self) -> np.ndarray[float]:
        """
        Set the decay times based on the time spacing, final time, and
        number of times provided.

        Returns
        -------
        decay_times : np.ndarray[float]
            The array of time values
        
        Raises
        ------
        ValueError
            If the provided decay time spacing is invalid
        
        """
        if self.decay_time_spacing == 'linear':
            self.decay_times: np.ndarray = np.linspace(
                0, self.decay_time, self.num_times)
        elif self.decay_time_spacing == 'log':
            self.decay_times: np.ndarray = np.geomspace(
                1e-2, self.decay_time, self.num_times-1)
            self.decay_times = np.append([0], self.decay_times)
        else:
            raise ValueError(
                f"Decay time spacing '{self.decay_time_spacing}' not supported.")
        return self.decay_times

    
    def _update_t_net(self) -> float:
        """
        Changes the net irradiation time to ensure the sample is in the core outlet.
        This greatly improves the condition number of the least squares solve.

        Returns
        -------
        t_net : float
            The updated net irradiation time
        """
        cycle = self.t_in + self.t_ex
        k = np.ceil((self.t_net - self.t_in) / cycle)
        k = max(k, 0)
        return k * cycle + self.t_in
    
    def get_irrad_index(self, single_time_val: bool) -> int:
        """
        Calculate the burnup index at which the sample starts decay

        Parameters
        ----------
        single_time_val : bool
            If there is only a single time value, the index is 0

        Returns
        -------
        post_irrad_index: int
            Integer index position of initial decay
        """
        if single_time_val:
            return 0
        
        in_use_time = np.min((self.openmc_settings['max_timestep'], self.t_in))
        ex_use_time = np.min((self.openmc_settings['max_timestep'], self.t_ex))

        if self.t_in == 0 and self.t_ex != 0:
            ratio = self.t_net / ex_use_time
            return int(np.floor(ratio + (1 - 1e-12)))
        elif self.t_ex == 0 and self.t_in != 0:
            ratio = self.t_net / in_use_time
            return int(np.floor(ratio + (1 - 1e-12)))
        elif self.t_in == 0 and self.t_ex == 0:
            raise ValueError('Residence times cannot all be zero')

        cycle_time = in_use_time + ex_use_time
        n_full = np.floor(self.t_net / cycle_time)
        post_irrad_index = int(2 * n_full)

        remainder = self.t_net - n_full * cycle_time
        eps = 1e-12 * max(1.0, self.t_net)

        if remainder > eps:
            post_irrad_index += 1

        return post_irrad_index

    def load_post_data(self) -> dict[str: float | str | list]:
        """
        Load post-processing data

        Returns
        -------
        dict[str: float|str|list]
            The post-processing data loaded from the JSON file.
        """
        if Path(self.postproc_path).exists():
            with open(self.postproc_path, 'r') as f:
                self.post_data = json.load(f)
        else:
            self.post_data: dict[str: float | str | list] = dict()
        return self.post_data

    def clear_post_data(self) -> None:
        """
        Clear the post-processing data.
        """
        self.post_data = {}
        with open(self.postproc_path, 'w') as f:
            json.dump(self.post_data, f, indent=4)
        return None

    def save_postproc(self) -> None:
        """
        Save post-processing data
        """
        try:
            with open(self.postproc_path, 'r') as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            self.clear_post_data()
            with open(self.postproc_path, 'r') as f:
                existing_data = json.load(f)
        try:
            existing_data.update(self.post_data)
        except AttributeError:
            self.post_data = dict()
        data_to_write = existing_data
        with open(self.postproc_path, 'w') as f:
            json.dump(data_to_write, f, indent=4)
        return None
    

    def _write_processed_data(self, data_type: str, overwrite: bool, data: dict[str, dict[str, float]]) -> None:
        """
        Read the processed data for a given fissile nuclide.

        Parameters
        ----------
        data_type : str
            The type of data to read (e.g., "fission_yield", "half_life",
                "cross_section", "emission_probability").
        overwrite : bool
            Whether or not to overwrite existing data
        data : dict[str: dict[str: float]]
            The processed data for the fissile nuclide.

        """
        data_path = os.path.join(self.processed_data_dir, f'{data_type}.csv')
        csv_handler = CSVHandler(data_path, create=False, overwrite=overwrite)
        if not csv_handler._file_exists():
            raise FileNotFoundError(
                f"Processed data file {data_path} does not exist.")
        data = csv_handler.write_csv(data)
        return None

    def _read_processed_data(self,
                             data_type: str) -> dict[str: dict[str: float]]:
        """
        Read the processed data for a given fissile nuclide.

        Parameters
        ----------
        data_type : str
            The type of data to read (e.g., "fission_yield", "half_life",
                "cross_section", "emission_probability").

        Returns
        -------
        data : dict[str: dict[str: float]]
            The processed data for the fissile nuclide.
        """
        data_path = os.path.join(self.processed_data_dir, f'{data_type}.csv')
        csv_handler = CSVHandler(data_path, create=False)
        if not csv_handler._file_exists():
            raise FileNotFoundError(
                f"Processed data file {data_path} does not exist.")
        data = csv_handler.read_csv()
        return data

    def _get_element_from_nuclide(self, nuclide: str) -> str:
        """
        Get the element from a given nuclide of the form `XX##`

        Parameters
        ----------
        nuclide : str
            Nuclide, such as `Xe135`

        Returns
        -------
        element : str
            Element, such as `Xe`
        """
        element = ''.join([i for i in nuclide if i.isalpha()])
        return element
