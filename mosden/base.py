from mosden.utils.input_handler import InputHandler
from pathlib import Path
from mosden.utils.csv_handler import CSVHandler
import os
import logging
import json
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
        self.endf_data_words: list[str] = ['nfy', 'decay']
        self.iaea_data_words: list[str] = ['iaea']
        self.jeff_data_words: list[str] = ['jeff']

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
        self.output_dir: str = self.input_data['file_options']['output_dir']
        self.logger.info(f'{self.name = }')

        self.energy_MeV: float = data_options.get('energy_MeV', 0.0)
        self.fissiles: dict[str, float] = data_options.get(
            'fissile_fractions', {})
        self.fissile_targets: list = list(self.fissiles.keys())

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
        self.conc_overwrite: bool = overwrite_options.get('concentrations', False)
        self.reprocessing: dict[str: float] = modeling_options.get(
            'reprocessing', {})
        self.reprocess: bool = (sum(self.reprocessing.values()) > 0)
        self.reprocess_locations: list[str] = modeling_options.get(
            'reprocessing_locations', [])
        self.t_in: float = modeling_options.get('incore_s', 0.0)
        self.t_ex: float = modeling_options.get('excore_s', 0.0)
        self.t_net: float = modeling_options.get('net_irrad_s', 0.0)
        self.irrad_type: str = modeling_options.get('irrad_type', 'saturation')
        self.spatial_scaling: str = modeling_options.get(
            'spatial_scaling', 'unscaled')
        
        self.count_overwrite: bool = overwrite_options.get('count_rate', False)
        self.num_times: int = modeling_options['num_decay_times']
        self.decay_time: float = modeling_options['decay_time']
        self.decay_time_spacing: str = data_options['decay_time_spacing']
        self.count_method: str = self.input_data['modeling_options']['count_rate_handling']
        self.irrad_type: str = self.input_data['modeling_options']['irrad_type']
        self.seed: int = group_options.get('seed', 0)

        self.group_method: str = group_options.get('method', 'nlls')
        self.num_groups: int = group_options.get('num_groups', 6)
        self.group_overwrite: bool = overwrite_options.get('group_fitting', False)
        self.MC_samples: int = group_options.get('samples', 1)
        self.sample_func: str = group_options.get('sample_func', 'normal')

        self.processed_data_dir: str = file_options['processed_data_dir']
        self.concentration_path: str = os.path.join(
            file_options['output_dir'], 'concentrations.csv')
        self.countrate_path: str = os.path.join(
            file_options['output_dir'], 'count_rate.csv')
        self.group_path: str = os.path.join(
            file_options['output_dir'], 'group_parameters.csv')
        self.postproc_path: str = os.path.join(
            file_options['output_dir'], 'postproc.json')

        self.img_dir: str = self.output_dir + 'images/'
        self.post_overwrite: bool = overwrite_options.get('postprocessing', False)
        self.sens_subplot: bool = post_options.get('sensitivity_subplots', True)
        self.lit_data: list[str] = post_options.get('lit_data', ['keepin'])
        self.nuclides: list[str] = post_options.get('nuclides', ['Br87'])
        self.num_top = post_options.get('top_num_nuclides', 3)
        self.num_stack = post_options.get('num_stacked_nuclides', 2)

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
