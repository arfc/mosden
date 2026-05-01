import os
from pathlib import Path
import json
import itertools
import shutil
from copy import deepcopy
import subprocess
from mosden.utils.chemical_schemes import Reprocessing
import numpy as np

base_input_file = './input.json'
analysis_list = list()

name = 'tintex'
residence_time_analysis = {
    'meta': {
        'name': name,
        'run_full': True,
        'run_post': True,
        'overwrite': True,
    },
    'net_irrad_s': [100],
    'incore_s': [0.5, 1, 1.5, 2, 2.5, 3, 3.5],
    'excore_s': [0, 0.5, 1, 1.5, 2, 2.5, 3],
    'multi_id': [name]
}
analysis_list.append(residence_time_analysis)

name = 'decay_time_nodes'
decay_times_analysis = {
    'meta': {
        'name': name,
        'run_full': True,
        'run_post': True,
        'overwrite': True
    },
    'num_decay_times': [50, 100, 150, 200, 250, 400, 800],
    'multi_id': [name]
}
analysis_list.append(decay_times_analysis)

name = 'omc_timestep'
omc_timestep_analysis = {
    'meta': {
        'name': name,
        'run_full': True,
        'run_post': True,
        'overwrite': True
    },
    'max_timestep': [0.1, 0.05, 0.01, 0.005, 0.001],
    'multi_id': [name]
}
analysis_list.append(omc_timestep_analysis)

name = 'total_decay_time'
total_decay_analysis = {
    'meta': {
        'name': name,
        'run_full': True,
        'run_post': True,
        'overwrite': True
    },
    'decay_time': [150, 300, 600, 1200, 2400, 4800],
    'multi_id': [name]
}
analysis_list.append(total_decay_analysis)

name = 'flux_scaling'
flux_analysis = {
    'meta': {
        'name': name,
        'run_full': True,
        'run_post': True,
        'overwrite': True
    },
    'incore_s': [9],
    'excore_s': [16],
    'net_irrad_s': [100],
    'flux': [True, False],
    'multi_id': [name]
}
analysis_list.append(flux_analysis)

name = 'chem_scaling'
chem_analysis = {
    'meta': {
        'name': name,
        'run_full': True,
        'run_post': True,
        'overwrite': True
    },
    'incore_s':[9],
    'excore_s': [16],
    'net_irrad_s': [100],
    'reprocessing_scheme': [Reprocessing(base_input_file).removal_scheme()],
    'reprocessing': [True, False],
    'multi_id': [name]
}
analysis_list.append(chem_analysis)

name = 'vf_scaling'
vf_analysis = {
    'meta': {
        'name': name,
        'run_full': True,
        'run_post': True,
        'overwrite': True
    },
    'incore_s': [9],
    'excore_s': [16],
    'net_irrad_s': [100],
    'reprocessing_scheme': [Reprocessing(base_input_file).removal_scheme(vf_scaling=0.1),
                            Reprocessing(base_input_file).removal_scheme(vf_scaling=1.0),
                            Reprocessing(base_input_file).removal_scheme(vf_scaling=10.)],
    'multi_id': [name]
}
analysis_list.append(vf_analysis)


name = 'spectra_compare'
spectra_analysis = {
    'meta': {
        'name': name,
        'run_full': False,
        'run_post': False,
        'overwrite': True
    },
    'incore_s': [9],
    'excore_s': [16],
    'net_irrad_s': [100],
    'num_groups': [8],
    'reprocessing_scheme': [Reprocessing(base_input_file).removal_scheme(rate_scaling=0.0),
                            Reprocessing(base_input_file).removal_scheme()],
    'energy_groups_MeV': [[0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.7,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1,1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09,1.1,1.11,1.12,1.13,1.14,1.15,1.16,1.17,1.18,1.19,1.2,1.21,1.22,1.23,1.24,1.25,1.26,1.27,1.28,1.29,1.3,1.31,1.32,1.33,1.34,1.35,1.36,1.37,1.38,1.39,1.4,1.41,1.42,1.43,1.44,1.45,1.46,1.47,1.48,1.49,1.5,1.51,1.52,1.53,1.54,1.55,1.56,1.57,1.58,1.59,1.6]],
    'multi_id': [name]
}
analysis_list.append(spectra_analysis)

name = 'group_compare'
group_analysis = {
    'meta': {
        'name': name,
        'run_full': False,
        'run_post': True,
        'overwrite': True
    },
    'incore_s': [9],
    'excore_s': [16],
    'net_irrad_s': [100],
    'num_groups': [8],
    'reprocessing_scheme': [Reprocessing(base_input_file).removal_scheme()],
    'energy_groups_MeV': [[0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.7,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1,1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09,1.1,1.11,1.12,1.13,1.14,1.15,1.16,1.17,1.18,1.19,1.2,1.21,1.22,1.23,1.24,1.25,1.26,1.27,1.28,1.29,1.3,1.31,1.32,1.33,1.34,1.35,1.36,1.37,1.38,1.39,1.4,1.41,1.42,1.43,1.44,1.45,1.46,1.47,1.48,1.49,1.5,1.51,1.52,1.53,1.54,1.55,1.56,1.57,1.58,1.59,1.6]],
    'num_groups': [6, 8, 10, 12],
    'multi_id': [name]
}
analysis_list.append(group_analysis)


def replace_value(input_data: dict, key: str, new_val: str|float|int) -> bool:
    """
    Recursively search through dict d to find key and replace its value.
    Returns True if replacement was made, False otherwise.

    Parameters
    ----------
    input_data : dict
        The dictionary of input data
    key : str
        The key to search for
    new_val : str|float|int
        The new value to assign to the key

    """
    if key in input_data.keys():
        input_data[key] = new_val
        return True

    for val in input_data.values():
        if isinstance(val, dict):
            if replace_value(val, key, new_val):
                return True
    return False

def create_directory(dir_name: str) -> None:
    """
    Create directory if it does not exist. If it does exist and overwrite is
    True, delete and recreate.

    Parameters
    ----------
    dir_name : str
        The name of the directory to create

    """
    if os.path.isdir(dir_name) and analysis['meta']['overwrite']:
        shutil.rmtree(dir_name)
    os.makedirs(dir_name, exist_ok=analysis['meta']['overwrite'])
    return None

def set_data(new_data: dict, dir_path: str, idx: int, combination: tuple) -> tuple[dict, str]:
    """
    Set the data for the new input file.

    Parameters
    ----------
    new_data : dict
        The new data to write to the input file
    dir_path : str
        The directory path where the input file will be created
    idx : int
        The index of the current input file
    combination : tuple
        The combination of parameters for this input file

    Returns
    -------
    tuple[dict, str]
        The updated data and the path to the input file
    """
    filename = 'input.json'
    file_dir = dir_path / str(idx)
    file_path = file_dir / filename
    new_data['file_options']['processed_data_dir'] = str(file_dir) + '/'
    new_data['file_options']['output_dir'] = str(file_dir) + '/'
    new_data['file_options']['log_file'] = str(file_dir) + '/log.log'
    new_data['modeling_options']['openmc_settings']['omc_dir'] = str(file_dir) + '/omc'
    new_data['name'] = str(combination)
    if analysis['meta']['run_full']:
        create_directory(file_dir)

    with open(file_path, "w") as f:
        json.dump(new_data, f, indent=2)

    return new_data, file_path

def populate_inputs(analysis: dict, dir_path: str) -> list[str]:
    """
    Populate the input files for each case in the analysis dict.

    Parameters
    ----------
    analysis : dict
        The analysis dictionary containing the parameters for each case
    dir_path : str
        The directory path where the input files will be created

    Returns
    -------
    list[str]
        A list of paths to the created input files

    Raises
    ------
    KeyError
        If a key in the analysis dict is not found in the input file
    """
    paths = []
    dir_path = Path(dir_path)

    with open("input.json", "r") as f:
        base_data = json.load(f)

    component_keys = [k for k in analysis.keys() if k != "meta"]
    value_combinations = itertools.product(*(analysis[k] for k in component_keys))
    for idx, combination in enumerate(value_combinations, start=1):
    
        new_data = deepcopy(base_data)

        for key, val in zip(component_keys, combination):
            replaced = replace_value(new_data, key, val)
            if not replaced:
                raise KeyError(f'{key} not found in input file')
        
        new_data, file_path = set_data(new_data, dir_path, idx, combination)

        paths.append(str(file_path))
    return paths

def run_mosden(analysis: dict, input_paths: list[str]) -> None:
    """
    Run mosden for the given analysis and input paths.

    Parameters
    ----------
    analysis : dict
        The analysis dictionary containing the parameters for the run
    input_paths : list[str]
        The list of input file paths to process

    """
    if analysis['meta']['run_full']:
        command = ['mosden', '-a'] + input_paths
    elif analysis['meta']['run_post']:
        command = ['mosden', '-post'] + input_paths
    else:
        return None
    subprocess.run(command)
    return None

if __name__ == '__main__':
    for analysis in analysis_list:
        dir_name = f'{os.getcwd()}/{analysis["meta"]["name"]}'
        if analysis['meta']['run_full'] or analysis['meta']['run_post']:
            input_paths = populate_inputs(analysis, dir_name)
            run_mosden(analysis, input_paths)
    pass
