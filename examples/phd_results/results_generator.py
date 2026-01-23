import os
from pathlib import Path
import json
import itertools
import shutil
from copy import deepcopy
import subprocess
from mosden.utils.chemical_schemes import Reprocessing

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
    'incore_s': [100, 50],
    'excore_s': [0, 50],
    'multi_id': [name]
}
analysis_list.append(residence_time_analysis)


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
    new_data['file_options']['processed_data_dir'] = str(file_dir)
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