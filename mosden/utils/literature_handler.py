from mosden.base import BaseClass
from uncertainties import ufloat
import numpy as np
import pandas as pd
import os

class Literature(BaseClass):
    def __init__(self, input_path) -> None:
        """
        This class holds data from the literature for passing into MoSDeN post processing.

        Parameters
        ----------
        input_path : str
            Path to the input file containing fissile data.
        """
        super().__init__(input_path)
        self.available_names: list[str] = ['keepin', 'charlton', 'endfb6', 'mills', 'saleh', 'synetos', 'tuttle', 'waldo', 'brady', 'Modified 0D Scaled']
        return None
    
    def get_group_data(self, names:list[str]=None) -> dict[dict[str: list[float]]]:
        """
        Get countrate data from various sources and compile into a dictionary

        Parameters
        ----------
        names : list[str]
            List of names for which to retrieve data. Default is ['keepin'].
        """
        data_holder = dict()
        self.logger.warning('Current literature energy binning could be improved')
        if self.energy_MeV < 1e-3:
            energy = 'thermal'
        else:
            energy = 'fast'

        if not names:
            names = self.available_names
        for name in names:
            data_holder[name] = dict()
            for fiss, frac in self.fissiles.items():
                data = self._get_group_data_helper(fiss, frac, name, energy)
                if data is None:
                    del data_holder[name]
                    continue
                data_holder[name][fiss] = data
        data_holder = self._merge_fiss(data_holder)
        return data_holder
    
    def _merge_fiss(self, data: dict[str: dict[str: list[float]]]) -> dict[dict[str: list[float]]]:
        """
        Merge same name fissile data into a single dictionary.

        Parameters
        ----------
        data : dict[str: dict[str: list[float]]]
            Dictionary containing group data for different names, fissile nuclides, and energies.
        """
        merged_data = dict()
        for name, name_data in data.items():
            merged_data[name] = dict()
            yields = dict()
            halflives = dict()
            for fiss, params in name_data.items():
                for i in range(len(params['yield'])):
                    yields[i] = yields.get(i, 0.0) + ufloat(params['yield'][i], params['sigma yield'][i])
                    halflives[i] = halflives.get(i, 0.0) + ufloat(params['half_life'][i], params['sigma half_life'][i])
            merged_data[name]['yield'] = [yields[i].n for i in yields]
            merged_data[name]['sigma yield'] = [yields[i].s for i in yields]
            merged_data[name]['half_life'] = [halflives[i].n for i in halflives]
            merged_data[name]['sigma half_life'] = [halflives[i].s for i in halflives]

        return merged_data


    def _get_group_data_helper(self, fiss: str, frac: float, name: str, energy: str) -> dict[str: list[float]]:
        """
        Helper function to retrieve group data for a specific fissile nuclide and energy.
        Parameters
        ----------
        fiss : str
            Fissile nuclide for which to retrieve data (e.g., 'U235').
        frac : float
            Fraction of the fissions from the given fissile nuclide.
        name : str
            Name of the literature source (e.g., 'keepin').
        energy : str
            Energy level ('thermal' or 'fast').

        Returns
        -------
        group_params : dict[str: list[float]]
            Group data for the specified fissile nuclide and energy.
        
        Raises
        ------
        FileNotFoundError
            If the fissile nuclide or energy level is not found in the specified literature source
        """
        data_path: str = os.path.join(self.lit_data_dir, fiss, energy, 'six_group.csv')
        try:
            data_file: pd.DataFrame = pd.read_csv(data_path)
        except FileNotFoundError:
            self.logger.warning(f"Data for {fiss} in {name} at {energy} energy not found")
            return None
        cur_data: pd.DataFrame = data_file.loc[data_file['name'] == name]
        yields = cur_data.loc[cur_data['category'] == 'yield']
        half_lives = cur_data.loc[cur_data['category'] == 'half_life']

        group_params = {
            'yield': [a*frac for a in yields['value']],
            'sigma yield': [a*frac for a in yields['uncertainty']],
            'half_life': [hl*frac for hl in half_lives['value']],
            'sigma half_life': [hl*frac for hl in half_lives['uncertainty']]
        }
        return group_params
            
if __name__ == "__main__":
    input_path = "../../examples/huynh_2014/input.json"
    lit = Literature(input_path)
    data = lit.get_group_data(lit.available_names)
    target_key = 'half_life'
    target_name = None
    for name, val in data.items():
        if name != target_name and target_name is not None:
            continue
        print(name)
        for key, item_val in val.items():
            if key != target_key and target_key is not None:
                continue
            print(key)
            yield_val = sum([ufloat(val['yield'][i], val['sigma yield'][i])*100 for i in range(len(item_val))])
            halflife_val = 1/yield_val * sum([ufloat(val['yield'][i], val['sigma yield'][i])*100*ufloat(val['half_life'][i], val['sigma half_life'][i]) for i in range(len(item_val))])
            if target_key == 'yield':
                print(f'{yield_val = }')
            elif target_key == 'half_life':
                print(f'{halflife_val = }')
            for item in item_val:
                print(f'{round(item, 5):.5f} & ', end='')
            print()
        print('\n')
