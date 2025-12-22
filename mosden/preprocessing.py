from mosden.utils.csv_handler import CSVHandler
from mosden.base import BaseClass
import os
import numpy as np
import re
from uncertainties import ufloat
from time import time


class Preprocess(BaseClass):
    def __init__(self, input_path: str) -> None:
        """
        This class preprocesses files into a common format.

        Parameters
        ----------
        input_path : str
            Path to the input file.
        """
        super().__init__(input_path)

        data_keys: list[str] = [
            'half_life',
            'cross_section',
            'emission_probability',
            'fission_yield'
        ]
        self.data_to_proc: dict[str: str] = {
            key: self.input_data['data_options'][key] for key in data_keys
        }

        return None

    def run(self) -> None:
        """
        Run the preprocessing.
        """
        start = time()
        datasource_list: list[list[str]] = [
            self.omc_data_words,
            self.endf_data_words,
            self.iaea_data_words,
            self.jeff_data_words
        ]
        func_list: list = [
            self.openmc_preprocess,
            self.endf_preprocess,
            self.iaea_preprocess,
            self.jeff_preprocess
        ]

        func_selector: list[zip] = list(zip(datasource_list,
                                            func_list))
        for data_val, path in self.data_to_proc.items():
            for ids, func in func_selector:
                if any(word in path for word in ids):
                    func(data_val, path)
        self.save_postproc()
        self.time_track(start, 'Preprocessing')
        return None

    def openmc_preprocess(self, data_val: str, unprocessed_path: str) -> None:
        """
        Processes OpenMC data

        Parameters
        ----------
        data_val : str
            Type of data to process
        unprocessed_path : str
            Path to the unprocessed data
        """
        if data_val == 'fission_yield' or data_val == 'half_life':
            self._openmc_chain_preprocess(data_val, unprocessed_path) 
        else:
            self.logger.error(f'{data_val} not available in OpenMC chain')
        return None

    def endf_preprocess(self, data_val: str, unprocessed_path: str) -> None:
        """
        Processes ENDF data

        Parameters
        ----------
        data_val : str
            Type of data to process
        unprocessed_path : str
            Path to the unprocessed data
        """
        if data_val == 'fission_yield':
            self._endf_nfy_preprocess(data_val, unprocessed_path)
        elif data_val == 'half_life' or data_val == 'emission_probability':
            self._endf_decay_preprocess(data_val, unprocessed_path)
        else:
            self.logger.error(f'{data_val} not available in ENDF')
        return None
    
    def jeff_preprocess(self, data_val: str, unprocessed_path: str) -> None:
        """
        Processes JEFF data

        Parameters
        ----------
        data_val : str
            Type of data to process
        unprocessed_path : str
            Path to the unprocessed data
        """
        if data_val == 'fission_yield':
            self._jeff_nfy_preprocess(data_val, unprocessed_path)
        else:
            self.logger.error(f'{data_val} not available in JEFF')
        return None

    def iaea_preprocess(self, data_val: str, unprocessed_path: str) -> None:
        """
        Processes IAEA data

        Parameters
        ----------
        data_val : str
            Type of data to process
        unprocessed_path : str
            Path to the unprocessed data

        """
        if data_val == 'half_life' or data_val == 'emission_probability':
            self._iaea_dn_preprocess(data_val, unprocessed_path)
        else:
            self.logger.error(f'{data_val} not available in IAEA')
        return None

    def _iaea_dn_preprocess(self, data_val: str, path: str) -> None:
        """
        Processes IAEA data for emission probabilities and half-lives.

        Parameters
        ----------
        data_val : str
            Type of data to process
        path : str
            Path to the unprocessed data
        """
        data_file: str = os.path.join(self.data_dir, path)
        out_file: str = os.path.join(self.processed_data_dir, f'{data_val}.csv')

        data = CSVHandler(data_file, create=False).read_csv(raw_iaea=True)
        csv_path: str = os.path.join(out_file)
        CSVHandler(csv_path, self.preprocess_overwrite).write_csv(data)
        return None

    def _openmc_chain_preprocess(self, data_val: str, path: str) -> None:
        """
        Processes OpenMC all chain_* files

        Parameters
        ----------
        data_val : str
            Type of data to process
        path : str
            Path to the unprocessed data
        """
        data_path: str = os.path.join(self.data_dir, path)
        out_path: str = os.path.join(self.processed_data_dir, f'{data_val}.csv')
        file_data: dict[str: dict[str: float]
                        ] = self._process_chain_file(data_path)
        CSVHandler(out_path, self.preprocess_overwrite).write_csv(file_data)
        return None

    def _jeff_nfy_preprocess(self, data_val: str, path: str) -> None:
        """
        Processes JEFF fission yield data for the specified fissile target.

        Parameters
        ----------
        data_val : str
            Type of data to process
        path : str
            Path to the unprocessed data
        """
        data_dir: str = os.path.join(self.data_dir, path)
        out_path: str = os.path.join(self.processed_data_dir, f'{data_val}.csv')
        pre_treated_data: dict[str: dict[str: dict[str: float]]] = dict()
        for fissile in self.fissile_targets:
            for file in os.listdir(data_dir):
                fissile_endf: str = self._endf_fissile_name(fissile)
                fissile_jeff = fissile_endf[1:].replace('_', '-')
                if not fissile_jeff in file:
                    continue
                full_path: str = os.path.join(data_dir, file)
                file_data: dict[str: dict[str: float]
                                ] = self._process_jeff_nfy_file(full_path)
            pre_treated_data[fissile] = file_data
        treated_data: dict[str: dict[str: float]
                           ] = self._treat_endf_data(pre_treated_data)
        csv_path: str = os.path.join(out_path)
        CSVHandler(csv_path, self.preprocess_overwrite).write_csv(treated_data)
        return None

    def _endf_nfy_preprocess(self, data_val: str, path: str) -> None:
        """
        Processes ENDF fission yield data for the specified fissile target.

        Parameters
        ----------
        data_val : str
            Type of data to process
        path : str
            Path to the unprocessed data
        """
        data_dir: str = os.path.join(self.data_dir, path)
        out_path: str = os.path.join(self.processed_data_dir, f'{data_val}.csv')
        pre_treated_data: dict[str: dict[str: dict[str: float]]] = dict()
        for fissile in self.fissile_targets:
            for file in os.listdir(data_dir):
                adjusted_fissile: str = self._endf_fissile_name(fissile)
                if not file.startswith(f'nfy-{adjusted_fissile}'):
                    continue
                full_path: str = os.path.join(data_dir, file)
                file_data: dict[str: dict[str: float]
                                ] = self._process_endf_nfy_file(full_path)
            pre_treated_data[fissile] = file_data
        treated_data: dict[str: dict[str: float]
                           ] = self._treat_endf_data(pre_treated_data)
        csv_path: str = os.path.join(out_path)
        CSVHandler(csv_path, self.preprocess_overwrite).write_csv(treated_data)
        return None
    
    def _endf_decay_preprocess(self, data_val: str, path: str) -> None:
        """
        Processes ENDF decay data for the specified fissile target.

        Parameters
        ----------
        data_val : str
            Type of data to process
        path : str
            Path to the unprocessed data
        """
        data_dir: str = os.path.join(self.data_dir, path)
        out_path: str = os.path.join(self.processed_data_dir, f'{data_val}.csv')
        file_data: dict[str: dict[str: float]
                        ] = self._process_endf_decay_file(data_dir)
        csv_path: str = os.path.join(out_path)
        CSVHandler(csv_path, self.preprocess_overwrite).write_csv(file_data)
        return None


    def _treat_endf_data(
            self,
            pre_treated_data: dict[str: dict[str: dict[str: float]]]
            ) -> dict[str: dict[str: float]]:
        """
        Take endf data for each nuclide and scale it by fissile fraction
          for each nuclide

        Parameters
        ----------
        pre_treated_data : dict[str: dict[str: dict[str: float]]]
            Dictionary containing the pre-treated data for each fissile target

        Returns
        -------
        treated_data : dict[str: dict[str: float]]
            Dictionary containing the data with fissile fractions applied.
        """
        treated_data: dict[str: dict[str: float]] = dict()
        for fissile in pre_treated_data.keys():
            frac = self.fissiles[fissile]
            for nuc in pre_treated_data[fissile].keys():
                try:
                    cur_dict = treated_data[nuc]
                except KeyError:
                    treated_data[nuc] = dict()
                cur_dict = treated_data[nuc]
                if pre_treated_data[fissile][nuc]['sigma CFY'] == 0.0:
                    pre_treated_data[fissile][nuc]['sigma CFY'] = 1e-12
                CFY_vals = ufloat(pre_treated_data[fissile][nuc]['CFY'],
                                  pre_treated_data[fissile][nuc]['sigma CFY'])
                treated_val = frac * CFY_vals
                cur_dict['CFY'] = cur_dict.get('CFY', 0) + treated_val.n
                cur_dict['sigma CFY'] = cur_dict.get(
                    'sigma CFY', 0) + treated_val.s
        return treated_data

    def _endf_fissile_name(self, fissile: str) -> str:
        """
        Adjusts the fissile target name for ENDF processing.

        Parameters
        ----------
        fissile : str
            Name of the fissile target to adjust.

        Returns
        -------
        adjusted_fissile : str
            Adjusted fissile target name.
        """
        match = re.match(r'([A-Za-z]+)(\d+)', fissile)
        if not match:
            raise ValueError(f"Invalid nuclide format: {fissile}")

        symbol, A = match.groups()
        A = int(A)

        periodic_table = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
            'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
            'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
            'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
            'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35,
            'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
            'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48,
            'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55,
            'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61,
            'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67,
            'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
            'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
            'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86,
            'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93,
            'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99,
            'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105,
            'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111,
            'Cn': 112, 'Fl': 114, 'Lv': 116
        }

        symbol = symbol.capitalize() if len(
            symbol) == 1 else symbol[0].upper() + symbol[1:].lower()

        Z = periodic_table.get(symbol)
        if Z is None:
            raise ValueError(f"Unknown element symbol: {symbol}")

        return f"{Z:03d}_{symbol}_{A}"

    def _process_endf_nfy_file(self, file: str) -> dict[str, dict[str: float]]:
        """
        Processes a single ENDF NFY file and returns the data as a dictionary.

        Parameters
        ----------
        file : str
            Name of the NFY file to process.

        Returns
        -------
        data : dict[str, dict[str: float]]
            Dictionary containing the processed data.
        """
        import openmc.data
        evaluation = openmc.data.endf.Evaluation(file)
        fpys = openmc.data.FissionProductYields(evaluation)
        energies = fpys.energies
        fys = fpys.cumulative
        endf_nucs: list = list(fys[0].keys())
        fit_FY_nfy = self._fit_fy_endf(energies, fys)

        data: dict[str: dict[str: float]] = dict()
        for nuc in endf_nucs:
            data[nuc] = {}
            data[nuc]['CFY'] = fit_FY_nfy[nuc].n
            data[nuc]['sigma CFY'] = fit_FY_nfy[nuc].s
        return data
    

    def _process_jeff_nfy_file(self, file: str) -> dict[str, dict[str: float]]:
        """
        Processes a single JEFF NFY file and returns the data as a dictionary.

        Parameters
        ----------
        file : str
            Name of the NFY file to process.

        Returns
        -------
        data : dict[str, dict[str: float]]
            Dictionary containing the processed data.
        """
        import openmc.data
        evaluation = openmc.data.endf.Evaluation(file)
        fpys = openmc.data.FissionProductYields(evaluation)
        energies = fpys.energies
        fys = fpys.cumulative
        endf_nucs: list = list(fys[0].keys())
        fit_FY_nfy = self._fit_fy_endf(energies, fys)

        data: dict[str: dict[str: float]] = dict()
        for nuc in endf_nucs:
            data[nuc] = {}
            data[nuc]['CFY'] = fit_FY_nfy[nuc].n
            data[nuc]['sigma CFY'] = fit_FY_nfy[nuc].s
        return data
    
    
    def _process_endf_decay_file(self, dir: str) -> dict[str, dict[str: float]]:
        """
        Processes all ENDF decay files and returns the data as a dictionary.

        Parameters
        ----------
        file : str
            Name of the ENDF decay directory to process.

        Returns
        -------
        data : dict[str, dict[str: float]]
            Dictionary containing the processed data.
        """
        import openmc.data
        data = dict()
        for file in os.listdir(dir):
            Pn = ufloat(0, 1e-12)
            half_life = ufloat(0, 1e-12)
            if not file.startswith(f'dec-'):
                continue
            decay = openmc.data.Decay.from_endf(dir+file)
            half_life = decay.half_life
            nuc_name = decay.nuclide['name']
            data[nuc_name] = dict()
            modes = decay.modes
            for mode in modes:
                products = mode.modes
                if 'n' in products:
                    multiplier = products.count('n')
                    Pn += mode.branching_ratio * multiplier
            if Pn.n > 0 and half_life.n > 0 and half_life.n != np.inf:
                data[nuc_name]['emission probability'] = Pn.n
                data[nuc_name]['sigma emission probability'] = Pn.s
                data[nuc_name]['half_life'] = half_life.n
                data[nuc_name]['sigma half_life'] = half_life.s
        return data

    def _process_chain_file(self, file: str) -> dict[str, dict[str: float]]:
        """
        Processes an OpenMC chain file and returns the data as a dictionary.

        Parameters
        ----------
        file : str
            Name of the chain file to process.

        Returns
        -------
        data : dict[str, dict[str, float]]
            Dictionary containing the processed data.
        """
        import openmc.deplete
        chain: openmc.deplete.chain = openmc.deplete.Chain.from_xml(file)
        nuclides: list[openmc.deplete.Nuclide] = chain.nuclides
        nuc_dict: dict[str, int] = chain.nuclide_dict
        FY_chain: dict[str: dict[float: float]] = {}

        for fissile, frac in self.fissiles.items():
            target_index: int = nuc_dict[fissile]
            target_nuc: openmc.deplete.Nuclide = nuclides[target_index]
            FY_data: openmc.deplete.FissionYieldDistribution = target_nuc.yield_data
            energies = FY_data.energies
            products = FY_data.products
            for product in products:
                try:
                    FY_chain[product]
                except KeyError:
                    FY_chain[product] = {}
                for energy in energies:
                    cur_dict = FY_chain[product]
                    cur_dict[energy] = cur_dict.get(
                        energy, 0) + FY_data[energy][product] * frac

        fit_FY_chain: dict[str: float] = self._fit_fy_chain(FY_chain, order=1)
        chain_nucs: list[str] = list(fit_FY_chain.keys())
        nuclide_halflives: dict[str: float] = {
            nuc.name: nuc.half_life for nuc in nuclides}

        data: dict[str: dict[str: float]] = dict()
        for nuc in chain_nucs:
            data[nuc] = {}
            if nuclide_halflives[nuc] is not None:
                data[nuc]['half_life'] = nuclide_halflives[nuc]
            else:
                data[nuc]['half_life'] = np.inf

            data[nuc]['IFY'] = fit_FY_chain[nuc]

        return data

    def _fit_fy_endf(self,
                     energies: list[float],
                     fys: list[dict[str: ufloat]]) -> dict[str: ufloat]:
        """
        Fit the fission yield data from ENDF files.
        Uses the closest fit, not an interpolation.

        Parameters
        ----------
        energies : list[float]
            List of energy values.
        ifys : list[dict[str: ufloat]]
            Dictionary of fission yields or uncertainties at energy indices.

        Returns
        -------
        dict[str: ufloat]
            Dictionary containing the fitted fission yield data.
        """
        fit_FY_endf: dict[str: float] = {}
        endf_nucs: list[str] = list(fys[0].keys())
        energy_index: int = np.argmin(
            np.abs(
                np.array(energies) -
                self.energy_MeV *
                1e6))
        for i, nuc in enumerate(endf_nucs):
            fission_yield = fys[energy_index][nuc]
            uncert = fission_yield.s
            if fission_yield.s == 0.0:
                uncert = 1e-12
            fit_FY_endf[nuc] = ufloat(fission_yield.n, uncert)
        return fit_FY_endf

    def _fit_fy_chain(self,
                      FY_chain: dict[str: dict[float: float]],
                      order: int = 1) -> dict[str: float]:
        """
        Fit the fission yield chain data.

        Parameters
        ----------
        FY_chain : dict[str: dict[float, float]]
            Dictionary containing the fission yield chain data.
        order : int
            Order of the polynomial fit.

        Returns
        -------
        fit_FY_chain : dict[str: float]
            Dictionary containing the fit fission yield chain data.
        """
        fit_FY_chain: dict[str: float] = {}
        for product in FY_chain:
            fit_FY_chain[product] = self._energy_fit(
                energies=list(
                    FY_chain[product].keys()), values=list(
                    FY_chain[product].values()), order=order)
        return fit_FY_chain

    def _energy_fit(
            self,
            energies: list[float],
            values: list[float],
            order: int = 1) -> float:
        """
        Fit the energy values to a polynomial of the specified order.
        Evaluates at `self.energy_MeV`.

        Parameters
        ----------
        energies : list[float]
            List of energy values.
        values : list[float]
            List of values corresponding to the energies.
        order : int
            Order of the polynomial fit.

        Returns
        -------
        fit_value : float
            The fitted value at `self.energy_MeV`.
        """
        xs = [energy * 1e-6 for energy in energies]
        if order == 1:
            fit_value = np.interp(self.energy_MeV, xs, values)
        else:
            coeffs = np.polyfit(xs, values, order)
            fit_value = np.polyval(coeffs, self.energy_MeV)
        return fit_value


if __name__ == "__main__":
    preproc = Preprocess('../examples/keepin_1957/input.json')
    preproc.run()
