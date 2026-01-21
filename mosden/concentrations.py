import numpy as np
from mosden.utils.csv_handler import CSVHandler
from uncertainties import ufloat
from mosden.base import BaseClass
from time import time
import os
from jinja2 import Environment, PackageLoader
import subprocess
import sys
import openmc.deplete
import json

class Concentrations(BaseClass):
    def __init__(self, input_path: str) -> None:
        """
        This class generates concentrations from data.

        Parameters
        ----------
        input_path : str
            Path to the input file
        """
        super().__init__(input_path)

        try:
            self.f_in: float = self.t_in / (self.t_in + self.t_ex)
            self.f_ex: float = self.t_ex / (self.t_in + self.t_ex)
        except ZeroDivisionError:
            self.logger.error('No in-core or ex-core time')
            self.f_in = 1.0
            self.f_ex = 1.0
        self.fission_term = 1.0 * self.f_in

        self.repr_scale = 0.0
        if 'incore' in self.reprocess_locations:
            self.repr_scale += self.f_in
        if 'excore' in self.reprocess_locations:
            self.repr_scale += self.f_ex
        self.repr_scale /= self.base_repr_scale

        if not self.flux_scaling:
            self.fission_term = 1.0
            self.f_in = 1.0
            self.f_ex = 1.0
        
        if self.chem_scaling:
            self.repr_scale = 1.0
        
        if self.repr_scale <= 0.0:
            self.logger.info(f'{self.repr_scale = }')
            self.logger.error('No valid chemical removal region provided')
            self.logger.warning('Setting reprocessing scale to 1.0')
            self.repr_scale = 1.0

        return None

    def generate_concentrations(self) -> None:
        """
        Generate the concentrations of each nuclide based on
        irradiation of the sample for the irradiation times.

        The 0D scaled model used the cumulative fission yield, which allows for
        calculation of the equilibrium concentration by dividing it by the
        decay constant.

        The independent fission yield exists as a way to collect pulse
        irradiation concentrations, but this is not an accurate method as it
        does not track decay chains. This exists primarily for testing purposes.
        """
        start = time()
        data: dict[str: dict[str: float]] = dict()
        if self.conc_method == 'CFY':
            if self.irrad_type != 'saturation':
                self.logger.error(
                    'CFY is intended for a saturation irradiation')
            data = self.CFY_concentrations()
        elif self.conc_method == 'IFY':
            if self.t_ex > 0.0:
                raise NotImplementedError(
                    'Excore residence not available for IFY')
            if self.reprocess:
                raise NotImplementedError('Reprocessing not available for IFY')
            if self.irrad_type != 'pulse':
                self.logger.error(
                    'IFY method does not use cumulative fission yields')
            self.logger.error(
                'IFY method has not been verified. Use with caution')
            data = self.IFY_concentrations()
        elif self.conc_method == 'OMC':
            data = self.OMC_concentrations()
        else:
            raise NotImplementedError(
                f"Concentration handling method '{
                    self.conc_method}' is not implemented")

        CSVHandler(self.concentration_path, self.conc_overwrite).write_csv_with_time(data)
        self.save_postproc()
        self.time_track(start, 'Concentrations')
        return

    def CFY_concentrations(self) -> list[dict]:
        """
        Generate the concentrations of each nuclide using the CFY method.

        Returns
        -------
        data : list[dict]
            Concentration and uncertainty for each nuclide post-irradiation
        """
        concentrations: dict[str: dict[str: ufloat]] = dict()
        all_nucs: set[str] = set()
        CFY_data = self._read_processed_data('fission_yield')
        half_life_data = self._read_processed_data('half_life')
        for nuclide in CFY_data.keys():
            concs = ufloat(
                CFY_data[nuclide]['CFY'],
                CFY_data[nuclide]['sigma CFY'])
            try:
                hl = ufloat(
                    half_life_data[nuclide]['half_life'],
                    half_life_data[nuclide]['sigma half_life'])
            except KeyError:
                continue
            repr_term = 0.0
            if self.reprocess:
                nuc_element = self._get_element_from_nuclide(nuclide)
                repr_term = self.reprocessing.get(nuc_element, 0.0)
            lam = np.log(2) / hl
            reprocessing = repr_term * self.repr_scale
            loss_term = lam + reprocessing
            concentrations[nuclide] = self.f_in * concs / loss_term
            all_nucs.add(nuclide)

        data = list()
        for t in range(1):
            for nuc in all_nucs:
                data.append({
                    'Time': t,
                    'Nuclide': nuc,
                    'Concentration': concentrations[nuc].n,
                    'sigma Concentration': concentrations[nuc].s
                })
        return data

    def OMC_concentrations(self) -> list[dict]:
        """
        Generate the concentrations of each nuclide using OpenMC.
        """
        env = Environment(loader=PackageLoader('mosden'))
        file = self.openmc_settings['omc_file']
        template = env.get_template(file)
        chain_file = self.unprocessed_data_dir + self.openmc_settings['chain']
        cross_sections = self.unprocessed_data_dir + self.openmc_settings['x_sections']
        omc_dir = self.openmc_settings['omc_dir']
        render_data = {
            'nps': self.openmc_settings['nps'],
            'mode': self.openmc_settings['mode'],
            'batches': self.openmc_settings['batches'],
            'source': self.openmc_settings['source'],
            'seed': self.seed,
            'energy': self.energy_MeV,
            'density': self.density_g_cc,
            'temperature': self.temperature_K,
            'fissiles': self.fissiles,
            't_in': self.t_in,
            't_ex': self.t_ex,
            'total_irrad_s': self.t_net,
            'decay_times': self.decay_times,
            'repr_locations': self.reprocess_locations,
            'reprocessing': self.reprocessing,
            'repr_scale': self.repr_scale,
            'chain_file': chain_file,
            'cross_sections': cross_sections,
            'omc_dir': omc_dir,
            'flux_scaling': self.flux_scaling,
            'chem_scaling': self.chem_scaling,
            'f_in': self.f_in
        }
        rendered_template = template.render(render_data)
        fname = 'omc.py'
        full_name = f'{omc_dir}/{fname}'

        if self.openmc_settings['run_omc']:
            if not os.path.exists(omc_dir):
                os.makedirs(omc_dir)
            with open(full_name, mode='w') as output:
                output.write(rendered_template)

            try:
                completed_process = subprocess.run([sys.executable, full_name],
                                                capture_output=True, text=True,
                                                check=True)
                with open(f'{omc_dir}/omc_output.txt', 'w') as f:
                    f.write(completed_process.stdout)
            except subprocess.CalledProcessError as e:
                print(f'OpenMC failed with return code {e.returncode}')
                print(f'Error output: {e.stderr}')
            except FileNotFoundError:
                print(f'{full_name} not found')
        
        data = list()
        results = openmc.deplete.Results(f'{omc_dir}/depletion_results.h5')
        times = results.get_times(time_units='s')
        nucs = list(results[0].index_nuc.keys())
        for nuc in nucs:
            _, concs = results.get_atoms('1', nuc)
            for ti, t in enumerate(times):
                data.append(
                    {
                        'Time': t,
                        'Nuclide': nuc,
                        'Concentration': concs[ti],
                        'sigma Concentration': 1e-12
                    }
                )
        if self.openmc_settings['write_fission_json']:
            self._collect_omc_fissions()
        return data
    
    def read_omc_fission_json(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """
        Collect the fission rate history from a json file

        Returns
        -------
        fissions : dict[str, np.ndarray]
            Fission rate history as a function of time for each fissile nuclide
            and the 'net'
        times : np.ndarray
            The times the fission rate history is recorded
        """
        with open(f'{self.output_dir}/omc_fissions.json', 'r') as f:
            full_data = json.load(f)
        fissions = full_data['fissions']
        times = np.array(full_data['times'])
        for nuc in fissions.keys():
            fissions[nuc] = np.array(fissions[nuc])
        return fissions, times

    
    def _collect_omc_fissions(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """
        Collect the fission rate history from OpenMC output and writes to a
        json file

        Returns
        -------
        fissions : dict[str, np.ndarray]
            Fission rate history as a function of time for each fissile nuclide
            and the 'net'
        times : np.ndarray
            The times the fission rate history is recorded
        """
        if self.openmc_settings['mode'] != 'fixed source':
            raise NotImplementedError('Only fixed source fission tracking enabled')
        fissions = dict()
        num_files = self.get_irrad_index(False) - 1
        for i in range(num_files):
            sp = openmc.StatePoint(f'{self.openmc_settings["omc_dir"]}/openmc_simulation_n{i}.h5')
            for tally in sp.tallies.keys():
                tally_data = sp.get_tally(id=tally)
                if 'fissionrate' in tally_data.name:
                    df = tally_data.get_pandas_dataframe(filters=False, scores=False, derivative=False, paths=False)
                    try:
                        df['mean'] = df['mean'] * self.openmc_settings['source']
                    except KeyError:
                        continue
                    df_sorted = df.sort_values(by='mean', ascending=False)
                    df_sorted = df_sorted.reset_index(drop=True)
                    if i == 0:
                        for nuc in df_sorted['nuclide']:
                            fissions[nuc] = np.zeros(num_files)
                        fissions['net'] = np.zeros(num_files)
                    for nuc_i, nuc in enumerate(df_sorted['nuclide']):
                        fissions[nuc][i] = df_sorted['mean'][nuc_i]
                        fissions['net'][i] += fissions[nuc][i]

        fiss_keys = list(fissions.keys())
        for nuc in fiss_keys:
            if np.all(fissions[nuc] <= 1e-12 * np.ones(num_files)):
                del fissions[nuc]
        results = openmc.deplete.Results(f'{self.openmc_settings["omc_dir"]}/depletion_results.h5')
        times = results.get_times(time_units='s')[:num_files+1]
        full_data = {}
        full_data['fissions'] = fissions
        full_data['times'] = times

        def json_default(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()

        with open(f'{self.output_dir}/omc_fissions.json', 'w') as f:
            json.dump(full_data, f, indent=4, default=json_default)
        return fissions, times


    def IFY_concentrations(self) -> list[dict]:
        """
        Generate the concentrations of each nuclide using the IFY method.
        
        Returns
        -------
        data : list[dict]
            Concentration and uncertainty for each nuclide post-irradiation
        """
        concentrations: dict[str: dict[str: ufloat]] = dict()
        all_nucs: set[str] = set()
        IFY_data = self._read_processed_data('fission_yield')
        for nuclide in IFY_data.keys():
            concs = IFY_data[nuclide]['IFY']
            concentrations[nuclide] = concs
            all_nucs.add(nuclide)

        data = list()
        for t in range(1):
            for nuc in all_nucs:
                data.append({
                    'Time': t,
                    'Nuclide': nuc,
                    'Concentration': concentrations[nuc],
                    'sigma Concentration': 1e-12
                })
        return data


if __name__ == "__main__":
    input_path = "../examples/keepin_1957/input.json"
    concentrations = Concentrations(input_path)
    concentrations.generate_concentrations()
