from mosden.utils.csv_handler import CSVHandler
from mosden.base import BaseClass
import os
import numpy as np
from uncertainties import ufloat, unumpy
from time import time


class CountRate(BaseClass):
    def __init__(self, input_path: str) -> None:
        """
        This class generates the delayed neutron count rate from concentrations
        and nuclear data.

        Parameters
        ----------
        input_path : str
            Path to the input file
        """
        super().__init__(input_path)

        if self.decay_time_spacing == 'linear':
            self.decay_times: np.ndarray = np.linspace(
                0, self.decay_time, self.num_times)
        elif self.decay_time_spacing == 'log':
            self.decay_times: np.ndarray = np.geomspace(
                1e-2, self.decay_time, self.num_times)
        else:
            raise ValueError(
                f"Decay time spacing '{self.decay_time_spacing}' not supported.")

        np.random.seed(self.seed)

        return None

    def calculate_count_rate(self,
                             MC_run: bool = False,
                             sampler_func: str = None,
                             write_data: bool = True) -> dict[str: list[float]]:
        """
        Calculate the delayed neutron count rate from
        concentrations using various methods

        Parameters
        ----------
        MC_run : bool, optional
            Whether to run in Monte Carlo mode, by default False
        sampler_func : str, optional
            The sampling function to use for Monte Carlo, by default None
        write_data : bool, optional
            Whether to write the data to a CSV file, by default True

        Returns
        -------
        data : dict[str: list[float]]
            Dictionary containing the times, count rates, and uncertainties
        """
        start = time()
        data: dict[str: list[float]] = dict()
        if self.count_method == 'data':
            pn_path = os.path.join(
                self.processed_data_dir,
                'emission_probability.csv')
            self.emission_prob_data = CSVHandler(
                pn_path, create=False).read_csv()
            half_life_path = os.path.join(
                self.processed_data_dir, 'half_life.csv')
            self.half_life_data = CSVHandler(
                half_life_path, create=False).read_csv()
            self.concentration_data = CSVHandler(
                self.concentration_path, create=False).read_csv()
            data = self._count_rate_from_data(MC_run, sampler_func)
        elif self.count_method == 'groupfit':
            self.group_params = CSVHandler(
                self.group_path, create=False).read_vector_csv()
            data = self._count_rate_from_groups()
        else:
            raise NotImplementedError(f'{self.count_method} not available')

        if not MC_run and write_data:
            CSVHandler(
                self.countrate_path,
                self.count_overwrite).write_count_rate_csv(data)
            self.save_postproc()
            self.time_track(start, 'Countrate')
        return data

    def _count_rate_from_groups(self) -> dict[str: list[float]]:
        """
        Calculate the delayed neutron count rate from group parameters

        Returns
        -------
        data : dict[str: list[float]]
            Dictionary containing the times, count rates, and uncertainties
        """
        from mosden.groupfit import Grouper
        data: dict[str: list[float]] = dict()
        count_rate: np.ndarray = np.zeros(len(self.decay_times))
        sigma_count_rate: np.ndarray = np.zeros(len(self.decay_times))
        msg = f'{self.irrad_type} not supported in nonlinear least squares'
        grouper = Grouper(self.input_path)
        if self.irrad_type == 'pulse':
            fit_function = grouper._pulse_fit_function
        elif self.irrad_type == 'saturation':
            fit_function = grouper._saturation_fit_function
        else:
            raise NotImplementedError(msg)

        parameters = np.zeros(grouper.num_groups * 2, dtype=object)
        for i in range(grouper.num_groups):
            yield_val = ufloat(
                self.group_params['yield'][i],
                self.group_params['sigma yield'][i])
            half_life = ufloat(
                self.group_params['half_life'][i],
                self.group_params['sigma half_life'][i])
            parameters[i] = yield_val
            parameters[grouper.num_groups + i] = half_life

        counts = fit_function(self.decay_times, parameters)
        count_rate = np.asarray(unumpy.nominal_values(counts), dtype=float)
        sigma_count_rate = np.asarray(unumpy.std_devs(counts), dtype=float)

        data = {
            'times': self.decay_times,
            'counts': count_rate,
            'sigma counts': sigma_count_rate
        }
        return data

    def _count_rate_from_data(self,
                              MC_run: bool = False,
                              sampler_func: str = None
                              ) -> dict[str: list[float]]:
        """
        Calculate the delayed neutron count rate from existing data

        Parameters
        ----------
        MC_run : bool, optional
            Whether to run in Monte Carlo mode, by default False
        sampler_func : str, optional
            The sampling function to use for Monte Carlo, by default None

        Returns
        -------
        data : dict[str: list[float]]
            Dictionary containing the times, count rates, and uncertainties
        """
        def sample_parameter(val: ufloat, dist: str) -> float:
            if isinstance(val, float):
                return val
            if val.s == 0.0:
                return val.n
            if dist == 'normal':
                return np.random.normal(val.n, val.s)
            elif dist == 'uniform':
                return np.random.uniform(val.n - val.s, val.n + val.s)
            else:
                raise NotImplementedError(f'{dist} sampling not implemented')

        data: dict[str: list[float]] = dict()
        count_rate: np.ndarray = np.zeros(len(self.decay_times))
        sigma_count_rate: np.ndarray = np.zeros(len(self.decay_times))

        emission_nucs = list(self.emission_prob_data.keys())
        half_life_nucs = list(self.half_life_data.keys())
        conc_nucs = list(self.concentration_data.keys())
        net_unique_nucs = list(
            set(emission_nucs + half_life_nucs + conc_nucs))
        net_similar_nucs = list(
            set(emission_nucs) & set(half_life_nucs) & set(conc_nucs))
        if not MC_run:
            self.logger.info(
                f'Data contains {
                    len(net_unique_nucs)} unique nuclides')
            self.logger.info(
                f'Only {
                    len(net_similar_nucs)} are in all datasets')

        if len(net_similar_nucs) == 0:
            raise Exception(
                'Error: no data exists for given data combination')

        Pn_post_data = dict()
        lam_post_data = dict()
        conc_post_data = dict()

        for nuc in net_similar_nucs:
            Pn_data = self.emission_prob_data[nuc]
            Pn = ufloat(
                Pn_data['emission probability'],
                Pn_data['sigma emission probability'])

            hl_data = self.half_life_data[nuc]
            try:
                halflife = ufloat(
                    hl_data['half_life'],
                    hl_data['sigma half_life'])
            except KeyError:
                self.logger.warning('Half-life does not have uncertainties')
                halflife = hl_data['half_life']
            decay_const = np.log(2) / halflife

            conc_data = self.concentration_data[nuc]
            conc = ufloat(
                conc_data['Concentration'],
                conc_data['sigma Concentration'])

            if MC_run and sampler_func:
                Pn = sample_parameter(Pn, sampler_func)
                halflife = sample_parameter(halflife, sampler_func)
                decay_const = np.log(2) / halflife
                conc = sample_parameter(conc, sampler_func)

                if conc < 0.0:
                    conc = 1e-12
                if decay_const < 0.0:
                    decay_const = 1e-12
                if Pn < 0.0:
                    Pn = 1e-12

                counts = Pn * decay_const * conc * \
                    np.exp(-decay_const * self.decay_times)
                count_rate += counts
            else:
                counts = Pn * decay_const * conc * \
                    unumpy.exp(-decay_const * self.decay_times)
                count_rate += unumpy.nominal_values(counts)
                sigma_count_rate += unumpy.std_devs(counts)

            Pn_post_data[nuc] = Pn
            lam_post_data[nuc] = np.log(2) / decay_const
            conc_post_data[nuc] = conc

        if MC_run:
            if 'PnMC' not in self.post_data.keys():
                self.post_data['PnMC'] = list()
            if 'hlMC' not in self.post_data.keys():
                self.post_data['hlMC'] = list()
            if 'concMC' not in self.post_data.keys():
                self.post_data['concMC'] = list()
            self.post_data['PnMC'].append(Pn_post_data)
            self.post_data['hlMC'].append(lam_post_data)
            self.post_data['concMC'].append(conc_post_data)
            self.save_postproc()

        data = {
            'times': self.decay_times,
            'counts': count_rate,
            'sigma counts': sigma_count_rate
        }
        return data


if __name__ == '__main__':
    delayed_neutrons = CountRate('../examples/keepin_1957/input.json')
    delayed_neutrons.calculate_count_rate()
