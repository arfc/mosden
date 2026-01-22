import numpy as np
from mosden.utils.csv_handler import CSVHandler
from mosden.concentrations import Concentrations
from uncertainties import unumpy
from mosden.base import BaseClass
from scipy.optimize import least_squares
from typing import Callable
from math import ceil
from time import time
import warnings
from tqdm import tqdm
from scipy.linalg import svd


class Grouper(BaseClass):
    def __init__(self, input_path: str) -> None:
        """
        This class forms DNP group parameters from count rates.

        Parameters
        ----------
        input_path : str
            Path to the input file
        """
        super().__init__(input_path)
        self.output_dir: str = self.input_data['file_options']['output_dir']

        self.fission_term, self.fission_times = self._calculate_fission_term()
        return None

    def _calculate_fission_term(self) -> list[float]:
        """
        Calculate the fission rate or number of fissions.
        The fission rate is used for saturation irradiations, while the number
        of fissions is used for pulse irradiations.

        Returns
        -------
        fission_term : list[float]
            The number/rate of fissions in the sample. The list is length > 1
            when using OMC for concentration calculation due to variation in the
            fission rate history.
        
        times : list[float]
            The times at which the fission history is evaluated (None unless OMC
            is used)

        Raises
        ------
        NotImplementedError
            Pulse irradiation not yet available.

        NameError
            Type of irradiation provided does not match any of the available
            irradiation types.
        """
        conc_handler = Concentrations(self.input_path)
        fission_term = [1.0]
        times = None

        if self.omc:
            fission_term, times = conc_handler.read_omc_fission_json(only_incore=True)
            fission_term = fission_term['net']

        if self.irrad_type == 'pulse':
            fission_term = [sum(fission_term)]
            if not self.omc:
                self.logger.error('Pulse irradiation fission term not treated')
        elif self.irrad_type == 'saturation':
            if conc_handler.spatial_scaling == 'scaled':
                fission_term = [conc_handler.f_in * f for f in fission_term]
        else:
            raise NameError(f'{self.irrad_type = } not available')

        return fission_term, times

    def generate_groups(self) -> None:
        """
        Generate some number of groups based on the selected method
        """
        start = time()
        data: dict[str: dict[str: float]] = dict()
        if self.group_method == 'nlls':
            data = self._nonlinear_least_squares()
        else:
            raise NotImplementedError(
                f'{self.group_method} is not implemented')
        CSVHandler(
            self.group_path,
            self.group_overwrite).write_groups_csv(
            data,
            sortby='half_life')
        self.save_postproc()
        self.time_track(start, 'Groupfit')
        self.logger.info(f'Ran in {time() - start} s')
        return None

    def _residual_function(
            self,
            parameters: np.ndarray[float],
            times: np.ndarray[float],
            counts: np.ndarray[float],
            count_err: np.ndarray[float],
            fit_func: Callable) -> float:
        """
        Calculate the residual of the current set of parameters

        Parameters
        ----------
        parameters : np.ndarray[float]
            Half life and yield parameters [yield1, yield2, ..., h1, h2, ...]
        times : np.ndarray[float]
            List of times
        counts : np.ndarray[float]
            List of delayed neutron counts
        count_err : np.ndarray[float]
            List of count errors
        fit_func : Callable
            Function that takes times and parameters to return list of counts

        Returns
        -------
        residual : float
            Value of the residual
        """
        residual = (counts - fit_func(times, parameters)) / (counts)
        return residual

    def _pulse_fit_function(self,
                            times: np.ndarray[float | object],
                            parameters: np.ndarray[float | object]
                            ) -> np.ndarray[float | object]:
        """
        Fit function for a pulse irradiation

        Parameters
        ----------
        times : np.ndarray[float|object]
            Times at which to evaluate the fit function
        parameters : np.ndarray[float|object]
            Fit parameters for the model

        Returns
        -------
        counts : np.ndarray[float|object]
            Array of counts for each time point (can be float or ufloat)
        """
        yields = parameters[:self.num_groups]
        half_lives = parameters[self.num_groups:]
        counts: np.ndarray[float] = np.zeros(len(times))
        for group in range(self.num_groups):
            lam = np.log(2) / half_lives[group]
            a = yields[group]
            try:
                counts += (a * lam * np.exp(-lam * times))
            except TypeError:
                if group == 0:
                    counts: np.ndarray[object] = np.zeros(
                        len(times), dtype=object)
                counts += (a * lam * unumpy.exp(-lam * times))
        return counts * self.fission_term[0]

    def _saturation_fit_function(self,
                                 times: np.ndarray[float | object],
                                 parameters: np.ndarray[float | object]
                                 ) -> np.ndarray[float | object]:
        """
        Fit function for a saturation irradiation

        Parameters
        ----------
        times : np.ndarray[float|object]
            Times at which to evaluate the fit function
        parameters : np.ndarray[float|object]
            Fit parameters for the model

        Returns
        -------
        counts : np.ndarray[float|object]
            Array of counts for each time point (can be float or ufloat)
        """
        yields = parameters[:self.num_groups]
        half_lives = parameters[self.num_groups:]
        counts: np.ndarray[float] = np.zeros(len(times))
        t_sum: float = self.t_in + self.t_ex
        try:
            recircs: int = int(np.floor(self.t_net/t_sum))
            irrad_circs: int = int(np.floor((self.t_net-self.t_in)/t_sum))
        except ZeroDivisionError:
            recircs = 0
            irrad_circs = 0
        
        try:
            np.exp(-np.log(2)/half_lives[0])
            exp = np.exp
        except TypeError:
            exp = unumpy.exp
            counts: np.ndarray[object] = np.zeros(
                len(times), dtype=object)

        for group in range(self.num_groups):
            lam = np.log(2) / half_lives[group]
            nu = yields[group]
            group_counts = 0

            if self.t_ex == 0:
                group_counts += exp(-lam*times) * (1 - exp(-lam*self.t_net))
                counts += nu * group_counts
                continue

            for j in range(0, irrad_circs+1):
                group_counts += exp(-lam*(times+self.t_net-j*t_sum-self.t_in)) - exp(-lam*(times+self.t_net-j*t_sum))
            for j in range(irrad_circs+1, recircs+1):
                group_counts += exp(-lam*times) - exp(-lam*(times+self.t_net-j*t_sum))
            counts += nu * group_counts
        return self.refined_fission_term * counts
    
    def _set_refined_fission_term(self, fine_times: np.ndarray[float]) -> float: 
        """
        Sets the `refined_fission_term` to a finer set of times.
        Currently takes the average value rather than using a time dependent
        version due to limitations in the derived equation.

        Parameters
        ----------
        fine_times : np.ndarray[float]
            The finer set of times over which to apply the fission history

        Returns
        -------
        refined_fission_term : float
            The fission term calculated using the mean (future work may use the
            explicit fission rate history tracking, returning a np.ndarray)
        """
        if not self.omc:
            self.refined_fission_term = np.mean(self.fission_term)
            return

        refined_term = list()
        for t in fine_times:
            for i in range(len(self.fission_term)):
                if self.fission_times[i] <= t < self.fission_times[i+1]:
                    refined_term.append(self.fission_term[i])
                    break
        self.refined_fission_term = np.asarray(refined_term)
        self.logger.debug('Time dependent fission rate history not enabled')
        self.refined_fission_term = np.mean(self.fission_term)
        return self.refined_fission_term

    def _nonlinear_least_squares(self,
                                 count_data: dict[str: np.ndarray[float]] = None
                                 ) -> dict[str: dict[str: float]]:
        """
        Run nonlinear least squares fit on the delayed neutron count rate curve
        to generate group half-lives and yields

        Parameters
        ----------
        count_data : dict[str: np.ndarray[float]], optional
            Dictionary containing the count data, by default None

        Returns
        -------
        data : dict[str: dict[str: float]]
            Dictionary containing the group parameters
              (yield, sigma yield, half_life, sigma half_life)
        """
        from mosden.countrate import CountRate
        initial_parameter_guess = np.ones(self.num_groups * 2)
        if count_data is None:
            count_data = CSVHandler(self.countrate_path).read_vector_csv()
        times = np.asarray(count_data['times'])
        self._set_refined_fission_term(times)
        counts = np.asarray(count_data['counts'])
        count_err = np.asarray(count_data['sigma counts'])
        if self.irrad_type == 'pulse':
            fit_function = self._pulse_fit_function
        elif self.irrad_type == 'saturation':
            fit_function = self._saturation_fit_function
        else:
            raise NotImplementedError(
                f'{self.irrad_type} not supported in nonlinear least squares')

        min_half_life = 1e-3
        max_half_life = 1e3
        max_yield = 1.0
        lower_bounds = np.concatenate(
            (np.zeros(
                self.num_groups), np.ones(
                self.num_groups) * min_half_life))
        upper_bounds = np.concatenate(
            (np.ones(
                self.num_groups) *
                max_yield,
                np.ones(
                self.num_groups) *
                max_half_life))

        bounds = (lower_bounds, upper_bounds)
        result = least_squares(self._residual_function,
                               initial_parameter_guess,
                               bounds=bounds,
                               method='trf',
                               ftol=1e-12,
                               gtol=1e-12,
                               xtol=1e-12,
                               verbose=0,
                               max_nfev=1e5,
                               args=(times, counts, count_err, fit_function))
        J = result.jac
        s = svd(J, compute_uv=False)
        condition_number = s[0] / s[-1]
        self.logger.info(f'{condition_number = }')
        sampled_params: list[float] = list()
        tracked_counts: list[float] = list()
        sorted_params = self._sort_params_by_half_life(result.x)
        sampled_params.append(sorted_params)
        countrate = CountRate(self.input_path)
        self.logger.info(f'Currently using {self.sample_func} sampling')
        post_data_save = []
        for _ in tqdm(range(1, self.MC_samples), desc='Solving least-squares'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                data, post_data = countrate.calculate_count_rate(
                    MC_run=True, sampler_func=self.sample_func)
                post_data_save.append(post_data)
                count_sample = data['counts']
                count_sample_err = data['sigma counts']
                result = least_squares(
                    self._residual_function,
                    result.x,
                    bounds=bounds,
                    method='trf',
                    ftol=1e-12,
                    gtol=1e-12,
                    xtol=1e-12,
                    verbose=0,
                    max_nfev=1e3,
                    args=(
                        times,
                        count_sample,
                        count_sample_err,
                        fit_function))
            tracked_counts.append([i for i in count_sample])
            sorted_params = self._sort_params_by_half_life(result.x)
            sampled_params.append(sorted_params)
        sampled_params: np.ndarray[float] = np.asarray(sampled_params)

        if 'PnMC' not in self.post_data.keys():
            self.post_data['PnMC'] = list()
        if 'hlMC' not in self.post_data.keys():
            self.post_data['hlMC'] = list()
        if 'concMC' not in self.post_data.keys():
            self.post_data['concMC'] = list()
        for post_data_vals in post_data_save:
            for key in self.post_data.keys():
                self.post_data[key].append(post_data_vals[key])
        self.save_postproc()

        yields = np.zeros((self.num_groups, self.MC_samples))
        half_lives = np.zeros((self.num_groups, self.MC_samples))

        for MC_i, params in enumerate(sampled_params):
            yield_val = params[:self.num_groups]
            half_life_val = params[self.num_groups:]
            sort_idx = np.argsort(half_life_val)[::-1]
            yields[:, MC_i] = np.asarray(yield_val)[sort_idx]
            half_lives[:, MC_i] = np.asarray(half_life_val)[sort_idx]

        groupMCdata = list()
        for iterval in range(self.MC_samples):
            groupMCdata.append([i for i in sampled_params[iterval]])
        try:
            self.post_data['groupfitMC'] = groupMCdata
            self.post_data['countsMC'] = tracked_counts
        except AttributeError:
            self.load_post_data()
            self.post_data['groupfitMC'] = groupMCdata
            self.post_data['countsMC'] = tracked_counts

        data: dict[str: dict[str: float]] = dict()
        for group in range(self.num_groups):
            data[group] = dict()
            data[group]['yield'] = np.mean(yields[group])
            data[group]['sigma yield'] = np.std(yields[group])
            data[group]['half_life'] = np.mean(half_lives[group])
            data[group]['sigma half_life'] = np.std(half_lives[group])
        return data

    def _sort_params_by_half_life(
            self, params: np.ndarray[float]) -> np.ndarray[float]:
        """
        Sorts yields and half-lives in params by half-life (descending).

        Parameters
        ----------
        params : np.ndarray[float]
            Array of parameters containing yields and half-lives

        Returns
        -------
        sorted_params : np.ndarray[float]
            Array of parameters sorted by half-life in descending order
        """
        yields = params[:self.num_groups]
        half_lives = params[self.num_groups:]
        sort_idx = np.argsort(half_lives)[::-1]
        sorted_yields = np.asarray(yields)[sort_idx]
        sorted_half_lives = np.asarray(half_lives)[sort_idx]
        return np.concatenate([sorted_yields, sorted_half_lives])


if __name__ == "__main__":
    input_path = "../examples/keepin_1957/input.json"
    groupcalc = Grouper(input_path)
    groupcalc.generate_groups()
    data = CSVHandler(groupcalc.group_path).read_vector_csv()
    print(data)
