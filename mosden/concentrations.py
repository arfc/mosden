import numpy as np
from mosden.utils.csv_handler import CSVHandler
from uncertainties import ufloat
from mosden.base import BaseClass
from time import time


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


        if self.spatial_scaling == 'unscaled':
            self.repr_scale = 1.0
            self.f_in = 1.0
            self.f_ex = 1.0
        elif self.spatial_scaling == 'scaled':
            try:
                self.f_in: float = self.t_in / (self.t_in + self.t_ex)
                self.f_ex: float = self.t_ex / (self.t_in + self.t_ex)
            except ZeroDivisionError:
                self.logger.error('No in-core or ex-core time')
            self.repr_scale = 0.0
            if 'incore' in self.reprocess_locations:
                self.repr_scale += self.f_in
            if 'excore' in self.reprocess_locations:
                self.repr_scale += self.f_ex
            self.repr_scale /= self.base_repr_scale
        else:
            raise NotImplementedError(
                f'{self.spatial_scaling} not implemented')
        self.fission_term = 1.0 * self.f_in
        if self.repr_scale <= 0.0:
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
    
    def OMC_concentrations(self) -> None:
        """
        Generate the concentrations of each nuclide using OpenMC.
        """
        self.repr_scale # multiply chemical removal rates by this term
        self.reprocess: bool # if reprocessing is present
        self.reprocess_locations: list[str] # Where reprocessing occurs
        self.t_in: float
        self.t_ex: float
        self.t_net: float
        self.decay_times: np.ndarray[float]
        self.reprocessing: dict[str, float] # elemental reprocessing rates
        self.energy_MeV
        self.fissiles
        self.temperature_K
        self.density_g_cc
        self.openmc_settings['nps']
        self.openmc_settings['mode']
        self.openmc_settings['batches']
        self.openmc_settings['flux']
        self.seed



        return None


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
