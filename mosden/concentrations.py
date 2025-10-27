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
        self.output_dir: str = self.input_data['file_options']['output_dir']
        modeling_options: dict = self.input_data.get('modeling_options', {})
        file_options: dict = self.input_data.get('file_options', {})
        overwrite: dict = file_options.get('overwrite', {})

        self.model_method: str = modeling_options.get(
            'concentration_handling', 'CFY')
        self.overwrite: bool = overwrite.get('concentrations', False)

        self.reprocessing: dict[str: float] = modeling_options.get(
            'reprocessing', {})
        self.reprocess: bool = (sum(self.reprocessing.values()) > 0)
        self.reprocess_locations: list[str] = modeling_options.get(
            'reprocessing_locations', [])
        self.t_in: float = modeling_options.get('incore_s', 0.0)
        self.t_ex: float = modeling_options.get('excore_s', 0.0)
        self.t_net: float = modeling_options.get('net_irrad_s', 0.0)
        self.irrad_type: str = modeling_options.get('irrad_type', 'saturation')
        self.f_in = 1.0
        self.f_ex = 1.0
        try:
            self.f_in: float = self.t_in / (self.t_in + self.t_ex)
            self.f_ex: float = self.t_ex / (self.t_in + self.t_ex)
        except ZeroDivisionError:
            self.logger.error('No in-core or ex-core time')
        self.spatial_scaling: str = modeling_options.get(
            'spatial_scaling', 'unscaled')

        if self.spatial_scaling == 'unscaled':
            self.repr_scale = 1.0
            self.f_in = 1.0
            self.f_ex = 1.0
        elif self.spatial_scaling == 'scaled':
            self.repr_scale = 0.0
            if 'incore' in self.reprocess_locations:
                self.repr_scale += self.f_in
            if 'excore' in self.reprocess_locations:
                self.repr_scale += self.f_ex
        else:
            raise NotImplementedError(
                f'{self.spatial_scaling} not implemented')
        self.fission_term = 1.0 * self.f_in

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
        if self.model_method == 'CFY':
            if self.irrad_type != 'saturation':
                self.logger.error(
                    'CFY is intended for a saturation irradiation')
            data = self.CFY_concentrations()
        elif self.model_method == 'IFY':
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
        else:
            raise NotImplementedError(
                f"Concentration handling method '{
                    self.model_method}' is not implemented")

        CSVHandler(self.concentration_path, self.overwrite).write_csv(data)
        self.save_postproc()
        self.time_track(start, 'Concentrations')
        return

    def CFY_concentrations(self) -> None:
        """
        Generate the concentrations of each nuclide using the CFY method.
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
            loss_term = lam + self.repr_scale * repr_term
            concentrations[nuclide] = self.f_in * concs / loss_term
            all_nucs.add(nuclide)

        data: dict[str: dict[str: float]] = dict()
        for nuc in all_nucs:
            data[nuc] = {}
            data[nuc]['Concentration'] = concentrations[nuc].n
            data[nuc]['sigma Concentration'] = concentrations[nuc].s

        return data

    def IFY_concentrations(self) -> None:
        """
        Generate the concentrations of each nuclide using the IFY method.
        """
        concentrations: dict[str: dict[str: ufloat]] = dict()
        all_nucs: set[str] = set()
        IFY_data = self._read_processed_data('fission_yield')
        for nuclide in IFY_data.keys():
            concs = IFY_data[nuclide]['IFY']
            concentrations[nuclide] = concs
            all_nucs.add(nuclide)

        data: dict[str: dict[str: float]] = dict()
        for nuc in all_nucs:
            data[nuc] = {}
            data[nuc]['Concentration'] = concentrations[nuc]
            data[nuc]['sigma Concentration'] = 1e-12

        return data


if __name__ == "__main__":
    input_path = "../examples/keepin_1957/input.json"
    concentrations = Concentrations(input_path)
    concentrations.generate_concentrations()
