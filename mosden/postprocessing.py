from logging import INFO
from uncertainties import ufloat, unumpy
import numpy as np
import os
from mosden.utils.literature_handler import Literature
from mosden.countrate import CountRate
from mosden.concentrations import Concentrations
from mosden.utils.csv_handler import CSVHandler
from mosden.base import BaseClass
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
import re
import pandas as pd
from scipy.stats import linregress
from armi import configure
from armi.nucDirectory import nuclideBases
plt.style.use('mosden.plotting')


class PostProcess(BaseClass):
    def __init__(self, input_path: str) -> None:
        """
        This class creates figures and performs analysis of generated data.

        Parameters
        ----------
        input_path : str
            Path to the input file
        """
        super().__init__(input_path)
        file_options: dict = self.input_data.get('file_options', {})
        modeling_options: dict = self.input_data.get('modeling_options', {})
        data_options: dict = self.input_data['data_options']
        overwrite: dict = file_options.get('overwrite', {})
        self.processed_data_dir: str = file_options.get('processed_data_dir',
                                                        '')
        self.output_dir: str = os.path.join(file_options.get('output_dir', ''),
                                            'images/')
        self.overwrite: bool = overwrite.get('postprocessing', False)
        self.num_groups: int = self.input_data['group_options']['num_groups']
        self.MC_samples: int = self.input_data['group_options']['samples']
        self.irrad_type: str = self.input_data['modeling_options']['irrad_type']
        self.use_data: list[str] = [
            'keepin', 'brady', 'synetos']#, 'Modified 0D Scaled']
        self.self_relative_data: bool = False
        self.nuclides: list[str] = [
            'Br87',
            'I137',
            'Br88',
            'Br89',
            'I138',
            'Rb94',
            'Rb93',
            'Te136',
            'Ge86',
            'As86',
            'Br90',
            'As85']
        self.markers: list[str] = ['v', 'o', 'x', '^', 's', 'D']
        self.linestyles: list[str] = ['-', '--', ':', '-.']
        self.load_post_data()
        self.decay_times: np.ndarray[float] = CountRate(input_path).decay_times
        self.num_decay_times = modeling_options['num_decay_times']
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.t_in: float = modeling_options.get('incore_s', 0.0)
        self.t_ex: float = modeling_options.get('excore_s', 0.0)
        self.decay_time_spacing: str = data_options['decay_time_spacing']
        self.total_decay_time: float = modeling_options['decay_time']
        self.group_data = None

        try:
            self.MC_yields, self.MC_half_lives = self._get_MC_group_params()
        except KeyError:
            self.logger.warning('Postdata does not exist')
        self.fission_term = Concentrations(self.input_path).fission_term

        return None

    def get_colors(self, num_colors: int, colormap: str = None,
                   min_val: float = 0.0, max_val: float = 1.0) -> list[tuple[float]]:
        """
        Get a list of colors from a colormap

        Parameters
        ----------
        num_colors : int
            Number of colors to retrieve
        colormap : str, optional
            Name of the colormap to use, by default None
        min_val : float, optional
            Minimum value for the colormap, by default 0.05
        max_val : float, optional
            Maximum value for the colormap, by default 1.0

        Returns
        -------
        colors : list[tuple[float]]
            List of colors
        """
        cmap = plt.get_cmap(colormap)
        colors = [cmap(i) for i in np.linspace(min_val, max_val, num_colors+2)]
        return colors[1:-1]

    def _convert_nuc_to_latex(self, nuc: str) -> str:
        """
        Convert a nuclide string to a LaTeX formatted string.

        Parameters
        ----------
        nuc : str
            Nuclide string to convert

        Returns
        -------
        str
            LaTeX formatted string

        Raises
        ------
        ValueError
            Unexpected format error
        """
        match = re.match(r"([A-Za-z]+)(\d+)", nuc)
        if not match:
            raise ValueError(f"Unexpected format: {nuc}")
        elem, mass = match.groups()
        return rf"$^{{{mass}}}${elem}"

    def run(self) -> None:
        """
        The main run function for postprocessing

        """
        self.compare_yields()
        self.compare_group_to_data()
        self.MC_NLLS_analysis()
        return None

    def compare_group_to_data(self) -> None:
        """
        Runs functions that compare the group parameters to the data
        """
        self._plot_group_vs_counts()
        return None

    def _plot_group_vs_counts(self) -> None:
        """
        Plot the relative difference between the group fit and the counts
        """
        group_data = CSVHandler(
            self.group_path,
            create=False).read_vector_csv()
        countrate = CountRate(self.input_path)
        countrate.group_params = group_data
        group_counts = countrate._count_rate_from_groups()['counts']
        summed_counts = CSVHandler(
            self.countrate_path).read_vector_csv()['counts']
        pcnt_diff = (summed_counts - group_counts) / summed_counts * 100
        plt.plot(self.decay_times, pcnt_diff)
        plt.xlabel('Time [s]')
        plt.xscale('log')
        plt.ylabel('Relative Difference [\\%]')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}pcnt_diff_counts.png')
        plt.close()
        return None
    
    def _chart_form(self, name: str, data: dict, cbar_label: str) -> None:
        """
        Create a chart of the nuclides with file name and with data

        Parameters
        ----------
        name : str
            Name of image
        data : dict[str, float]
            Data to plot, using the nuclide name as a key and the value to plot
            (of the form "XE135")
        """
        configure(permissive=True)
        plt.figure(figsize=(12, 8))
        N = list()
        Z = list()
        C = list()
        for nuc, base in nuclideBases.byName.items():
            try:
                value = data[nuc.capitalize()]
                N.append(base.a - base.z)
                Z.append(base.z)
                C.append(value)
            except KeyError:
                continue
        norm = 'log'
        plt.scatter(N, Z, c=C, norm=norm, marker="s", s=60)
        plt.set_cmap('viridis')
        cbar = plt.colorbar()
        cbar.set_label(cbar_label)
        plt.xlabel("Number of neutrons (N)")
        plt.ylabel("Number of protons (Z)")
        plt.savefig(f'{self.output_dir}chart_{name}.png')
        plt.close()
        return None 

    def MC_NLLS_analysis(self) -> None:
        """
        Analyze Monte Carlo Non-linear Least Squares results
        """
        self._plot_counts()
        if self.MC_samples > 1:
            self._plot_MC_group_params()
            self._get_sens_coeffs(write=True)
            self._plot_sensitivities(
                off_nominal=True,
                relative_diff=True,
                subplot=False)
        return None

    def _get_sens_coeffs(self, write=False) -> tuple[list[dict[str, float]],
                                                     list[dict[str, float]], list[dict[str, float]], list[str]]:
        """
        Get the sensitivity coefficients for the Monte Carlo samples

        Parameters
        ----------
        write : bool, optional
            Whether to write the results to a file, by default False

        Returns
        -------
        tuple[list[dict[str, float]], list[dict[str, float]], list[dict[str, float]], list[str]]
            A tuple containing the sensitivity coefficients for the Monte Carlo samples
        """
        Pn_data = self.post_data['PnMC']
        hl_data = self.post_data['hlMC']
        conc_data = self.post_data['concMC']
        uncert_data = self._get_data()
        nuclides = list(Pn_data[0].keys())
        data_sets = [Pn_data, hl_data, conc_data]
        uncert_names = ['emission_probability',
                        'concentration',
                        'half_life']
        data_names = ['Emission Probability',
                      'Half-life',
                      'Concentration']
        group_data = [self.MC_yields,
                      self.MC_half_lives]
        group_names = ['Yield',
                       'Half-life']
        nucs_with_pcc = list()
        pcc_cutoff = 0.2
        summed_pcc_data = dict()
        scaled_uncert_pcc = dict()
        pcc_data = dict()
        if write:
            self.logger.info(f'Writing nuclides with PCC > {pcc_cutoff}')
        for nuc in nuclides:
            for group in range(self.num_groups):
                for gname, gdata in zip(group_names, group_data):
                    for data, name, uname in zip(data_sets, data_names, uncert_names):
                        try:
                            rel_uncertainty = (uncert_data['nucs'][nuc][uname].s / 
                                            (uncert_data['nucs'][nuc][uname].n))
                        except ZeroDivisionError:
                            rel_uncertainty = 1e-12
                        data_vals = [data[nuc] for data in data]
                        group_vals = gdata[group, 1:]
                        mean_group_val = np.mean(group_vals)
                        mean_data_val = np.mean(data_vals)
                        data_val = (
                            (data_vals - mean_data_val) / mean_data_val)
                        group_val = (
                            (group_vals - mean_group_val) / mean_group_val)
                        result = linregress(data_val, group_val)
                        current_pcc_val = summed_pcc_data.setdefault(nuc, 0.0)
                        current_uncert_val = scaled_uncert_pcc.setdefault(nuc, 0.0)
                        summed_pcc_data[nuc] = current_pcc_val + abs(result.rvalue)
                        scaled_uncert_pcc[nuc] = current_uncert_val + abs(result.rvalue) * rel_uncertainty
                        if abs(result.rvalue) > pcc_cutoff:
                            nuc_lab, group_lab = self._configure_x_y_labels(name,
                                                                            gname,
                                                                            False,
                                                                            False,
                                                                            group_val=group + 1)
                            nuc_name = self._convert_nuc_to_latex(nuc)
                            pcc_data.setdefault('Nuclide', []).append(nuc_name)
                            pcc_data.setdefault(
                                'Group Value', []).append(group_lab)
                            pcc_data.setdefault(
                                'DNP Value', []).append(nuc_lab)
                            pcc_data.setdefault(
                                'PCC', []).append(
                                result.rvalue)
                            nucs_with_pcc.append(nuc)
        pcc_df_data: pd.DataFrame = pd.DataFrame.from_dict(
            pcc_data, orient='columns')
        pcc_latex = pcc_df_data.to_latex(index=False)
        if write:
            self.logger.info(f'\n{pcc_latex}')
            self.logger.info('Completed writing nuclides \n')
            self._chart_form(name='PCC', data=summed_pcc_data, cbar_label='Sum of Pearson Correlation Coefficient Magnitudes')
            self._chart_form(name='PCC_uncertainty', data=scaled_uncert_pcc, cbar_label='Sum of Relative Uncertainties Scaled by PCC Magnitudes')
            sorted_summed_pccs = sorted(summed_pcc_data.items(), key=lambda item: item[1], reverse=True)
            top = 10
            self.logger.info(f'Writing {top = } summed |PCC| nuclides')
            for nuc,sum_PCC in sorted_summed_pccs[:top]:
                self.logger.info(f'{nuc = }    {sum_PCC = }')
            sorted_uncert_pccs = sorted(scaled_uncert_pcc.items(), key=lambda item: item[1], reverse=True)
            self.logger.info(f'Writing {top = } summed uncertainty times |PCC| nuclides')
            for nuc,sum_PCC in sorted_uncert_pccs[:top]:
                self.logger.info(f'{nuc = }    {sum_PCC = }')
        return Pn_data, hl_data, conc_data, nucs_with_pcc

    def _configure_x_y_labels(self, xlab: str, ylab: str, off_nominal: bool, relative_diff: bool,
                              group_val: str = 'k') -> tuple[str, str]:
        """
        Configure the x and y labels for the sensitivity plots

        Parameters
        ----------
        xlab : str
            The label for the x-axis
        ylab : str
            The label for the y-axis
        off_nominal : bool
            Whether the plot is off-nominal
        relative_diff : bool
            Whether to plot the relative difference
        group_val : str, optional
            The group value for the plot, by default 'k'

        Returns
        -------
        xlab, ylab : tuple[str, str]
            The configured x and y labels
        """
        xlabel_replace = {
            "Half-life": fr"$\tau_i [s]$",
            "Decay Constant": fr"$\lambda_i [s^{{-1}}]$",
            "Concentration": fr"$N_i [-]$",
            "Emission Probability": fr'$P_{{n, i}} [-]$'
        }
        ylabel_replace = {
            "Half-life": fr"$\tau_{group_val} [s]$",
            "Decay Constant": fr"$\lambda_{group_val} [s^{{-1}}]$",
            "Yield": fr"$\bar{{\nu}}_{{d, {group_val}}} [-]$",
        }
        offnom_ylabel_replace = {
            "Half-life": fr"$\Delta \tau_{group_val} [s]$",
            "Decay Constant": fr"$\Delta \lambda_{group_val} [s^{{-1}}]$",
            "Yield": fr"$\Delta \bar{{\nu}}_{{d, {group_val}}} [-]$",
        }
        pcnt_ylabel_replace = {
            "Half-life": fr"$\Delta \tau_{group_val} / \tau_{group_val} [\%]$",
            "Decay Constant": fr"$\Delta \lambda_{group_val} / \lambda_{group_val} [\%]$",
            "Yield": fr"$\Delta \bar{{\nu}}_{{d, {group_val}}} / \bar{{\nu}}_{{d, {group_val}}} [\%]$",
        }
        pcnt_xlabel_replace = {
            "Half-life": fr"$\Delta \tau_i / \tau_i [\%]$",
            "Decay Constant": fr"$\Delta \lambda_i / \lambda_i [\%]$",
            "Concentration": fr"$\Delta N_i / N_i [\%]$",
            "Emission Probability": fr'$\Delta P_{{n, i}} / P_{{n, i}} [\%]$'
        }

        if off_nominal:
            if relative_diff:
                ylab = pcnt_ylabel_replace[ylab]
            else:
                ylab = offnom_ylabel_replace[ylab]
        else:
            ylab = ylabel_replace[ylab]
        if not (off_nominal and relative_diff):
            xlab = xlabel_replace[xlab]
        else:
            xlab = pcnt_xlabel_replace[xlab]
        return xlab, ylab

    def _get_sens_data(self, nuc: str,
                       off_nominal: bool,
                       relative_diff: bool,
                       group_params: np.ndarray,
                       group: int,
                       indiv_dnp_data: dict,
                       actual_data_val: float=None) -> tuple[list[float, list[float]]]:
        """
        Get the sensitivity data for a given nuclide and group

        Parameters
        ----------
        nuc : str
            The nuclide to get data for
        off_nominal : bool
            Whether the plot is off-nominal
        relative_diff : bool
            Whether to plot the relative difference
        group_params : np.ndarray
            The group parameters for the plot
        group : int
            The group index
        indiv_dnp_data : dict
            The individual delayed neutron precursor data
        actual_data_val : float, optional
            The actual value of the data for the given nuclide

        Returns
        -------
        data_val, plot_val : tuple[list[float, list[float]]]
            The sensitivity data for the given nuclide and group
        """
        data_vals = [data[nuc] for data in indiv_dnp_data]
        group_vals = group_params[group, 1:]
        plot_val = group_vals
        mean_group_val = np.mean(group_vals)
        if not actual_data_val:
            mean_data_val = np.mean(data_vals)
        else:
            mean_data_val = np.float64(actual_data_val)
        if off_nominal:
            plot_val = group_vals - mean_group_val
            if relative_diff:
                data_val = ((data_vals - mean_data_val) / mean_data_val) * 100
                plot_val = (
                    (group_vals - mean_group_val) / mean_group_val) * 100
        return data_val, plot_val

    def _scatter_helper(self,
                        data: dict,
                        group_params: np.ndarray[float],
                        xlab: str,
                        ylab: str,
                        savename: str,
                        savedir: str,
                        off_nominal: bool = True,
                        nuclides: list[str] = None,
                        relative_diff: bool = True,
                        processed_data_dict: dict=None) -> None:
        """
        Helper function to create scatter plots of sensitivity data

        Parameters
        ----------
        data : dict
            The sensitivity data for the given nuclide and group
        group_params : np.ndarray[float]
            The group parameters for the plot
        xlab : str
            The label for the x-axis
        ylab : str
            The label for the y-axis
        savename : str
            The name to save the plot as
        savedir : str
            The directory to save the plot in
        off_nominal : bool, optional
            Whether the plot is off-nominal, by default True
        nuclides : list[str], optional
            The list of nuclides to plot, by default None
        relative_diff : bool, optional
            Whether to plot the relative difference, by default True
        processed_data_dict : dict, optional
            Dictionary of processed data for Pn, N, and half-lives.
            Mean values are used if this is not provided

        """

        nuclides = nuclides or self.nuclides or list(data[0].keys())
        xlab_new, ylab_new = self._configure_x_y_labels(
            xlab, ylab, off_nominal, relative_diff)
        num_colors = self.num_groups
        colors = self.get_colors(num_colors)
        label_mapping = {'Emission Probability': 'emission_probability',
                         'Concentration': 'concentration',
                         'Half-life': 'half_life'}

        for nuc in nuclides:
            actual_data_val = None
            if processed_data_dict:
                item = label_mapping[xlab]
                actual_data_val = processed_data_dict['nucs'][nuc][item]
                if type(actual_data_val) is not float:
                    actual_data_val = actual_data_val.n
                    
            for group in range(self.num_groups):
                data_val, plot_val = self._get_sens_data(nuc, off_nominal,
                                                         relative_diff,
                                                         group_params, group,
                                                         data,
                                                         actual_data_val=actual_data_val)
                plt.scatter(
                    data_val,
                    plot_val,
                    label=f'Group {group + 1}',
                    alpha=0.5,
                    s=4,
                    marker=self.markers[group],
                    color=colors[group])
                plt.xlabel(xlab_new)
                plt.ylabel(ylab_new)
                plt.savefig(f'{savedir}{savename}_{nuc}_{group+1}.png')
                plt.close()
        return None

    def _plot_sensitivities(self, off_nominal: bool = True,
                            relative_diff: bool = True,
                            subplot: bool = True) -> None:
        """
        Plot the sensitivities of emission probabilities, concentrations,
          and half-lives

        Parameters
        ----------
        off_nominal : bool, optional
            Whether to plot off-nominal sensitivities, by default True
        relative_diff : bool, optional
            Whether to use the relative difference, by default False
        subplot : bool, optional
            Whether to create subplots for each nuclide, by default True
        """
        pn_save_dir = os.path.join(self.output_dir, 'sens_pn/')
        if not os.path.exists(pn_save_dir):
            os.makedirs(pn_save_dir)
        lam_save_dir = os.path.join(self.output_dir, 'sens_hl/')
        if not os.path.exists(lam_save_dir):
            os.makedirs(lam_save_dir)
        conc_save_dir = os.path.join(self.output_dir, 'sens_conc/')
        if not os.path.exists(conc_save_dir):
            os.makedirs(conc_save_dir)
        processed_data_dict = self._get_data()

        Pn_data, hl_data, conc_data, nuclides = self._get_sens_coeffs()
        if not subplot:
            self._scatter_helper(
                Pn_data,
                self.MC_yields,
                'Emission Probability',
                'Yield',
                'sens_pn_yield',
                pn_save_dir,
                off_nominal=off_nominal,
                nuclides=nuclides,
                relative_diff=relative_diff,
                processed_data_dict=processed_data_dict)
            self._scatter_helper(
                hl_data,
                self.MC_yields,
                'Half-life',
                'Yield',
                'sens_lam_yield',
                lam_save_dir,
                off_nominal=off_nominal,
                nuclides=nuclides,
                relative_diff=relative_diff,
                processed_data_dict=processed_data_dict)
            self._scatter_helper(
                conc_data,
                self.MC_yields,
                'Concentration',
                'Yield',
                'sens_conc_yield',
                conc_save_dir,
                off_nominal=off_nominal,
                nuclides=nuclides,
                relative_diff=relative_diff,
                processed_data_dict=processed_data_dict)
            self._scatter_helper(
                Pn_data,
                self.MC_half_lives,
                'Emission Probability',
                'Half-life',
                'sens_pn_halflife',
                pn_save_dir,
                off_nominal=off_nominal,
                nuclides=nuclides,
                relative_diff=relative_diff,
                processed_data_dict=processed_data_dict)
            self._scatter_helper(
                hl_data,
                self.MC_half_lives,
                'Half-life',
                'Half-life',
                'sens_lam_halflife',
                lam_save_dir,
                off_nominal=off_nominal,
                nuclides=nuclides,
                relative_diff=relative_diff,
                processed_data_dict=processed_data_dict)
            self._scatter_helper(
                conc_data,
                self.MC_half_lives,
                'Concentration',
                'Half-life',
                'sens_conc_halflife',
                conc_save_dir,
                off_nominal=off_nominal,
                nuclides=nuclides,
                relative_diff=relative_diff,
                processed_data_dict=processed_data_dict)
        else:
            subplot_save_dir = os.path.join(self.output_dir, 'sens_subplots/')
            if not os.path.exists(subplot_save_dir):
                os.makedirs(subplot_save_dir)
            group_data = [self.MC_yields, self.MC_half_lives]
            group_name = ['Yield', 'Half-life']
            dnp_data = [Pn_data, hl_data, conc_data]
            dnp_name = ['Emission Probability', 'Half-life', 'Concentration']
            label_mapping = {'Emission Probability': 'emission_probability',
                            'Concentration': 'concentration',
                            'Half-life': 'half_life'}
            num_colors = self.num_groups
            colors = self.get_colors(num_colors)
            for nuc in nuclides:
                for group_val, gname in zip(group_data, group_name):
                    fig = plt.figure()
                    gs = fig.add_gridspec(
                        self.num_groups, 3, hspace=0.1, wspace=0.05)
                    axs = gs.subplots(sharex='col', sharey='row')
                    for group_i in range(self.num_groups):
                        for dnp_i, (dnp, name_dnp) in enumerate(
                                zip(dnp_data, dnp_name)):
                            actual_data_val = None
                            if processed_data_dict:
                                item = label_mapping[name_dnp]
                                actual_data_val = processed_data_dict['nucs'][nuc][item]
                                if type(actual_data_val) is not float:
                                    actual_data_val = actual_data_val.n
                            dataval, plotval = self._get_sens_data(
                                nuc, off_nominal, relative_diff, group_val, group_i, dnp, actual_data_val=actual_data_val)
                            cur_ax = axs[group_i, dnp_i]
                            cur_ax.scatter(
                                dataval,
                                plotval,
                                label=f'Group {group_i + 1}',
                                alpha=0.5,
                                s=4,
                                marker=self.markers[group_i],
                                color=colors[group_i])
                            xlab, ylab = self._configure_x_y_labels(
                                name_dnp, gname, off_nominal, relative_diff)
                            if group_i == self.num_groups - 1:
                                cur_ax.set_xlabel(xlab, fontsize=8)
                            else:
                                cur_ax.set_xlabel('')
                            cur_ax.tick_params(
                                axis='both', which='major', labelsize=8)
                    lines_labels = [ax.get_legend_handles_labels()
                                    for ax in fig.axes]
                    lines, labels = [sum(lol, [])
                                     for lol in zip(*lines_labels)]

                    unique = dict(zip(labels, lines))
                    fig.legend(unique.values(), unique.keys(),
                               loc='center right', fontsize=8)

                    plt.subplots_adjust(right=0.85)
                    fig.supylabel(ylab)
                    plt.savefig(f'{subplot_save_dir}{gname}_{nuc}.png')
                    plt.close()
        return None

    def compare_yields(self) -> None:
        """
        Compare the total DN yields from summing individuals and from
          group parameters
        """
        num_top = 4
        num_stack = 2
        summed_yield, summed_avg_halflife = self._get_summed_params(num_top)
        group_yield, group_avg_halflife = self._get_group_params()

        self.summed_yield = summed_yield
        self.summed_avg_halflife = summed_avg_halflife
        self.group_yield = group_yield
        self.group_avg_halflife = group_avg_halflife

        self._plot_nuclide_count_rates(num_stack)
        self.logger.info(f'{summed_yield = }')
        self.logger.info(f'{summed_avg_halflife = } s')
        self.logger.info(f'{group_yield = }')
        self.logger.info(f'{group_avg_halflife = } s')
        return None

    def _plot_nuclide_count_rates(self, num_stack: int = 1):
        """
        Plot the most important nuclide (by delayed neutron counts emitted) at
            each time step.

        Parameters
        ----------
        num_stack : int, optional
            number of nuclides to plot stacked at each time, by default 1
        """
        data_dict = self._get_data()
        net_nucs = data_dict['net_nucs']
        count_rates = dict()

        for nuc in net_nucs:
            Pn = data_dict['nucs'][nuc]['emission_probability']
            N = data_dict['nucs'][nuc]['concentration']
            hl = data_dict['nucs'][nuc]['half_life']
            lam = np.log(2) / hl
            count_rates[nuc] = list()
            counts = Pn * lam * N * unumpy.exp(-lam * self.decay_times)
            count_rates[nuc] = counts

        biggest_nucs_list = list()
        nuc_names = list()
        for ti in range(len(self.decay_times)):
            cur_t_counts = dict()
            for nuc in net_nucs:
                cur_t_counts[nuc] = count_rates[nuc][ti].n
            for nuc in range(num_stack):
                try:
                    max_nuc = max(cur_t_counts, key=cur_t_counts.get)
                except ValueError:
                    self.logger.warning("Max nuc evaluation failed")
                    break
                biggest_nucs_list.append(max_nuc)
                nuc_names.append(self._convert_nuc_to_latex(max_nuc))
                del cur_t_counts[max_nuc]
        biggest_nucs = list(dict.fromkeys(biggest_nucs_list))
        nuc_names = list(dict.fromkeys(nuc_names))

        colors = self.get_colors(len(biggest_nucs))
        for nuci, nuc in enumerate(biggest_nucs):
            rate_n = unumpy.nominal_values(count_rates[nuc])
            rate_s = unumpy.std_devs(count_rates[nuc])
            upper = rate_n + rate_s
            lower = rate_n - rate_s
            plt.fill_between(self.decay_times, lower, upper, color=colors[nuci],
                             alpha=0.5)
            plt.plot(self.decay_times, rate_n, color=colors[nuci], label=nuc_names[nuci],
                     linestyle='--', marker=self.markers[nuci % len(self.markers)], markevery=5,
                     markersize=3)

        plt.xlabel('Time [s]')
        plt.ylabel(r'Delayed Neutron Rate $[s^{-1}]$')
        plt.xscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}individual_nuclide_rates.png')
        plt.close()

        stacked_data = list()
        for nuci, nuc in enumerate(biggest_nucs):
            rate_n = unumpy.nominal_values(count_rates[nuc])
            rate_s = unumpy.std_devs(count_rates[nuc])
            upper = cumulative_trapezoid(
                rate_n + rate_s, self.decay_times, initial=0)
            lower = cumulative_trapezoid(
                rate_n - rate_s, self.decay_times, initial=0)
            rate_n = cumulative_trapezoid(rate_n, self.decay_times, initial=0)
            rate_s = cumulative_trapezoid(rate_s, self.decay_times, initial=0)
            stacked_data.append(rate_n)

            plt.fill_between(
                self.decay_times,
                lower,
                upper,
                color=colors[nuci],
                alpha=0.5)
            plt.plot(self.decay_times, rate_n, color=colors[nuci], label=nuc_names[nuci],
                     linestyle='--', marker=self.markers[nuci % len(self.markers)], markevery=5,
                     markersize=3)
        plt.xlabel('Time [s]')
        plt.ylabel('Relative Delayed Neutron Counts')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}individual_nuclide_counts.png')
        plt.close()

        plt.stackplot(self.decay_times, stacked_data, labels=nuc_names,
                      colors=colors)
        plt.xlabel('Time [s]')
        plt.ylabel('Relative Delayed Neutron Counts')
        plt.xscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}individual_nuclide_counts_stacked.png')
        plt.close()

        return None

    def _group_param_helper(self,
                            name: str,
                            items: np.ndarray[float],
                            group_data: dict[str: object]) -> None:
        """

        Parameters
        ----------
        name : str
            Name of the parameter to plot ('yield' or 'half_life')
        items : np.ndarray[float]
            Items to plot
        group_data : dict[str: object]
            Group data containing parameter values and uncertainties

        Raises
        ------
        NotImplementedError
            If the name is not 'yield' or 'half_life'
        """
        if name == 'yield':
            label_name = 'Yield'
            xlabel = 'Yield'
            scaler = 1
            scale_label = ''
        elif name == 'half_life':
            label_name = 'Half-life'
            xlabel = 'Half-life [s]'
            scaler = 1
            scale_label = ''
        else:
            raise NotImplementedError(f'{name} not defined')

        group_item = [scaler * ufloat(y,
                                      std) for y,
                      std in zip(group_data[name],
                                 group_data[f'sigma {name}'])]
        for group, item in enumerate(items):
            item = item * scaler
            bins = np.linspace(
                min(item), max(item), int(
                    np.sqrt(
                        len(item))))
            counts, edges = np.histogram(item, bins=bins)
            normalized_counts = counts
            bin_centers = 0.5 * (edges[:-1] + edges[1:])
            plt.bar(
                bin_centers,
                normalized_counts,
                width=np.diff(edges),
                label=f'Sampled {label_name}',
                alpha=0.5,
                color='red',
                edgecolor='black')

            plt.axvline(
                group_item[group].n,
                color='blue',
                linestyle='--',
                label=fr'Group {label_name} ± $1\sigma$')
            plt.axvspan(
                group_item[group].n -
                group_item[group].s,
                group_item[group].n +
                group_item[group].s,
                color='blue',
                alpha=0.25)

            plt.axvline(items[group, 0], color='black',
                        linestyle='-', label=f'Nominal {label_name}')

            plt.xlabel(xlabel + scale_label)
            plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(
                useMathText=True))
            plt.ticklabel_format(style='sci', axis='x', scilimits=(-2, 2))
            plt.ylabel('Frequency')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}MC_group{group + 1}_{name}.png')
            plt.close()
        return None

    def _plot_MC_group_params(self) -> None:
        """
        Plot the group parameters from the Monte Carlo NLLS analysis
        """

        self.group_data = CSVHandler(
            self.group_path,
            create=False).read_vector_csv()

        self._group_param_helper('yield', self.MC_yields, self.group_data)
        self._group_param_helper(
            'half_life',
            self.MC_half_lives,
            self.group_data)

        return None

    def _get_MC_group_params(
            self) -> tuple[np.ndarray[float], np.ndarray[float]]:
        """
        Get the Monte Carlo group parameters from the postprocessing data
        Returns yields and half-lives as numpy arrays

        Returns
        -------
        yields, half_lives : tuple[np.ndarray[float], np.ndarray[float]]
            Tuple containing the yields and half-lives as numpy arrays
        """
        parameters = self.post_data[self.names['groupfitMC']]
        yields = np.zeros((self.num_groups, self.MC_samples))
        half_lives = np.zeros((self.num_groups, self.MC_samples))
        for MC_i, params in enumerate(parameters):
            yield_val = params[:self.num_groups]
            half_life_val = params[self.num_groups:]
            sort_idx = np.argsort(half_life_val)[::-1]
            yields[:, MC_i] = np.asarray(yield_val)[sort_idx]
            half_lives[:, MC_i] = np.asarray(half_life_val)[sort_idx]

        return yields, half_lives

    def _plot_counts(self) -> None:
        """
        Plot the counts from all sources
        """
        sample_color = 'red'
        mean_color = 'black'
        group_color = 'blue'
        mc_label = 'Sample, This Work'

        counts = self.post_data[self.names['countsMC']]
        countrate = CountRate(self.input_path)
        times = countrate.decay_times
        alpha_MC: float = 1 / np.sqrt(self.MC_samples)
        for MC_iterm, count_val in enumerate(counts):
            label = mc_label if MC_iterm == 0 else None
            plt.plot(
                times,
                count_val,
                alpha=alpha_MC,
                color=sample_color,
                label=label)
        count_data = CSVHandler(self.countrate_path).read_vector_csv()
        plt.errorbar(
            times,
            count_data['counts'],
            count_data['sigma counts'],
            color=mean_color,
            linestyle='',
            marker='x',
            label='Mean, This Work',
            markersize=5,
            markevery=5)
        countrate.method = 'groupfit'
        if self.self_relative_data:
            base_name = mc_label
            base_counts = np.asarray(count_data['counts'])
        group_counts = countrate.calculate_count_rate(write_data=False)
        plt.plot(
            times,
            group_counts['counts'],
            color=group_color,
            alpha=0.75,
            label='Group Fit, This Work',
            linestyle='--',
            zorder=3)
        plt.fill_between(
            times,
            group_counts['counts'] -
            group_counts['sigma counts'],
            group_counts['counts'] +
            group_counts['sigma counts'],
            color=group_color,
            alpha=0.3,
            zorder=2,
            edgecolor='black')
        literature_data = Literature(
            self.input_path).get_group_data(
            self.use_data)
        first: bool = True
        colors = self.get_colors(len(literature_data.keys()))

        for index, (name, lit_data) in enumerate(literature_data.items()):
            if name == 'endfb6':
                name = 'ENDF/B-VI'
            else:
                name = name.capitalize()
            countrate.group_params = lit_data
            data = countrate._count_rate_from_groups()
            plt.plot(times, data['counts'], label=f'{name} 6-Group Fit',
                     color=colors[index],
                     linestyle=self.linestyles[index%len(self.linestyles)])
            plt.fill_between(
                times,
                data['counts'] - data['sigma counts'],
                data['counts'] + data['sigma counts'],
                alpha=0.3,
                zorder=2,
                edgecolor='black',
                color=colors[index])
            if first and not self.self_relative_data:
                base_name = name
                base_counts = data['counts']
                first = False

        plt.xlabel('Time [s]')
        plt.ylabel(r'Count Rate $[n \cdot s^{-1}]$')
        plt.yscale('log')
        leg = plt.legend()
        for line in leg.legend_handles:
            if line.get_label() == mc_label:
                line.set_alpha(0.5)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}MC_counts.png')
        plt.close()

        for MC_iterm, count_val in enumerate(counts):
            label = mc_label if MC_iterm == 0 else None
            plt.plot(
                times,
                count_val /
                base_counts,
                alpha=alpha_MC,
                color=sample_color,
                label=label)
        plt.errorbar(
            times,
            count_data['counts'] /
            base_counts,
            count_data['sigma counts'],
            color=mean_color,
            linestyle='',
            marker='x',
            label='Mean, This Work',
            markersize=5,
            markevery=5)

        plt.plot(
            times,
            group_counts['counts'] /
            base_counts,
            color=group_color,
            alpha=0.75,
            label='Group Fit, This Work',
            linestyle='--',
            zorder=3)
        plt.fill_between(
            times,
            (group_counts['counts'] - group_counts['sigma counts']) /
            base_counts,
            (group_counts['counts'] + group_counts['sigma counts']) /
            base_counts,
            color=group_color,
            alpha=0.3,
            zorder=2,
            edgecolor='black')
        for index, (name, lit_data) in enumerate(literature_data.items()):
            if name == 'endfb6':
                name = 'ENDF/B-VI'
            elif name == 'Modified 0D Scaled':
                pass
            else:
                name = name.capitalize()
            countrate.group_params = lit_data
            data = countrate._count_rate_from_groups()
            plt.plot(
                times,
                data['counts'] /
                base_counts,
                label=f'{name} 6-Group Fit',
                color=colors[index],
                linestyle=self.linestyles[index%len(self.linestyles)])
            plt.fill_between(
                times,
                (data['counts'] - data['sigma counts']) / base_counts,
                (data['counts'] + data['sigma counts']) / base_counts,
                alpha=0.3,
                zorder=2,
                edgecolor='black',
                color=colors[index])
        plt.xlabel('Time [s]')
        plt.ylabel(fr'{base_name} Normalized Count Rate')
        leg = plt.legend()
        for line in leg.legend_handles:
            if line.get_label() == mc_label:
                line.set_alpha(0.5)
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}{base_name}_counts.png')
        plt.close()

        return None

    def _get_group_params(self) -> tuple[float, float]:
        """
        Get the group parameters from the postprocessing data

        returns
        -------
        net_yield, avg_half_life : tuple[float, float]
            net yield and average half-life of the group.

        """
        group_data = CSVHandler(
            self.group_path,
            create=False).read_vector_csv()
        yields = [
            ufloat(
                y, std) for y, std in zip(
                group_data['yield'], group_data['sigma yield'])]
        halflives = [
            ufloat(
                hl, std) for hl, std in zip(
                group_data['half_life'], group_data['sigma half_life'])]
        net_yield = sum(yields)
        if net_yield.n <= 0.0:
            net_yield = ufloat(1e-12, 1e-12)
        lam_vals = np.log(2) / halflives
        avg_halflife = sum(yields * np.asarray(halflives) / (net_yield))
        return net_yield, avg_halflife

    def _get_data(self) -> dict[str: dict]:
        """
        Collect the data from the processed data files and add them to a
            dictionary

        Returns
        -------
        data_dict : dict[str: dict]
            Dictionary of processed data name to its data
        """
        data_dict = dict()
        emission_prob_data = CSVHandler(
            os.path.join(
                self.processed_data_dir,
                'emission_probability.csv'),
            create=False).read_csv()
        data_dict['emission_probability'] = emission_prob_data
        halflife_data = CSVHandler(
            os.path.join(
                self.processed_data_dir,
                'half_life.csv'),
            create=False).read_csv()
        data_dict['half_life'] = halflife_data
        concentration_data = CSVHandler(
            self.concentration_path,
            create=False).read_csv()
        data_dict['concentration'] = concentration_data

        emission_nucs = list(emission_prob_data.keys())
        conc_nucs = list(concentration_data.keys())
        net_nucs = list(set(emission_nucs) & set(conc_nucs))
        data_dict['net_nucs'] = net_nucs
        data_dict['nucs'] = {}

        for nuc in net_nucs:
            data_dict['nucs'][nuc] = {}
            emission_data = emission_prob_data[nuc]
            Pn = ufloat(emission_data['emission probability'],
                        emission_data['sigma emission probability'])
            conc_data = concentration_data[nuc]
            N = ufloat(conc_data['Concentration'],
                       conc_data['sigma Concentration'])
            hl_data = halflife_data[nuc]
            uncert = hl_data.get('sigma half_life', 1e-12)
            hl = ufloat(hl_data['half_life'], uncert)
            data_dict['nucs'][nuc]['emission_probability'] = Pn
            data_dict['nucs'][nuc]['concentration'] = N
            data_dict['nucs'][nuc]['half_life'] = hl
        return data_dict

    def _get_sorted_dnp_data(self) -> tuple[dict, dict, dict]:
        """
        Get the sorted delayed neutron precursor data by yield

        Returns
        -------
        tuple[dict, dict, dict]
            Sorted dictionaries of yields, concentrations, and halflife times
        """
        nuc_yield: dict[str, float] = dict()
        data_dict = self._get_data()
        halflife_times_yield: dict = dict()
        net_nucs = data_dict['net_nucs']

        self.total_delayed_neutrons: float = 0.0
        nuc_concs: dict[str, float] = dict()

        for nuc in net_nucs:
            Pn = data_dict['nucs'][nuc]['emission_probability']
            N = data_dict['nucs'][nuc]['concentration']
            hl = data_dict['nucs'][nuc]['half_life']
            lam_val = np.log(2) / hl
            nuc_yield[nuc] = Pn * N * lam_val / self.fission_term
            self.total_delayed_neutrons += (Pn * N).n
            halflife_times_yield[nuc] = nuc_yield[nuc] * np.log(2) / lam_val
            nuc_concs[nuc] = N

        sorted_yields = dict(
            sorted(
                nuc_yield.items(),
                key=lambda item: item[1].n,
                reverse=True))
        sorted_concs = dict(
            sorted(
                nuc_concs.items(),
                key=lambda item: item[1].n,
                reverse=True))
        return sorted_yields, sorted_concs, halflife_times_yield

    def _get_summed_params(self, num_top: int = 10) -> tuple[float, float]:
        """
        Get the summed parameters from the postprocessing data

        Parameters
        ----------
        num_top : int, optional
            Number of top contributors to consider, by default 10

        returns
        -------
        net_yield, avg_half_life : tuple[float, float]
            net yield and average half-life of the group.
        """
        data_dict = self._get_data()
        sorted_yields, sorted_concs, halflife_times_yield = self._get_sorted_dnp_data()
        net_yield = np.sum([i for i in sorted_yields.values()])
        net_N = np.sum([i for i in sorted_concs.values()])
        if net_yield.n <= 0.0:
            net_yield = ufloat(1e-12, 1e-12)
        if net_N.n <= 0.0:
            net_N = ufloat(1e-12, 1e-12)
        # Parish 1999 uses relative alpha_i values, not yields
        avg_halflife = np.sum(
            [i / net_yield for i in halflife_times_yield.values()])
        extracted_vals = dict()
        running_sum = 0
        sizes = list()
        labels = list()
        counter = 0
        self.logger.info(
            f'Writing nuclide emission times concentration (net yield)')
        for nuc, yield_val in sorted_yields.items():
            self.logger.info(
                f'{nuc} - {round(yield_val.n, 5)} +/- {round(yield_val.s, 5)}')
            sizes.append(yield_val.n)
            labels.append(self._convert_nuc_to_latex(nuc))
            running_sum += yield_val
            counter += 1
            extracted_vals[nuc] = yield_val
            if counter > num_top:
                break
        self.logger.info(
            f'Finished nuclide emission times concentration (net yield)')
        remainder = net_yield.n - running_sum.n
        sizes.append(remainder)
        labels.append('Other')
        colors = self.get_colors(num_top + 2)
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%',
               pctdistance=0.7, labeldistance=1.1,
               colors=colors)
        ax.axis('equal')
        plt.tight_layout()
        fig.savefig(f'{self.output_dir}dnp_yield.png')
        plt.close()

        sizes = list()
        labels = list()
        counter = 0
        running_sum = 0
        for nuc, conc_val in sorted_concs.items():
            N = data_dict['nucs'][nuc]['concentration']
            sizes.append(conc_val.n)
            labels.append(self._convert_nuc_to_latex(nuc))
            running_sum += conc_val
            counter += 1
            if counter > num_top:
                break
        remainder = net_N.n - running_sum.n
        sizes.append(remainder)
        labels.append('Other')
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%',
               pctdistance=0.7, labeldistance=1.1,
               colors=colors)
        ax.axis('equal')
        plt.tight_layout()
        fig.savefig(f'{self.output_dir}dnp_conc.png')
        plt.close()

        labels = [
            self._convert_nuc_to_latex(
                i.capitalize()) for i in self.fissiles.keys()]
        sizes = list(self.fissiles.values())
        remainder = 1 - sum(sizes)
        if remainder > 0.0:
            labels.append('Other')
            sizes.append(remainder)
        colors = self.get_colors(len(labels))
        fig, ax = plt.subplots()
        wedges, _, _ = ax.pie(sizes, autopct='%1.1f%%',
                              pctdistance=0.7, labeldistance=1.1,
                              colors=colors, textprops={'fontsize': 12})

        ax.legend(
            wedges,
            labels,
            title="Relative Fission Rates",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1)
        )

        ax.axis('equal')

        plt.tight_layout()
        fig.savefig(f'{self.output_dir}fission_fraction.png')
        plt.close()
        return net_yield, avg_halflife
