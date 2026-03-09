from mosden.postprocessing import PostProcess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mosden.countrate import CountRate
import os


class MultiPostProcess():
    def __init__(self, input_paths: list[str]) -> None:
        """
        This class creates figures and performs analysis on multiple PostProcess
        objects.

        Parameters
        ----------
        input_paths : list[str]
            A list of file paths to the input data files.
        """
        self.posts: list[PostProcess] = [PostProcess(p) for p in input_paths]
        self.output_dir = self.posts[0].output_dir
        self.fig_post_name = ''
        if len(self.posts) > 1:
            self.output_dir = f'./{self.posts[0].multi_id}/images/'
            self.fig_post_name = f'{self.posts[0].multi_id}'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.do_heatmap = False

        self.hm_x_vals: list = list()
        self.hm_y_vals: list = list()
        self._post_heatmap_setup()
        self._initialize_posts()
        self.hm_z_names: dict[str, str] = {
            'summed_yield': r'$\bar{\nu}_d$',
            'group_yield': r'$\bar{\nu}_d$',
            'summed_avg_halflife': r'$\bar{T} [s]$',
            'group_avg_halflife': r'$\bar{T} [s]$'
        }
        for i in range(1, self.posts[0].num_groups + 1):
            self.hm_z_names[f'group_{i}_yield'] = rf'$\bar{{\nu}}_{{d,{i}}}$'
            self.hm_z_names[f'group_{i}_halflife'] = rf'$T_{i} [s]$'
        self._set_post_names()
        return None

    def _is_name(self, name: str) -> bool:
        """
        Check if all PostProcess objects have the same multi_id.

        Parameters
        ----------
        name : str
            The multi_id to check against.

        Returns
        -------
        bool
            True if all PostProcess objects have the same multi_id, False otherwise.
        """
        return np.all([post.multi_id == name for post in self.posts])

    def _set_post_names(self):
        if self._is_name('tintex'):
            for post in self.posts:
                post.name = f'({post.t_in}, {post.t_ex})'
        elif self._is_name('chem_long'):
            self.posts[0].name = 'Full MSBR'
            self.posts[1].name = 'Partial MSBR'
        elif self._is_name('chem_bool'):
            self.posts[0].name = 'Full MSBR'
            self.posts[1].name = 'No removal'
        elif self._is_name('spacing_times'):
            for post in self.posts:
                post.name = post.decay_time_spacing.capitalize()
        elif self._is_name('decay_time_nodes'):
            for post in self.posts:
                post.name = f'{post.num_decay_times} nodes'
        elif self._is_name('total_decay_time'):
            for post in self.posts:
                post.name = f'T = {post.total_decay_time}s'
        elif self._is_name('detailed_decay'):
            for post in self.posts:
                post.name = f'T = {post.total_decay_time}s with {post.num_decay_times} nodes'
        return None

    def _post_heatmap_setup(self) -> None:
        """
        Setup heatmap parameters based on the multi_id of the PostProcess objects.
        """
        if np.all([post.multi_id == 'tintex' for post in self.posts]):
            self.heatmap_key: str = 'modeling_options'
            self.heatmap_x: str = 'incore_s'
            self.heatmap_y: str = 'excore_s'
            self.hm_x_name = r'$\tau_{in}$'
            self.hm_x_units = r'$[s]$'
            self.hm_y_name = r'$\tau_{ex}$'
            self.hm_y_units = r'$[s]$'
            self.do_heatmap = True
        return None

    def _initialize_posts(self) -> None:
        """
        Setup heatmap parameters based on the multi_id of the PostProcess objects.
        """
        for post in self.posts:
            post.run()
            try:
                modeling_options: dict = post.input_data.get(
                    self.heatmap_key, {})
                post.hm_x = modeling_options.get(self.heatmap_x, 0.0)
                post.hm_y = modeling_options.get(self.heatmap_y, 0.0)
                self.hm_x_vals.append(post.hm_x)
                self.hm_y_vals.append(post.hm_y)
            except AttributeError:
                self.do_heatmap = False
        return None

    def run(self):
        """
        Run the multi-post processing analysis and generate figures.
        """
        if len(self.posts) <= 1:
            return None
        if self.do_heatmap:
            self.heatmap_gen()
        self.group_param_histogram()
        self.group_fit_counts()
        return None

    def _collect_post_data(self) -> dict[str: list[float]]:
        """
        Collect data from each PostProcess object and return it as a dictionary.

        Returns
        -------
        dict[str: list[float]]
            A dictionary containing lists of data from each PostProcess object.
        """

        post_data = dict()
        summed_yield = list()
        summed_avg_halflife = list()
        group_yield = list()
        group_avg_halflife = list()
        group_yields = np.zeros((len(self.posts), self.posts[0].num_groups))
        group_halflives = np.zeros((len(self.posts), self.posts[0].num_groups))


        for posti, post in enumerate(self.posts):
            if post.num_groups != self.posts[0].num_groups:
                raise ValueError(f"Post {posti} has {post.num_groups} groups")

            summed_yield.append(post.summed_yield.n)
            summed_avg_halflife.append(post.summed_avg_halflife.n)
            group_yield.append(post.group_yield.n)
            group_avg_halflife.append(post.group_avg_halflife.n)

            group_yield_values = post.MC_yields
            group_halflife_values = post.MC_half_lives

            group_yields[posti, :] = group_yield_values.ravel()
            group_halflives[posti, :] = group_halflife_values.ravel()

        post_data['summed_yield'] = summed_yield
        post_data['summed_avg_halflife'] = summed_avg_halflife
        post_data['group_yield'] = group_yield
        post_data['group_avg_halflife'] = group_avg_halflife

        for i in range(self.posts[0].num_groups):
            post_data[f'group_{i+1}_yield'] = group_yields[:, i].tolist()
            post_data[f'group_{i+1}_halflife'] = group_halflives[:, i].tolist()

        return post_data

    def heatmap_gen(self) -> None:
        """
        Generate heatmaps and ratio plots for the collected data.
        """
        z_values: dict[str, list[float]] = self._collect_post_data()
        for z_id in self.hm_z_names.keys():
            x_name = self.hm_x_name + self.hm_x_units
            X = self.hm_x_vals
            y_name = self.hm_y_name + self.hm_y_units
            Y = self.hm_y_vals
            z_name = self.hm_z_names[z_id]
            Z = z_values[z_id]

            df = pd.DataFrame.from_dict(np.array([X, Y, Z]).T)
            df.columns = [x_name, y_name, z_name]
            df[z_name] = pd.to_numeric(df[z_name])
            pivotted = df.pivot(index=x_name, columns=y_name, values=z_name)
            color = sns.color_palette("viridis", as_cmap=True)
            ax = sns.heatmap(pivotted, cmap=color)
            ax.invert_yaxis()
            ax.collections[0].colorbar.set_label(z_name)
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}surf_{z_id}.png')
            plt.close()

            x_to_y_ratio = np.asarray(X) / np.asarray(Y)
            sorted_indices = np.argsort(x_to_y_ratio)
            x_to_y_ratio = np.array(x_to_y_ratio)[sorted_indices]
            sorted_z = np.array(Z)[sorted_indices]

            plt.plot(x_to_y_ratio, sorted_z, marker='.', markersize=5)
            plt.xlabel(f'{self.hm_x_name}/{self.hm_y_name}')
            plt.xscale('log')
            plt.ylabel(z_name)
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}ratio_{z_id}.png')
            plt.close()
        return None

    def group_param_histogram(self) -> None:
        """
        Generate histograms for group parameters (yield and half-life) from each PostProcess object.
        """
        data = dict()

        label_locations = np.arange(self.posts[0].num_groups)
        n_bars = len(self.posts)
        base_width = 0.8
        width = base_width / n_bars

        def offset(index): return np.linspace(-base_width / 2 + width / 2,
                                              base_width / 2 - width / 2,
                                              n_bars)[index]
        group_labels = [f'{x}' for x in range(1, self.posts[0].num_groups + 1)]
        for post in self.posts:
            data[post.name] = {}
            if post.group_data is not None:
                post_data = post.group_data
                yield_val = post_data['yield']
                halflife_val = post_data['half_life']
                sig_yield = post_data['sigma yield']
                sig_halflife = post_data['sigma half_life']
            else:
                yield_val, halflife_val = post._get_MC_group_params()
                yield_val = yield_val.flatten()
                halflife_val = halflife_val.flatten()
            data[post.name]['Yield'] = yield_val
            data[post.name]['Halflife [s]'] = halflife_val
            if post.group_data is not None:
                data[post.name]['Yield Uncertainty'] = sig_yield
                data[post.name]['Halflife Uncertainty'] = sig_halflife

        fig, ax = plt.subplots()
        colors = self.posts[0].get_colors(len(self.posts))
        for post_i, post in enumerate(self.posts):
            if post.group_data is not None:
                ax.bar(label_locations + offset(post_i), data[post.name]['Yield'], width, label=post.name, yerr=data[post.name]['Yield Uncertainty'],
                       color=colors[post_i])
            else:
                ax.bar(label_locations + offset(post_i), data[post.name]['Yield'], width, label=post.name,
                       color=colors[post_i])
        ax.set_ylabel(r'$\bar{\nu}_{d, k}$')
        ax.set_xticks(label_locations)
        ax.set_xlabel('Groups')
        ax.set_xticklabels(group_labels)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}yields_{self.fig_post_name}.png')
        plt.close()

        fig, ax = plt.subplots()
        for post_i, post in enumerate(self.posts):
            if post.group_data is not None:
                ax.bar(label_locations + offset(post_i), data[post.name]['Halflife [s]'], width, label=post.name, yerr=data[post.name]['Halflife Uncertainty'],
                       color=colors[post_i])
            else:
                ax.bar(label_locations + offset(post_i), data[post.name]['Halflife [s]'], width, label=post.name,
                       color=colors[post_i])
        ax.set_ylabel(r'$\tau_k$ $[s]$')
        ax.set_xticks(label_locations)
        ax.set_xlabel('Groups')
        ax.set_xticklabels(group_labels)
        plt.legend()
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}halflives_{self.fig_post_name}.png')
        plt.close()
        return None

    def group_fit_counts(self) -> None:
        """
        Generate count rate plots for each PostProcess object
        using group fitting.
        """
        colors = self.posts[0].get_colors(len(self.posts))
        for pi, post in enumerate(self.posts):
            times = post.decay_times
            countrate = CountRate(post.input_path)
            countrate.method = 'groupfit'
            group_counts = countrate.calculate_count_rate(write_data=False)
            plt.plot(
                times,
                group_counts['counts'],
                color=colors[pi],
                alpha=0.75,
                label=post.name,
                linestyle='--',
                zorder=3,
                marker=post.markers[pi % len(post.markers)],
                markersize=3,
                markevery=5)
            plt.fill_between(
                times,
                group_counts['counts'] -
                group_counts['sigma counts'],
                group_counts['counts'] +
                group_counts['sigma counts'],
                color=colors[pi],
                alpha=0.3,
                zorder=2,
                edgecolor='black')
        plt.xlabel('Time [s]')
        plt.ylabel(r'Count Rate $[n \cdot s^{-1}]$')
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}group_counts_{self.fig_post_name}.png')
        plt.close()
