import json
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
plt.style.use('mosden.plotting')
from mosden.postprocessing import PostProcess


class PRKE:
    def __init__(self, input_path: str, output_path: str):
        self.post = PostProcess(input_path)
        self.input_path = input_path
        self.output_path = output_path
        self.data = self._parse_input()
        return
    
    def _parse_input(self):
        with open(self.input_path, 'r') as file:
            data = json.load(file)
        return data
    
    
    def _solve_handler(self, dt, rho, problem):
        cur_data = self.data[problem]
        total_yield = self.data['neutrons_per_fission']
        betas = np.asarray(cur_data['yields']) / total_yield
        beta_eff = np.sum(betas)
        self.betaeff = beta_eff
        lams = np.log(2) / cur_data['hls']
        self.betas = betas
        self.lams = lams
        p0 = self.data['p0']
        gen = self.data['gen_time']
        self.gen = gen

        times = np.arange(0, self.data['tf']+dt, dt)
        power_vals = list()
        prec_vals = list()
        num_groups = len(cur_data['yields'])
        self.num_groups = num_groups

        if self.data['equilibrium_dnps']:
            prec_initial_val = [betas[i]*p0/(lams[i]*gen) for i in range(num_groups)]
        else:
            prec_initial_val = [0] * num_groups

        for ti, t in enumerate(times):
            if ti == 0:
                power_vals.append(p0)
                prec_vals.append(prec_initial_val)
                continue
            prev_power = power_vals[ti-1]
            prev_conc = prec_vals[ti-1]

            prec_sum = 0
            for k in range(num_groups):
                prec_sum += lams[k] * prev_conc[k]
            
            if self.data['euler_mode'] == 'forward':
                new_power = (prev_power + dt * ((rho(t) - beta_eff) / gen * prev_power + prec_sum))
                new_precs = list()
                for k in range(num_groups):
                    cur_conc = (prev_conc[k] + dt * (betas[k]/gen * prev_power - lams[k] * prev_conc[k]))
                    new_precs.append(cur_conc)

            elif self.data['euler_mode'] == 'backward':
                new_power = (prev_power + dt * prec_sum) / (1 - dt*(rho(t+dt)-beta_eff)/gen)
                new_precs = list()
                for k in range(num_groups):
                    cur_conc = ((prev_conc[k] + betas[k]*dt/gen*prev_power) / (1 + dt*lams[k]))
                    new_precs.append(cur_conc)

            else:
                raise Exception('Not implemented')
            power_vals.append(new_power)
            prec_vals.append(new_precs)
            
        return times, power_vals, prec_vals
    
    def _conc_plot(self, time, conc_data):
        num_groups = len(conc_data[0])
        colors = self.post.get_colors(num_groups)
        for k in range(num_groups):
            conc_vals = [C[k] for C in conc_data]
            plt.plot(time, conc_vals, label=f'Group {k+1}', color=colors[k])
        plt.xlabel('Time [s]')
        plt.ylabel('Precursor Concentration')
        plt.legend()
        plt.tight_layout()
        plt.savefig('conc.png')
        plt.close()
        return

    
    def _power_reativity_plot(self, time, power, reactivity_data):
        colors = self.post.get_colors(2)

        fig, ax1 = plt.subplots()

        color = colors[0]
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel(r'$n(t)/n_0$', color=color)
        ax1.plot(time, power, color=color)
        ax1.set_yscale('log')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()

        color = colors[1]
        ax2.set_ylabel('Reactivity [\$]', color=color)
        time = np.append([0], time)
        reactivity_data = np.append([0], reactivity_data)
        ax2.plot(time, np.asarray(reactivity_data)/self.betaeff, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        fig.savefig('power_reactivity.png')
        plt.close()
        return
    
    def _get_reactivity(self, problem, reactivity_form: str):
        total_yield = self.data['neutrons_per_fission']
        betaeff = np.sum(self.data[problem]['yields']) /  total_yield
        if reactivity_form == 'step':
            rho = lambda t: 50e-5
            return rho
        elif reactivity_form == 'step_relative':
            rho = lambda t: self.data['step_relative_insertion'] * betaeff
            return rho
        elif reactivity_form == 'ramp':
            rho_max = self.data['rho_max_dollars'] * betaeff
            rho = lambda t: min(rho_max*t, rho_max)
            return rho
        elif reactivity_form == 'sine':
            rho_0 = self.data['rho_amplitude']
            omega = self.data['rho_frequency']
            rho = lambda t: rho_0 * np.sin(omega * t)
            return rho
        elif reactivity_form == 'sine_relative':
            rho_0 = self.data['rho_relative_amplitude'] * betaeff
            omega = self.data['rho_frequency']
            rho = lambda t: rho_0 * np.sin(omega * t)
            return rho
        else:
            raise Exception('Reactivity form provided not implemented')
        
    def _dt_plot(self, time_collections, power_collections, dt_list):
        for index_val, times in enumerate(time_collections):
            power = np.asarray(power_collections[index_val])
            dt = dt_list[index_val]
            label_val = rf'$\Delta t = {dt*1000:.1f} ms$'
            if index_val == 0:
                base_y = power
                base_x = times
            t_common = np.intersect1d(base_x, times)
            base_common = base_y[np.isin(base_x, t_common)]
            cur_common = power[np.isin(times, t_common)]
            pcnt_diff = ((np.asarray(base_common) - np.asarray(cur_common)) / 
                         np.asarray(cur_common) * 100)
            plt.plot(t_common, pcnt_diff, label=label_val)
        plt.xlabel('Time [s]')
        plt.ylabel('Difference [%]')
        plt.legend()
        plt.tight_layout()
        plt.savefig('dtcompare.png')
        plt.close()
        return
    

    def _write_output(self, write_path, times, powers, precs, rho, cur_data, dt,
                      dti):
        data_dict = copy.deepcopy(cur_data)
        data_dict['time_steps'] = dt
        header_df = pd.DataFrame(data_dict)
        
        num_precs = len(precs[0])
        data = dict()
        data['Times [s]'] = times
        data['Relative Power'] = powers
        for k in range(num_precs):
            data[f'Precursor Group {k} Concentration'] = [C[k] for C in precs]
        reactivity_data = list()
        for t in times:
            reactivity_data.append(rho(t))
        data['Reactivity'] = reactivity_data
        

        df = pd.DataFrame(data)
        with open(write_path, 'w', newline='') as f:
            header_df.to_csv(f, index=False)
            df.to_csv(f, index=False)
        
        if dti == 0:
            self._power_reativity_plot(times, powers, reactivity_data)
            self._conc_plot(times, precs)        
        return
    
    def compare_results(self, full_data):
        linestyles = [':', '-.', '--']
        colors = self.post.get_colors(len(full_data))
        for pi, (problem, data) in enumerate(full_data.items()):
            label: str = problem.title()
            times = data['times']
            power = data['power']
            plt.plot(times, power, label=label, linestyle=linestyles[pi%len(linestyles)], color=colors[pi])
            print(f'{label} {power[-1] = }')
        plt.legend()
        #plt.xscale('log')
        plt.xlabel('Time [s]')
        plt.ylabel(r'$n$')
        plt.savefig(f'compare_power.png')
        plt.close()


        for pi, (problem, data) in enumerate(full_data.items()):
            label: str = problem.title()
            times = data['times']
            power = np.asarray(data['power'])
            if pi == 0:
                base_label = label
                base_power = power
            power_diff = (base_power - power) / ((base_power + power) / 2) * 100
            plt.plot(times, power_diff, label=f'{label}', linestyle=linestyles[pi%len(linestyles)], color=colors[pi])
        plt.legend()
        plt.xlabel('Time [s]')
        plt.xscale('log')
        plt.ylabel(rf'$\Delta n$ from {base_label} [\%]')
        plt.savefig(f'compare_power_percent.png')
        plt.close()
        
        for group in range(self.num_groups):
            for pi, (problem, data) in enumerate(full_data.items()):
                label: str = problem.title()
                times = data['times']
                concs = data['concs']
                conc = np.asarray(concs)[:, group]

                plt.plot(times, conc, label=label, linestyle=linestyles[pi%len(linestyles)], color=colors[pi])
                print(f'{label} {conc[-1] = }')
            plt.legend()
            plt.xlabel('Time [s]')
            plt.ylabel('Atoms [\#]')
            plt.savefig(f'compare_conc_{group+1}.png')
            plt.close()

        for pi, (problem, data) in enumerate(full_data.items()):
            label: str = problem.title()
            times = data['times']
            power = np.asarray(data['power'])
            base_rho = data['reactivity']
            reactivity_data = list()
            for t in times:
                reactivity_data.append(base_rho(t))
            concs = data['concs']
            conc_contribution = 0
            for group in range(self.num_groups):
                conc = np.asarray(concs)[:, group]
                conc_contribution += self.gen/power * self.betas[group] * self.lams[group]
            reactivity = np.asarray(reactivity_data) - self.betaeff + np.asarray(conc_contribution)
            print(f'{reactivity_data[-1] = }')
            print(f'{conc_contribution[-1] = }')
            print(f'{self.betaeff = }')
            plt.plot(times, reactivity, label=f'{label}', linestyle=linestyles[pi%len(linestyles)], color=colors[pi])
        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel(f'Reactivity')
        plt.savefig(f'compare_reactivity.png')
        plt.close()
        

        return
    
    def solve(self):
        problems = self.data['selections']
        compare_data = dict()
        for problem in problems:
            compare_data[problem] = dict()
            cur_data = self.data[problem]
            reactivity_form = self.data['problem']
            rho = self._get_reactivity(problem, reactivity_form)
            dt_list = self.data['time_steps']
            time_collections = list()
            power_collections = list()
            precursor_collections = list()

            for dti, dt in enumerate(dt_list):
                times, powers, precs = self._solve_handler(dt, rho, problem)
                self._write_output(f'{self.output_path}_{problem}.csv',
                                times, powers, precs, rho, cur_data, dt, dti)
                time_collections.append(times)
                power_collections.append(powers)
                precursor_collections.append(precs)
            compare_data[problem]['times'] = times
            compare_data[problem]['power'] = powers
            compare_data[problem]['concs'] = precs
            compare_data[problem]['reactivity'] = rho
            
            self._dt_plot(time_collections, power_collections, dt_list)
        if len(problems) > 1:
            self.compare_results(compare_data)
        return
    



if __name__ == '__main__':
    input_data = './input.json'
    output_data = './output'
    solver = PRKE(input_data, output_data)
    solver.solve()