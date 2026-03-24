import matplotlib.pyplot as plt
from collections import defaultdict
from mosden.utils.csv_handler import CSVHandler
import glob
import numpy as np
import os
from mosden.postprocessing import PostProcess
plt.style.use('mosden.plotting')

def _cleanup_data(data_vals):
    formatted_data = defaultdict(list)
    formatted_data['yields'] = defaultdict(list)
    formatted_data['hls'] = defaultdict(list)
    formatted_data["xs"] = []

    total_yields = []
    avg_halflives = []
    for t, params in data_vals.items():
        for key, vals in params.items():
            if key == 'yields':
                total_yield = sum(vals)
                total_yields.append(total_yield)
        avg_hl = 1/total_yield * np.sum(np.asarray(params['yields']) * np.asarray(params['hls']))
        avg_halflives.append(avg_hl)

    for t_net, params in data_vals.items():
        formatted_data['xs'].append(t_net)
        for name, data in params.items():
            for group, val in enumerate(data):
                formatted_data[name][group].append(val)
    return formatted_data, total_yields, avg_halflives


def plot_data(data_vals, namemod='', xlab=r'Irradiation Time $[s]$', actual_yield=None, actual_hl=None):
    if data_vals == {}:
        return None

    post = PostProcess(None)
    markers = post.markers
    formatted_data, total_yields, avg_halflives = _cleanup_data(data_vals)
    xscale = 'log'
    max_index = total_yields.index(max(total_yields))
    min_index = total_yields.index(min(total_yields))

    colors = post.get_colors(1)
    print(f'Max - min yield of {round(1e5*(total_yields[max_index] - total_yields[min_index]), 4)} pcm ({namemod})')
    plt.plot(formatted_data['xs'], total_yields, label=r"Group", color=colors[0])
    if actual_yield:
        plt.hlines(actual_yield, min(formatted_data['xs']), max(formatted_data['xs']), label='Actual',
                   linestyle='--', color='red')
        plt.legend()
    plt.xscale(xscale)
    plt.xlabel(xlab)
    plt.ylabel(r'$\nu_d$')
    plt.savefig(f'total_yield{namemod}.png')
    plt.close()


    plt.plot(formatted_data['xs'], avg_halflives, label="Group", color=colors[0])
    if actual_hl:
        plt.hlines(actual_hl, min(formatted_data['xs']), max(formatted_data['xs']), label=r'Actual',
                   linestyle='--', color='red')
        plt.legend()
    plt.xscale(xscale)
    plt.xlabel(xlab)
    plt.ylabel(r'$\bar{\tau}$')
    plt.savefig(f'average_hl{namemod}.png')
    plt.close()



    for name, data in formatted_data.items():
        if type(data) is list:
            continue
        colors = post.get_colors(len(data.values()))
        for group, params in data.items():
            plt.plot(formatted_data['xs'], params, label=f'Group {group+1}',
                    marker=markers[group % len(markers)], linestyle='--', markersize=5,
                    linewidth=1, color=colors[group])
        plt.legend()
        plt.xlabel(xlab)
        plt.xscale(xscale)
        if name == 'yields':
            ylab = 'Group Yield'
        elif name == 'hls':
            ylab = 'Group Half-life [s]'
            plt.yscale('log')
        plt.ylabel(ylab)
        plt.savefig(f'{name}{namemod}.png')
        plt.close()
    
    xs = formatted_data['xs']
    yields = formatted_data['yields']

    y_arrays = [yields[group] for group in sorted(yields.keys())]
    labels = [f'Group {group + 1}' for group in sorted(yields.keys())]
    colors = post.get_colors(len(y_arrays))

    plt.stackplot(xs, y_arrays, labels=labels, colors=colors)

    plt.xlabel(xlab)
    plt.xscale(xscale)
    plt.ylabel('Yield')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(f'stack_yields{namemod}.png')
    plt.close()

def plot_accumulated_data(accumulated_data, actual_yield=None, actual_hl=None):
    data = list()
    groups = list()
    time_vals = list()
    hl_data = list()
    for group, data_vals in accumulated_data.items():
        formatted_data, total_yields, avg_halflives = _cleanup_data(data_vals)
        times = formatted_data['xs']
        time_vals.append(times)
        groups.append(group)
        data.append(total_yields)
        hl_data.append(avg_halflives)
        
    
    
    post = PostProcess(None)
    colors = post.get_colors(len(data))
    markers = post.markers

    for group in range(len(data)):
        plt.plot(time_vals[group], data[group], label=f"{groups[group]} Groups",
                 marker=markers[group % len(markers)],
                 color=colors[group], linestyle='--',
                 linewidth=0.75,
                 markersize=5)
    if actual_yield:
        plt.hlines(actual_yield, min(formatted_data['xs']), max(formatted_data['xs']), label='Actual',
                   linestyle='--', color='red')
    plt.xlabel(r'Irradiation Time $[s]$')
    plt.ylabel(r'$\nu_d$')
    plt.xscale('log')
    plt.legend()
    plt.savefig(f'multiple_group_yields.png')
    plt.close()


    for group in range(len(hl_data)):
        plt.plot(time_vals[group], hl_data[group], label=f"{groups[group]} Groups",
                 marker=markers[group % len(markers)],
                 color=colors[group], linestyle='--',
                 linewidth=0.75,
                 markersize=5)
    if actual_hl:
        plt.hlines(actual_hl, min(formatted_data['xs']), max(formatted_data['xs']), label='Actual',
                   linestyle='--', color='red')
    plt.xlabel(r'Irradiation Time $[s]$')
    plt.ylabel(r'$\bar{\tau}$ $[s]$')
    plt.xscale('log')
    plt.legend()
    plt.savefig(f'multiple_group_hls.png')
    plt.close()


    if actual_yield:
        for group in range(len(data)):
            diff = 1e5*np.abs(np.asarray(data[group]) - actual_yield)
            plt.plot(time_vals[group], diff, label=f"{groups[group]} Groups",
                    marker=markers[group % len(markers)],
                    color=colors[group], linestyle='--',
                    linewidth=0.75,
                    markersize=5)
        plt.xlabel(r'Irradiation Time $[s]$')
        #plt.yscale('log')
        plt.xscale('log')
        plt.ylabel(r'$|\Delta \nu_d |$ $[pcm]$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'total_yield_diff_actual.png')
        plt.close()

    if actual_hl:
        for group in range(len(hl_data)):
            diff = np.abs(np.asarray(hl_data[group]) - actual_hl)
            plt.plot(time_vals[group], diff, label=f"{groups[group]} Groups",
                    marker=markers[group % len(markers)],
                    color=colors[group], linestyle='--',
                    linewidth=0.75,
                    markersize=5)
        plt.xlabel(r'Irradiation Time $[s]$')
        #plt.yscale('log')
        plt.xscale('log')
        plt.ylabel(r'$|\Delta \bar{\tau} |$ $[s]$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'avg_hl_diff_actual.png')
        plt.close()


    data = np.asarray(data).T
    hl_data = np.asarray(hl_data).T
    colors = post.get_colors(len(data))

    for i in range(len(data)):
        plt.plot(groups, data[i], label=f"T = {times[i]}",
                 marker=markers[i % len(markers)],
                 color=colors[i], linestyle='--',
                 linewidth=0.75,
                 markersize=5)
    if actual_yield:
        plt.hlines(actual_yield, min(groups), max(groups), label='Actual',
                   linestyle='--', color='red')
    plt.xlabel('Number of Groups')
    plt.ylabel(r'$\nu_d$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'total_yield_groups.png')
    plt.close()


    for i in range(len(hl_data)):
        plt.plot(groups, hl_data[i], label=f"T = {times[i]}",
                 marker=markers[i % len(markers)],
                 color=colors[i], linestyle='--',
                 linewidth=0.75,
                 markersize=5)
    if actual_hl:
        plt.hlines(actual_hl, min(groups), max(groups), label='Actual',
                   linestyle='--', color='red')
    plt.xlabel('Number of Groups')
    plt.ylabel(r'$\bar{\tau}$ $[s]$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'avg_hl_groups.png')
    plt.close()


    diffs = list()
    for i in range(len(data[0])):
        diff = max(data[:, i]) - min(data[:, i])
        diffs.append(1e5 * diff)
    
    plt.plot(groups, diffs, color='black')
    plt.xlabel('Number of Groups')
    plt.yscale('log')
    plt.ylabel(r'$\Delta \nu_d $ $[pcm]$')
    plt.tight_layout()
    plt.savefig(f'total_yield_diff.png')
    plt.close()


    diffs = list()
    for i in range(len(hl_data[0])):
        diff = max(hl_data[:, i]) - min(hl_data[:, i])
        diffs.append(diff)
    
    plt.plot(groups, diffs, color='black')
    plt.xlabel('Number of Groups')
    plt.yscale('log')
    plt.ylabel(r'$\Delta \bar{\tau} $ $[s]$')
    plt.tight_layout()
    plt.savefig(f'avg_hl_diff.png')
    plt.close()






    return None



def build_data_dict(data_path=r'./dataNet/', post_name='_post', all_name='_all'):
    def helper(pathmod):
        files = glob.glob(os.path.join(data_path, f"*{pathmod}.csv"))
        data = {}
        split_offset = 1
        if '_' in data_path:
            split_offset += 1
        for file in files:
            file: str = file
            time = float(file.split('_')[split_offset])
            data[time] = dict()
            file_data = CSVHandler(file).read_vector_csv()
            data[time]['yields'] = file_data['yield']
            data[time]['hls'] = file_data['half_life']
        data = dict(sorted(data.items()))
        return data
    
    post_data = helper(post_name)
    all_data = helper(all_name)

    return post_data, all_data

if __name__ == '__main__':
    actual_yield = 0.01609639439

    groups = [4, 5, 6, 7, 8, 9, 12, 20]
    accumulated_data = dict()

    for group in groups:
        post_data, all_data = build_data_dict(f'./dataNet_{group}', post_name='_post-irrad')
        plot_data(post_data, f'_{group}_post-irrad', actual_yield=actual_yield)
        plot_data(all_data, f'_{group}_all', actual_yield=actual_yield)
        accumulated_data[group] = post_data
    
    plot_accumulated_data(accumulated_data, actual_yield)