import matplotlib.pyplot as plt
from collections import defaultdict
from mosden.utils.csv_handler import CSVHandler
import glob
import os
plt.style.use('mosden.plotting')

def plot_data(data_vals, namemod=''):
    formatted_data = defaultdict(list)
    formatted_data['yields'] = defaultdict(list)
    formatted_data['hls'] = defaultdict(list)
    formatted_data["xs"] = []
    xlab = 'Irradiation Time [s]'
    xscale = 'log'

    for t_net, params in data_vals.items():
        formatted_data['xs'].append(t_net)
        for name, data in params.items():
            for group, val in enumerate(data):
                formatted_data[name][group].append(val)

    markers = ['.', '*', '>', '<', 'v', '^']
    for name, data in formatted_data.items():
        if type(data) is list:
            continue
        for group, params in data.items():
            plt.plot(formatted_data['xs'], params, label=f'Group {group+1}',
                    marker=markers[group], linestyle='--', markersize=5,
                    linewidth=1)
        plt.legend()
        plt.xlabel(xlab)
        plt.xscale(xscale)
        if name == 'yields':
            ylab = 'Group Yield'
        elif name == 'hls':
            ylab = 'Group Half-life [s]'
        plt.ylabel(ylab)
        plt.savefig(f'{name}{namemod}.png')
        plt.close()
    
    xs = formatted_data['xs']
    yields = formatted_data['yields']

    y_arrays = [yields[group] for group in sorted(yields.keys())]
    labels = [f'Group {group + 1}' for group in sorted(yields.keys())]

    plt.stackplot(xs, y_arrays, labels=labels)

    plt.xlabel(xlab)
    plt.xscale(xscale)
    plt.ylabel('Yield')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(f'stack_yields{namemod}.png')
    plt.close()

def build_data_dict(data_path=r'./data/'):
    def helper(pathmod):
        files = glob.glob(os.path.join(data_path, f"*{pathmod}.csv"))
        data = {}
        for file in files:
            file: str = file
            time = float(file.split('_')[1])
            data[time] = dict()
            file_data = CSVHandler(file).read_vector_csv()
            data[time]['yields'] = file_data['yield']
            data[time]['hls'] = file_data['half_life']
        data = dict(sorted(data.items()))
        return data
    
    post_data = helper('_post')
    all_data = helper('_all')

    return post_data, all_data

if __name__ == '__main__':
    post_data, all_data = build_data_dict()
    plot_data(post_data, '_post')
    plot_data(all_data, '_all')