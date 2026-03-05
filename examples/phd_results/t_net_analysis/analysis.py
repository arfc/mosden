import matplotlib.pyplot as plt
from collections import defaultdict
from mosden.utils.csv_handler import CSVHandler
import glob
import os
plt.style.use('mosden.plotting')

def plot_data(data_vals, namemod='', xlab=r'Irradiation Time $[s]$'):
    formatted_data = defaultdict(list)
    formatted_data['yields'] = defaultdict(list)
    formatted_data['hls'] = defaultdict(list)
    formatted_data["xs"] = []
    xscale = 'log'

    total_yields = []
    for t, params in data_vals.items():
        for key, vals in params.items():
            if key == 'yields':
                total_yield = sum(vals)
                total_yields.append(total_yield)
    max_index = total_yields.index(max(total_yields))

    for t_net, params in data_vals.items():
        formatted_data['xs'].append(t_net)
        for name, data in params.items():
            for group, val in enumerate(data):
                formatted_data[name][group].append(val)

    markers = ['.', '*', '>', '<', 'v', '^']
    print(f'Maximum yield of {total_yields[max_index]} at {formatted_data["xs"][max_index]}s')
    plt.plot(formatted_data['xs'], total_yields)
    plt.xscale(xscale)
    plt.xlabel(xlab)
    plt.ylabel('Total Yield')
    plt.savefig(f'total_yield{namemod}.png')
    plt.close()



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
            plt.yscale('log')
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

def build_data_dict(data_path=r'./dataNet/'):
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

    post_data, all_data = build_data_dict('./dataDt5s/')
    xlab = r'Irradiation Time Step $[s]$'
    plot_data(post_data, '_post_dt', xlab)
    plot_data(all_data, '_all_dt', xlab)