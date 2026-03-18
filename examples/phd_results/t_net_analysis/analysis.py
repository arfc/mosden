import matplotlib.pyplot as plt
from collections import defaultdict
from mosden.utils.csv_handler import CSVHandler
import glob
import os
plt.style.use('mosden.plotting')

def plot_data(data_vals, namemod='', xlab=r'Irradiation Time $[s]$', actual_yield=None):
    if data_vals == {}:
        return None
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
    plt.plot(formatted_data['xs'], total_yields, label="Yields")
    if actual_yield:
        plt.hlines(actual_yield, min(formatted_data['xs']), max(formatted_data['xs']), label='Actual Yield',
                   linestyle='--', color='orange')
        plt.legend()
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
    actual_yield = None

    groups = [4, 6]

    for group in groups:
        post_data, all_data = build_data_dict(f'./dataNet_{group}', post_name='_post-irrad')
        plot_data(post_data, f'_{group}_post-irrad', actual_yield=actual_yield)
        plot_data(all_data, f'_{group}_all', actual_yield=actual_yield)

    post_data, all_data = build_data_dict()
    plot_data(post_data, '_post', actual_yield=actual_yield)
    plot_data(all_data, '_all', actual_yield=actual_yield)

    post_data, all_data = build_data_dict('./dataNumSteps5s/')
    xlab = r'Number of Irradiation Time Steps'
    plot_data(post_data, '_post_dt', xlab, actual_yield=actual_yield)
    plot_data(all_data, '_all_dt', xlab, actual_yield=actual_yield)