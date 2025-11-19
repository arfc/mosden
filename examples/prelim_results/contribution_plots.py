from mosden.postprocessing import PostProcess
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

yields = {
    'Ge86': 12.6,
    'I137': 10.4,
    'As86': 8.5,
    'Rb94': 7.6,
    'Br89': 6.6,
    'Br90': 6.4,
    'As85': 6.1,
    'Br88': 5.3,
    'Other': 36.6
}

counts = {
    'I137': 40.0,
    'Br87': 19.9,
    'Br88': 13.5,
    'Br89': 4.5,
    'Other': 22.1
}

concs = {
    'Br87': 24.8,
    'Cs141': 22.7,
    'I137': 16.5,
    'Br88': 6.3,
    'Te136': 5.2,
    'Other': 24.5
}


all_data = [yields, counts, concs]
all_nucs = [
    'Ge86',
    'I137',
    'As86',
    'Rb94',
    'Br89',
    'Br90',
    'As85',
    'Br88',
    'Br87',
    'Cs141',
    'Te136',
    'Other'
]
all_nucs.reverse()
postobj = PostProcess(None)
num_nucs = len(all_nucs)
colors = postobj.get_colors(num_nucs)
color_nucs = dict(zip(all_nucs, colors))
counter = 0
for dataset in all_data:
    sizes = list()
    labels = list()
    colors = list()
    text_colors = list()
    for k, v in dataset.items():
        sizes.append(v)
        colors.append(color_nucs[k])
        if k != 'Other':
            k = postobj._convert_nuc_to_latex(k)
        labels.append(k + ', ' + str(round(v)) + '\%')

    fig, ax = plt.subplots(subplot_kw=dict(aspect='equal'))
    _, _ = ax.pie(sizes, labels=labels, labeldistance=1.1,
            colors=colors)
    ax.axis('equal')

    plt.tight_layout()
    fig.savefig(f'{counter}dnp_yield.png')
    plt.close()
    counter += 1