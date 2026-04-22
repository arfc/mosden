from mosden.utils.csv_handler import CSVHandler
import numpy as np


def counts_from_concs(conc_data: dict[str, dict[float, tuple[float, float]]],
                      pn_data,
                      hl_data) -> tuple[list[float], dict[str, list[float]]]:
    counts = dict()
    emission_nucs = list(pn_data.keys())
    half_life_nucs = list(hl_data.keys())
    conc_nucs = list(conc_data.keys())
    net_similar_nucs = list(
        set(emission_nucs) & set(half_life_nucs) & set(conc_nucs))
    for nuc in net_similar_nucs:
        data = conc_data[nuc]
        Pn = pn_data[nuc]['emission probability']
        hl = hl_data[nuc]['half_life']
        decay_const = np.log(2) / hl
        if np.allclose(list(data.values()), 0.0):
            continue
        counts[nuc] = list()
        for _, (conc, _) in data.items():
            count = Pn * decay_const * conc
            counts[nuc].append(count)

    times = list(conc_data[nuc].keys())
            
    return times, counts


def trim_counts(counts, index):
    counts_new = dict()
    for nuc, data in counts.items():
        target_data = data[index]
        counts_new[nuc] = target_data
    return counts_new

def calc_relative_diff(dict1, dict2):
    diff_dict = dict()
    for nuc in dict1.keys():
        rel_diff = (dict1[nuc] - dict2[nuc])/((dict1[nuc]+dict2[nuc])/2)
        diff_dict[nuc] = rel_diff
    return diff_dict



if __name__ == "__main__":
    pn_file = './emission_probability.csv'
    hl_file = './half_life.csv'
    long_irrad_file = './concentrations_(0.1,10).csv'
    short_irrad_file = './concentrations_(0.01,1).csv'
    final_irrad_index = 100
    top_num = 5

    pn_data = CSVHandler(pn_file, create=False).read_csv()
    hl_data = CSVHandler(hl_file, create=False).read_csv()
    short_conc_data = CSVHandler(short_irrad_file).read_csv_with_time()
    times, counts = counts_from_concs(short_conc_data, pn_data=pn_data, hl_data=hl_data)
    counts_short = trim_counts(counts, final_irrad_index)

    long_conc_data = CSVHandler(long_irrad_file).read_csv_with_time()
    times, counts = counts_from_concs(long_conc_data, pn_data=pn_data, hl_data=hl_data)
    counts_long = trim_counts(counts, final_irrad_index)

    diff_dict = calc_relative_diff(counts_short, counts_long)
    sorted_keys = list(sorted(diff_dict, key=diff_dict.get, reverse=True))
    
    for each_index in range(top_num):
        nuc = sorted_keys[each_index]
        diff_val = np.round(diff_dict[nuc] * 100, 1)
        hl = hl_data[nuc]['half_life']
        pn = pn_data[nuc]['emission probability']
        print(f'{nuc = }\n{diff_val = }%\n{hl = }\n{pn = }\n')


    print()
    sorted_keys = list(sorted(diff_dict, key=diff_dict.get, reverse=False))
    
    for each_index in range(top_num):
        nuc = sorted_keys[each_index]
        diff_val = np.round(diff_dict[nuc] * 100, 1)
        hl = hl_data[nuc]['half_life']
        pn = pn_data[nuc]['emission probability']
        print(f'{nuc = }\n{diff_val = }%\n{hl = }\n{pn = }\n')


