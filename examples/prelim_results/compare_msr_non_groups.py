import numpy as np
import matplotlib.pyplot as plt
import time


def msr_group(T, t, tin, tex, f_rate, yields, halflives):
    tnet = tin+tex
    J = int(np.floor(T/tnet))
    Jin = int(np.floor((T-tin)/tnet))
    full_sum = 0
    j_sum = 0
    for k in range(len(yields)):
        lam = np.log(2) / halflives[k]
        nu = yields[k]
        for j in range(0, Jin+1):
            j_sum += np.exp(-lam*(t+T-j*tnet-tin)) - np.exp(-lam*(t+T-j*tnet))
        for j in range(Jin+1, J+1):
            j_sum += np.exp(-lam*t) - np.exp(-lam*(t+T-j*tnet))
        non_j_sum = nu
        full_sum += non_j_sum * j_sum
    full_sum = full_sum * f_rate
    return full_sum

def nonmsr_group(T, t, f_rate, yields, halflives):
    full_sum = 0
    for k in range(len(yields)):
        lam = np.log(2) / halflives[k]
        nu = yields[k]
        full_sum += nu * np.exp(-lam * t) * (1 - np.exp(-lam*T))
    full_sum = full_sum * f_rate
    return full_sum


def get_heatmap_data(T, t, tins, texs, f_rate, yields, halflives):
    hm_data = np.zeros((len(tins), len(texs)))
    for tini, tin in enumerate(tins):
        for texi, tex in enumerate(texs):
            msr_res = msr_group(T, t, tin, tex, f_rate, yields, halflives)
            stat_res = nonmsr_group(T, t, f_rate, yields, halflives)
            hm_data[tini, texi] = pcnt_diff(stat_res, msr_res)
    return hm_data


def pcnt_diff(a, b):
    avg_sum = (np.mean(a) + np.mean(b)) / 2
    diff = np.mean(a) - np.mean(b)
    val = diff/avg_sum * 100
    return val

def heaviside_func(T, t, tin, tex):
    summation = 0
    tnet = tin+tex
    J = int(np.ceil(T/(tnet)))
    for j in range(0, J):
        sum_add = np.heaviside(t-j*tnet, 0) - np.heaviside(t-tin-j*tnet, 0)
        summation += sum_add
    return summation

T = 1200
t = np.arange(0, 300, 0.01)
irrad_times = np.arange(0, T, 0.01)
tin = 10
tex = 0
tnet = tin+tex
halflives = [1]#[100]
yields = [1]#[0.00693]
f_rate = 1
min_tin = 5
min_tex = 5
max_tin = 7
max_tex = 7
num_nodes = 200
tins = np.linspace(min_tin, max_tin, num_nodes)
texs = np.linspace(min_tex, max_tex, num_nodes)
J = int(np.floor(T/tnet))
Jin = int(np.floor((T-tin)/tnet))
print(f'{J = }\n{Jin = }')

plot_t = True
plot_hm = not plot_t



start = time.time()

if plot_hm:
    t = 0
    hm_data = get_heatmap_data(T, t, tins, texs, f_rate, yields, halflives)
    plt.imshow(hm_data, cmap='viridis', origin='lower', aspect='auto', extent=[min_tex, max_tex, min_tin, max_tin])
    plt.colorbar(label='Difference [%]')
    plt.ylabel(r'$\tau_{in}$ $[s]$')
    plt.xlabel(r'$\tau_{ex}$ $[s]$')
    plt.savefig('diff.png')
    plt.close()

if plot_t:
    msr_res = msr_group(T, t, tin, tex, f_rate, yields, halflives)
    stat_res = nonmsr_group(T, t, f_rate, yields, halflives)
    diff = pcnt_diff(stat_res, msr_res)
    heaviside = heaviside_func(T, irrad_times, tin, tex)

    print(f'{round(diff,3)}%')

    print(f'{msr_res = }')
    print(f'{stat_res = }')
    plt.plot(t, msr_res, label='MSR', linestyle='--')
    plt.plot(t, stat_res, label='Traditional', marker='.', linestyle='', markersize=5, markevery=0.1)
    plt.xlabel('Time [s]')
    plt.ylabel('Relative Delayed Neutron Count Rate')
    plt.legend()
    plt.savefig('countrates.png')
    plt.close()

    plt.plot(irrad_times, heaviside, label='Fission History')
    plt.xlabel('Time [s]')
    plt.ylabel('Relative Fission History')
    plt.savefig('irrad.png')
    plt.close()


print(f'Took {round(time.time() - start)}s')