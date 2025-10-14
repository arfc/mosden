import numpy as np
import matplotlib.pyplot as plt
import time


def msr_group(T, t, tin, tex, f_rate, yields, halflives):
    J = int(np.ceil(T/(tin+tex)))
    full_sum = 0
    for k in range(len(yields)):
        lam = np.log(2) / halflives[k]
        nu = yields[k]
        j_sum = 0
        for j in range(1, J):
            j_sum += np.exp(-lam * (T - j*tin - (j-1) * tex))
        full_sum += nu * np.exp(-lam*t) * (1 - np.exp(-lam*T) + (1-np.exp(lam*tex)) * j_sum)
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


T = 100
t = np.arange(0, 300, 0.01)
tin = 20
tex = 20
halflives = [60]
min_tin = 1
min_tex = 1
max_tin = 20
max_tex = 20
num_nodes = 200
tins = np.linspace(min_tin, max_tin, num_nodes)
texs = np.linspace(min_tex, max_tex, num_nodes)

plot_t = False
plot_hm = True


yields = [1]
f_rate = 1

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

    print(f'{round(diff,3)}%')

    print(msr_res)
    print(stat_res)
    plt.plot(t, msr_res, label='MSR')
    plt.plot(t, stat_res, label='Traditional')
    plt.xlabel('Time [s]')
    plt.ylabel('Relative Delayed Neutron Count Rate')
    plt.legend()
    plt.savefig('countrates.png')
    plt.close()

print(f'Took {round(time.time() - start)}s')