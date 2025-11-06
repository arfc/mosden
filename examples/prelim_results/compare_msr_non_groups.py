import numpy as np
import matplotlib.pyplot as plt
import time


def msr_group(T: float, t: float|np.ndarray[float], tin: float,
              tex: float, f_rate: float, yields: np.ndarray[float],
              halflives: np.ndarray[float]) -> float|np.ndarray[float]:
    """
    Calculates the delayed neutron count rate for an irradiated sample in a
    circulating-fuel reactor.

    Parameters
    ----------
    T : float
        Total length of time the sample is circulating
    t : float | np.ndarray[float]
        Post-irradiation time(s) over which the sample decays
    tin : float
        The in-core residence time
    tex : float
        The ex-core residence time
    f_rate : float
        The fission rate the sample experiences
    yields : np.ndarray[float]
        The yield/yields of each DNP group
    halflives : np.ndarray[float]
        The half-life/lives of each DNP group

    Returns
    -------
    full_sum : np.ndarray[float]|float
        The delayed neutron count rate evaluted at each `t` value
    """
    tnet = tin+tex
    try:
        J = int(np.floor(T/tnet))
        Jin = int(np.floor((T-tin)/tnet))
    except OverflowError:
        J = 0
        Jin = 0
    j_sum = 0
    full_sum = 0
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

def nonmsr_group(T: float, t: float|np.ndarray[float], f_rate: float,
                 yields: np.ndarray[float],
                 halflives: np.ndarray[float]) -> float|np.ndarray[float]:
    """
    Calculates the delayed neutron count rate for an irradiated sample in a
    stationary reactor.

    Parameters
    ----------
    T : float
        Total length of time the sample is circulating
    t : float | np.ndarray[float]
        Post-irradiation time(s) over which the sample decays
    f_rate : float
        The fission rate the sample experiences
    yields : np.ndarray[float]
        The yield/yields of each DNP group
    halflives : np.ndarray[float]
        The half-life/lives of each DNP group

    Returns
    -------
    full_sum : np.ndarray[float]|float
        The delayed neutron count rate evaluted at each `t` value
    """
    full_sum = 0
    for k in range(len(yields)):
        lam = np.log(2) / halflives[k]
        nu = yields[k]
        full_sum += nu * np.exp(-lam * t) * (1 - np.exp(-lam*T))
    full_sum = full_sum * f_rate
    return full_sum


def get_heatmap_data(T, t, tins, texs, f_rate, yields, halflives):
    """
    Calculates the delayed neutron count rates for an irradiated sample in both
    a circulating-fuel and a stationary reactor. Generate heatmap data of these
    mean differences for varying in-core and ex-core residence times.

    Parameters
    ----------
    T : float
        Total length of time the sample is circulating
    t : float | np.ndarray[float]
        Post-irradiation time(s) over which the sample decays
    tins : np.ndarray[float]
        The in-core residence times
    texs : np.ndarray[float]
        The ex-core residence times
    f_rate : float
        The fission rate the sample experiences
    yields : np.ndarray[float]
        The yield/yields of each DNP group
    halflives : np.ndarray[float]
        The half-life/lives of each DNP group

    Returns
    -------
    hm_data : np.ndarray[float, float]
        The delayed neutron count rate mean difference at each in and ex-
        residence time
    """
    hm_data = np.zeros((len(tins), len(texs)))
    for tini, tin in enumerate(tins):
        for texi, tex in enumerate(texs):
            msr_res = msr_group(T, t, tin, tex, f_rate, yields, halflives)
            stat_res = nonmsr_group(T, t, f_rate, yields, halflives)
            hm_data[tini, texi] = pcnt_diff(stat_res, msr_res)
    return hm_data


def pcnt_diff(a, b):
    """
    Find the average percent difference between the two arrays.

    Parameters
    ----------
    a : np.ndarray[float]
        Array of values
    b : np.ndarray[float]
        Array of values

    Returns
    -------
    val : float
        Percent difference of average between a and b
    """
    avg_sum = (np.mean(a) + np.mean(b)) / 2
    diff = np.mean(a) - np.mean(b)
    val = diff/avg_sum * 100
    return val

def heaviside_func(T, t, tin, tex):
    """
    Defines the Heaviside function

    Parameters
    ----------
    T : float
        Total length of time the sample is circulating
    t : float | np.ndarray[float]
        Post-irradiation time(s) over which the sample decays
    tin : float
        The in-core residence time
    tex : float
        The ex-core residence time

    Returns
    -------
    summation : float|np.ndarray[float]
         Value of Heaviside at each value of `t`
    """
    summation = 0
    tnet = tin+tex
    J = int(np.ceil(T/(tnet)))
    for j in range(0, J):
        sum_add = np.heaviside(t-j*tnet, 0) - np.heaviside(t-tin-j*tnet, 0)
        summation += sum_add
    return summation


if __name__ == '__main__':
    T = 1200
    t = np.arange(0, 300, 0.01)
    irrad_times = np.arange(0, T, 0.01)
    tin = 10
    tex = 0
    tnet = tin+tex
    halflives = [1]#[100]
    yields = [0.00693]
    f_rate = 1
    min_tin = 8
    min_tex = 8
    max_tin = 10
    max_tex = 10
    num_nodes = 400
    tins = np.linspace(min_tin, max_tin, num_nodes)
    texs = np.linspace(min_tex, max_tex, num_nodes)
    J = int(np.floor(T/tnet))
    Jin = int(np.floor((T-tin)/tnet))
    print(f'{J = }\n{Jin = }')

    plot_t = False
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