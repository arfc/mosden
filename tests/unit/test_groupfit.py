from mosden.groupfit import Grouper
import numpy as np
import pytest
from math import ceil

def test_grouper_init():
    """
    Test the initialization of the grouper class
    """
    input_path = './tests/unit/input/input.json'
    countrate = Grouper(input_path)
    assert countrate.input_path == input_path, f"Expected input path {input_path}, but got {countrate.input_path}"
    assert countrate.output_dir == './tests/unit/output', f"Expected output directory './tests/unit/output', but got {countrate.output_dir}"
    assert countrate.energy_MeV == 1.0, f"Expected energy 1.0, but got {countrate.energy_MeV}"
    assert countrate.fissiles == {'U235': 0.8, 'U238': 0.2}, f"Expected fissile targets {{'U235': 0.8, 'U238': 0.2}}, but got {countrate.fissiles}"

    return

def run_grouper_fit_test(irrad_type: str, grouper: Grouper):
    half_lives = [60, 50, 40, 30, 20, 10]
    yields = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    times = np.linspace(0, 600, 100)
    counts = np.zeros(len(times))
    t_sum: float = grouper.t_in + grouper.t_ex
    for a, hl in zip(yields, half_lives):
        lam = np.log(2) / hl
        if irrad_type == 'pulse':
            counts += a * lam * np.exp(-lam * times)
        elif irrad_type == 'saturation' or irrad_type == 'intermediate':
            counts += a * np.exp(-lam * times)
        elif irrad_type == 'saturation_ex' or irrad_type == 'intermediate_ex':
            recircs: int = int(np.floor(grouper.t_net/t_sum))
            irrad_circs: int = int(np.floor((grouper.t_net-grouper.t_in)/t_sum))
            group_counts = 0
            for j in range(0, irrad_circs+1):
                group_counts += np.exp(-lam*(times+grouper.t_net-j*t_sum-grouper.t_in)) - np.exp(-lam*(times+grouper.t_net-j*t_sum))
            for j in range(irrad_circs+1, recircs+1):
                group_counts += np.exp(-lam*times) - np.exp(-lam*(times+grouper.t_net-j*t_sum))
            counts += a * group_counts
        else:
            raise ValueError(f'Unknown irrad_type: {irrad_type}')

    if '_ex' in irrad_type:
        irrad_type = irrad_type.replace('_ex', '')

    if irrad_type == 'pulse':
        fit_func = grouper._pulse_fit_function
    elif irrad_type == 'saturation':
        fit_func = grouper._saturation_fit_function
    elif irrad_type == 'intermediate':
        fit_func = grouper._intermediate_numerical_fit_function

    grouper.irrad_type = irrad_type
    grouper.num_groups = 6
    grouper._set_refined_fission_term(times)
    if irrad_type == 'intermediate':
        grouper.fission_times = np.arange(0, 1201, 1)
        if grouper.t_ex == 0:
            grouper.full_fission_term = [1] * 1200
        elif grouper.t_ex == 10:
            grouper.full_fission_term = [1, 0] * 600
    parameters = yields + half_lives
    func_counts = fit_func(times, parameters)
    assert np.isclose(func_counts, counts).all(), f'{irrad_type.capitalize()} counts mismatch between hand calculation and function evaluation'

    count_data = {
        'times': times,
        'counts': counts,
        'sigma counts': np.zeros(len(counts))
    }

    data = grouper._nonlinear_least_squares(count_data=count_data, set_refined_fiss=False)
    test_yields = sorted([data[key]['yield'] for key in data], reverse=True)
    test_half_lives = sorted([data[key]['half_life'] for key in data], reverse=True)

    for group in range(grouper.num_groups):
        assert np.isclose(test_yields[group], yields[group], rtol=1e-4, atol=1e-8), \
            f'Group {group+1} yields mismatch - {test_yields[group]=} != {yields[group]=}'
        assert np.isclose(test_half_lives[group], half_lives[group], rtol=1e-4, atol=1e-8), \
            f'Group {group+1} half lives mismatch - {test_half_lives[group]=} != {half_lives[group]=}'
    return None

def test_grouper_pulse_fitting():
    input_path = './tests/unit/input/input.json'
    grouper = Grouper(input_path)
    
    run_grouper_fit_test('pulse', grouper)


@pytest.mark.slow
def test_grouper_saturation_noex_fitting():
    input_path = './tests/unit/input/input.json'
    grouper = Grouper(input_path)
    grouper.t_ex = 0
    run_grouper_fit_test('saturation', grouper)

@pytest.mark.slow
def test_grouper_intermediate_noex_fitting():
    input_path = './tests/unit/input/input.json'
    grouper = Grouper(input_path)
    grouper.t_ex = 0
    run_grouper_fit_test('intermediate', grouper)


@pytest.mark.slow
def test_grouper_saturation_ex_fitting():
    input_path = './tests/unit/input/input.json'
    grouper = Grouper(input_path)
    grouper.t_ex = 10

    run_grouper_fit_test('saturation_ex', grouper)


@pytest.mark.slow
def test_grouper_intermediate_ex_fitting():
    input_path = './tests/unit/input/input.json'
    grouper = Grouper(input_path)
    grouper.t_ex = 10
    run_grouper_fit_test('intermediate_ex', grouper)