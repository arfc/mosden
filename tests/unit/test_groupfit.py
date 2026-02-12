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

def get_yield_halflives(which='default'):
    if which == 'default':
        half_lives = [60, 50, 40, 30, 20, 10]
        yields = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    elif which == 'standard':
        half_lives = [55.4, 22.8, 7.2, 2.3, 0.58, 0.12]
        yields = [0.00058, 0.00308, 0.00237, 0.00722, 0.00386, 0.00157]
    return half_lives, yields

def run_grouper_fit_test(irrad_type: str, grouper: Grouper,
                         data_type: str='default'):
    expected_residual = 0.0
    half_lives, yields = get_yield_halflives(data_type)
    lams = np.log(2)/half_lives
    times = np.linspace(0, 600, 100)
    counts = np.zeros(len(times))
    fission_times = np.linspace(0, grouper.t_net, 10000)
    dt = np.diff(fission_times)[0]
    concs = np.zeros(6)
    if irrad_type == 'pulse':
        concs = np.asarray(yields)
    else:
        period = grouper.t_in + grouper.t_ex
        fiss_rate = np.asarray(((fission_times % period) < grouper.t_in).astype(int))
        for ti in range(len(fission_times)-1):
            concs = (concs + np.asarray(yields)*dt*fiss_rate[ti+1])/(1+lams*dt)
    # One group per row, one time per column
    counts_groups = concs[:, None] * np.exp(-lams[:, None] * times[None, :]) * lams[:, None]
    counts = np.sum(counts_groups, axis=0)

    check_scaling_term = True
    if '_ex' in irrad_type:
        irrad_type = irrad_type.replace('_ex', '')
        check_scaling_term = False

    if irrad_type == 'pulse':
        check_scaling_term = False
        fit_func = grouper._pulse_fit_function
    elif irrad_type == 'saturation':
        fit_func = grouper._saturation_fit_function
    elif irrad_type == 'intermediate':
        fit_func = grouper._intermediate_numerical_fit_function

    if irrad_type != 'pulse':
        grouper.fission_times = fission_times
        grouper.full_fission_term = fiss_rate[:-1]
        grouper.refined_fission_term = np.mean(fiss_rate)

    if check_scaling_term:
        sat_term = grouper.refined_fission_term * (1 - np.exp(-lams * grouper.t_net))
        inter_term = grouper._get_effective_fission(lams, np.exp, np.expm1)
        assert np.all(np.isclose(sat_term, inter_term)), "Terms do not agree"

    grouper.irrad_type = irrad_type
    grouper.num_groups = 6
    grouper._set_refined_fission_term(times)
    if irrad_type != 'pulse':
        grouper.fission_times = fission_times
        grouper.full_fission_term = fiss_rate[:-1]
        grouper.refined_fission_term = np.mean(fiss_rate)
        assert grouper.full_fission_term is not None, "Full fission term is none"
        assert grouper.fission_times is not None, "Fission times are none"
    parameters = yields + half_lives
    func_counts = fit_func(times, parameters)
    assert np.isclose(func_counts, counts, atol=1e-2, rtol=1e-2).all(), f'{irrad_type.capitalize()} counts mismatch between hand calculation and function evaluation'

    count_data = {
        'times': times,
        'counts': counts,
        'sigma counts': np.zeros(len(counts))
    }

    data = grouper._nonlinear_least_squares(count_data=count_data, set_refined_fiss=False)
    test_yields = sorted([data[key]['yield'] for key in data], reverse=True)
    test_half_lives = sorted([data[key]['half_life'] for key in data], reverse=True)
    parameters = test_yields + test_half_lives

    for group in range(grouper.num_groups):
        assert np.isclose(test_yields[group], yields[group], rtol=1e-1, atol=1e-1), \
            f'Group {group+1} yields mismatch - {test_yields[group]=} != {yields[group]=}'
        assert np.isclose(test_half_lives[group], half_lives[group], rtol=1e-1, atol=1e-1), \
            f'Group {group+1} half lives mismatch - {test_half_lives[group]=} != {half_lives[group]=}'
        
    residual = np.linalg.norm(grouper._residual_function(parameters, times, counts, None, fit_func))
    grouper.logger.error(f'{parameters = }')
    grouper.logger.error(f'{counts = }')
    grouper.logger.error(f'{fit_func(times, parameters) = }')
    grouper.logger.error(f'{residual = }')
    assert np.isclose(residual, expected_residual), "Residual did not match expected value"
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
def test_grouper_saturation_noex_fitting_standard_params():
    input_path = './tests/unit/input/input.json'
    grouper = Grouper(input_path)
    grouper.t_ex = 0
    run_grouper_fit_test('saturation', grouper, 'standard')

@pytest.mark.slow
def test_grouper_intermediate_noex_fitting():
    input_path = './tests/unit/input/input.json'
    grouper = Grouper(input_path)
    grouper.t_ex = 0
    run_grouper_fit_test('intermediate', grouper)

@pytest.mark.slow
def test_grouper_intermediate_noex_fitting_standard_params():
    input_path = './tests/unit/input/input.json'
    grouper = Grouper(input_path)
    grouper.t_ex = 0
    run_grouper_fit_test('intermediate', grouper, 'standard')


@pytest.mark.slow
def test_grouper_saturation_ex_fitting():
    input_path = './tests/unit/input/input.json'
    grouper = Grouper(input_path)
    grouper.t_ex = 10
    run_grouper_fit_test('saturation_ex', grouper)

def test_grouper_saturation_ex_short_fitting():
    input_path = './tests/unit/input/input.json'
    grouper = Grouper(input_path)
    grouper.t_ex = 10
    grouper.t_net = 30
    run_grouper_fit_test('saturation_ex', grouper)

def test_grouper_intermediate_ex_short_fitting():
    input_path = './tests/unit/input/input.json'
    grouper = Grouper(input_path)
    grouper.t_ex = 10
    grouper.t_net = 30
    run_grouper_fit_test('intermediate_ex', grouper)

@pytest.mark.slow
def test_grouper_intermediate_ex_fitting():
    input_path = './tests/unit/input/input.json'
    grouper = Grouper(input_path)
    grouper.t_ex = 10
    run_grouper_fit_test('intermediate_ex', grouper)