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
    elif which == 'few_groups':
        half_lives = [60]
        yields = [0.6]

    return half_lives, yields

def run_grouper_fit_test(irrad_type: str, grouper: Grouper,
                         data_type: str='default'):
    half_lives, yields = get_yield_halflives(data_type)
    yields = np.asarray(yields)
    half_lives = np.asarray(half_lives)
    num_groups = len(half_lives)
    grouper.num_groups = num_groups
    lams = np.log(2)/half_lives
    times = np.linspace(0, 600, 100)
    counts = np.zeros(len(times))
    fission_times = np.linspace(0, grouper.t_net, 10000)
    dt = np.diff(fission_times)[0]
    concs = np.zeros(num_groups)

    if irrad_type == 'pulse':
        concs = yields
    else:
        period = grouper.t_in + grouper.t_ex
        fiss_rate = np.asarray(((fission_times % period) < grouper.t_in).astype(int))
        for ti in range(len(fission_times)-1):
            fiss = fiss_rate[ti]
            concs = (concs + np.asarray(yields)*dt*fiss)/(1+lams*dt)
    
    if irrad_type == 'saturation' or irrad_type == 'intermediate':
        alt_concs = np.asarray(yields) / np.asarray(lams)
        expected_concs = alt_concs * (1 - np.exp(-np.asarray(lams) * grouper.t_net))
        assert np.all(np.isclose(expected_concs, concs, rtol=1e-1)), "Expected concs don't match"

    # One group per row, one time per column
    counts_groups = concs[:, None] * np.exp(-lams[:, None] * times[None, :]) * lams[:, None]
    if num_groups > 1:
        counts = np.sum(counts_groups, axis=0)
    else:
        counts = counts_groups[0]

    check_scaling_term = True
    if '_ex' in irrad_type:
        irrad_type = irrad_type.replace('_ex', '')
        check_scaling_term = False

    grouper.irrad_type = irrad_type
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
        grouper.logger.error(f'{grouper.refined_fission_term = }')

    if irrad_type == 'saturation':
        initial_count_rate = 0
        grouper._set_refined_fission_term(fission_times)
        for group in range(num_groups):
            lam = lams[group]
            fiss_term = grouper._get_saturation_fission_term(lam, np.exp)
            initial_count_rate += fiss_term * yields[group]
        if num_groups == 1:
            desired_fiss_term = counts[0] / yields[0]
            assert np.isclose(desired_fiss_term, fiss_term, rtol=1e-3), "Fission term not within expected tolerance"
        assert np.isclose(initial_count_rate, counts[0], rtol=1e-1), "Saturation initial count rate mismatch"
        grouper.logger.error(f'{fiss_term = }')


    if check_scaling_term:
        grouper.fission_times = fission_times
        inter_terms = grouper._get_effective_fission(lams, np.exp, np.expm1)
        for group in range(num_groups):
            inter_term = inter_terms[group]
            lam = lams[group]
            sat_term = grouper._get_saturation_fission_term(lam, np.exp)
            grouper.logger.error(f'{sat_term = }')
            assert np.all(np.isclose(sat_term, inter_term)), f"Fission terms do not agree in group {group+1}"

    grouper._set_refined_fission_term(fission_times)
    if irrad_type != 'pulse':
        grouper.fission_times = fission_times
        grouper.full_fission_term = fiss_rate[:-1]
        #grouper.refined_fission_term = np.mean(fiss_rate)
        assert grouper.full_fission_term is not None, "Full fission term is none"
        assert grouper.fission_times is not None, "Fission times are none"
    base_parameters = np.concatenate((yields, half_lives))
    base_inter_parameters = grouper._restructure_intermediate_yields(base_parameters, False)
    grouper.logger.error(f'{base_parameters = }')
    if irrad_type == 'intermediate':
        assert np.isclose(counts[0], np.sum(base_inter_parameters[:num_groups]), rtol=1e-2), "Summed parameters don't match counts"
    func_counts = fit_func(times, base_inter_parameters)

    if irrad_type == 'saturation':
        assert np.isclose(func_counts[0], initial_count_rate, rtol=1e-4), "Initial count rate mismatch"

    assert np.allclose(func_counts, counts, atol=1e-2, rtol=1e-2), f'{irrad_type.capitalize()} counts mismatch between hand calculation and function evaluation'

    count_data = {
        'times': times,
        'counts': counts,
        'sigma counts': np.zeros(len(counts))
    }

    assert grouper.irrad_type == irrad_type
    data = grouper._nonlinear_least_squares(count_data=count_data, set_refined_fiss=False)
    test_yields = [data[key]['yield'] for key in range(grouper.num_groups)]
    test_half_lives = [data[key]['half_life'] for key in range(grouper.num_groups)]
    parameters = test_yields + test_half_lives
    adjusted_parameters = grouper._restructure_intermediate_yields(parameters)
    residual_known = np.linalg.norm(grouper._residual_function(adjusted_parameters, times, counts, None, fit_func))
    residual_previous = np.linalg.norm(grouper._residual_function(adjusted_parameters, times, func_counts, None, fit_func))
    grouper.logger.error(f'{base_parameters = }')
    grouper.logger.error(f'{base_inter_parameters = }')
    grouper.logger.error(f'{parameters = }')
    grouper.logger.error(f'{adjusted_parameters = }')
    grouper.logger.error(f'{residual_known = }')

    assert np.isclose(residual_known, residual_previous, atol=1e-1), "Same counts should have the same residual"
    
    original_half_lives = np.asarray(half_lives)
    original_yields = np.asarray(yields)
    sort_idx = np.argsort(original_half_lives)[::-1]
    sorted_original_yields = original_yields[sort_idx]
    sorted_original_half_lives = original_half_lives[sort_idx]
    
    for group in range(grouper.num_groups):
        assert np.isclose(test_yields[group], sorted_original_yields[group], rtol=1e-1, atol=1e-1), \
            f'Group {group+1} yields mismatch - {test_yields[group]=} != {sorted_original_yields[group]=}'
        assert np.isclose(test_half_lives[group], sorted_original_half_lives[group], rtol=1e-1, atol=1e-1), \
            f'Group {group+1} half lives mismatch - {test_half_lives[group]=} != {sorted_original_half_lives[group]=}'
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
def test_grouper_saturation_noex_short_fitting_few():
    input_path = './tests/unit/input/input.json'
    grouper = Grouper(input_path)
    grouper.t_ex = 0
    grouper.t_net = 30
    run_grouper_fit_test('saturation', grouper, 'few_groups')

@pytest.mark.slow
def test_grouper_saturation_ex_short_fitting_few():
    input_path = './tests/unit/input/input.json'
    grouper = Grouper(input_path)
    grouper.t_ex = 10
    grouper.t_net = 30
    run_grouper_fit_test('saturation_ex', grouper, 'few_groups')

@pytest.mark.slow
def test_grouper_intermediate_noex_short_fitting_few():
    input_path = './tests/unit/input/input.json'
    grouper = Grouper(input_path)
    grouper.t_ex = 0
    grouper.t_net = 30
    run_grouper_fit_test('intermediate', grouper, 'few_groups')

@pytest.mark.slow
def test_grouper_intermediate_ex_short_fitting_few():
    input_path = './tests/unit/input/input.json'
    grouper = Grouper(input_path)
    grouper.t_ex = 10
    grouper.t_net = 30
    run_grouper_fit_test('intermediate_ex', grouper, 'few_groups')

@pytest.mark.slow
def test_grouper_saturation_noex_fitting_standard_params():
    input_path = './tests/unit/input/input.json'
    grouper = Grouper(input_path)
    grouper.t_ex = 0
    run_grouper_fit_test('saturation', grouper, 'standard')

@pytest.mark.slow
def test_grouper_saturation_ex_fitting_standard_params():
    input_path = './tests/unit/input/input.json'
    grouper = Grouper(input_path)
    grouper.t_ex = 10
    run_grouper_fit_test('saturation_ex', grouper, 'standard')

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
def test_grouper_intermediate_ex_fitting_standard_params():
    input_path = './tests/unit/input/input.json'
    grouper = Grouper(input_path)
    grouper.t_ex = 10
    run_grouper_fit_test('intermediate_ex', grouper, 'standard')

@pytest.mark.slow
def test_grouper_intermediate_ex_short_fitting_standard_params():
    input_path = './tests/unit/input/input.json'
    grouper = Grouper(input_path)
    grouper.t_ex = 10
    grouper.t_net = 30
    run_grouper_fit_test('intermediate_ex', grouper, 'standard')


@pytest.mark.slow
def test_grouper_saturation_ex_fitting():
    input_path = './tests/unit/input/input.json'
    grouper = Grouper(input_path)
    grouper.t_ex = 10
    run_grouper_fit_test('saturation_ex', grouper)

def test_grouper_saturation_noex_short_fitting():
    input_path = './tests/unit/input/input.json'
    grouper = Grouper(input_path)
    grouper.t_ex = 0
    grouper.t_net = 30
    run_grouper_fit_test('saturation', grouper)

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

def test_effective_fiss():
    input_path = './tests/unit/input/input.json'
    grouper = Grouper(input_path)
    grouper.fission_times = np.asarray([0, 1, 2, 3])
    grouper.t_net = grouper.fission_times[-1]
    grouper.full_fission_term = np.asarray([1, 0, 1])
    hl = 250e3
    lams = np.asarray([np.log(2)/hl])
    grouper.refined_fission_term = 2/3
    eff_fiss = grouper._get_effective_fission(lams, np.exp, np.expm1)
    stat_fiss = grouper._get_saturation_fission_term(lams[0], np.exp)
    assert np.isclose(eff_fiss, stat_fiss), "Fission terms not equal"

    grouper.full_fission_term = np.asarray([1, 1, 1])
    hl = 250e3
    lams = np.asarray([np.log(2)/hl])
    grouper.refined_fission_term = 1
    eff_fiss = grouper._get_effective_fission(lams, np.exp, np.expm1)
    stat_fiss = grouper._get_saturation_fission_term(lams[0], np.exp)
    assert np.isclose(eff_fiss, stat_fiss), "Fission terms not equal"

    hl = 1e10
    lams = np.asarray([np.log(2)/hl])
    eff_fiss = grouper._get_effective_fission(lams, np.exp, np.expm1) / lams[0]
    stat_fiss = grouper._get_saturation_fission_term(lams[0], np.exp) / lams[0]
    assert np.isclose(eff_fiss, grouper.t_net), "Limit for long-lived incorrect"
    assert np.isclose(stat_fiss, grouper.t_net), "Limit for long-lived incorrect"

    hl = 1e-10
    lams = np.asarray([np.log(2)/hl])
    eff_fiss = grouper._get_effective_fission(lams, np.exp, np.expm1) / lams[0]
    stat_fiss = grouper._get_saturation_fission_term(lams[0], np.exp) / lams[0]
    assert np.isclose(eff_fiss, 0), "Limit for long-lived incorrect"
    assert np.isclose(stat_fiss, 0), "Limit for long-lived incorrect"

    hl = [1e-10, 1e10]
    lams = np.asarray(np.log(2)/hl)
    eff_fiss = grouper._get_effective_fission(lams, np.exp, np.expm1) / lams
    assert np.allclose(eff_fiss, [0.0, grouper.t_net]), "Limit for multiple incorrect"


    
def test_effective_fiss_many_ts():
    input_path = './tests/unit/input/input.json'
    grouper = Grouper(input_path)

    hl = 250e-3
    lam = np.log(2) / hl
    lams = np.asarray([lam])

    tau_d = 1 / lam
    dt = np.min((0.1, tau_d / 25.0))
    t_net = 3

    grouper.fission_times = np.arange(0.0, t_net + dt, dt)
    grouper.t_net = t_net

    t_mid = 0.5 * (grouper.fission_times[:-1] + grouper.fission_times[1:])
    full_fission = np.zeros_like(t_mid)

    full_fission[(t_mid >= 0.0) & (t_mid < 1)] = 1.0
    full_fission[(t_mid >= 2) & (t_mid < 3)] = 1.0

    grouper.full_fission_term = full_fission

    grouper.refined_fission_term = 2/3

    eff_fiss = grouper._get_effective_fission(lams, np.exp, np.expm1)
    stat_fiss = grouper._get_saturation_fission_term(lam, np.exp)

    assert np.isclose(eff_fiss, 0.9424, rtol=1e-2), "Effective fission mismatch"
    assert np.isclose(stat_fiss, 0.666, rtol=1e-2), "Static effective fission mismatch"


    full_fission[(t_mid >= 1) & (t_mid < 2)] = 1.0
    grouper.full_fission_term = full_fission
    grouper.refined_fission_term = 1

    eff_fiss = grouper._get_effective_fission(lams, np.exp, np.expm1)
    stat_fiss = grouper._get_saturation_fission_term(lam, np.exp)

    assert np.isclose(eff_fiss, stat_fiss, atol=1e-2)