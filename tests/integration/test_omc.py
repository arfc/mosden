import pytest
import numpy as np
from mosden.utils.csv_handler import CSVHandler
from mosden.preprocessing import Preprocess
from mosden.concentrations import Concentrations
from mosden.countrate import CountRate
from mosden.groupfit import Grouper
import os
from scipy.integrate import simpson

@pytest.fixture
def setup_classes():
    input_path = 'tests/integration/test-data/input_omc.json'
    preproc = Preprocess(input_path)

    preproc.run()
    Pn_data = CSVHandler(f'{preproc.processed_data_dir}/emission_probability.csv')

    new_Pn_data = {
        'Br87': {'emission probability': 1.0,
                 'sigma emission probability': 1.0},
        'Br89': {'emission probability': 1.0,
                'sigma emission probability': 1.0},
        'I137': {'emission probability': 1.0,
                'sigma emission probability': 1.0},
        'Rb94': {'emission probability': 1.0,
                'sigma emission probability': 1.0},
        'As85': {'emission probability': 1.0,
                'sigma emission probability': 1.0},
    }

    #Pn_data.write_csv(new_Pn_data) 

    return input_path

@pytest.mark.slow
def test_in_ex_no_diff(setup_classes):
    """
    This test checks that altering the in and ex-core residence times does not 
    affect the results (when no chemical removal is included).
    1. Compare nuclide yields to total delayed neutrons over total fissions
    2. Compare nuclide yields using group fit calculation
    3. Compare stationary and 50% in 50% ex residence time (1 second in, 1
    second ex-core)
    """ 
    nuc_data = {
        'Br87': {
            'yield': 0.00054,
            'lam': np.log(2) / 55.65,
            'pn': 0.026
        },
        'I137' : {
            'yield': 0.00215,
            'lam': np.log(2) / 24.5,
            'pn': 0.0714
        },
        'Ge86': {
            'yield': 0.00032,
            'lam': np.log(2) / 9.5e-2,
            'pn': 0.051965
        }
    }

    input_path = setup_classes
    source_mult = 1.18
    name_mod = '_stationary'
    concs = Concentrations(input_path)
    concs.openmc_settings['omc_dir'] = os.path.join(os.path.dirname(__file__), f'output_omc/omc_inex{name_mod}')
    concs.t_in = 1
    concs.t_ex = 0
    concs.t_net = 30
    concs.openmc_settings['source'] = source_mult
    t_net_new = concs._update_t_net()
    assert t_net_new == 30
    assert concs.t_in == 1
    assert concs.t_ex == 0
    assert concs.t_net == 30
    assert concs.openmc_settings['source'] == source_mult
    assert concs.get_irrad_index(False) == 30
    concs.concentration_path = concs.output_dir + f'concentrations{name_mod}.csv'
    concs.generate_concentrations()
    conc_data = CSVHandler(concs.concentration_path).read_csv_with_time(False)

    counts = CountRate(input_path)
    counts.t_in = 1
    counts.t_ex = 0
    counts.t_net = 30
    counts.countrate_path = counts.output_dir + f'counts{name_mod}.csv'
    counts.concentration_path = concs.output_dir + f'concentrations{name_mod}.csv'
    base_counts = counts.calculate_count_rate()

    groups = Grouper(input_path)
    groups.t_in = 1
    groups.t_ex = 0
    groups.t_net = 30
    groups.countrate_path = counts.output_dir + f'counts{name_mod}.csv'
    groups.concentration_path = concs.output_dir + f'concentrations{name_mod}.csv'
    groups.group_path = concs.output_dir + f'groups{name_mod}.csv'
    groups.full_fission_term, groups.fission_times = concs._calculate_fission_term(False)
    groups.fission_term, _ = concs._calculate_fission_term()
    groups.generate_groups()
    base_group_data = CSVHandler(groups.group_path).read_vector_csv()

    yields = base_group_data['yield']
    half_lives = base_group_data['half_life']
    base_params = yields + half_lives
    fit_func = groups._intermediate_numerical_fit_function
    base_residual_intermediate = np.linalg.norm(groups._residual_function(base_params, groups.decay_times, base_counts['counts'], None, fit_func))
    fit_func = groups._saturation_fit_function
    base_residual_saturation = np.linalg.norm(groups._residual_function(base_params, groups.decay_times, base_counts['counts'], None, fit_func))
    groups.logger.error(f'{groups.decay_times = }')
    groups.logger.error(f'{base_counts["counts"] = }')
    groups.logger.error(f'{base_params = }')
    assert np.isclose(base_residual_saturation, base_residual_intermediate, rtol=1e-1), "Residual from intermediate fit does not match the saturation fit for constant irradiation"

    yield_assertions(nuc_data, concs, groups, conc_data)

    name_mod = '_flowing'
    concs = Concentrations(input_path)
    concs.openmc_settings['omc_dir'] = os.path.join(os.path.dirname(__file__), f'output_omc/omc_inex{name_mod}')
    irrad_cutoff = concs.get_irrad_index(False)
    concs.t_in = 1
    concs.t_ex = 1
    concs.t_net = 30
    concs.openmc_settings['source'] = source_mult
    t_net_new = concs._update_t_net()
    assert t_net_new == 30
    assert concs.t_in == 1
    assert concs.t_ex == 1
    assert concs.t_net == 30
    assert concs.openmc_settings['source'] == source_mult
    assert concs.get_irrad_index(False) == 30
    concs.concentration_path = concs.output_dir + f'concentrations{name_mod}.csv'
    concs.generate_concentrations()
    flow_concs = CSVHandler(concs.concentration_path).read_csv_with_time(False)

    counts.t_in = 1
    counts.t_ex = 1
    counts.t_net = 30
    counts = CountRate(input_path)
    counts.countrate_path = counts.output_dir + f'counts{name_mod}.csv'
    counts.concentration_path = concs.output_dir + f'concentrations{name_mod}.csv'
    flow_counts = counts.calculate_count_rate()

    groups = Grouper(input_path)
    groups.t_in = 1
    groups.t_ex = 1
    groups.t_net = 30
    groups.countrate_path = counts.output_dir + f'counts{name_mod}.csv'
    groups.concentration_path = concs.output_dir + f'concentrations{name_mod}.csv'
    groups.group_path = concs.output_dir + f'groups{name_mod}.csv'
    groups.full_fission_term, groups.fission_times = concs._calculate_fission_term(False)
    groups.fission_term, _ = concs._calculate_fission_term()
    groups.generate_groups()
    flow_groups = CSVHandler(groups.group_path).read_vector_csv()

    yields = flow_groups['yield']
    half_lives = flow_groups['half_life']
    flow_params = yields + half_lives
    fit_func = groups._intermediate_numerical_fit_function
    flow_residual_intermediate = np.linalg.norm(groups._residual_function(flow_params, groups.decay_times, flow_counts['counts'], None, fit_func))
    stat_params_on_flow_residual_intermediate = np.linalg.norm(groups._residual_function(base_params, groups.decay_times, flow_counts['counts'], None, fit_func))
    fit_func = groups._saturation_fit_function
    flow_residual_saturation = np.linalg.norm(groups._residual_function(flow_params, groups.decay_times, flow_counts['counts'], None, fit_func))
    stat_params_on_flow_residual_saturation = np.linalg.norm(groups._residual_function(base_params, groups.decay_times, flow_counts['counts'], None, fit_func))
    assert np.isclose(flow_residual_saturation, flow_residual_intermediate), "Residual from intermediate fit does not match the saturation fit for (1,1) irradiation"

    assert np.isclose(stat_params_on_flow_residual_saturation, stat_params_on_flow_residual_intermediate), "Stationary parameters applied to (1,1) don't have matching residuals"

    assert flow_residual_intermediate < stat_params_on_flow_residual_intermediate, "Stationary params provide a superior fit than (1,1) params"

    yield_assertions(nuc_data, concs, groups, flow_concs)
    





@pytest.mark.slow
def test_chemical_removal(setup_classes):
    """
    This test checks multiple things:
    1. Chemical removal takes place in OpenMC
    2. Chemical removing a small amount of uranium does not affect the results significantly
    3. Using the intermediate method (which accounts for varying fission rates), the change in fission rates does not affect the result
    """
    input_path = setup_classes
    name_mod = '_base'
    concs = Concentrations(input_path)
    concs.openmc_settings['omc_dir'] = os.path.join(os.path.dirname(__file__), f'output_omc/omc_chem{name_mod}')
    concs.reprocess_locations = ['excore', 'incore']
    irrad_cutoff = concs.get_irrad_index(False)
    assert concs.t_in == 5
    assert concs.t_ex == 2
    assert concs.t_net == 33
    assert concs.get_irrad_index(False) == 9
    assert concs.repr_scale == 1.0
    assert concs.reprocess_locations == ['excore', 'incore']
    concs.concentration_path = concs.output_dir + f'concentrations{name_mod}.csv'

    concs.generate_concentrations()

    conc_data = CSVHandler(concs.concentration_path).read_csv_with_time(False)
    u235_conc = conc_data['U235']
    conc_over_time = list()
    for N in u235_conc.values():
        conc_over_time.append(N[0])
    uranium_constant = np.allclose(conc_over_time, conc_over_time[0])
    assert uranium_constant, f'{uranium_constant} values should be constant in time'

    counts = CountRate(input_path)
    counts.countrate_path = counts.output_dir + f'counts{name_mod}.csv'
    counts.concentration_path = concs.output_dir + f'concentrations{name_mod}.csv'
    base_counts = counts.calculate_count_rate()

    groups = Grouper(input_path)
    groups.countrate_path = counts.output_dir + f'counts{name_mod}.csv'
    groups.concentration_path = concs.output_dir + f'concentrations{name_mod}.csv'
    groups.group_path = concs.output_dir + f'groups{name_mod}.csv'
    groups.generate_groups()
    base_group_data = CSVHandler(groups.group_path).read_vector_csv()

    name_mod = '_small_chem'
    uranium_removal_rate = 1e-2
    concs.reprocessing = {
        'U': uranium_removal_rate
    }
    assert concs.t_in == 5
    assert concs.t_ex == 2
    assert concs.t_net == 33
    assert concs.get_irrad_index(False) == 9
    assert concs.repr_scale == 1.0
    assert concs.reprocess_locations == ['excore', 'incore']
    concs.openmc_settings['omc_dir'] = os.path.join(os.path.dirname(__file__), f'output_omc/omc_chem{name_mod}')
    concs.concentration_path = concs.output_dir + f'concentrations{name_mod}.csv'
    concs.generate_concentrations()
    conc_data = CSVHandler(concs.concentration_path).read_csv_with_time(False)
    u235_conc = conc_data['U235']
    conc_over_time = list()
    times = list()
    for t, N in u235_conc.items():
        conc_over_time.append(N[0])
        times.append(t)
    uranium_constant = np.allclose(conc_over_time, conc_over_time[0])
    assert not uranium_constant, f'{uranium_constant} values should decrease'

    predicted_uranium_conc = conc_over_time[0] * np.exp(-uranium_removal_rate * np.asarray(times[:irrad_cutoff]))
    proper_chem = np.allclose(predicted_uranium_conc, conc_over_time[:irrad_cutoff])
    assert proper_chem, f'Removal not matching: {predicted_uranium_conc = } != {conc_over_time = }'

    counts.countrate_path = counts.output_dir + f'counts{name_mod}.csv'
    counts.concentration_path = concs.output_dir + f'concentrations{name_mod}.csv'
    chem_counts = counts.calculate_count_rate()

    groups.countrate_path = counts.output_dir + f'counts{name_mod}.csv'
    groups.concentration_path = concs.output_dir + f'concentrations{name_mod}.csv'
    groups.group_path = concs.output_dir + f'groups{name_mod}.csv'
    groups.generate_groups()
    chem_group_data = CSVHandler(groups.group_path).read_vector_csv()
    for key in base_group_data.keys():
        assert np.all(np.isclose(chem_group_data[key], base_group_data[key], rtol=1e-1)), f"Data mismatch for {key}"


    name_mod = '_large_chem_noex'
    uranium_removal_rate = 3e-1
    concs.irrad_type = 'intermediate'
    counts.irrad_type = 'intermediate'
    groups.irrad_type = 'intermediate'
    concs.t_in = 1
    concs.t_ex = 0
    concs.t_net = 30
    concs.reprocessing = {
        'U': uranium_removal_rate
    }
    assert concs.t_in == 1
    assert concs.t_ex == 0
    assert concs.t_net == 30
    assert concs.get_irrad_index(False) == 30
    assert concs.repr_scale == 1.0
    assert concs.reprocess_locations == ['excore', 'incore']
    assert concs.conc_method == 'OMC'
    concs.openmc_settings['omc_dir'] = os.path.join(os.path.dirname(__file__), f'output_omc/omc_chem{name_mod}')
    concs.concentration_path = concs.output_dir + f'concentrations{name_mod}.csv'
    concs.generate_concentrations()
    conc_data = CSVHandler(concs.concentration_path).read_csv_with_time(False)
    u235_conc = conc_data['U235']
    conc_over_time = list()
    times = list()
    for t, N in u235_conc.items():
        conc_over_time.append(N[0])
        times.append(t)
    uranium_constant = np.allclose(conc_over_time, conc_over_time[0])
    assert not uranium_constant, f'{uranium_constant} values should decrease'

    predicted_uranium_conc = conc_over_time[0] * np.exp(-uranium_removal_rate * np.asarray(times[:irrad_cutoff]))
    proper_chem = np.allclose(predicted_uranium_conc, conc_over_time[:irrad_cutoff])
    assert proper_chem, f'Removal not matching: {predicted_uranium_conc = } != {conc_over_time = }'

    counts.t_in = 1
    counts.t_ex = 0
    counts.t_net = 30
    counts.countrate_path = counts.output_dir + f'counts{name_mod}.csv'
    counts.concentration_path = concs.output_dir + f'concentrations{name_mod}.csv'
    assert counts.count_method == 'data'
    chem_counts = counts.calculate_count_rate()

    groups.countrate_path = counts.output_dir + f'counts{name_mod}.csv'
    groups.concentration_path = concs.output_dir + f'concentrations{name_mod}.csv'
    groups.group_path = concs.output_dir + f'groups{name_mod}.csv'
    assert groups.irrad_type == 'intermediate'
    groups.full_fission_term, groups.fission_times = concs._calculate_fission_term(False)
    groups.generate_groups()
    chem_group_data = CSVHandler(groups.group_path).read_vector_csv()
    for key in base_group_data.keys():
        assert np.all(np.isclose(chem_group_data[key], base_group_data[key], rtol=1e-1)), f"Data mismatch for {key}"


    name_mod = '_large_chem'
    uranium_removal_rate = 3e-1
    concs.irrad_type = 'intermediate'
    counts.irrad_type = 'intermediate'
    groups.irrad_type = 'intermediate'
    concs.reprocessing = {
        'U': uranium_removal_rate
    }
    concs.t_ex = 2
    concs.t_in = 5
    concs.t_net = 33
    assert concs.t_in == 5
    assert concs.t_ex == 2
    assert concs.t_net == 33
    assert concs.get_irrad_index(False) == 9
    assert concs.repr_scale == 1.0
    assert concs.reprocess_locations == ['excore', 'incore']
    assert concs.conc_method == 'OMC'
    assert concs.irrad_type == 'intermediate'
    concs.openmc_settings['omc_dir'] = os.path.join(os.path.dirname(__file__), f'output_omc/omc_chem{name_mod}')
    concs.concentration_path = concs.output_dir + f'concentrations{name_mod}.csv'
    concs.generate_concentrations()
    conc_data = CSVHandler(concs.concentration_path).read_csv_with_time(False)
    u235_conc = conc_data['U235']
    conc_over_time = list()
    times = list()
    for t, N in u235_conc.items():
        conc_over_time.append(N[0])
        times.append(t)
    uranium_constant = np.allclose(conc_over_time, conc_over_time[0])
    assert not uranium_constant, f'{uranium_constant} values should decrease'

    predicted_uranium_conc = conc_over_time[0] * np.exp(-uranium_removal_rate * np.asarray(times[:irrad_cutoff]))
    proper_chem = np.allclose(predicted_uranium_conc, conc_over_time[:irrad_cutoff])
    assert proper_chem, f'Removal not matching: {predicted_uranium_conc = } != {conc_over_time = }'

    counts.t_in = 5
    counts.t_ex = 2
    counts.t_net = 33
    counts.countrate_path = counts.output_dir + f'counts{name_mod}.csv'
    counts.concentration_path = concs.output_dir + f'concentrations{name_mod}.csv'
    assert counts.count_method == 'data'
    chem_counts = counts.calculate_count_rate()

    groups.countrate_path = counts.output_dir + f'counts{name_mod}.csv'
    groups.concentration_path = concs.output_dir + f'concentrations{name_mod}.csv'
    groups.group_path = concs.output_dir + f'groups{name_mod}.csv'
    assert groups.irrad_type == 'intermediate'
    groups.full_fission_term, groups.fission_times = concs._calculate_fission_term(False)
    groups.generate_groups()
    chem_group_data = CSVHandler(groups.group_path).read_vector_csv()
    for key in base_group_data.keys():
        assert np.all(np.isclose(chem_group_data[key], base_group_data[key], rtol=1e-1)), f"Data mismatch for {key}"


def yield_assertions(nuc_data: dict, concs: Concentrations, groups: Grouper, conc_data: dict):
    groups.full_fission_term, groups.fission_times = concs._calculate_fission_term(False)
    dts = np.diff(groups.fission_times)
    groups.logger.error(f'{groups.fission_times = }')
    # Constant fission source over each time step
    irrad_conc_index = groups.get_irrad_index(False)
    fission_integral = [groups.full_fission_term[ti] * dts[ti] for ti in range(len(dts))]
    for nuc, data in nuc_data.items():
        yield_val = data['yield']
        lam = data['lam']
        pn = data['pn']
        exp_decays = np.exp(-lam*(groups.t_net - groups.fission_times))
        effective_fission_integral = [groups.full_fission_term[ti] * exp_decays[ti] * dts[ti] for ti in range(len(dts))]
        nuc_conc = conc_data[nuc]
        conc_times = []
        conc_vals = []
        groups.logger.error(f'{nuc_conc = }')
        for time, conc_with_uncert in nuc_conc.items():
            conc_times.append(time)
            conc_vals.append(conc_with_uncert[0])
        
        # Calculation 1 - total delnu over total fissions
        total_delnu = simpson(lam*pn*conc_vals, conc_times)
        total_fissions = fission_integral
        calc_1_yield = total_delnu / total_fissions
        assert calc_1_yield == yield_val, f'{nuc = } delnu over fission yield does not match'

        # Calculation 2 - Delayed neutrons post-irrad over effective fission integral form
        post_irrad_delnu = pn*conc_vals[irrad_conc_index]
        calc_2_yield = post_irrad_delnu / effective_fission_integral
        assert calc_2_yield == yield_val, f'{nuc = } post-irrad delnu over effective fission yield does not match'