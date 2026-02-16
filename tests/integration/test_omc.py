import pytest
import numpy as np
from mosden.utils.csv_handler import CSVHandler
from mosden.preprocessing import Preprocess
from mosden.concentrations import Concentrations
from mosden.countrate import CountRate
from mosden.groupfit import Grouper
import os
from scipy.integrate import simpson, trapezoid

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
    groups.irrad_type = 'intermediate'
    adjusted_params = groups._restructure_intermediate_yields(base_params, to_yield=False)
    groups.irrad_type = 'saturation'
    assert np.allclose(base_counts['counts'], fit_func(groups.decay_times, adjusted_params), rtol=1e-2), "Intermediate counts do not match"
    base_residual_intermediate = np.linalg.norm(groups._residual_function(adjusted_params, groups.decay_times, base_counts['counts'], None, fit_func))
    fit_func = groups._saturation_fit_function
    assert np.allclose(base_counts['counts'], fit_func(groups.decay_times, base_params), rtol=1e-2), "Saturation counts do not match"
    base_residual_saturation = np.linalg.norm(groups._residual_function(base_params, groups.decay_times, base_counts['counts'], None, fit_func))
    assert np.isclose(base_residual_saturation, base_residual_intermediate, atol=1e-1), "Residual from intermediate fit does not match the saturation fit for constant irradiation"

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
    concs.t_net = t_net_new
    assert t_net_new == 31
    assert concs.t_in == 1
    assert concs.t_ex == 1
    assert concs.t_net == 31
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
    groups.irrad_type = 'intermediate'
    adjusted_flow_params = groups._restructure_intermediate_yields(base_params)
    assert np.allclose(base_counts['counts'], fit_func(groups.decay_times, adjusted_flow_params), rtol=1e-2), "Intermediate counts do not match"
    flow_residual_intermediate = np.linalg.norm(groups._residual_function(adjusted_flow_params, groups.decay_times, flow_counts['counts'], None, fit_func))
    stat_params_on_flow_residual_intermediate = np.linalg.norm(groups._residual_function(adjusted_params, groups.decay_times, flow_counts['counts'], None, fit_func))
    fit_func = groups._saturation_fit_function
    groups.irrad_type = 'saturation'
    assert np.allclose(base_counts['counts'], fit_func(groups.decay_times, flow_params), rtol=1e-2), "Saturation counts do not match"
    flow_residual_saturation = np.linalg.norm(groups._residual_function(flow_params, groups.decay_times, flow_counts['counts'], None, fit_func))
    stat_params_on_flow_residual_saturation = np.linalg.norm(groups._residual_function(base_params, groups.decay_times, flow_counts['counts'], None, fit_func))
    assert np.isclose(flow_residual_saturation, flow_residual_intermediate), "Residual from intermediate fit does not match the saturation fit for (1,1) irradiation"

    assert np.isclose(stat_params_on_flow_residual_saturation, stat_params_on_flow_residual_intermediate), "Stationary parameters applied to (1,1) don't have matching residuals"

    assert flow_residual_intermediate < stat_params_on_flow_residual_intermediate, "Stationary params provide a superior fit than (1,1) params"

    yield_assertions(nuc_data, concs, groups, flow_concs)
    

def set_attrs(class_obj, tin, tex, tnet, irrad_type, name_mod, uranium_removal_rate=0.0):
    class_obj.t_in = tin
    class_obj.t_ex = tex
    class_obj.t_net = tnet
    class_obj.irrad_type = irrad_type
    class_obj.openmc_settings['omc_dir'] = os.path.join(os.path.dirname(__file__), f'output_omc/omc_chem{name_mod}')
    class_obj.concentration_path = class_obj.output_dir + f'concentrations{name_mod}.csv'
    class_obj.countrate_path = class_obj.output_dir + f'counts{name_mod}.csv'
    class_obj.group_path = class_obj.output_dir + f'groups{name_mod}.csv'
    class_obj.reprocess_locations = ['excore', 'incore']
    class_obj.repr_scale = 1.0
    class_obj.reprocessing = {
        'U': uranium_removal_rate
    }
    assert class_obj.t_in == tin
    assert class_obj.t_ex == tex
    assert class_obj.t_net == tnet
    assert class_obj.irrad_type == irrad_type
    assert class_obj.repr_scale == 1.0
    assert class_obj.reprocess_locations == ['excore', 'incore']
    return class_obj

def set_attrs_from_obj(new_obj, old_obj, name_mod):
    tin = old_obj.t_in
    tex = old_obj.t_ex
    tnet = old_obj.t_net
    irrad_type = old_obj.irrad_type
    uranium_removal_rate = old_obj.reprocessing['U']
    return set_attrs(new_obj, tin, tex, tnet, irrad_type, name_mod, uranium_removal_rate)


def check_uranium_conc(concs, uranium_removal_rate, irrad_cutoff):
    conc_data = CSVHandler(concs.concentration_path).read_csv_with_time(False)
    u235_conc = conc_data['U235']
    conc_over_time = list()
    times = list()
    for t, N in u235_conc.items():
        conc_over_time.append(N[0])
        times.append(t)
    predicted_uranium_conc = conc_over_time[0] * np.exp(-uranium_removal_rate * np.asarray(times[:irrad_cutoff]))
    proper_chem = np.allclose(predicted_uranium_conc, conc_over_time[:irrad_cutoff])
    assert proper_chem, f'Removal not matching: {predicted_uranium_conc = } != {conc_over_time = }'

def get_counts_and_groups(input_path, name_mod, concs):
    counts = CountRate(input_path)
    counts = set_attrs_from_obj(counts, concs, name_mod)
    counts.countrate_path = counts.output_dir + f'counts{name_mod}.csv'
    counts.concentration_path = concs.output_dir + f'concentrations{name_mod}.csv'
    count_data = counts.calculate_count_rate()

    groups = Grouper(input_path)
    groups = set_attrs_from_obj(groups, concs, name_mod)
    groups.full_fission_term, groups.fission_times = concs._calculate_fission_term(False)
    groups.countrate_path = counts.output_dir + f'counts{name_mod}.csv'
    groups.concentration_path = concs.output_dir + f'concentrations{name_mod}.csv'
    groups.group_path = concs.output_dir + f'groups{name_mod}.csv'
    groups.generate_groups()
    group_data = CSVHandler(groups.group_path).read_vector_csv()
    return count_data, group_data

def compare_counts(new_counts, old_counts):
    assert np.all(np.isclose(new_counts['counts'], old_counts['counts'])), "Counts do not match"

def compare_group_params(new_group, old_group):
    keys = ['half_life', 'yield']
    for key in keys:
        assert np.all(np.isclose(new_group[key], old_group[key], rtol=1e-1)), f"Data mismatch for {key}"


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
    uranium_removal_rate = 0.0
    concs = Concentrations(input_path)
    concs = set_attrs(concs, tin=5, tex=2, tnet=33, irrad_type='intermediate', name_mod=name_mod, uranium_removal_rate=uranium_removal_rate)
    irrad_cutoff = concs.get_irrad_index(False)
    assert concs.get_irrad_index(False) == 9
    concs.generate_concentrations()
    check_uranium_conc(concs, uranium_removal_rate, irrad_cutoff)
    base_counts, base_group_data = get_counts_and_groups(input_path, name_mod, concs)
    compare_counts(base_counts, base_counts)

    name_mod = '_small_chem'
    uranium_removal_rate = 1e-2
    concs = set_attrs(concs, tin=5, tex=2, tnet=33, irrad_type='intermediate', name_mod=name_mod, uranium_removal_rate=uranium_removal_rate)
    assert concs.get_irrad_index(False) == 9
    concs.generate_concentrations()
    check_uranium_conc(concs, uranium_removal_rate, irrad_cutoff)
    small_chem_counts, small_chem_groups = get_counts_and_groups(input_path, name_mod, concs)
    compare_counts(small_chem_counts, base_counts)
    compare_group_params(small_chem_groups, base_group_data)


    name_mod = '_large_chem_noex'
    uranium_removal_rate = 3e-1
    concs = set_attrs(concs, tin=1, tex=0, tnet=30, irrad_type='intermediate', name_mod=name_mod, uranium_removal_rate=uranium_removal_rate)
    assert concs.get_irrad_index(False) == 30
    concs.generate_concentrations()
    check_uranium_conc(concs, uranium_removal_rate, irrad_cutoff)
    large_chem_noex_counts, large_chem_noex_groups = get_counts_and_groups(input_path, name_mod, concs)
    compare_counts(large_chem_noex_counts, base_counts)
    compare_group_params(large_chem_noex_groups, base_group_data)


    name_mod = '_large_chem'
    uranium_removal_rate = 3e-1
    concs = set_attrs(concs, tin=5, tex=2, tnet=33, irrad_type='intermediate', name_mod=name_mod, uranium_removal_rate=uranium_removal_rate)
    assert concs.get_irrad_index(False) == 9
    concs.generate_concentrations()
    check_uranium_conc(concs, uranium_removal_rate, irrad_cutoff)
    large_chem_counts, large_chem_groups = get_counts_and_groups(input_path, name_mod, concs)
    compare_counts(large_chem_counts, base_counts)
    compare_group_params(large_chem_groups, base_group_data)


def yield_assertions(nuc_data: dict, concs: Concentrations, groups: Grouper, conc_data: dict):
    groups.full_fission_term, groups.fission_times = concs._calculate_fission_term(False)
    dts = np.diff(groups.fission_times)
    # Constant fission source over each time step
    irrad_conc_index = groups.get_irrad_index(False)
    fission_integral = [groups.full_fission_term[ti] * dts[ti] for ti in range(len(dts))]
    for nuc, data in nuc_data.items():
        yield_val = data['yield']
        lam = data['lam']
        pn = data['pn']
        effective_fission_integral = groups._get_effective_fission(np.asarray([lam]), np.exp, np.expm1)[0] / lam
        nuc_conc = conc_data[nuc]
        conc_times = []
        conc_vals = []
        for time, conc_with_uncert in nuc_conc.items():
            conc_times.append(time)
            conc_vals.append(conc_with_uncert[0])
        conc_vals = np.asarray(conc_vals)

        # Calculation 1 - total delnu over total fissions
        total_delnu = trapezoid(lam*pn*conc_vals, conc_times)
        total_fissions = np.sum(fission_integral)
        calc_1_yield = total_delnu / total_fissions
        assert np.isclose(calc_1_yield, yield_val, atol=1e-3), f'{nuc = } delnu over fission yield does not match'

        # Calculation 2 - Delayed neutrons post-irrad over effective fission integral form
        post_irrad_delnu = pn*conc_vals[irrad_conc_index]
        calc_2_yield = post_irrad_delnu / effective_fission_integral
        assert np.isclose(calc_2_yield, yield_val, atol=1e-3), f'{nuc = } post-irrad delnu over effective fission yield does not match'