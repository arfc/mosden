import pytest
import numpy as np
from mosden.utils.csv_handler import CSVHandler
from mosden.preprocessing import Preprocess
from mosden.concentrations import Concentrations
from mosden.countrate import CountRate
from mosden.groupfit import Grouper
import subprocess
import sys
import os
from jinja2 import Environment, PackageLoader

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

    Pn_data.write_csv(new_Pn_data) 

    return input_path

@pytest.mark.slow
def test_chemical_removal(setup_classes):
    input_path = setup_classes
    concs = Concentrations(input_path)
    concs.t_in = 27
    concs.t_ex = 3
    concs.reprocess_locations = ['excore', 'incore']

    env = Environment(loader=PackageLoader('mosden'))
    file = concs.openmc_settings['omc_file']
    template = env.get_template(file)
    chain_file = os.path.join(concs.unprocessed_data_dir, concs.openmc_settings['chain'])
    cross_sections = os.path.join(concs.unprocessed_data_dir, concs.openmc_settings['x_sections'])
    omc_dir = concs.openmc_settings['omc_dir']
    render_data = {
        'nps': concs.openmc_settings['nps'],
        'mode': concs.openmc_settings['mode'],
        'batches': concs.openmc_settings['batches'],
        'source': concs.openmc_settings['source'],
        'seed': concs.seed,
        'energy': concs.energy_MeV,
        'density': concs.density_g_cc,
        'temperature': concs.temperature_K,
        'fissiles': concs.fissiles,
        't_in': concs.t_in,
        't_ex': concs.t_ex,
        'total_irrad_s': concs.t_net,
        'decay_times': concs.decay_times,
        'repr_locations': concs.reprocess_locations,
        'reprocessing': concs.reprocessing,
        'repr_scale': concs.repr_scale,
        'chain_file': chain_file,
        'cross_sections': cross_sections,
        'omc_dir': omc_dir,
        'flux_scaling': concs.flux_scaling,
        'chem_scaling': concs.chem_scaling,
        'f_in': concs.f_in
    }
    rendered_template = template.render(render_data)
    omc_dir = concs.openmc_settings['omc_dir']
    full_name = f'{omc_dir}/omc.py'
    if not os.path.exists(omc_dir):
        os.makedirs(omc_dir)
    with open(full_name, mode='w') as output:
        output.write(rendered_template)
    completed_process = subprocess.run([sys.executable, full_name],
                                capture_output=True, text=True,
                                check=True)
    CSVHandler(concs.concentration_path, concs.conc_overwrite).write_csv_with_time(data)
    conc_data = CSVHandler(concs.concentration_path).read_csv_with_time(False)
    u235_conc = conc_data['U235']
    conc_over_time = list()
    for N in u235_conc.values():
        conc_over_time.append(N)
    uranium_constant = np.allclose(conc_over_time, conc_over_time[0])
    assert uranium_constant, f'{uranium_constant} values should be constant in time'

    counts = CountRate(input_path)
    counts.calculate_count_rate()
    base_count_data = CSVHandler(counts.countrate_path).read_vector_csv()
    groups = Grouper(input_path)
    groups.generate_groups()
    base_group_data = CSVHandler(groups.group_path).read_vector_csv()

    concs.reprocessing = {
        'U': 1.0
    }
    concs.generate_concentrations()
    conc_data = CSVHandler(concs.concentration_path).read_csv_with_time(False)
    u235_conc = conc_data['U235']
    conc_over_time = list()
    times = list()
    for t, N in u235_conc.items():
        conc_over_time.append(N)
        times.append(t)
    uranium_constant = np.allclose(conc_over_time, conc_over_time[0])
    assert not uranium_constant, f'{uranium_constant} values should decrease'

    predicted_uranium_conc = N[0] * np.exp(1.0 * times)
    proper_chem = np.allclose(predicted_uranium_conc, conc_over_time)
    assert proper_chem, f'{predicted_uranium_conc = } != {conc_over_time = }'

    counts.calculate_count_rate()
    chem_count_data = CSVHandler(counts.countrate_path).read_vector_csv()
    assert chem_count_data.keys() == base_count_data.keys(), 'Count rate times do not match'
    for key in chem_count_data.keys():
        assert np.all(np.isclose(chem_count_data[key], base_count_data[key])), f"Data mismatch for {key}"
    
    groups.generate_groups()
    chem_group_data = CSVHandler(groups.group_path).read_vector_csv()
    for key in base_group_data.keys():
        assert np.all(np.isclose(chem_group_data[key], base_group_data[key], atol=1e-3)), f"Data mismatch for {key}"

