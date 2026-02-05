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
    concs.openmc_settings['omc_dir'] = os.path.join(os.path.dirname(__file__), 'output_omc/omc')
    concs.reprocess_locations = ['excore', 'incore']
    irrad_cutoff = concs.get_irrad_index(False)
    assert concs.repr_scale == 1.0
    assert concs.reprocess_locations == ['excore', 'incore']

    concs.generate_concentrations()

    conc_data = CSVHandler(concs.concentration_path).read_csv_with_time(False)
    u235_conc = conc_data['U235']
    conc_over_time = list()
    for N in u235_conc.values():
        conc_over_time.append(N[0])
    uranium_constant = np.allclose(conc_over_time, conc_over_time[0])
    assert uranium_constant, f'{uranium_constant} values should be constant in time'

    groups = Grouper(input_path)
    groups.generate_groups()
    base_group_data = CSVHandler(groups.group_path).read_vector_csv()

    concs.reprocessing = {
        'U': 1.0
    }
    assert concs.repr_scale == 1.0
    assert concs.reprocess_locations == ['excore', 'incore']
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

    concs.logger.error(f'{times[:irrad_cutoff] = }')
    concs.logger.error(f'{conc_over_time[0]}')
    predicted_uranium_conc = conc_over_time[0] * np.exp(-1.0 * np.asarray(times[:irrad_cutoff]))
    proper_chem = np.allclose(predicted_uranium_conc, conc_over_time[:irrad_cutoff])
    assert proper_chem, f'Removal not matching: {predicted_uranium_conc = } != {conc_over_time = }'

    groups.generate_groups()
    chem_group_data = CSVHandler(groups.group_path).read_vector_csv()
    for key in base_group_data.keys():
        assert np.all(np.isclose(chem_group_data[key], base_group_data[key], atol=1e-3)), f"Data mismatch for {key}"

