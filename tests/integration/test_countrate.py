from mosden.countrate import CountRate
from pathlib import Path
from mosden.utils.csv_handler import CSVHandler
from mosden.preprocessing import Preprocess
from mosden.concentrations import Concentrations
import numpy as np
import pytest
import os

@pytest.mark.parametrize("input_path, reference_output_path", [
    ("tests/integration/test-data/input1.json", "tests/integration/test-data/reference/test1"),
    ("tests/integration/test-data/input2.json", "tests/integration/test-data/reference/test2"),
    ("tests/integration/test-data/input3.json", "tests/integration/test-data/reference/test3"),
    ("tests/integration/test-data/input4.json", "tests/integration/test-data/reference/test4")
] )
def test_calculate_count_rate(input_path, reference_output_path):
    """
    Test the count rate calculation.
    """
    countrate = CountRate(input_path)
    countrate.processed_data_dir = reference_output_path
    countrate.postproc_path = os.path.join(reference_output_path, 'postproc.json')
    countrate.concentration_path = os.path.join(countrate.input_data['file_options']['output_dir'], 'concentrations.csv')

    countrate.calculate_count_rate()
    
    output_path = Path(countrate.output_dir) / "count_rate.csv"
    assert output_path.exists(), f"Output file {output_path} does not exist."

    data = CSVHandler(output_path).read_vector_csv()
    assert data, "Output file is empty."

    reference_path = Path(reference_output_path) / "count_rate.csv"
    reference_data = CSVHandler(reference_path).read_vector_csv()

    assert data.keys() == reference_data.keys(), "Reference keys do not match output keys"

    for key in data.keys():
        assert np.all(np.isclose(data[key], reference_data[key])), f"Data mismatch for {key}"

    return


def test_spectra_counts():
    input_path = 'tests/integration/test-data/input9.json'
    Preprocess(input_path).run()
    Concentrations(input_path).generate_concentrations()
    countrate = CountRate(input_path)
    countrate.calculate_count_rate()

    counts_path = Path(countrate.output_dir) / "count_rate.csv"
    assert counts_path.exists(), "Count rate does not exist"

    spectra_path = Path(countrate.output_dir) / "spectra_counts.csv"
    assert spectra_path.exists(), "Spectral count rate does not exist"

    countrate_data = CSVHandler(counts_path, create=False).read_vector_csv()
    spectra_data = CSVHandler(spectra_path, create=False).read_vector_csv()

    assert countrate_data['times'] == spectra_data['times'], "Times do not match"
    
    for ti in range(len(countrate_data['times'])):
        counts = countrate_data['counts'][ti]
        spectral_energies = countrate.eV_midpoints
        spectral_rates = [spectra_data[str(e)][ti] for e in spectral_energies]
        spectral_counts = np.sum(spectral_rates)
        assert np.isclose(counts, spectral_counts), f"Count rates do not match at {ti}"

    countrate.energy_groups_MeV = [0,3.00E-09,5.00E-09,6.90E-09,1.00E-08,1.50E-08,2.00E-08,2.50E-08,3.00E-08,3.50E-08,4.20E-08,5.00E-08,5.80E-08,6.70E-08,7.70E-08,8.00E-08,9.50E-08,1.00E-07,1.15E-07,1.34E-07,1.40E-07,1.60E-07,1.80E-07,1.89E-07,2.20E-07,2.48E-07,2.80E-07,3.00E-07,3.15E-07,3.20E-07,3.50E-07,3.91E-07,4.00E-07,4.33E-07,4.85E-07,5.00E-07,5.40E-07,6.25E-07,7.05E-07,7.80E-07,7.90E-07,8.50E-07,8.60E-07,9.10E-07,9.30E-07,9.50E-07,9.72E-07,9.86E-07,9.96E-07,1.02E-06,1.04E-06,1.05E-06,1.07E-06,1.10E-06,1.11E-06,1.13E-06,1.15E-06,1.17E-06,1.24E-06,1.30E-06,1.34E-06,1.37E-06,1.44E-06,1.48E-06,1.50E-06,1.59E-06,1.67E-06,1.76E-06,1.84E-06,1.93E-06,2.02E-06,2.10E-06,2.13E-06,2.36E-06,2.55E-06,2.60E-06,2.72E-06,2.77E-06,3.30E-06,3.38E-06,4.00E-06,4.13E-06,5.04E-06,5.35E-06,6.16E-06,7.52E-06,8.32E-06,9.19E-06,9.91E-06,1.12E-05,1.37E-05,1.59E-05,1.95E-05,2.26E-05,2.50E-05,2.76E-05,3.05E-05,3.37E-05,3.73E-05,4.02E-05,4.55E-05,4.83E-05,5.16E-05,5.56E-05,6.79E-05,7.57E-05,9.17E-05,1.37E-04,1.49E-04,2.04E-04,3.04E-04,3.72E-04,4.54E-04,6.77E-04,7.49E-04,9.14E-04,1.01E-03,1.23E-03,1.43E-03,1.51E-03,2.03E-03,2.25E-03,3.35E-03,3.53E-03,5.00E-03,5.50E-03,7.47E-03,9.12E-03,1.11E-02,1.50E-02,1.66E-02,2.48E-02,2.74E-02,2.93E-02,3.70E-02,4.09E-02,5.52E-02,6.74E-02,8.23E-02,1.11E-01,1.23E-01,1.83E-01,2.47E-01,2.73E-01,3.02E-01,4.08E-01,4.50E-01,4.98E-01,5.50E-01,6.08E-01,8.21E-01,9.07E-01,1.00E+00,1.11E+00,1.22E+00,1.35E+00,1.65E+00,2.02E+00,2.23E+00,2.47E+00,3.01E+00,3.68E+00,4.49E+00,5.49E+00,6.07E+00,6.70E+00,8.19E+00,1.00E+01,1.00E+03]
    countrate.eV_midpoints = countrate._get_midpoint_eVs(countrate.energy_groups_MeV)
    os.remove(counts_path)
    os.remove(spectra_path)

    assert not counts_path.exists(), "Count rate was not removed"
    assert not spectra_path.exists(), "Spectral count rate was not removed"
    

    preproc = Preprocess(input_path)
    preproc.eV_midpoints = countrate.eV_midpoints
    preproc.run()
    countrate.calculate_count_rate()
    countrate_data = CSVHandler(counts_path, create=False).read_vector_csv()
    spectra_data = CSVHandler(spectra_path, create=False).read_vector_csv()
    
    assert countrate_data['times'] == spectra_data['times'], "Times do not match"
    
    for ti in range(len(countrate_data['times'])):
        counts = countrate_data['counts'][ti]
        spectral_energies = countrate.eV_midpoints
        spectral_rates = [spectra_data[str(e)][ti] for e in spectral_energies]
        spectral_counts = np.sum(spectral_rates)
        assert np.isclose(counts, spectral_counts), f"Count rates do not match at {ti}"