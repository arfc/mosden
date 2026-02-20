import subprocess
import numpy as np
import pytest
from pathlib import Path
from mosden.utils.csv_handler import CSVHandler
from mosden.base import BaseClass

@pytest.mark.parametrize("input_path, reference_output_path, output_path", [
    ("tests/integration/test-data/input1.json", "tests/integration/test-data/reference/test1", "tests/integration/output1"),
    ("tests/integration/test-data/input2.json", "tests/integration/test-data/reference/test2", "tests/integration/output2"),
    pytest.param("tests/integration/test-data/input3.json", "tests/integration/test-data/reference/test3", "tests/integration/output3", marks=pytest.mark.slow),
    pytest.param("tests/integration/test-data/input4.json", "tests/integration/test-data/reference/test4", "tests/integration/output4", marks=pytest.mark.slow),
    pytest.param("tests/integration/test-data/input5.json", "tests/integration/test-data/reference/test5", "tests/integration/output5", marks=pytest.mark.slow),
    pytest.param("tests/integration/test-data/input6.json", "tests/integration/test-data/reference/test6", "tests/integration/output6", marks=pytest.mark.slow),
    ("tests/integration/test-data/input7.json", "tests/integration/test-data/reference/test7", "tests/integration/output7"),
    ("tests/integration/test-data/input8.json", "tests/integration/test-data/reference/test8", "tests/integration/output8"),
])
def test_mosden_cli(input_path, reference_output_path, output_path):

    def check_output_file_exists_and_matches(output_path, filename, reference_path):
        output_file = Path(output_path) / filename
        reference_file = Path(reference_path) / filename
        assert output_file.exists(), f"Expected output file {output_file} not found."
        assert reference_file.exists(), f"Reference file {reference_file} not found."
        rtol = 1e-5
        atol = 1e-8
        if filename == 'group_parameters.csv':
            output_data = CSVHandler(output_file).read_vector_csv()
            reference_data = CSVHandler(reference_file).read_vector_csv()
            rtol = 1
            atol = 1
        elif filename == 'postproc.json':
            output_data = BaseClass(output_file).load_post_data()
            reference_data = BaseClass(reference_file).load_post_data()
        elif filename == 'concentrations.csv':
            output_data = CSVHandler(output_file).read_csv_with_time()
            reference_data = CSVHandler(reference_file).read_csv_with_time()
        else:
            output_data = CSVHandler(output_file).read_csv()
            reference_data = CSVHandler(reference_file).read_csv()
        for key in output_data.keys():
            if not isinstance(output_data[key], dict):
                assert np.allclose(output_data[key], reference_data[key], rtol=rtol, atol=atol), f"Data mismatch for {key} in {filename}"
            else:
                for subkey in output_data[key].keys():
                    assert np.all(np.isclose(output_data[key][subkey], reference_data[key][subkey], rtol=rtol, atol=atol)), f"Data mismatch for {subkey} in {key} of {filename}"


    result = subprocess.run(["mosden", "-pre", input_path])
    assert result.returncode == 0, f"mosden failed: {result.stderr}"
    result = subprocess.run(["mosden", "-m", input_path])
    assert result.returncode == 0, f"mosden failed: {result.stderr}"
    files = ['group_parameters.csv',
             'concentrations.csv',
             'fission_yield.csv',
             'count_rate.csv',
             'half_life.csv',
             'emission_probability.csv',
             'postproc.json']

    for filename in files:
        check_output_file_exists_and_matches(output_path, filename, reference_output_path)
