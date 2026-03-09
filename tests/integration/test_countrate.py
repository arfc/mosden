from mosden.countrate import CountRate
from pathlib import Path
from mosden.utils.csv_handler import CSVHandler
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