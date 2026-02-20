from mosden.groupfit import Grouper
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
def test_fit_groups(input_path, reference_output_path):
    """
    Test the group fitting.
    """
    grouper = Grouper(input_path)

    grouper.postproc_path = os.path.join(reference_output_path, 'postproc.json')
    grouper.generate_groups()

    output_path = Path(grouper.output_dir) / "group_parameters.csv"
    assert output_path.exists(), f"Output file {output_path} does not exist."

    data = CSVHandler(output_path).read_vector_csv()
    assert data, "Output file is empty."

    reference_path = Path(reference_output_path) / "group_parameters.csv"
    reference_data = CSVHandler(reference_path).read_vector_csv()

    assert data.keys() == reference_data.keys(), "Reference keys do not match output keys"

    for key in data.keys():
        assert np.all(np.isclose(data[key], reference_data[key], atol=1e-3)), f"Data mismatch for {key}"

    return