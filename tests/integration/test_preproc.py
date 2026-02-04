from pathlib import Path
from mosden.preprocessing import Preprocess
import pytest
from mosden.utils.csv_handler import CSVHandler
import numpy as np
import os

@pytest.mark.parametrize("input_path, reference_output_path", [
    ("tests/integration/test-data/input1.json", "tests/integration/test-data/reference/test1/"),
    ("tests/integration/test-data/input2.json", "tests/integration/test-data/reference/test2"),
    ("tests/integration/test-data/input3.json", "tests/integration/test-data/reference/test3"),
    ("tests/integration/test-data/input4.json", "tests/integration/test-data/reference/test4")
] )
def test_preprocess(input_path, reference_output_path):
    """
    Test the preprocessing functionality.
    """
    preproc = Preprocess(input_path)
    preproc.postproc_path = os.path.join(reference_output_path, 'postproc.json')
    preproc.run()

    output_dir = Path(preproc.processed_data_dir)
    assert output_dir.exists(), f"Output directory {output_dir} does not exist."
    
    file = Path(preproc.processed_data_dir) / 'half_life.csv'
    assert file.exists(), f'Output file {file} does not exist.'
    reference_file = Path(reference_output_path) / 'half_life.csv'
    assert reference_file.exists(), f"Reference file {reference_file} does not exist."
    reference_data = CSVHandler(reference_file).read_csv()
    data = CSVHandler(file).read_csv()
    assert data == reference_data, f"Output data for {file} does not match reference data."

    file = Path(preproc.processed_data_dir) / 'fission_yield.csv'
    assert file.exists(), f'Output file {file} does not exist.'
    reference_file = Path(reference_output_path) / 'fission_yield.csv'
    assert reference_file.exists(), f"Reference file {reference_file} does not exist."
    reference_data = CSVHandler(reference_file).read_csv()
    data = CSVHandler(file).read_csv()
    assert data == reference_data, f"Output data for {file} does not match reference data." 

    file = Path(preproc.processed_data_dir) / 'emission_probability.csv'
    assert file.exists(), f'Output file {file} does not exist.'
    data = CSVHandler(file).read_csv() 
    assert data, "Output CSV file is empty."
    reference_file = Path(reference_output_path) / 'emission_probability.csv'
    assert reference_file.exists(), f"Reference file {reference_file} does not exist."
    reference_data = CSVHandler(reference_file).read_csv()
    
    assert data == reference_data, "Output data does not match the expected reference data."

    return

