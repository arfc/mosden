import os
from mosden.concentrations import Concentrations
from pathlib import Path
from mosden.utils.csv_handler import CSVHandler
from mosden.preprocessing import Preprocess
import numpy as np
import pytest

@pytest.mark.parametrize("input_path, reference_output_path", [
    ("tests/integration/test-data/input1.json", "tests/integration/test-data/reference/test1"),
    ("tests/integration/test-data/input2.json", "tests/integration/test-data/reference/test2"),
    ("tests/integration/test-data/input3.json", "tests/integration/test-data/reference/test3"),
    ("tests/integration/test-data/input4.json", "tests/integration/test-data/reference/test4")
] )
def test_generate_concentrations(input_path, reference_output_path):
    """
    Test the concentration generation method.
    """
    preproc = Preprocess(input_path)
    preproc.run()
    concentrations = Concentrations(input_path)
    concentrations.postproc_path = os.path.join(reference_output_path, 'postproc.json')
    concentrations.processed_data_dir = reference_output_path
    
    concentrations.generate_concentrations()
    
    output_path = Path(concentrations.output_dir) / "concentrations.csv"
    assert output_path.exists(), f"Output file {output_path} does not exist."

    data = CSVHandler(output_path).read_csv()
    assert data, "Output file is empty."
    
    for nuclide, values in data.items():
        assert isinstance(values['Concentration'], float), f"Concentration for {nuclide} is not a float."
        assert isinstance(values['sigma Concentration'], float), f"Sigma Concentration for {nuclide} is not a float."

    reference_path = Path(reference_output_path) / "concentrations.csv"
    reference_data = CSVHandler(reference_path).read_csv()

    assert data == reference_data, "Output concentrations do not match the expected reference concentrations."

    return


def test_chem_removal():
    """
    Test chemical removal using the basic test with and without chemical removal
    """
    input_path = "tests/integration/test-data/input7.json"
    preproc = Preprocess(input_path)
    preproc.run()
    concentrations = Concentrations(input_path)
    concentrations.generate_concentrations()
    output_path = Path(concentrations.output_dir) / "concentrations.csv"
    assert output_path.exists(), f"Output file {output_path} does not exist."
    data_nochem = CSVHandler(output_path).read_csv()
    assert data_nochem, "Output file is empty."

    input_path = "tests/integration/test-data/input8.json"
    preproc = Preprocess(input_path)
    preproc.run()
    concentrations = Concentrations(input_path)
    concentrations.generate_concentrations()
    output_path = Path(concentrations.output_dir) / "concentrations.csv"
    assert output_path.exists(), f"Output file {output_path} does not exist."
    data_chem = CSVHandler(output_path).read_csv()
    assert data_chem, "Output file is empty."

    chems_removed = concentrations.reprocessing.keys()


    
    for nuclide in data_nochem.keys():
        for chem in chems_removed:
            if chem in nuclide:
                assert (data_nochem[nuclide]['Concentration'] > data_chem[nuclide]['Concentration']), f'{nuclide} increased with removal of Xe'
