import os
import subprocess
from pathlib import Path
import numpy as np
import pytest
from mosden.utils.csv_handler import CSVHandler
from mosden.base import BaseClass

@pytest.mark.slow
def test_ex_no_ex_cfy():

    # Run input_ex_cfy.json and input_no_ex_cfy.json
    ex_file = "tests/integration/test-data/input_ex_cfy.json"
    no_ex_file = "tests/integration/test-data/input_no_ex_cfy.json"

    # Run mosden for both files
    result_ex = subprocess.run(["mosden", "-pre", ex_file])
    assert result_ex.returncode == 0, f"mosden failed for ex file: {result_ex.stderr}"
    result_ex = subprocess.run(["mosden", "-m", ex_file])
    assert result_ex.returncode == 0, f"mosden failed for ex file: {result_ex.stderr}"


    result_noex = subprocess.run(["mosden", "-pre", no_ex_file])
    assert result_noex.returncode == 0, f"mosden failed for ex file: {result_noex.stderr}"
    result_noex = subprocess.run(["mosden", "-m", no_ex_file])
    assert result_noex.returncode == 0, f"mosden failed for ex file: {result_noex.stderr}"

    # Access data for each
    base_ex = BaseClass(ex_file)
    base_no_ex = BaseClass(no_ex_file)

    # ex should have concentrations 1/2 that of no_ex
    ex_concs = CSVHandler(Path(base_ex.output_dir) / "concentrations.csv").read_csv_with_time()
    no_ex_concs = CSVHandler(Path(base_no_ex.output_dir) / "concentrations.csv").read_csv_with_time()
    assert ex_concs, "Output file is empty for ex file."
    assert no_ex_concs, "Output file is empty for no_ex file."
    # Ex concs should be half of no_ex concs
    for nuc in ex_concs:
        assert nuc in no_ex_concs, f"Nucleus {nuc} not found in no_ex concentrations."
        for time in ex_concs[nuc]:
            assert time in no_ex_concs[nuc], f"Time {time} not found for nucleus {nuc} in no_ex concentrations."
            ex_value = ex_concs[nuc][time]
            no_ex_value = no_ex_concs[nuc][time]
            assert np.isclose(ex_value[0], 0.5 * no_ex_value[0], rtol=1e-5, atol=1e-8), f"Concentration mismatch for {nuc} at time {time}: expected {0.5 * no_ex_value}, got {ex_value}"

    # ex should have count rate 1/2 that of no_ex
    ex_count_rate = CSVHandler(Path(base_ex.output_dir) / "count_rate.csv").read_csv()
    no_ex_count_rate = CSVHandler(Path(base_no_ex.output_dir) / "count_rate.csv").read_csv()
    assert ex_count_rate, "Output file is empty for ex file."
    assert no_ex_count_rate, "Output file is empty for no_ex file."
    # Ex count rate should be half of no_ex count rate
    for nuc in ex_count_rate:
        assert nuc in no_ex_count_rate, f"Nucleus {nuc} not found in no_ex count rate."
        for time in ex_count_rate[nuc]:
            assert time in no_ex_count_rate[nuc], f"Time {time} not found for nucleus {nuc} in no_ex count rate."
            ex_value = ex_count_rate[nuc][time]
            no_ex_value = no_ex_count_rate[nuc][time]
            assert np.isclose(ex_value, 0.5 * no_ex_value, rtol=1e-5, atol=1e-8), f"Count rate mismatch for {nuc} at time {time}: expected {0.5 * no_ex_value}, got {ex_value}"

    # Calculate total yield (summed group yields)
    ex_groups = CSVHandler(Path(base_ex.output_dir) / "group_parameters.csv").read_vector_csv()
    no_ex_groups = CSVHandler(Path(base_no_ex.output_dir) / "group_parameters.csv").read_vector_csv()
    yield_header = 'yield'
    ex_yields = ex_groups[yield_header]
    no_ex_yields = no_ex_groups[yield_header]
    assert ex_yields, "Group parameters file is empty for ex file."
    assert no_ex_yields, "Group parameters file is empty for no_ex file."
    # Summed ex yields should be identical to no_ex yields
    summed_ex_yield = sum(ex_yields)
    summed_no_ex_yield = sum(no_ex_yields)
    assert np.isclose(summed_ex_yield, summed_no_ex_yield, rtol=1e-5, atol=1e-8), f"Yield mismatch: expected {summed_no_ex_yield}, got {summed_ex_yield}"