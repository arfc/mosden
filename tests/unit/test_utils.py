from mosden.utils.csv_handler import CSVHandler
from mosden.utils.input_handler import InputHandler
from mosden.utils.literature_handler import Literature
import os
import pytest

def test_csv_handler():
    """
    Test the CSVHandler class for reading and writing CSV files.
    """
    test_path = './tests/unit/output/test.csv'
    csv_handler = CSVHandler(test_path, overwrite=True)

    # Test writing to a CSV file
    data = {'Xe135': {'value': 1.0, 'uncertainty': 0.1},
            'U235': {'value': 100.0, 'uncertainty': 5.0}}
    csv_handler.write_csv(data)
    
    # Test reading from the same CSV file
    read_data = csv_handler.read_csv()
    
    assert read_data == data, f"Expected {data}, but got {read_data}"
 
    return None

def test_input_handler():
    """
    Test the InputHandler class for reading input files.
    """
    from mosden.utils.input_handler import DEFAULTS
    import json
    input_path = 'test_input.json'
    
    # Create a test input file
    with open(input_path, 'w') as f:
        f.write('{"key": "value"}')
    
    input_handler = InputHandler(input_path)
    
    # Test reading the input file
    data = input_handler.read_input(check=False, apply_defaults=False)
    
    assert data == {"key": "value"}, f"Expected {{'key': 'value'}}, but got {data}"

    # Test exceptions in input handling
    with pytest.raises(KeyError):
        data = input_handler.read_input(check=True, apply_defaults=False)
        data = input_handler.read_input(check=False, apply_defaults=False)
        data = input_handler.read_input(check=True, apply_defaults=False)
    
    data = input_handler.read_input(check=True, apply_defaults=True)
    data.pop('key')
    default_data = json.loads(json.dumps(DEFAULTS))
    assert default_data == data, "Default application failed"


    os.remove(input_path)
    
    return None

def test_literature_handler():
    """
    Test the Literature class for handling literature data.
    """
    input_path = './tests/unit/input/input.json'
    lit_data = Literature(input_path)
    zero_data = lit_data._get_group_data_helper('U235', 0, 'keepin', 'thermal')
    assert zero_data['yield'] == [0.0]*6, "Expected zero yields for zero fraction"
    assert zero_data['sigma yield'] == [0.0]*6, "Expected zero uncertainties for zero fraction"
    assert zero_data['half_life'] == [0.0]*6, "Expected zero half-lives for zero fraction"
    assert zero_data['sigma half_life'] == [0.0]*6, "Expected zero uncertainties for zero half-lives"

    U235_data = lit_data._get_group_data_helper('U235', 1, 'keepin', 'thermal')
    U238_data = lit_data._get_group_data_helper('U238', 1, 'keepin', 'thermal')
    lit_data.fissiles['U235'] = 0.5
    lit_data.fissiles['U238'] = 0.5
    lit_data.energy_MeV = 0.0253e-6
    split_235_238_data = lit_data.get_group_data(names=['keepin'])
    for i in range(len(U235_data['yield'])):
        assert split_235_238_data['keepin']['yield'][i] == (0.5*U235_data['yield'][i] + 0.5*U238_data['yield'][i])
        assert split_235_238_data['keepin']['half_life'][i] == (0.5*U235_data['half_life'][i] + 0.5*U238_data['half_life'][i])



    none_data = lit_data._get_group_data_helper('nonexistant', 1, 'nonexistant', 'nonexistant')
    assert none_data == None

    return None

