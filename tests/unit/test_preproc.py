from mosden.preprocessing import Preprocess




def test_preprocess_init():
    """    
    Test the initialization of the Preprocess class.
    """

    input_path = './tests/unit/input/input.json'
    preproc = Preprocess(input_path)
    assert preproc.input_path == input_path, f"Expected input path {input_path}, but got {preproc.input_path}"
    assert preproc.processed_data_dir == './tests/unit/data_output', f"Expected output directory './tests/unit/data_output', but got {preproc.processed_data_dir}"
    assert preproc.energy_MeV == 1.0, f"Expected energy_MeV 1.0, but got {preproc.energy_MeV}"
    assert preproc.fissile_targets == ['U235', 'U238'], f"Expected fissile targets ['U235', 'U238'], but got {preproc.fissile_targets}"

    return

