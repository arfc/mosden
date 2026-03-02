from mosden.base import BaseClass
import numpy as np

def test_get_irrad_index():
    """
    Test the get_irrad_index method
    """
    input_path = './tests/unit/input/input.json'
    base = BaseClass(input_path)
    base.t_net = 30
    base.t_in = 0.1
    base.t_ex = 29.9
    index = base.get_irrad_index(False)
    assert index == 2

    base.t_net = base._update_t_net()
    assert base.t_net == 30.1
    index = base.get_irrad_index(False)
    assert index == 3

    base.t_net = 10
    base.t_in = 5
    base.t_ex = 5
    index = base.get_irrad_index(False)
    assert index == 2

    base.t_net = base._update_t_net()
    assert base.t_net == 15
    index = base.get_irrad_index(False)
    assert index == 3

def test_get_irrad_index_with_min_time():
    input_path = './tests/unit/input/input.json'
    base = BaseClass(input_path)
    base.t_net = 5
    base.t_in = 2
    base.t_ex = 1
    index = base.get_irrad_index(False)
    assert index == 3

    base.openmc_settings['min_timestep'] = 1
    index = base.get_irrad_index(False)
    assert index == 5

    base.t_net = 30
    base.t_in = 1
    base.t_ex = 0
    base.openmc_settings['min_timestep'] = 0.25
    index = base.get_irrad_index(False)
    assert index == 120



def test_update_t_net():
    input_path = './tests/unit/input/input.json'
    base = BaseClass(input_path)
    base.t_net = 30
    base.t_in = 0.1
    base.t_ex = 29.9

    t_net = base._update_t_net()
    assert t_net == 30.1

def test_irrad_and_update():
    input_path = './tests/unit/input/input.json'
    base = BaseClass(input_path)
    base.t_net = 10
    base.t_in = 5
    base.t_ex = 5
    base.openmc_settings['min_timestep'] = 1e10
    assert base.t_in == 5
    assert base.t_ex == 5
    assert base.t_net == 10
    assert base.openmc_settings['min_timestep'] > 5
    index = base.get_irrad_index(False)
    assert index == 2

    base.t_net = base._update_t_net()
    index = base.get_irrad_index(False)
    assert index == 3


    base.t_net = 30
    base.t_in = 10
    base.t_ex = 0
    index = base.get_irrad_index(False)
    assert index == 3

    base.t_net = base._update_t_net()
    index = base.get_irrad_index(False)
    assert index == 3

    base.t_in = 27
    base.t_ex = 3
    index = base.get_irrad_index(False)
    assert index == 2

    base.t_net = base._update_t_net()
    index = base.get_irrad_index(False)
    assert index == 3


    base.t_net = 30
    base.t_in = 1
    base.t_ex = 0.2
    base.t_net = base._update_t_net()
    assert base.t_net == 31
    index = base.get_irrad_index(False)
    assert index == 51


    base.t_net = 30
    base.t_in = 1
    base.t_ex = 0.1
    base.t_net = base._update_t_net()
    assert np.isclose(base.t_net, 30.7)
    index = base.get_irrad_index(False)
    assert index == 55

def test_times_rates_mask():
    input_path = './tests/unit/input/input.json'
    base = BaseClass(input_path)
    base.t_in = 1
    base.t_ex = 1
    base.t_net = 10
    base.openmc_settings['min_timestep'] = 1e10
    base.flux_scaling = False
    assert not base.flux_scaling
    data = base._get_times_and_rates()
    assert base.t_in == 1
    assert base.t_ex == 1
    assert base.t_net == 10
    assert base.openmc_settings['min_timestep'] > 1
    assert base._set_cycle_times(1) == [1]

    assert data['timesteps'][:10] == [1]*10
    assert data['source_rates'][:10] == [1.0, 0]*5
    assert data['removal_indeces'][:10] == list(np.arange(0, 10))
    assert data['irrad_mask'] == [0]*10

    base.residual_masks = ['post-irrad']
    data = base._get_times_and_rates()

    assert data['timesteps'][:10] == [1]*10
    assert data['source_rates'][:10] == [1.0, 0]*5
    assert data['removal_indeces'][:10] == list(np.arange(0, 10))
    assert data['irrad_mask'] == [0]*10

    base.residual_masks = ['incore']
    data = base._get_times_and_rates()

    assert data['timesteps'][:10] == [1]*10
    assert data['source_rates'][:10] == [1.0, 0]*5
    assert data['removal_indeces'][:10] == list(np.arange(0, 10))
    assert data['irrad_mask'] == [1,0]*5

    base.residual_masks = ['excore']
    data = base._get_times_and_rates()

    assert data['timesteps'][:10] == [1]*10
    assert data['source_rates'][:10] == [1.0, 0]*5
    assert data['removal_indeces'][:10] == list(np.arange(0, 10))
    assert data['irrad_mask'] == [0,1]*5


    base.residual_masks = ['all']
    data = base._get_times_and_rates()

    assert data['timesteps'][:10] == [1]*10
    assert data['source_rates'][:10] == [1.0, 0]*5
    assert data['removal_indeces'][:10] == list(np.arange(0, 10))
    assert data['irrad_mask'] == [1]*10


def test_openmc_time_setting():
    input_path = './tests/unit/input/input.json'
    base = BaseClass(input_path)
    base.openmc_settings['min_timestep'] = 1
    base.t_in = 1
    base.t_ex = 1
    base.t_net = 5
    in_vals = base._set_cycle_times(base.t_in)
    assert np.allclose(in_vals, [1])

    base.openmc_settings['min_timestep'] = 0.5
    in_vals = base._set_cycle_times(base.t_in)
    assert np.allclose(in_vals, [0.5, 0.5])

    base.openmc_settings['min_timestep'] = 0.3
    in_vals = base._set_cycle_times(base.t_in)
    assert np.allclose(in_vals, [0.3, 0.3, 0.3, 0.1])

    base.openmc_settings['min_timestep'] = 1/3
    in_vals = base._set_cycle_times(base.t_in)
    assert np.allclose(in_vals, [1/3, 1/3, 1/3])
