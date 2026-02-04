from mosden.base import BaseClass

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

