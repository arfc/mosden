from mosden.concentrations import Concentrations
import numpy as np
from scipy.integrate import trapezoid

def test_concentrations_init():
    """
    Test the initialization of the Concentrations class.
    """

    input_path = './tests/unit/input/input.json'
    conc = Concentrations(input_path)
    assert conc.input_path == input_path, f"Expected input path {input_path}, but got {conc.input_path}"
    assert conc.output_dir == './tests/unit/output', f"Expected output directory './tests/unit/output', but got {conc.output_dir}"
    assert conc.energy_MeV == 1.0, f"Expected energy 1.0, but got {conc.energy_MeV}"
    assert conc.fissiles == {'U235': 0.8, 'U238': 0.2}, f"Expected fissile targets {{'U235': 0.8, 'U238': 0.2}}, but got {conc.fissiles}"

    return

def test_evaluate_conc():
    input_path = './tests/unit/input/input.json'
    conc = Concentrations(input_path)
    dt = np.diff([0, 1, 2])

    # N(dt) = (F*y/lam) * (1 - exp(-lam*dt))
    name = 'Fission and decay'
    lam = np.log(2) / 10
    lam_p = np.log(2) / 10
    new_conc, new_p_conc = conc._evaluate_conc(
        cur_conc=0, cur_p_conc=0, lam_p=lam_p, lam=lam, ti=0,
        dt=dt, fission_rates=[1, 1, 1], concs=[0], p_concs=[0],
        y_p=0.0, y=1.0
    )
    assert new_p_conc == 0
    expected = (1.0 / lam) * (1 - np.exp(-lam * 1))
    assert np.isclose(new_conc, expected), f"{name}"

    # N(dt) = N0 * exp(-lam*dt)
    name = 'Pure decay'
    N0  = 5.0
    lam = np.log(2) / 10
    new_conc, new_p_conc = conc._evaluate_conc(
        cur_conc=N0, cur_p_conc=0, lam_p=0, lam=lam, ti=0,
        dt=dt, fission_rates=[0, 0, 0], concs=[N0], p_concs=[0],
        y_p=0.0, y=1.0
    )
    expected = N0 * np.exp(-lam * 1)
    assert np.isclose(new_conc, expected), f"{name}"

    # N(dt) = N0*exp(-l*dt) + (lp*Np0/(l-lp)) * (exp(-lp*dt) - exp(-l*dt))
    name = 'Pure decay with parent'
    lam_p = np.log(2) / 30
    lam   = np.log(2) / 10
    Np0, Nc0 = 10.0, 0.0
    new_conc, new_p_conc = conc._evaluate_conc(
        cur_conc=Nc0, cur_p_conc=Np0, lam_p=lam_p, lam=lam, ti=0,
        dt=dt, fission_rates=[0, 0, 0], concs=[Nc0], p_concs=[Np0],
        y_p=0.0, y=0.0
    )
    expected_p = Np0 * np.exp(-lam_p * 1)
    expected_c = (Nc0 * np.exp(-lam * 1)
                  + (lam_p * Np0 / (lam - lam_p))
                  * (np.exp(-lam_p * 1) - np.exp(-lam * 1)))
    assert np.isclose(new_p_conc, expected_p), f'{name} Parent incorrect'
    assert np.isclose(new_conc, expected_c), f'{name} Daughter incorrect'

    # Same but equal half-lives
    name = 'Pure decay with parent (same hl)'
    lam_p = lam = np.log(2) / 10
    Np0, Nc0 = 10.0, 0.0
    new_conc, new_p_conc = conc._evaluate_conc(
        cur_conc=Nc0, cur_p_conc=Np0, lam_p=lam_p, lam=lam, ti=0,
        dt=dt, fission_rates=[0, 0, 0], concs=[Nc0], p_concs=[Np0],
        y_p=0.0, y=0.0
    )
    expected_p = Np0 * np.exp(-lam_p * 1)
    expected_c = lam_p * Np0 * 1 * np.exp(-lam * 1)
    assert np.isclose(new_p_conc, expected_p), f'{name} Parent incorrect'
    assert np.isclose(new_conc,   expected_c), f'{name} Daughter incorrect'

    # N(dt) = N0 + F*y*dt
    name = 'Fission no decay'
    lam = 0.0
    lam_p = np.log(2) / 10
    new_conc, new_p_conc = conc._evaluate_conc(
        cur_conc=0, cur_p_conc=0, lam_p=lam_p, lam=lam, ti=0,
        dt=dt, fission_rates=[2, 2, 2], concs=[0], p_concs=[0],
        y_p=0.0, y=3.0
    )
    expected = 2.0 * 3.0 * 1
    assert np.isclose(new_conc, expected), f'{name}'

    # Pulse irradiation
    lam = np.log(2) / 10
    lam_p = 1.0
    y, y_p = 1.0, 0.0
    F = 846.364

    irrad_times = [0, 1e-5]
    decay_times = list(np.geomspace(1e-2, 600, 300))
    all_times = irrad_times + (decay_times + 1e-5)
    dt_all = np.diff(all_times)

    fission_rates = [F] + [0] * (len(dt_all))

    concs_all   = [0]
    p_concs_all = [0]
    cur_conc = cur_p_conc = 0.0

    for ti in range(len(dt_all)):
        cur_conc, cur_p_conc = conc._evaluate_conc(
            cur_conc, cur_p_conc, lam_p, lam, ti,
            dt_all, fission_rates, concs_all, p_concs_all, y_p, y
        )
        concs_all.append(cur_conc)
        p_concs_all.append(cur_p_conc)

    times_arr = np.array(all_times)
    concs_arr = np.array(concs_all)
    total_fissions = F * 1e-5
    total_delnus = trapezoid(lam * concs_arr, times_arr)
    yield_val = total_delnus / total_fissions

    assert np.isclose(yield_val, 1.0, atol=1e-6), 'Yield incorrect'


def test_evaluate_conc():
    input_path = './tests/unit/input/input.json'
    conc = Concentrations(input_path)
    conc.conc_method = 'IFY'
    times = [0, 1, 2]
    dt = np.diff(times)
    cur_conc = 0
    cur_p_conc = 0
    lam_p = np.log(2) / 10
    lam = np.log(2) / 10
    ti = 0
    fission_rates = [1, 1, 1]
    concs = [0]
    p_concs = [0]
    y_p = 0.0
    y = 1.0
    new_conc, new_p_conc = conc._evaluate_conc(cur_conc, cur_p_conc, lam_p, lam, ti, dt, fission_rates, concs, p_concs, y_p, y)
    assert new_p_conc == 0
    new_conc_expected = 1/lam * (1 - np.exp(-lam * 1))
    assert new_conc == new_conc_expected, "Concentrations don't match"

    conc.conc_method = 'CFY'
    new_conc, new_p_conc = conc._evaluate_conc(cur_conc, cur_p_conc, lam_p, lam, ti, dt, fission_rates, concs, p_concs, y_p, y)
    assert new_p_conc == y_p / lam_p
    new_conc_expected = (y_p + y) / lam
    assert new_conc == new_conc_expected, "Concentrations don't match"

