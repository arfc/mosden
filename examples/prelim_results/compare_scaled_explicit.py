import matplotlib.pyplot as plt
import numpy as np

s = 1
hl = 0.222
lam = np.log(2) / hl
f_in = 0.25
f_ex = 0.75
tau = 20
paras = 1e-12
tf = 10 * tau
dt = 1e-2
times = np.arange(0, tf, dt)
Na = [0]
Ns = [0]
Npa = [0]
Nps = [0]

for ti, t in enumerate(times[1:]):
    if (t % tau) <= f_in * tau:
        actual_conc = Na[ti] + dt * (s - lam * Na[ti] - paras * Na[ti])
        actual_parasitic = Npa[ti] + dt * paras * Na[ti]
    else:
        actual_conc = Na[ti] + dt * (0 - lam*Na[ti])
        actual_parasitic = Npa[ti]
    scaled_conc = s*f_in/(lam+f_in*paras) * (1 - np.exp(-(lam+f_in*paras) * t))
    scaled_parasitic = Nps[ti] + dt * f_in * paras * Ns[ti]
    Na.append(actual_conc)
    Ns.append(scaled_conc)
    Npa.append(actual_parasitic)
    Nps.append(scaled_parasitic)

plt.plot(times, Na, label='Actual')
plt.plot(times, Ns, label='Scaled')
plt.legend()
plt.show()

        
plt.plot(times, Npa, label='Actual')
plt.plot(times, Nps, label='Scaled')
plt.legend()
plt.show()

