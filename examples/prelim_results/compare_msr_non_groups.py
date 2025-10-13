import numpy as np
import matplotlib.pyplot as plt


def msr_group(T, t, tin, tex, f_rate, yields, halflives):
    J = int(np.ceil(T/(tin+tex)))
    print(J)
    full_sum = 0
    for k in range(len(yields)):
        lam = np.log(2) / halflives[k]
        nu = yields[k]
        j_sum = 0
        for j in range(1, J):
            j_sum += np.exp(-lam * (T - j*tin - (j-1) * tex))
        full_sum += nu * np.exp(-lam*t) * (1 - np.exp(-lam*T) + (1-np.exp(lam*tex)) * j_sum)
    full_sum = full_sum * f_rate
    return full_sum

def nonmsr_group(T, t, f_rate, yields, halflives):
    full_sum = 0
    for k in range(len(yields)):
        lam = np.log(2) / halflives[k]
        nu = yields[k]
        full_sum += nu * np.exp(-lam * t) * (1 - np.exp(-lam*T))
    full_sum = full_sum * f_rate
    return full_sum


T = 600
t = np.arange(0, 300, 0.01)
tin = 10
tex = 10
f_rate = 1
yields = [1]
halflives = [60]

msr_res = msr_group(T, t, tin, tex, f_rate, yields, halflives)
stat_res = nonmsr_group(T, t, f_rate, yields, halflives)

print(msr_res)
print(stat_res)

plt.plot(t, msr_res, label='MSR')
plt.plot(t, stat_res, label='Traditional')
plt.legend()
plt.yscale('log')
plt.show()
