import numpy as np
from scipy.optimize import least_squares

def residual_func(parameters, tot_times, counts, post_residual, insitu_residual, split_index):
    calculated_counts = fit_func(tot_times, parameters, split_index)
    residual_val = (counts - calculated_counts) / counts
    if post_residual and insitu_residual:
        residual = residual_val
    elif post_residual:
        residual = residual_val[split_index:]
    elif insitu_residual:
        residual = residual_val[:split_index]
    else:
        raise ValueError
    return residual

def fit_func(tot_times, parameters, split_index):
    num_groups = int(len(parameters) / 2)
    yields = np.asarray(parameters[:num_groups])
    half_lives = parameters[num_groups:]
    lam = np.log(2) / half_lives
    # Insitu component
    times = tot_times[:split_index]
    t = np.asarray(times)
    t1 = times[:-1]
    t2 = times[1:]
    dt = t2 - t1

    lam = lam[:, None, None]
    t_eval = t[None, :, None]
    t2 = t2[None, None, :]
    dt = dt[None, None, :]

    a = -lam * (t_eval - t2)
    b = -lam * dt

    exponential_term = np.nan_to_num(np.exp(a) * -np.expm1(b))

    mask = t_eval >= t2
    exponential_term *= mask

    scaled = exponential_term
    fission_component = np.sum(scaled, axis=2)
    insitu_counts = np.sum(yields[:, None] * fission_component, axis=0)



    t1 = times[:-1]
    t2 = times[1:]
    dt = t2 - t1
    lam = np.log(2) / half_lives
    a = -lam[:, None] * np.asarray(times[-1] - t2)[None, :]
    b = -lam[:, None] * dt[None, :]
    exponential_term = np.exp(a) * -np.expm1(b)
    scaled_fission = exponential_term
    fission_component = np.sum(scaled_fission, axis=1)


    post_irrad_times = tot_times[split_index:]
    count_exponential = np.exp(-lam[:, None] * post_irrad_times[None, :])
    post_counts = np.sum(yields[:, None] * count_exponential * fission_component[:, None], axis=0)

    counts = np.concatenate((insitu_counts, post_counts))
    return counts


yields = [0.25, 0.25, 0.25, 0.25]
decay_constants = [0.001, 0.01, 0.1, 1]

# I want pulse, saturation, intermediate irradiation data
# I want the count rates collected at each point in time for that data
# Then, I want to be able to combine it in different ways
# pulse-post-irrad; pulse short-lived saturation long-lived; combined residual solve, etc.
post_residual = False
insitu_residual = True


fission_dt = 1
fission_tf = 10
decay_tf = 600
num_decay_times = 10
num_groups = 4


min_half_life = 1e-3
max_half_life = 1e3
max_yield = 1.0
lower_bounds = np.concatenate(
    (np.zeros(
        num_groups), np.ones(
        num_groups) * min_half_life))
upper_bounds = np.concatenate(
    (np.ones(
        num_groups) *
        max_yield,
        np.ones(
        num_groups) *
        max_half_life))
bounds = (lower_bounds, upper_bounds)
initial_fit = (upper_bounds + lower_bounds) / 2


fission_times = np.arange(0, fission_tf+fission_dt, fission_dt)
decay_times = np.geomspace(1e-2, decay_tf, num_decay_times)
tot_times = np.concatenate((fission_times, decay_times))
split_index = len(fission_times)
concs = np.zeros((len(tot_times), 4))
counts = np.zeros((len(tot_times), 4))
for group, (y, lam) in enumerate(zip(yields, decay_constants)):
    concs[:split_index, group] = y/lam * (1 - np.exp(-lam * fission_times))
    concs[split_index:, group] = ((y/lam) * (1 - np.exp(-lam * fission_times[-1]))) * np.exp(-lam * decay_times)
    counts[:split_index, group] = lam * y/lam * (1 - np.exp(-lam * fission_times))
    counts[split_index:, group] = lam * ((y/lam) * (1 - np.exp(-lam * fission_times[-1]))) * np.exp(-lam * decay_times)
counts = np.sum(counts, axis=1)

# Check fit functions
#temp_val = fit_func(tot_times, np.concatenate((yields, np.log(2)/decay_constants)), split_index)
#print(counts)
#print(temp_val)

result = least_squares(residual_func, initial_fit, bounds=bounds, method='trf',
                       ftol=1e-12, gtol=1e-12, xtol=1e-12,
                       verbose=0, max_nfev=1e6,
                       args=(tot_times, counts, post_residual, insitu_residual, split_index))
yields = result.x[:num_groups]
half_lives = result.x[num_groups:]
paired = zip(yields, half_lives)
sorted_pairs = sorted(paired, reverse=True)
yields, half_lives = zip(*sorted_pairs)
print(f'{yields = }')
print(f'{half_lives = }')