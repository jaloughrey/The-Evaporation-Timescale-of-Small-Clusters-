import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import glob
import functions as func

N, G, radius, softening, eta, alpha_virial, mass_segregated, t_end, n_snap, loc, seed, stem, IC_stem = func.read_param('param.csv')
seed_dirs = sorted(glob.glob(f"snapshots/{loc}/SEED*"))

#############################################################################################

#PLOT FOR LAGRANGE RADII (10% TO 90%)

mass_fractions = np.arange(0.1, 1.0, 0.1)  #10% to 90%

#one list per mass percentage
all_radii = {f: [] for f in mass_fractions}
all_times = []

#loop over each simulation in the ensemble
for seed_dir in sorted(glob.glob(f"snapshots/{loc}/SEED*")):
    snapshot_files = sorted(glob.glob(f"{seed_dir}/{stem}*.csv"))

    times = []
    lagrangian_radii = {f: [] for f in mass_fractions}

    for file in snapshot_files:
        ID, M, x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, KE_ind, PE_ind, E_ind, bound, E_total, KE, PE, alpha_virial, time, dt_initial, R90 = func.read_snapshot(file)
        times.append(time)

        #center of mass
        x_com, y_com, z_com = func.center_of_mass(x, y, z, M)
        x -= x_com
        y -= y_com
        z -= z_com

        #lagrange radius calculations
        r = np.sqrt(x**2 + y**2 + z**2)
        sorted_indices = np.argsort(r)
        r_sorted = r[sorted_indices]
        M_sorted = M[sorted_indices]
        cumulative_mass = np.cumsum(M_sorted)
        total_mass = cumulative_mass[-1]

        #for each mass fraction
        for f in mass_fractions:
            target_mass = f * total_mass
            idx = np.searchsorted(cumulative_mass, target_mass)
            lagrangian_radii[f].append(r_sorted[idx])

    times = np.array(times)
    all_times.append(times)
  
    for f in mass_fractions:
        #create interpolation functions
        interp_func = interp1d(times, lagrangian_radii[f], bounds_error=False, fill_value="extrapolate")
        all_radii[f].append(interp_func)

#define a common time grid
min_time = max(t[0] for t in all_times)
max_time = min(t[-1] for t in all_times)
common_time = np.linspace(min_time, max_time, 200)

#interpolate and stack data
mean_radii = {}
std_radii = {}

for f in mass_fractions:
    interpolated = np.array([func(common_time) for func in all_radii[f]])
    mean_radii[f] = np.mean(interpolated, axis=0)
    std_radii[f] = np.std(interpolated, axis=0)

plt.figure(figsize=(8, 4))

for f in mass_fractions:
    label = f"{int(f * 100)}% mass"
    plt.plot(common_time, mean_radii[f], label=label)
    
    #only add error band for the 90% mass shell
    if f == 0.9:
        plt.fill_between(common_time,
                         mean_radii[f] - std_radii[f],
                         mean_radii[f] + std_radii[f], color='y', alpha=0.2)

#format plot
plt.xlabel("Time (Myr)")
plt.ylabel("Lagrangian Radius (pc)")
plt.title(f"{loc}")
plt.legend()
plt.ylim(-5,60)
plt.grid(True)
plt.tight_layout()
plt.show()

#############################################################################################

#LARGRANGE RADIUS EVAPORTION INDEX TERM PLOT

all_times = []
all_evap_indices = []

for seed_dir in seed_dirs:
    snapshot_files = sorted(glob.glob(f"{seed_dir}/{stem}*.csv"))

    times = []
    R90_values = []

    #get R90_initial from first snapshot
    ID, M, x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, KE_ind, PE_ind, E_ind, bound, E_total, KE, PE, alpha_virial, time, dt_initial, R90_initial = func.read_snapshot(snapshot_files[0])

    for file in snapshot_files:
        ID, M, x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, KE_ind, PE_ind, E_ind, bound, E_total, KE, PE, alpha_virial, time, dt_initial, R90 = func.read_snapshot(file)
        times.append(time)
        R90_values.append(R90)

    times = np.array(times)
    R90_values = np.array(R90_values)
    evap_index = R90_values / R90_initial

    #interpolate
    interp_func = interp1d(times, evap_index, bounds_error=False, fill_value="extrapolate")
    all_evap_indices.append(interp_func)
    all_times.append(times)

#define common time grid
min_time = max(t[0] for t in all_times)
max_time = min(t[-1] for t in all_times)
common_time = np.linspace(min_time, max_time, 200)

#interpolate to common grid
interpolated_evap = np.array([f(common_time) for f in all_evap_indices])

#compute mean and std
mean_evap = np.mean(interpolated_evap, axis=0)
std_evap = np.std(interpolated_evap, axis=0)

#plot
plt.figure(figsize=(5, 3))
plt.plot(common_time, mean_evap, label='Mean Lagrange Term', color='purple')
plt.fill_between(common_time, mean_evap - std_evap, mean_evap + std_evap, alpha=0.3, color='purple', label='±1σ')
plt.axhline(5.0, color='red', linestyle='--', label='threshold ')
plt.xlabel("Time (Myr)")
plt.ylabel("Lagrange Term (R90(t) / R90_initial)")
plt.title(f"{loc}")
plt.grid(True)
plt.ylim(0,16)
plt.legend()
plt.tight_layout()
plt.show()

#############################################################################################

#BOUND FRACTION EVAPORATION INDEX TERM PLOT

#create storage for interpolation
all_times = []
all_bound_fractions = []

for seed_dir in seed_dirs:
    snapshot_files = sorted(glob.glob(f"{seed_dir}/{stem}*.csv"))

    times = []
    bound_fraction = []

    for file in snapshot_files:
        ID, M, x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, KE_ind, PE_ind, E_ind, bound, E_total, KE, PE, alpha_virial, time, dt_initial, R90 = func.read_snapshot(file)
        num_bound = np.sum(bound)
        bound_frac = num_bound / N
        times.append(time)
        bound_fraction.append(bound_frac)

    times = np.array(times)
    bound_fraction = np.array(bound_fraction)

    #interpolation function 
    interp_func = interp1d(times, bound_fraction, kind='linear', bounds_error=False, fill_value="extrapolate")
    all_times.append(times)
    all_bound_fractions.append(interp_func)

#define a common time grid 
min_time = max(times[0] for times in all_times)
max_time = min(times[-1] for times in all_times)
common_time = np.linspace(min_time, max_time, 200)  #200 time steps

#interpolate each run onto the common time grid
interpolated_bound_fractions = np.array([f(common_time) for f in all_bound_fractions])

#compute mean and std across simulations
mean_bound_fraction = np.mean(interpolated_bound_fractions, axis=0)
std_bound_fraction = np.std(interpolated_bound_fractions, axis=0)

#plot
plt.figure(figsize=(5, 3))
plt.plot(common_time, mean_bound_fraction, label='Mean Bound Fraction', color='blue')
plt.fill_between(common_time, 
                 mean_bound_fraction - std_bound_fraction, 
                 mean_bound_fraction + std_bound_fraction, 
                 color='blue', alpha=0.3, label='±1σ')
plt.xlabel("Time (Myr)")
plt.ylabel("Bound Fraction Term")
plt.title(f"{loc}")
plt.axhline(0.5, color='red', linestyle='--', label='threshold ')
plt.ylim(0.25,0.92)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#average ejection rate calculation

#total simulation time
total_time = common_time[-1] - common_time[0]

#final bound fractions from all runs
final_bound_fractions = interpolated_bound_fractions[:, -1]

#mean and std of final bound fraction
mean_final_bound = np.mean(final_bound_fractions)
std_final_bound = np.std(final_bound_fractions)

mean_initial_bound = mean_bound_fraction[0]
N_bound_initial = N* (mean_initial_bound)

#total ejected and average rate
total_ejected = N_bound_initial * (1 - mean_final_bound)

avg_ejection_rate = total_ejected / total_time

#error in ejection rate
ejection_rate_error = (N * std_final_bound) / total_time

print(f"Average ejection rate: {avg_ejection_rate:.2f} ± {ejection_rate_error:.2f} stars/Myr")

#############################################################################################

#DENSITY EVAPORATION INDEX TERM PLOT

seed_dirs = sorted(glob.glob(f"snapshots/{loc}/SEED*"))

#store data from all runs
all_density_terms = []
all_times = []

for seed_dir in seed_dirs:
    snapshot_files = sorted(glob.glob(f"{seed_dir}/{stem}*.csv"))

    times = []
    density_term = []

    ID, M, x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, KE_ind, PE_ind, E_ind, bound, E_total, KE, PE, alpha_virial, time, dt_initial, R90_initial = func.read_snapshot(snapshot_files[0])
    print(R90_initial)
    for file in snapshot_files:
        ID, M, x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, KE_ind, PE_ind, E_ind, bound, E_total, KE, PE, alpha_virial, time, dt_initial, R90 = func.read_snapshot(file)
        E, R90, rho, check_evap = func.evap_check(N, x, y, z, M, bound, R90_initial)
        times.append(time)
        density_term.append(rho)

    times = np.array(times)
    density_term = np.array(density_term)
    density_term = func.trailing_smooth(density_term, window_size =4)
    

    #interpolate
    interp_func = interp1d(times, density_term, bounds_error=False, fill_value="extrapolate")
    all_density_terms.append(interp_func)
    all_times.append(times)

#common time grid
min_time = max(t[0] for t in all_times)
max_time = min(t[-1] for t in all_times)
common_time = np.linspace(min_time, max_time, 200)

#interpolate all to common grid
interpolated = np.array([f(common_time) for f in all_density_terms])

#compute mean and std
mean_density = np.mean(interpolated, axis=0)
std_density = np.std(interpolated, axis=0)

#plot
plt.figure(figsize=(5, 3))
plt.plot(common_time, mean_density, label='Mean Density Term (ρ_ref/ρ(t))', color='teal')
plt.fill_between(common_time, mean_density - std_density, mean_density + std_density, alpha=0.3, color='teal', label='±1σ')
plt.axhline(0.5, color='red', linestyle='--', label='threshold ')
plt.xlabel("Time (Myr)")
plt.ylabel("Density Term (ρ_ref/ρ(t))")
plt.title(f"{loc}")
plt.grid(True)
plt.ylim(-0.2,4)
plt.legend()
plt.tight_layout()
plt.show()

#############################################################################################

#EVAPORATION INDEX PLOT 

all_times = []
all_evaps = []

evaporation_times = []

for seed_dir in seed_dirs:
    snapshot_files = sorted(glob.glob(f"{seed_dir}/{stem}*.csv"))

    times = []
    evaps = []

    for file in snapshot_files:
        *_, time, _, _, _, evap = func.read_snapshot(file)
        times.append(time)
        evaps.append(evap)

    times = np.array(times)
    evaps = np.array(evaps)

    #store interpolation function
    interp_func = interp1d(times, evaps, bounds_error=False, fill_value="extrapolate")
    all_evaps.append(interp_func)
    all_times.append(times)

    #track evaporation time (index crosses 1)
    if (evaps >= 1).any():
        evap_time = times[evaps >= 1][0]
        evaporation_times.append(evap_time)
        print(f"{seed_dir}: Evaporation time = {evap_time:.2f} Myr")
    else:
        print(f"{seed_dir}: No evaporation")

#common time range 
min_time = max(t[0] for t in all_times)
max_time = min(t[-1] for t in all_times)
common_time = np.linspace(min_time, max_time, 200)

#interpolate all
interpolated_evaps = np.array([f(common_time) for f in all_evaps])


#compute mean and standard error
mean_evap = np.mean(interpolated_evaps, axis=0)
std_error = np.std(interpolated_evaps, axis=0, ddof=1) / np.sqrt(len(interpolated_evaps))  #standard error


#plot
plt.figure(figsize=(6, 4))
plt.plot(common_time, mean_evap, color='orange', label='Mean Evaporation Index')
plt.fill_between(common_time, mean_evap - std_error, mean_evap + std_error, color='orange', alpha=0.3, label='±1σ')
plt.axhline(1.0, color='red', linestyle='--', label='Evaporation Threshold')
plt.xlabel("Time (Myr)")
plt.ylabel("Evaporation Index")
plt.title(f"{loc}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#############################################################################################

