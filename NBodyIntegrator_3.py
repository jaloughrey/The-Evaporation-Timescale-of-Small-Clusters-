import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob 
from numba import jit 

#FUNCTIONS:
@jit
def acceleration(x, y, z, vx, vy, vz, softening, M, G):
    ax = np.zeros(N)
    ay = np.zeros(N)
    az = np.zeros(N)
    adotx = np.zeros(N)
    adoty = np.zeros(N)
    adotz = np.zeros(N)
    for i in range(N):#current star
        for j in range(i + 1, N):  #sum each pair only once
            dx = x[i]- x[j]
            dy = y[i]- y[j]
            dz = z[i]- z[j]
            dvx = vx[j] - vx[i]
            dvy = vy[j] - vy[i]
            dvz = vz[j] - vz[i]
                
            r = np.sqrt(dx*dx+dy*dy+dz*dz+softening*softening)
            r3 = r*r*r
            r5 = r*r*r*r*r
            r3_inv = 1/r3
            r5_inv = 1 / r5

            rv_dot = dx * dvx + dy * dvy + dz * dvz  #dot product of r_ij and v_ij

            #acceleration
            ax[i] += -(G*M[j]*dx*r3_inv)
            ay[i] += -(G*M[j]*dy*r3_inv)
            az[i] += -(G*M[j]*dz*r3_inv)
                
            ax[j] += (G*M[i]*dx*r3_inv)
            ay[j] += (G*M[i]*dy*r3_inv)
            az[j] += (G*M[i]*dz*r3_inv)
                
            #jerk 
            adotx[i] += G * M[j] * (dvx * r3_inv - 3 * rv_dot * dx * r5_inv)
            adoty[i] += G * M[j] * (dvy * r3_inv - 3 * rv_dot * dy * r5_inv)
            adotz[i] += G * M[j] * (dvz * r3_inv - 3 * rv_dot * dz * r5_inv)

            adotx[j] += -G * M[i] * (dvx * r3_inv - 3 * rv_dot * dx * r5_inv)
            adoty[j] += -G * M[i] * (dvy * r3_inv - 3 * rv_dot * dy * r5_inv)
            adotz[j] += -G * M[i] * (dvz * r3_inv - 3 * rv_dot * dz * r5_inv)
                
                
    return ax, ay, az, adotx, adoty, adotz 

@jit
def energy(x, y, z, vx, vy, vz, M, G, softening):
    vxcom  = np.sum(vx*M)/np.sum(M)
    vycom  = np.sum(vy*M)/np.sum(M)
    vzcom  = np.sum(vz*M)/np.sum(M)
    vx_temp = vx-vxcom
    vy_temp = vy-vycom
    vz_temp = vz-vzcom
    #kinetic
    KE_individual = 0.5*M*(vx_temp*vx_temp + vy_temp*vy_temp + vz_temp*vz_temp)
    KE_total = np.sum(KE_individual)
    #potential
    PE_individual = np.zeros(N)
    PE_total = 0
    for i in range(N):#current star
        for j in range(i + 1, N):  #sum each pair only once
            dx = x[i]- x[j]
            dy = y[i]- y[j]
            dz = z[i]- z[j]
            r = np.sqrt(dx*dx + dy*dy + dz*dz + softening*softening)
            r_inv = 1/r
            PE = -G*M[j]*M[i]*r_inv
            PE_individual[i] += PE
            PE_individual[j] += PE
    PE_individual *= 0.5
    PE_total = np.sum(PE_individual)

    #total energy per star
    E_individual = KE_individual + PE_individual

    #boolean array, True = bound, False = unbound
    bound = E_individual < 0

    #total energy of the system
    E_total = PE_total + KE_total

    return KE_individual, KE_total, PE_individual, PE_total, E_individual, E_total, bound

def evap_check(bound, R90_initial):
    k_1, k_2, k_3 = 0.66, 1, 0.11
    N_bound = np.sum(bound)
    rho_ref = 0.04
    R90, rho = lagrange90(x, y, z, M, bound)
    print(rho)
    
    E = k_1*(1-N_bound/N)+k_2*(rho_ref/(rho+rho_ref))+k_3*(R90/R90_initial)
    print(f'term 1:{k_1*(1-N_bound/N)}')
    print(f'term 2:{k_2*(rho_ref/(rho+rho_ref))}')
    print(f'term 3:{k_3*(R90/R90_initial)}')

    if E >= 1:
        check_evap = True
    else:
        check_evap = False
        
    return E, R90, rho, check_evap

def lagrange90(x, y, z, M, bound):
    
    x_bound = x[bound]
    y_bound = y[bound]
    z_bound = z[bound]
    M_bound = M[bound]

    #center of mass (COM)
    x_com, y_com, z_com = center_of_mass(x_bound, y_bound, z_bound, M_bound)
        
    #shift bound positions to place COM at the origin
    x_bound -= x_com
    y_bound -= y_com
    z_bound -= z_com
    
    r = np.sqrt(x_bound**2 + y_bound**2 + z_bound**2)


    #sort by radius
    sorted_indices = np.argsort(r)
    r_sorted = r[sorted_indices]
    M_sorted = M_bound[sorted_indices]

    #cumulative mass
    cumulative_mass = np.cumsum(M_sorted)
    total_mass = np.sum(M_sorted)
    
    target_mass = 0.9 * total_mass
    R90 = r_sorted[cumulative_mass >= target_mass][0]

    #density calculations
    r_inside = r <= R90
    M_enc = np.sum(M_bound[r_inside])
    vol = (4/3)*np.pi*R90**3
    
    rho = M_enc / vol
    
    
    return R90, rho

def center_of_mass(x, y, z, M):
    total_mass = np.sum(M)
    x_com = np.sum(M * x) / total_mass
    y_com = np.sum(M * y) / total_mass
    z_com = np.sum(M * z) / total_mass
    return x_com, y_com, z_com


#velocity Verlet integration
def velocity_verlet(x, y, z, vx, vy, vz, ax, ay, az, M, G, dt):
    
    #update positions 
    x = x + vx * dt + 0.5 * ax * dt*dt
    y = y + vy * dt + 0.5 * ay * dt*dt
    z = z + vz * dt + 0.5 * az * dt*dt
    
    #new accelerations at the new positions
    ax_new, ay_new, az_new, adotx_new, adoty_new, adotz_new = acceleration(x, y, z, vx, vy, vz, softening, M, G)
    
    #update velocities 
    vx = vx + 0.5 * (ax + ax_new) * dt
    vy = vy + 0.5 * (ay + ay_new) * dt
    vz = vz + 0.5 * (az + az_new) * dt
    
    return x, y, z, vx, vy, vz, ax_new, ay_new, az_new, adotx_new, adoty_new, adotz_new

@jit
def hermite_4th_order(x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, M, G, dt, eta):
    dt2 = dt * dt
    dt3 = dt * dt * dt 
    
    #predict positions and velocities at t + dt
    x_pred = x + vx * dt + 0.5 * ax * dt2 + (1/6) * adotx * dt3
    y_pred = y + vy * dt + 0.5 * ay * dt2 + (1/6) * adoty * dt3
    z_pred = z + vz * dt + 0.5 * az * dt2 + (1/6) * adotz * dt3

    vx_pred = vx + ax * dt + 0.5 * adotx * dt2
    vy_pred = vy + ay * dt + 0.5 * adoty * dt2
    vz_pred = vz + az * dt + 0.5 * adotz * dt2

    #accelerations and jerks at predicted positions
    ax_pred, ay_pred, az_pred, adotx_pred, adoty_pred, adotz_pred = acceleration(x_pred, y_pred, z_pred, vx_pred, vy_pred, vz_pred, softening, M, G)

    dt_new = timestep(ax, ay, az, adotx, adoty, adotz, ax_pred, ay_pred, az_pred, adotx_pred, adoty_pred, adotz_pred, dt, eta)
    
    #corrected positions and velocities 
    x_corr = x + 0.5 * (vx + vx_pred) * dt + (1/12) * (ax - ax_pred) * dt2
    y_corr = y + 0.5 * (vy + vy_pred) * dt + (1/12) * (ay - ay_pred) * dt2
    z_corr = z + 0.5 * (vz + vz_pred) * dt + (1/12) * (az - az_pred) * dt2

    vx_corr = vx + 0.5 * (ax + ax_pred) * dt + (1/12) * (adotx - adotx_pred) * dt2
    vy_corr = vy + 0.5 * (ay + ay_pred) * dt + (1/12) * (adoty - adoty_pred) * dt2
    vz_corr = vz + 0.5 * (az + az_pred) * dt + (1/12) * (adotz - adotz_pred) * dt2
    
    return x_corr, y_corr, z_corr, vx_corr, vy_corr, vz_corr, ax_pred, ay_pred, az_pred, adotx_pred, adoty_pred, adotz_pred, dt_new

@jit
def timestep(ax, ay, az, adotx, adoty, adotz, ax_pred, ay_pred, az_pred, adotx_pred, adoty_pred, adotz_pred, dt, eta):
   
    #compute second derivative of acceleration (snap)
    a2x = (-6 * (ax - ax_pred) - dt * (4 * adotx + 2 * adotx_pred)) / dt**2
    a2y = (-6 * (ay - ay_pred) - dt * (4 * adoty + 2 * adoty_pred)) / dt**2
    a2z = (-6 * (az - az_pred) - dt * (4 * adotz + 2 * adotz_pred)) / dt**2

    #compute third derivative of acceleration (crackle)
    a3x = (-12 * (ax - ax_pred) - 6 * dt * (adotx + adotx_pred)) / dt**3
    a3y = (-12 * (ay - ay_pred) - 6 * dt * (adoty + adoty_pred)) / dt**3
    a3z = (-12 * (az - az_pred) - 6 * dt * (adotz + adotz_pred)) / dt**3

    #compute magnitudes
    a_mag = np.sqrt(ax_pred*ax_pred + ay_pred*ay_pred + az_pred*az_pred) 
    adot_mag = np.sqrt(adotx_pred*adotx_pred + adoty_pred*adoty_pred + adotz_pred*adotz_pred)  
    a2_mag = np.sqrt(a2x*a2x + a2y*a2y + a2z*a2z)  
    a3_mag = np.sqrt(a3x*a3x + a3y*a3y + a3z*a3z) 

    #compute the full hermite 4th order scheme timestep
    numerator = a_mag * a2_mag + adot_mag*adot_mag
    denominator = adot_mag * a3_mag + a2_mag*a2_mag
    
    dt_new = min(np.sqrt(eta * (numerator / denominator)))

    #if dt_new < dt_min:
    #    dt_new = dt_min
    
    return dt_new


def write_snapshot(snapshot_filename, x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, M, G, softening, time, dt, R90_initial):
    
    #calculate energies at the snapshot time (ensure energy conservation) 
    KE_individual, KE, PE_individual, PE, E_individual, E_total, bound = energy(x, y, z, vx, vy, vz, M, G, softening)
    evap, R90, rho, check_evap = evap_check(bound, R90_initial)

    Q = 2 * KE / abs(PE)  #virial ratio

    #create dataframe with star data
    df = pd.DataFrame({
        "ID": range(1, N+1),
        "Mass": M,
        "x": x, "y": y, "z": z,
        "vx": vx, "vy": vy, "vz": vz,
        "ax": ax, "ay": ay, "az": az,
        "adotx": adotx, "adoty": adoty, "adotz": adotz,
        "Kinetic": KE_individual, "Potential": PE_individual, "Total Energy": E_individual,
        "Bound": bound
    })

    #metadata row for energy
    metadata = pd.DataFrame({
        "ID": ["Total Energy:"],
        "Mass": [E_total], "x": ["Kinetic:"], "y": [KE],
        "z": ["Potential"], "vx": [PE], "vy": ["Virial Ratio"], 
        "vz": [Q], "ax": ["Time"], "ay": [t], "az": ["Timestep, dt"], "adotx": [dt],
        "adoty":[R90], "adotz": [evap], "Kinetic": [rho], "Potential":[check_evap]
    })

    #append metadata to snapshot file
    df = pd.concat([metadata, df], ignore_index=True)

    #save as csv
    df.to_csv(snapshot_filename, index=False)
    
    print(f"Snapshot saved: {snapshot_filename} | Time: {t:.2f} Myrs | total energy: {E_total:.3e} | Q: {Q:.3f}")

def read_snapshot(snapshot_filename):
    
    ICS = pd.read_csv(snapshot_filename)
    #read metadata (energy and time)
    metadata = ICS.iloc[0]
    E_total = float(metadata["Mass"])#total energy in mass column
    KE = float(metadata["y"])
    PE = float(metadata["vx"])
    alpha_virial = float(metadata["vz"])
    time = float(metadata["ay"]) #time in ay column 
    dt_initial = float(metadata["adotx"])
    R90_initial = float(metadata["adoty"])

    #read file data (x,v,a,adot)
    ICS = pd.read_csv(snapshot_filename, skiprows=[1])
    ID = ICS['ID'].values
    M = ICS['Mass'].values
    x = ICS['x'].values
    y = ICS['y'].values
    z = ICS['z'].values
    vx = ICS['vx'].values
    vy = ICS['vy'].values
    vz = ICS['vz'].values
    ax = ICS['ax'].values
    ay = ICS['ay'].values
    az = ICS['az'].values
    adotx = ICS['adotx'].values
    adoty = ICS['adoty'].values
    adotz = ICS['adotz'].values
    KE_ind = ICS['Kinetic'].values
    PE_ind = ICS['Potential'].values
    E_ind = ICS['Total Energy'].values
    bound = ICS['Bound'].values
    
    return ID, M, x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, KE_ind, PE_ind, E_ind, bound, E_total, KE, PE, alpha_virial, time, dt_initial, R90_initial




#read parameter file for constants 
param = pd.read_csv('param_new.csv')
N = param['N'].values[0]
G = param['G'].values[0]
radius = param['radius'].values[0]
softening = param['softening'].values[0]
eta = param['eta'].values[0]
alpha_virial = param['alpha_virial'].values[0]
mass_segregated = param['mass_segregated'].values[0]
t_end = param['t_end'].values[0]
n_snap = param['n_snap'].values[0]
loc = param['loc'].values[0]
seed = param['seed'].values[0]
stem = param['stem'].values[0]
IC_stem = param['IC'].values[0]

rho_ref = 0.04


#seed = np.array(seed.split(),dtype=int)



#run simulations for all directories in the config file (i.e. run simulation 10 times and save snapshots in different files)
for i in range(len(seed)):
    #read the initial conidtions 
    IC_filename = f"IC_files/{IC_stem}_{seed[i]}.csv"

snapshot_filename = f"snapshots/{loc}/SEED{seed[i]}/{stem}_{snapshot_count:04d}.csv"



snap_titles = np.array(['N30_R0.65_VIR0.5_MSEGFALSE_T150MYR','N30_R0.65_VIR0.5_MSEGTRUE_T150MYR','N30_R0.65_VIR1.0_MSEGFALSE_T150MYR',
                    'N30_R0.65_VIR1.0_MSEGTRUE_T150MYR','N30_R0.65_VIR2.0_MSEGFALSE_T150MYR','N30_R0.65_VIR2.0_MSEGTRUE_T150MYR',
                    'N60_R0.82_VIR0.5_MSEGFALSE_T150MYR','N60_R0.82_VIR0.5_MSEGTRUE_T150MYR','N60_R0.82_VIR1.0_MSEGFALSE_T150MYR',
                    'N60_R0.82_VIR1.0_MSEGTRUE_T150MYR','N60_R0.82_VIR2.0_MSEGFALSE_T150MYR','N60_R0.82_VIR2.0_MSEGTRUE_T150MYR',
                    'N120_R1.00_VIR0.5_MSEGFALSE_T150MYR','N120_R1.00_VIR0.5_MSEGTRUE_T150MYR','N120_R1.00_VIR1.0_MSEGFALSE_T150MYR',
                    'N120_R1.00_VIR1.0_MSEGTRUE_T150MYR','N120_R1.00_VIR2.0_MSEGFALSE_T150MYR','N120_R1.00_VIR2.0_MSEGTRUE_T150MYR'])
print(snap_titles)
seed_groups = np.array(['414 240 612 381 735 476 837 433 222 595','140 364 994 241 838 126 894 844 300 275','489 125 748 363 526 96 655 99 253 447',
                        '924 113 567 200 339 101 880 467 619 891','99 982 854 938 271 649 313 620 926 344','878 605 790 321 638 473 87 195 629 180',
                        '318 242 998 706 899 746 864 655 89 257','948 650 549 808 521 2 87 818 180 381','41 967 108 317 701 656 66 998 182 989',
                        '73 818 663 938 536 191 752 980 722 603','514 869 649 482 210 553 487 65 349 464','379 439 107 272 631 895 24 487 730 343',
                        '501 758 797 369 948,191 987 282 19 387','413 376 68 849 462,521 739 773 961 165','319 301 983 73 370,809 484 85 67 168'])
print(seed_groups)
N_groups = [30,30,30,30,30,30,60,60,60,60,60,60,120,120,120,120,120,120]
print(N_groups)
R_groups = [0.65,0.65,0.65,0.65,0.65,0.65,0.82,0.82,0.82,0.82,0.82,0.82,1.00,1.00,1.00,1.00,1.00,1.00]
VIR = [0.5,0.5,1.0,1.0,2.0,2.0,0.5,0.5,1.0,1.0,2.0,2.0,0.5,0.5,1.0,1.0,2.0,2.0]
MSEG = [False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True]
T_groups = [150,150,150,150,150,150,150,150,150,150,150,150,150,150,150,150,150,150]


for i in range()




#MAIN
snap_titles = np.array(['N120_R1.00_VIR0.5_MSEGFALSE_T150MYR','N120_R1.00_VIR0.5_MSEGTRUE_T150MYR','N120_R1.00_VIR1.0_MSEGFALSE_T150MYR',
                    'N120_R1.00_VIR1.0_MSEGTRUE_T150MYR','N120_R1.00_VIR2.0_MSEGFALSE_T150MYR','N120_R1.00_VIR2.0_MSEGTRUE_T150MYR'])

seed_groups = np.array(['501 758 797 369 947','190 987 282 19 387','413 376 68 849 462','520 739 773 961 165','319 301 983 72 370','809 484 85 67 168'])

N_groups = [120,120,120,120,120,120]
R_groups = [1.00,1.00,1.00,1.00,1.00,1.00]
VIR = [0.5,0.5,1.0,1.0,2.0,2.0]
MSEG = [False,True,False,True,False,True]



for j in range(len(snap_titles)):
    loc = snap_titles[j]
    seed = seed_groups[j]
    N = N_groups[j]
    radius = R_groups[j]
    alpha_virial = VIR[j]
    mass_segregated = MSEG[j]
    
    seed = np.array(seed.split(),dtype=int)

    #run simulations for all directories in the config file (i.e. run simulation 10 times and save snapshots in different files)
    for i in range(len(seed)):
        #read the initial conidtions 
        IC_filename = f"IC_files/{IC_stem}_{seed[i]}.csv"
        print(IC_filename)
        ID, M, x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, KE_ind, PE_ind, E_ind, bound, E_total, KE, PE, alpha_virial, t, dt, R90_initial = read_snapshot(IC_filename)
    

        #initialise time variables for loop

        dt_snap = t_end/(n_snap+1)

        t_snap = dt_snap 
        snapshot_count = 0 #problem for continuing from any snapshot 

        while t < t_end:  #main loop from 0s to chosen end point
    
            if t >= t_snap: #create a snapshot at chosen interval
                snapshot_filename = f"snapshots/{loc}/SEED{seed[i]}/{stem}_{snapshot_count:04d}.csv"
                write_snapshot(snapshot_filename, x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, M, G, softening, t, dt, R90_initial)
        
                t_snap += dt_snap
                snapshot_count += 1
    
            #velocity verlet 
            #x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz = velocity_verlet(x, y, z, vx, vy, vz, ax, ay, az, M, G, dt)
    
            #hermite 4th order
            x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, dt = hermite_4th_order(x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, M, G, dt, eta)
    
            t += dt
    


# MAIN SINGLE TEST RUN

#run simulations for all directories in the config file (i.e. run simulation 10 times and save snapshots in different files)
for i in range(1):
    #read the initial conidtions 
    IC_filename = f"IC_files/{IC_stem}_{seed}.csv"
    print(IC_filename)
    ID, M, x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, KE_ind, PE_ind, E_ind, bound, E_total, KE, PE, alpha_virial, t, dt, R90_initial = read_snapshot(IC_filename)
    

    #initialise time variables for loop

    dt_snap = t_end/(n_snap+1)

    t_snap = dt_snap 
    snapshot_count = 0 #problem for continuing from any snapshot 

    while t < t_end:  #main loop from 0s to chosen end point
    
        if t >= t_snap: #create a snapshot at chosen interval
            snapshot_filename = f"snapshots/{loc}/SEED{seed}/{stem}_{snapshot_count:04d}.csv"
            write_snapshot(snapshot_filename, x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, M, G, softening, t, dt, R90_initial)
        
            t_snap += dt_snap
            snapshot_count += 1
    
        #velocity verlet 
        #x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz = velocity_verlet(x, y, z, vx, vy, vz, ax, ay, az, M, G, dt)
    
        #hermite 4th order
        x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, dt = hermite_4th_order(x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, M, G, dt, eta)
    
        t += dt
    



#PLOT CLUSTER USING FILES

#plotting arrays to store trajectories
x_trajectory, y_trajectory, z_trajectory = [], [], []
bound_check = []

#snapshot files
snapshot_files = sorted(glob.glob(f"snapshots/{loc}/SEED880/{stem}*.csv"))

#Loop over each snapshot 
for file in snapshot_files:
    ID, M, x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, KE_ind, PE_ind, E_ind, bound, E_total, KE, PE, alpha_virial, t, dt, R90 = read_snapshot(file)
    df = pd.read_csv(file, skiprows=[1])  #Skip metadata row with energy and time information
    x = df['x'].astype(float).values
    y = df['y'].astype(float).values
    z = df['z'].astype(float).values
    
    #append positions at each snapshot
    x_trajectory.append(x)
    y_trajectory.append(y)
    z_trajectory.append(z)
    bound_check.append(bound)

#transpose 
x_trajectory = np.array(x_trajectory).T
y_trajectory = np.array(y_trajectory).T
z_trajectory = np.array(z_trajectory).T
bound_check = np.array(bound_check).T

#plot in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

#plot each body's trajectory
for i in range(N):
    is_bound = bound_check[i][-1]  
    color = 'blue' if is_bound else 'red'
    ax.plot(x_trajectory[i], y_trajectory[i], z_trajectory[i], color=color)


#add legend for bound/unbound and move it slightly toward the center
bound_patch = plt.Line2D([0], [0], color='blue', label='Bound')
unbound_patch = plt.Line2D([0], [0], color='red', label='Unbound')
ax.legend(handles=[bound_patch, unbound_patch], loc='upper left', bbox_to_anchor=(0.15, 0.8))


#labels and title
ax.set_title(f"{N}-Body Orbit Simulation ({t_end:.1f} Myrs)")
ax.set_xlabel("X (PC)")
ax.set_ylabel("Y (PC)")
ax.set_zlabel("Z (PC)")

#aspect ratio
ax.set_box_aspect([1, 1, 1])

#zoom limits
zoom_factor = 6*radius  #adjust this if needed
ax.set_xlim([-zoom_factor, zoom_factor])
ax.set_ylim([-zoom_factor, zoom_factor])
ax.set_zlim([-zoom_factor, zoom_factor])

plt.tight_layout()
plt.show()




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import glob
from mpl_toolkits.mplot3d import Axes3D  #3D projection

x_trajectory, y_trajectory, z_trajectory = [], [], []
bound_check = []


snapshot_files = sorted(glob.glob(f"snapshots/{loc}/SEED880/{stem}*.csv"))

#Load all snapshots 
for file in snapshot_files:
    df = pd.read_csv(file, skiprows=[1])
    x_trajectory.append(df['x'].astype(float).values)
    y_trajectory.append(df['y'].astype(float).values)
    z_trajectory.append(df['z'].astype(float).values)
    bound_check.append(df['Bound'].astype(bool).values)

x_trajectory = np.array(x_trajectory).T  #shape: (N, time)
y_trajectory = np.array(y_trajectory).T
z_trajectory = np.array(z_trajectory).T
bound_check = np.array(bound_check).T


zoom_factor = 6 * radius

#Set up plot 
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim([-zoom_factor, zoom_factor])
ax.set_ylim([-zoom_factor, zoom_factor])
ax.set_zlim([-zoom_factor, zoom_factor])

ax.set_xlabel("X (PC)")
ax.set_ylabel("Y (PC)")
ax.set_zlabel("Z (PC)")
ax.set_title(f"{N}-Body Orbit Simulation ({t_end:.1f} Myrs)")
ax.set_box_aspect([1, 1, 1])

#legend
bound_patch = plt.Line2D([0], [0], color='blue', label='Bound')
unbound_patch = plt.Line2D([0], [0], color='red', label='Unbound')
ax.legend(handles=[bound_patch, unbound_patch], loc='upper left', bbox_to_anchor=(0.15, 0.8))

#initialize lines 
lines = []
for i in range(N):
    color = 'blue' if bound_check[i, -1] else 'red'
    line, = ax.plot([], [], [], color=color, lw=1)
    lines.append(line)

#Animation update function 
def update(frame):
    for i, line in enumerate(lines):
        line.set_data(x_trajectory[i, :frame], y_trajectory[i, :frame])
        line.set_3d_properties(z_trajectory[i, :frame])


    return lines

#create animation 
anim = FuncAnimation(fig, update, frames=len(snapshot_files), interval=50, blit=False)

#Save as GIF 
anim.save("nbody_trajectories.gif", writer=PillowWriter(fps=15))

plt.close()



#PLOT CLUSTER USING FILES

#snapshot files
seed_dirs = sorted(glob.glob(f"snapshots/{loc}/SEED*"))

fig = plt.figure(figsize=(12, 24))
rows = 5
cols = 2

for i, seed_dir in enumerate(seed_dirs):

    snapshot_files = sorted(glob.glob(f"{seed_dir}/{stem}*.csv"))
    
    #plotting arrays to store trajectories
    x_trajectory, y_trajectory, z_trajectory = [], [], []
    bound_check = []


    #Loop over each snapshot 
    for file in snapshot_files:
        ID, M, x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, KE_ind, PE_ind, E_ind, bound, E_total, KE, PE, alpha_virial, time, dt_initial, R90_initial = read_snapshot(file)
        df = pd.read_csv(file, skiprows=[1])  #Skip metadata row with energy and time information
        x = df['x'].astype(float).values
        y = df['y'].astype(float).values
        z = df['z'].astype(float).values
    
        #append positions at each snapshot
        x_trajectory.append(x)
        y_trajectory.append(y)
        z_trajectory.append(z)
        bound_check.append(bound)

    #transpose 
    x_trajectory = np.array(x_trajectory).T
    y_trajectory = np.array(y_trajectory).T
    z_trajectory = np.array(z_trajectory).T
    bound_check = np.array(bound_check).T

    #create subplot
    ax = fig.add_subplot(rows, cols, i + 1, projection='3d')


    #plot each body's trajectory
    for i in range(N):
        is_bound = bound_check[i][-1]  
        color = 'blue' if is_bound else 'red'
        ax.plot(x_trajectory[i], y_trajectory[i], z_trajectory[i], color=color, label=f"Star {i+1}'s Orbit")


    #labels and title
    ax.set_title(f"{seed_dir.split('/')[-1]}", fontsize=10)
    ax.set_xlabel("X (PC)")
    ax.set_ylabel("Y (PC)")
    ax.set_zlabel("Z (PC)")

    #aspect ratio
    ax.set_box_aspect([1, 1, 1])

    #zoom limits
    zoom_factor = 5*radius  # Adjust this if needed
    ax.set_xlim([-zoom_factor, zoom_factor])
    ax.set_ylim([-zoom_factor, zoom_factor])
    ax.set_zlim([-zoom_factor, zoom_factor])


plt.suptitle(f"Trajectories from 10 SEED Simulations", fontsize=16)
plt.subplots_adjust(wspace=0.01, hspace=0.15, top=0.95, bottom=0.03)
plt.show()



#PLOT CLUSTER USING FILES

#snapshot files
seed_dirs = sorted(glob.glob(f"snapshots/{loc}/SEED880"))

fig = plt.figure(figsize=(12, 24))
rows = 5
cols = 2


snapshot_files = sorted(glob.glob(f"{88}/{stem}*.csv"))
    
#plotting arrays to store trajectories
x_trajectory, y_trajectory, z_trajectory = [], [], []
bound_check = []


#Loop each snapshot 
for file in snapshot_files:
    ID, M, x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, KE_ind, PE_ind, E_ind, bound, E_total, KE, PE, alpha_virial, time, dt_initial, R90_initial = read_snapshot(file)
    df = pd.read_csv(file, skiprows=[1])  #Skip metadata row with energy and time information
    x = df['x'].astype(float).values
    y = df['y'].astype(float).values
    z = df['z'].astype(float).values
    
    #append positions at each snapshot
    x_trajectory.append(x)
    y_trajectory.append(y)
    z_trajectory.append(z)
    bound_check.append(bound)

#transpose 
x_trajectory = np.array(x_trajectory).T
y_trajectory = np.array(y_trajectory).T
z_trajectory = np.array(z_trajectory).T
bound_check = np.array(bound_check).T

#create subplot
ax = fig.add_subplot(rows, cols, i + 1, projection='3d')


#plot each body's trajectory
for i in range(N):
    is_bound = bound_check[i][-1]  
    color = 'blue' if is_bound else 'red'
    ax.plot(x_trajectory[i], y_trajectory[i], z_trajectory[i], color=color, label=f"Star {i+1}'s Orbit")


#labels and title
ax.set_title(f"{seed_dir.split('/')[-1]}", fontsize=10)
ax.set_xlabel("X (PC)")
ax.set_ylabel("Y (PC)")
ax.set_zlabel("Z (PC)")

#aspect ratio
ax.set_box_aspect([1, 1, 1])

#zoom limits
zoom_factor = 5*radius  # Adjust this if needed
ax.set_xlim([-zoom_factor, zoom_factor])
ax.set_ylim([-zoom_factor, zoom_factor])
ax.set_zlim([-zoom_factor, zoom_factor])


plt.suptitle(f"Trajectories from 10 SEED Simulations", fontsize=16)
plt.subplots_adjust(wspace=0.01, hspace=0.15, top=0.95, bottom=0.03)
plt.show()



#ENERGY PLOTS
 

times = []
total_energies = []

for file in snapshot_files:
    df = pd.read_csv(file)

    #read metadata
    metadata = df.iloc[0]
    time = float(metadata["ay"]) #time in the ay column
    total_energy = float(metadata["Mass"])  #total energy in mass column

    times.append(time)
    total_energies.append(total_energy)

E0 = total_energies[0]  #initial total energy

fractional_energy_change = [(E - E0) / abs(E0) for E in total_energies]

#plot fractional energy change over time
plt.figure(figsize=(8, 5))
#plt.scatter(times, total_energies, marker='o', color='black', label="Fractional Energy Change")
plt.scatter(times, fractional_energy_change, marker='o', color='black', label="Fractional Energy Change")
plt.xlabel("Time (Myr)")
plt.ylabel("ΔE / E0")
plt.title("Fractional Energy Change Over Time")
plt.axhline(0, color='r', linestyle='--', linewidth=2)  #reference line at 0
plt.grid(True)
#plt.ylim(-1.2e-6,0.2e-6)
plt.legend()
plt.show()



#set up subplots
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12, 20))
axes = axes.flatten()

for i, seed_dir in enumerate(seed_dirs):
    snapshot_files = sorted(glob.glob(f"{seed_dir}/{stem}*.csv"))

    times = []
    total_energies = []

    for file in snapshot_files:
        df = pd.read_csv(file)

        metadata = df.iloc[0]
        time = float(metadata["ay"])           #time in ay column
        total_energy = float(metadata["Mass"]) #energy in Mass column

        times.append(time)
        total_energies.append(total_energy)

    E0 = total_energies[0]
    fractional_change = [(E - E0) / abs(E0) for E in total_energies]

    ax = axes[i]
    ax.plot(times, fractional_change, 'o', color='black', markersize=3)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_title(f"{seed_dir.split('/')[-1]}", fontsize=10)
    ax.set_xlabel("Time (Myr)")
    ax.set_ylabel("ΔE / E0")
    ax.grid(True)

#adjust layout
plt.suptitle("Fractional Energy Change Over Time", fontsize=16)
plt.subplots_adjust(wspace=0.3, hspace=0.4, top=0.94, bottom=0.04)
plt.show()



#plot kinetic energy
times = []
kinetic_energies = []
potential_energies = []

for file in snapshot_files:
    df = pd.read_csv(file)

    #read metadata
    metadata = df.iloc[0]
    time = float(metadata["ay"]) #time in the ay column
    kinetic_energy = float(metadata["y"])  #kinetic energy 
    potential_energy = float(metadata["vx"])  #potential energy 
    times.append(time)
    kinetic_energies.append(kinetic_energy)
    potential_energies.append(potential_energy)
    
KE0 = kinetic_energies[0]  #initial kinetic energy
frac_kinetic_energy_change = [(KE - KE0) / abs(KE0) for KE in kinetic_energies]
PE0 = potential_energies[0]  #initial potenial energy
frac_potential_energy_change = [(PE - PE0) / abs(PE0) for PE in potential_energies]

#plot fractional energy change over time
plt.figure(figsize=(8, 5))
#plt.scatter(times, total_energies, marker='o', color='black', label="Fractional Energy Change")
plt.scatter(times, frac_potential_energy_change, marker='o', color='black', label="Potential Energy Change")
plt.scatter(times, frac_kinetic_energy_change, marker='o', color='blue', label="Kinetic Energy Change")
plt.xlabel("Time (Years)")
plt.ylabel("ΔE / E0")
plt.title("Fractional Kinetic Energy Change Over Time")
plt.axhline(0, color='r', linestyle='--', linewidth=2)  #reference line at 0
plt.grid(True)
plt.legend()
plt.show()


#plot timestep over time 
times = []
kinetic_energies = []
timesteps = []

for file in snapshot_files:
    df = pd.read_csv(file)

    #read metadata
    metadata = df.iloc[0]
    time = float(metadata["ay"])  #time in the ay column
    timestep = float(metadata["adotx"]) #dt
    times.append(time)
    timesteps.append(timestep)
    

#plot fractional energy change over time
plt.figure(figsize=(8, 5))
plt.scatter(times, timesteps, marker='o', color='black', label="timesteps")
plt.xlabel("Time (Myr)")
plt.ylabel("dt (Myr)")
plt.title("Timestep tracking")
plt.axhline(0, color='r', linestyle='--', linewidth=2)  #reference line at 0
plt.grid(True)
plt.legend()
plt.show()



#set up subplots
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12, 20))
axes = axes.flatten()

for i, seed_dir in enumerate(seed_dirs):
    snapshot_files = sorted(glob.glob(f"{seed_dir}/{stem}*.csv"))

    times = []
    timesteps = []

    for file in snapshot_files:
        df = pd.read_csv(file)
        metadata = df.iloc[0]

        time = float(metadata["ay"])       #time in 'ay' column
        timestep = float(metadata["adotx"])  #dt in 'adotx' column

        times.append(time)
        timesteps.append(timestep)

    ax = axes[i]
    ax.plot(times, timesteps, 'o', color='black', markersize=3)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_title(f"{seed_dir.split('/')[-1]}", fontsize=10)
    ax.set_xlabel("Time (Myr)")
    ax.set_ylabel("dt (Myr)")
    ax.grid(True)

#adjust layout
plt.suptitle("Timestep Over Time", fontsize=16)
plt.subplots_adjust(wspace=0.3, hspace=0.4, top=0.94, bottom=0.04)
plt.show()
