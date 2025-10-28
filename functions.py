import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob 
from numba import jit 

#FUNCTIONS:

  #CALCULATION FUNCTIONS

    #ACCELERATION FUNCTION
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
  

    #ENERGY FUNCTION
@jit
def energy(x, y, z, vx, vy, vz, M, G, softening):
    #centre of velocity as temporary variable 
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

    #bound check
    bound = E_individual < 0 #boolean array, True = bound, False = unbound

    #total energy of the system
    E_total = PE_total + KE_total

    return KE_individual, KE_total, PE_individual, PE_total, E_individual, E_total, bound

    #CENTRE OF MASS 
def center_of_mass(x, y, z, M):
    total_mass = np.sum(M)
    x_com = np.sum(M * x) / total_mass
    y_com = np.sum(M * y) / total_mass
    z_com = np.sum(M * z) / total_mass
    return x_com, y_com, z_com


  #INITIAL CONDITIONS FUNCTIONS 

    #DEFINING INITIAL POSITIONS WITH MASS SEGREGATION
def mass_segregated_positions(N, M, base_sigma, k, segregation_fraction):

    #threshold mass for segregation (top 20% of masses)
    threshold_index = int((1 - segregation_fraction) * N)  
    mass_threshold = np.sort(M)[threshold_index]  

    #separate stars into segregated (top 20%) and non-segregated (bottom 80%)
    segregated = M >= mass_threshold   

    #compute width of gaussian to sample from 
    M_min = np.min(M) 
    sigmas = np.full(N, base_sigma)  #default sigma = radius of cluster
    sigmas[segregated] = base_sigma * (M_min / M[segregated]) ** k  #adjusted sigma for top 20% of masses

    #sample positions using variable value of sigma
    x = np.random.normal(0, sigmas, N)
    y = np.random.normal(0, sigmas, N)
    z = np.random.normal(0, sigmas, N)

    return x, y, z
    
    #DEFINING INITIAL MASSES

      #CHABRIER 1
def chabrier_log_normal(N):
    m_char = 0.079  #characteristic mass in solar masses
    sigma = 0.69    #log-normal dispersion
    
    #sample from log-normal distribution
    mass_dist = np.random.normal(loc=np.log10(m_char), scale=sigma, size=N)
    masses = 10**mass_dist
    
    return masses

      #CHABRIER 2
def chabrier_log_normal_scaled(N, m_min=0.01, m_max=1.0):
    m_char = 0.079
    sigma = 0.69

    #draw N log-normal masses within physical bounds
    masses = []
    while len(masses) < N:
        samples = 10**np.random.normal(loc=np.log10(m_char), scale=sigma, size=N)
        valid = samples[(samples >= m_min) & (samples <= m_max)]
        masses.extend(valid[:N - len(masses)])

    masses = np.array(masses)

    return masses

  #CHECK/ANALYSIS FUNCTIONS

    #EVAPORATION INDEX CALCULATION AND DEFINING EVAPORATED POINT OF CLUSTER
def evap_check(bound, R90_initial):
    k_1, k_2, k_3 = 0.66, 1, 0.11 #weighting constants
    N_bound = np.sum(bound) #number of bound stars 
    rho_ref = 0.04 #refrance density
    R90, rho = lagrange90(x, y, z, M, bound)#current lagrange and density 
    print(rho)
    
    E = k_1*(1-N_bound/N)+k_2*(rho_ref/(rho+rho_ref))+k_3*(R90/R90_initial)#calculation of index
    print(f'term 1:{k_1*(1-N_bound/N)}')
    print(f'term 2:{k_2*(rho_ref/(rho+rho_ref))}')
    print(f'term 3:{k_3*(R90/R90_initial)}')

    if E >= 1:
        check_evap = True
    else:
        check_evap = False
        
    return E, R90, rho, check_evap

    #90% LAGRANGIAN RADIUS CALCULATIONS 
def lagrange90(x, y, z, M, bound):

    #consider bound stars only
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
  
    #80% mass radius calculation
    target_mass = 0.9 * total_mass
    R90 = r_sorted[cumulative_mass >= target_mass][0]

    #density calculations
    r_inside = r <= R90
    M_enc = np.sum(M_bound[r_inside])
    vol = (4/3)*np.pi*R90**3
    rho = M_enc / vol
    
    return R90, rho

  #INTEGRATORS 

    #VELOCITY VERLET 
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

    #HERMITE 4TH ORDER INTEGRATION 
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
  
    #TIMESTEP FOR THE HERMITE SCHEME
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
    
    return dt_new


  #READ WRITE FILE FUNCTIONS 

    #WRITE SNAPSHOT
def write_snapshot(snapshot_filename, x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, M, G, softening, time, dt, R90_initial):
    
    #calculate energies and evaporation at the snapshot time (ensure energy conservation) 
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

    #energydata row for energy
    energydata = pd.DataFrame({
        "ID": ["Total Energy:"],
        "Mass": [E_total], "x": ["Kinetic:"], "y": [KE],
        "z": ["Potential"], "vx": [PE], "vy": ["Virial Ratio"], 
        "vz": [Q], "ax": ["Time"], "ay": [t], "az": ["Timestep, dt"], "adotx": [dt],
        "adoty":[R90], "adotz": [evap], "Kinetic": [rho], "Potential":[check_evap]
    })

    #append energydata to snapshot file
    df = pd.concat([energydata, df], ignore_index=True)

    #save as csv
    df.to_csv(snapshot_filename, index=False)
    
    print(f"Snapshot saved: {snapshot_filename} | Time: {t:.2f} Myrs | total energy: {E_total:.3e} | Q: {Q:.3f}")


    #READ SNAPSHOT
def read_snapshot(snapshot_filename):
  
    file = pd.read_csv(snapshot_filename)
    #read energy data (energy and time)
    energydata = file.iloc[0] #locate energy data
    E_total = float(energydata["Mass"])#total energy in mass column
    KE = float(energydata["y"])
    PE = float(energydata["vx"])
    alpha_virial = float(energydata["vz"])
    time = float(energydata["ay"])
    dt_initial = float(energydata["adotx"])
    R90_initial = float(energydata["adoty"])

    #read file data (x,v,a,adot)
    file = pd.read_csv(snapshot_filename, skiprows=[1])
    ID = file['ID'].values
    M = file['Mass'].values
    x = file['x'].values
    y = file['y'].values
    z = file['z'].values
    vx = file['vx'].values
    vy = file['vy'].values
    vz = file['vz'].values
    ax = file['ax'].values
    ay = file['ay'].values
    az = file['az'].values
    adotx = file['adotx'].values
    adoty = file['adoty'].values
    adotz = file['adotz'].values
    KE_ind = file['Kinetic'].values
    PE_ind = file['Potential'].values
    E_ind = file['Total Energy'].values
    bound = file['Bound'].values
    
    return ID, M, x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, KE_ind, PE_ind, E_ind, bound, E_total, KE, PE, alpha_virial, time, dt_initial, R90_initial
