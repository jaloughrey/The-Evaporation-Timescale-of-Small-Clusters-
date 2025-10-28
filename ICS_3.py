import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob 


#FUNCTIONS:

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
    
  
def center_of_mass(x, y, z, M):
    total_mass = np.sum(M)
    x_com = np.sum(M * x) / total_mass
    y_com = np.sum(M * y) / total_mass
    z_com = np.sum(M * z) / total_mass
    return x_com, y_com, z_com


def chabrier_log_normal(N):
    m_char = 0.079  #characteristic mass in solar masses
    sigma = 0.69    #log-normal dispersion
    
    #sample from log-normal distribution
    mass_dist = np.random.normal(loc=np.log10(m_char), scale=sigma, size=N)
    masses = 10**mass_dist
    
    return masses

def chabrier_log_normal_scaled(N, m_min=0.01, m_max=1.0):
    m_char = 0.079
    sigma = 0.69

    # Step 1: Draw N log-normal masses within physical bounds
    masses = []
    while len(masses) < N:
        samples = 10**np.random.normal(loc=np.log10(m_char), scale=sigma, size=N)
        valid = samples[(samples >= m_min) & (samples <= m_max)]
        masses.extend(valid[:N - len(masses)])

    masses = np.array(masses)

    
    return masses


def write_snapshot(snapshot_filename, x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, M, G, softening, time, dt,R90):
    
    #calculate energies at the snapshot time (ensure energy conservation) 
    KE_individual, KE, PE_individual, PE, E_individual, E_total, bound = energy(x, y, z, vx, vy, vz, M, G, softening)
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
        "adoty":[R90]
        
    })

    #append metadata to snapshot file
    df = pd.concat([metadata, df], ignore_index=True)

    #save as csv
    df.to_csv(snapshot_filename, index=False)
    
    print(f"Snapshot saved: {snapshot_filename} | Time: {t:.2f} years | total energy: {E_total:.3e} | Q: {Q:.3f}")



#read parameter file 
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

#seed = np.array(seed.split(),dtype=int)



for i in range(len(seed)):
    
    #set random seed 
    np.random.seed(seed[i])  

    #M = chabrier_log_normal(N)
    M = chabrier_log_normal_scaled(N) 

    std_dev = radius / 3

    if mass_segregated == True:
        x, y, z = mass_segregated_positions(N, M, base_sigma=std_dev, k=0.2, segregation_fraction=0.2)
    else:
        #initialize positions (normal distribution centered at the origin)
        x = np.random.normal(0, std_dev, N)  
        y = np.random.normal(0, std_dev, N)  
        z = np.random.normal(0, std_dev, N)  

    #center of mass (COM)
    x_com, y_com, z_com = center_of_mass(x, y, z, M)

    #shift positions to place COM at the origin
    x -= x_com
    y -= y_com
    z -= z_com

    r = np.sqrt(x**2 + y**2 + z**2)


    #initial lagrange radius
    #sort by radius
    sorted_indices = np.argsort(r)
    r_sorted = r[sorted_indices]
    M_sorted = M[sorted_indices]

    #cumulative mass
    cumulative_mass = np.cumsum(M_sorted)
    total_mass = np.sum(M_sorted)
    
    target_mass = 0.9 * total_mass
    R90 = r_sorted[cumulative_mass >= target_mass][0]
    print(R90)
    R_max = r_sorted[cumulative_mass >= target_mass][-1]
    print(R_max)
    
    #Step 2: Scale masses to match target density
    
    #density calculations to scale total mass
    M_enc = np.sum(M)
    
    vol = (4/3)*np.pi*R_max**3
    rho_sample = M_enc / vol
    rho_target = 10

    scale_factor = rho_target / rho_sample
    M *= scale_factor
    print(M)


    #random velocities (in m/s)
    vx = np.random.uniform(-30000, 30000, N)  
    vy = np.random.uniform(-30000, 30000, N)
    vz = np.random.uniform(-30000, 30000, N)

    #initial kinetic and potential energy
    _, KE_initial,_, PE_initial,_, E_total_initial,_ = energy(x, y, z, vx, vy, vz, M, G, softening)

    #scale velocities for desired virial ratio
    scaling_factor = np.sqrt(alpha_virial / (2*KE_initial / abs(PE_initial)))
    vx *= scaling_factor
    vy *= scaling_factor
    vz *= scaling_factor

    #intial virialised kinetic and potential energy
    _, KE,_, PE,_, E_total,_ = energy(x, y, z, vx, vy, vz, M, G, softening)

    #intial acceleration 
    ax, ay, az, adotx, adoty, adotz = acceleration(x, y, z, vx, vy, vz, softening, M, G)

    #start time 
    t=0

    #intial timestep calculation
    a_mag = np.sqrt(ax*ax + ay*ay + az*az)
    adot_mag = np.sqrt(adotx*adotx + adoty*adoty + adotz*adotz)
    dt = eta * min(a_mag/adot_mag)

    #update filename 
    IC_filename = f"IC_files/{IC_stem}_{seed[i]}.csv"
    print(IC_filename)

    #save initial conditions to the csv file
    write_snapshot(IC_filename, x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, M, G, softening, t, dt, R90)



#DEFINE INITIAL CONDITIONS FROM PARAMETER FILE 

for i in range(1):
    
    #set random seed 
    np.random.seed(seed)  

    #M = chabrier_log_normal(N)
    M = chabrier_log_normal_scaled(N) 

    std_dev = radius / 3

    if mass_segregated == True:
        x, y, z = mass_segregated_positions(N, M, base_sigma=std_dev, k=0.2, segregation_fraction=0.2)
    else:
        #initialize positions (normal distribution centered at the origin)
        x = np.random.normal(0, std_dev, N)  
        y = np.random.normal(0, std_dev, N)  
        z = np.random.normal(0, std_dev, N)  

    #center of mass (COM)
    x_com, y_com, z_com = center_of_mass(x, y, z, M)

    #shift positions to place COM at the origin
    x -= x_com
    y -= y_com
    z -= z_com

    r = np.sqrt(x**2 + y**2 + z**2)


    #initial lagrange radius
    #sort by radius
    sorted_indices = np.argsort(r)
    r_sorted = r[sorted_indices]
    M_sorted = M[sorted_indices]

    #cumulative mass
    cumulative_mass = np.cumsum(M_sorted)
    total_mass = np.sum(M_sorted)
    
    target_mass = 0.9 * total_mass
    R90 = r_sorted[cumulative_mass >= target_mass][0]
    print(R90)
    R_max = r_sorted[cumulative_mass >= target_mass][-1]
    print(R_max)
    
    #Step 2: Scale masses to match target density
    
    #density calculations to scale total mass
    M_enc = np.sum(M)
    
    vol = (4/3)*np.pi*R_max**3
    rho_sample = M_enc / vol
    rho_target = 10

    scale_factor = rho_target / rho_sample
    M *= scale_factor
    print(M)


    #random velocities (in m/s)
    vx = np.random.uniform(-30000, 30000, N)  
    vy = np.random.uniform(-30000, 30000, N)
    vz = np.random.uniform(-30000, 30000, N)

    #initial kinetic and potential energy
    _, KE_initial,_, PE_initial,_, E_total_initial,_ = energy(x, y, z, vx, vy, vz, M, G, softening)

    #scale velocities for desired virial ratio
    scaling_factor = np.sqrt(alpha_virial / (2*KE_initial / abs(PE_initial)))
    vx *= scaling_factor
    vy *= scaling_factor
    vz *= scaling_factor

    #intial virialised kinetic and potential energy
    _, KE,_, PE,_, E_total,_ = energy(x, y, z, vx, vy, vz, M, G, softening)

    #intial acceleration 
    ax, ay, az, adotx, adoty, adotz = acceleration(x, y, z, vx, vy, vz, softening, M, G)

    #start time 
    t=0

    #intial timestep calculation
    a_mag = np.sqrt(ax*ax + ay*ay + az*az)
    adot_mag = np.sqrt(adotx*adotx + adoty*adoty + adotz*adotz)
    dt = eta * min(a_mag/adot_mag)

    #update filename 
    IC_filename = f"IC_files/{IC_stem}_{seed}.csv"
    print(IC_filename)

    #save initial conditions to the csv file
    write_snapshot(IC_filename, x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, M, G, softening, t, dt, R90)


for i in range(len(seed)):
    
    #set random seed 
    np.random.seed(seed[i])  

IC_filename = f"IC_files/{IC_stem}_{seed[i]}.csv"




fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

#normalize mass for color mapping
norm = plt.Normalize(np.min(M), np.max(M))
cmap = plt.cm.inferno

#convert to parsec and to solar masses for clearer axis

print(x)
print(np.sum(M))
    
#scatter
sc = ax.scatter(x, y, z, c=M, cmap=cmap, s=40 * (M / np.max(M)), alpha=0.7, edgecolors='k')

#colourbar
cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
cbar.set_label("Stellar Mass (M☉)")

#labels & Formatting
ax.set_xlabel("X (pc)")
ax.set_ylabel("Y (pc)")
ax.set_zlabel("Z (pc)")
ax.view_init(elev=30, azim=45)  #viewing angle

zoom_factor = (radius)
ax.set_xlim([-zoom_factor, zoom_factor])
ax.set_ylim([-zoom_factor, zoom_factor])
ax.set_zlim([-zoom_factor, zoom_factor])

plt.show()




ICS = pd.read_csv(IC_filename)
print(ICS)




