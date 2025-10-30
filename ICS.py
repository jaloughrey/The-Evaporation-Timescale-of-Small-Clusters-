import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob 


#read parameter file for constants 
#N: number of stars
#G: gravitational constant ()
#radius: desired cluster radius (pc)
#softening: buffer to avoid singularties ()
#eta: accuracy parameter ()
#alpha_virial: virial ratio (1.0 = equilibrium)
#mass_segregated: boolean value to toggle mass segregation in clusters
#t_end: desired end time (Myr)
#n_snap: desired number of snapshot files
#loc: ensemble directory
#seed: desired random seed
#stem: snapshot file naming
#IC_stem: Initial condition file naming 
N, G, radius, softening, eta, alpha_virial, mass_segregated, t_end, n_snap, loc, seed, stem, IC_stem = read_param('param.csv')


#DEFINE INITIAL CLUSTER CONDITIONS FROM PARAMETERS
    
#set random seed 
np.random.seed(seed)  

#define masses based on chabrier distribution
#M = chabrier_log_normal(N)
M = chabrier_log_normal_scaled(N) 

#approximation of standard deviation
std_dev = radius / 3

#define positions of stars 
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


#initial lagrange radius (90% enclosed radius)
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
    
#scale masses to match target density
    
#density calculations 
M_enc = np.sum(M)
    
vol = (4/3)*np.pi*R_max**3
rho_sample = M_enc / vol
rho_target = 10

scale_factor = rho_target / rho_sample
M *= scale_factor

#Define initial velocities based on virial ratio

#start with random velocities (in m/s)
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

#save initial conditions to the csv file
write_snapshot(IC_filename, x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, M, G, softening, t, dt, R90)


#Plot the intial cluster
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

#normalize mass for color mapping
norm = plt.Normalize(np.min(M), np.max(M))
cmap = plt.cm.inferno

#convert to parsec and to solar masses for clear axis

print(x)
print(np.sum(M))
    
#scatter
sc = ax.scatter(x, y, z, c=M, cmap=cmap, s=40 * (M / np.max(M)), alpha=0.7, edgecolors='k')

#colourbar
cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
cbar.set_label("Stellar Mass (M☉)")

#labels & formatting
ax.set_xlabel("X (pc)")
ax.set_ylabel("Y (pc)")
ax.set_zlabel("Z (pc)")
ax.view_init(elev=30, azim=45)  #viewing angle

#adjust zoom factor if needed
zoom_factor = (radius)
ax.set_xlim([-zoom_factor, zoom_factor])
ax.set_ylim([-zoom_factor, zoom_factor])
ax.set_zlim([-zoom_factor, zoom_factor])

plt.show()
