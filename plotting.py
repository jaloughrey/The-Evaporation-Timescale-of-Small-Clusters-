import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob 

#################################################################################

#PLOT CLUSTER STATIC

#plotting arrays to store trajectories
x_trajectory, y_trajectory, z_trajectory = [], [], []
bound_check = []

#load snapshot files
snapshot_files = sorted(glob.glob(f"snapshots/{loc}/SEED_{seed}/{stem}*.csv"))

#loop over each snapshot and extract  
for file in snapshot_files:
    ID, M, x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, KE_ind, PE_ind, E_ind, bound, E_total, KE, PE, alpha_virial, t, dt, R90 = read_snapshot(file)
    df = pd.read_csv(file, skiprows=[1])  #Skip energydata row with energy and time information
    x = df['x'].astype(float).values
    y = df['y'].astype(float).values
    z = df['z'].astype(float).values
    
    #append positions at each snapshot  to array
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


#add legend for bound/unbound stars
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
zoom_factor = 6*radius  # Adjust this if needed
ax.set_xlim([-zoom_factor, zoom_factor])
ax.set_ylim([-zoom_factor, zoom_factor])
ax.set_zlim([-zoom_factor, zoom_factor])

plt.tight_layout()
plt.show()

#################################################################################

#PLOT CLUSTER ANIMATION

from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  #needed for 3D projection

#plotting arrays to store trajectories
x_trajectory, y_trajectory, z_trajectory = [], [], []
bound_check = []

#load snapshot files
snapshot_files = sorted(glob.glob(f"snapshots/{loc}/SEED880/{stem}*.csv"))

#load snapshots and extract position data
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

#plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

#adjust zoom factor if needed
zoom_factor = 6 * radius
ax.set_xlim([-zoom_factor, zoom_factor])
ax.set_ylim([-zoom_factor, zoom_factor])
ax.set_zlim([-zoom_factor, zoom_factor])

#formating
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

#animation update function
def update(frame):
    for i, line in enumerate(lines):
        line.set_data(x_trajectory[i, :frame], y_trajectory[i, :frame])
        line.set_3d_properties(z_trajectory[i, :frame])

    return lines

#create animation
anim = FuncAnimation(fig, update, frames=len(snapshot_files), interval=50, blit=False)

#save as GIF 
anim.save("nbody_trajectories.gif", writer=PillowWriter(fps=15))
plt.close()

#################################################################################

#ENERGY PLOTTING
 
#energy change over time (check for energy conservation)
times = []
total_energies = []

for file in snapshot_files:
    df = pd.read_csv(file)

    #read energydata
    energydata = df.iloc[0]
    time = float(energydata["ay"]) #time in the ay column
    total_energy = float(energydata["Mass"])  #total energy in mass column

    times.append(time)
    total_energies.append(total_energy)

E0 = total_energies[0]  #initial total energy

fractional_energy_change = [(E - E0) / abs(E0) for E in total_energies] #fractional energy change

#plot fractional energy change over time
plt.figure(figsize=(8, 5))
plt.scatter(times, fractional_energy_change, marker='o', color='black', label="Fractional Energy Change")
plt.xlabel("Time (Myr)")
plt.ylabel("ΔE / E0")
plt.title("Fractional Energy Change Over Time")
plt.axhline(0, color='r', linestyle='--', linewidth=2)  #reference line at 0
plt.grid(True)
plt.legend()
plt.show()

###################################################################################

#plot kinetic energy
times = []
kinetic_energies = []
potential_energies = []

for file in snapshot_files:
    df = pd.read_csv(file)

    #read energydata
    energydata = df.iloc[0]
    time = float(energydata["ay"]) #time in the ay column
    kinetic_energy = float(energydata["y"])  #kinetic energy 
    potential_energy = float(energydata["vx"])  #potential energy 
    times.append(time)
    kinetic_energies.append(kinetic_energy)
    potential_energies.append(potential_energy)
    
KE0 = kinetic_energies[0]  #initial kinetic energy
frac_kinetic_energy_change = [(KE - KE0) / abs(KE0) for KE in kinetic_energies]
PE0 = potential_energies[0]  #initial potenial energy
frac_potential_energy_change = [(PE - PE0) / abs(PE0) for PE in potential_energies]

#plot fractional energy change over time
plt.figure(figsize=(8, 5))
plt.scatter(times, frac_potential_energy_change, marker='o', color='black', label="Potential Energy Change")
plt.scatter(times, frac_kinetic_energy_change, marker='o', color='blue', label="Kinetic Energy Change")
plt.xlabel("Time (Years)")
plt.ylabel("ΔE / E0")
plt.title("Fractional Kinetic Energy Change Over Time")
plt.axhline(0, color='r', linestyle='--', linewidth=2)  #reference line at 0
plt.grid(True)
plt.legend()

#################################################################################

#plot timestep over time (timestep check)
times = []
kinetic_energies = []
timesteps = []

for file in snapshot_files:
    df = pd.read_csv(file)

    #read energydata
    energydata = df.iloc[0]
    time = float(energydata["ay"])  #time in the ay column
    timestep = float(energydata["adotx"]) #dt
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
plt.show()

#################################################################################
