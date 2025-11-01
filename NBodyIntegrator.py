import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob 
from numba import jit 
import functions as func

# MAIN

#read parameter file for constants 
N, G, radius, softening, eta, alpha_virial, mass_segregated, t_end, n_snap, loc, seed, stem, IC_stem = func.read_param('param.csv')

#read the initial conidtions 
IC_filename = f"IC_files/{IC_stem}_{seed}.csv"
print(IC_filename)
ID, M, x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, KE_ind, PE_ind, E_ind, bound, E_total, KE, PE, alpha_virial, t, dt, R90_initial = func.read_snapshot(IC_filename)
    
#initialise time variables for loop
dt_snap = t_end/(n_snap+1) #time between snapshot files to meet desired number (n_snap)
t_snap = dt_snap #first snapshot
snapshot_count = 0 #for snapshot file naming 

#start loop
while t < t_end:  #main loop from 0s to chosen end point
    
    if t >= t_snap: #create a snapshot at chosen interval
        snapshot_filename = f"snapshots/{loc}/SEED{seed}/{stem}_{snapshot_count:04d}.csv"
        func.write_snapshot(snapshot_filename, N, x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, M, G, softening, t, dt, R90_initial)

        #update variables
        t_snap += dt_snap 
        snapshot_count += 1
        
    #Integrators 
    
    #velocity verlet 
    #x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz = velocity_verlet(x, y, z, vx, vy, vz, ax, ay, az, M, G, dt)
    
    #hermite 4th order
    x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, dt = func.hermite_4th_order(N, x, y, z, vx, vy, vz, ax, ay, az, adotx, adoty, adotz, softening, M, G, dt, eta)
    
    t += dt #update current time

#end loop
    
