# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:17:36 2019

@author: foersterronny
"""
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt # Libraries for plotting

# In[]
def GenerateRandomWalk(diameter, num_particles, frames, frames_per_second, ep = 0, mass = 1, microns_per_pixel = 0.477, temp_water = 295, visc_water = 9.5e-16):
    # Generating particle tracks as comparison
    
    # diameter of particle in nm
    #frames length of track of simulated particles
    #num_particles amount of simulated particles
    
    const_Boltz = 1.38e-23

    
    sim_part_diff = (2*const_Boltz*temp_water/(6*math.pi *visc_water)*1e9) /diameter 
    # [mum^2/s] x-diffusivity of simulated particle 
    sim_part_sigma_x=np.sqrt(2*sim_part_diff / frames_per_second)/microns_per_pixel 
    # [pixel/frame] st.deviation for simulated particle's x-movement
    
    # Generating list to hold frames:
    sim_part_frame=[]
    for sim_frame in range(frames):
        sim_part_frame.append(sim_frame)
    sim_part_frame_list=sim_part_frame*num_particles
    
    # generating list to hold particle and
    # generating list to hold its x-position, coming from a Gaussian-distribution
    sim_part_part=[]
    sim_part_x=[]

    
    for sim_part in range(num_particles):
        for sim_frame in range(frames):
            sim_part_part.append(sim_part)
            sim_part_x.append(np.random.normal(loc=0,scale=sim_part_sigma_x)) #sigma given by sim_part_sigma as above
            # Is that possibly wrong??
    
    # Putting the results into a df and formatting correctly:
    sim_part_tm=pd.DataFrame({'x':sim_part_x, 'y':0, 'mass':mass, 'ep': 0, \
                              'frame':sim_part_frame_list, 'particle':sim_part_part})
    sim_part_tm.x=sim_part_tm.groupby('particle').x.cumsum()
    sim_part_tm.index=sim_part_tm.frame
    
    # check if tm is gaussian distributed
    my_mean = []
    my_var = []
    for sim_frame in range(frames):
        mycheck = sim_part_tm[sim_part_tm.frame == sim_frame].x.values
        my_mean.append(np.mean(mycheck))
        my_var.append(np.var(mycheck))
        
#    plt.plot(my_mean)
#    plt.plot(my_var)
    
    return sim_part_tm
