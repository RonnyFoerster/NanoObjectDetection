# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:17:36 2019

@author: Ronny Förster und Stefan Weidlich
"""
import math
import numpy as np
import pandas as pd
import matplotlib
# False since deprecated from version 3.0
#matplotlib.rcParams['text.usetex'] = False
#matplotlib.rcParams['text.latex.unicode'] = False

import sys
#import matplotlib.pyplot as plt # Libraries for plotting
import NanoObjectDetection as nd

from pdb import set_trace as bp #debugger


# In[]
def PrepareRandomWalk(ParameterJsonFile):
    """
    Configure the parameters for a randowm walk out of a JSON file
    """
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)    
    
    diameter            = settings["Simulation"]["DiameterOfParticles"]
    num_particles       = settings["Simulation"]["NumberOfParticles"]
    frames              = settings["Simulation"]["NumberOfFrames"]
    RatioDroppedFrames  = settings["Simulation"]["RatioDroppedFrames"]
    EstimationPrecision = settings["Simulation"]["EstimationPrecision"]
    mass                = settings["Simulation"]["mass"]
    
    
    frames_per_second   = settings["Exp"]["fps"]
    microns_per_pixel   = settings["Exp"]["Microns_per_pixel"]
    temp_water          = settings["Exp"]["Temperature"]


    solvent = settings["Exp"]["solvent"]
    
    if settings["Exp"]["Viscocity_auto"] == 1:
        visc_water = nd.handle_data.GetViscocity(temperature = temp_water, solvent = solvent)
        bp()
    else:
        visc_water = settings["Exp"]["Viscocity"]

    
    output = GenerateRandomWalk(diameter, num_particles, frames, frames_per_second, \
                                              RatioDroppedFrames = RatioDroppedFrames, \
                                              ep = EstimationPrecision, mass = mass, \
                                              microns_per_pixel = microns_per_pixel, temp_water = temp_water, \
                                              visc_water = visc_water)
       
    if 1 == 0:
        #check if buoyancy shall be considered
        DoBuoyancy = settings["Simulation"]["DoBuoyancy"]
        if DoBuoyancy == 1:
            # include Buoyancy
            rho_particle = settings["Simulation"]["Density_Particle"]
            rho_fluid    = settings["Simulation"]["Density_Fluid"]
            
            visc_water_m_Pa_s = visc_water * 1e12
            
            v_sedi = StokesVelocity(rho_particle, rho_fluid, diameter, visc_water_m_Pa_s)
            
            # convert in px per second
            v_sedi = v_sedi * 1e6 / microns_per_pixel 
            
            # sedimentation per frame
            delta_t = 1 / frames_per_second
            delta_x_sedi = v_sedi * delta_t
                   
            x_sedi = np.zeros([1,frames])
            x_sedi[:] = delta_x_sedi
            x_sedi = x_sedi.cumsum()
            
            bp()
            output.x = output.x + x_sedi
            
          
    
    nd.handle_data.WriteJson(ParameterJsonFile, settings) 

    return output


def StokesVelocity(rho_particle, rho_fluid, diameter, visc_water):
    #https://en.wikipedia.org/wiki/Stokes%27_law
    
    #g is the gravitational field strength (N/kg)
    g = 9.81
    
    #R is the radius of the spherical particle (m)
    R = diameter*1e-9 / 2
    
    #rho_particle is the mass density of the particles (kg/m3)
    
    #rho_fluid is the mass density of the fluid (kg/m3)
    
    #visc_water is the dynamic viscosity (Ns/m^2).
    
    #v_sedi sedimentation velocity (m/s)
    
    v_sedi = 2/9 * (rho_particle - rho_fluid) * g * R**2 / visc_water
    
    
    return v_sedi
    

def GenerateRandomWalk(diameter, num_particles, frames, frames_per_second, RatioDroppedFrames = 0, ep = 0, mass = 1, microns_per_pixel = 0.477, temp_water = 295, visc_water = 9.5e-16):
    """
    Simulate a random walk of brownian diffusion and return it in a panda like it came from real data
    
    diameter
    num_particles: number of particles to simular
    frames: frames simulated
    frames_per_second
    ep = 0 :estimation precision
    mass = 1: mass of the particle
    microns_per_pixel = 0.477
    temp_water = 295
    visc_water = 9.5e-16:
    """
    
    # Generating particle tracks as comparison
    
    # diameter of particle in nm
    #frames length of track of simulated particles
    #num_particles amount of simulated particles
    
    print("Do random walk with parameters: \
          \n diameter = {} \
          \n num_particles = {} \
          \n frames = {} \
          \n frames_per_second = {} \
          \n ep = {} \
          \n mass = {} \
          \n microns_per_pixel = {} \
          \n temp_water = {} \
          \n visc_water = {}" \
          .format(diameter,num_particles,frames,frames_per_second,ep,mass,\
          microns_per_pixel,temp_water,visc_water))
    
#    diameter, num_particles, frames, frames_per_second, ep = 0, mass = 1,
#    microns_per_pixel = 0.477, temp_water = 295, visc_water = 9.5e-16
    
    const_Boltz = 1.38e-23

    #diffusion constant of the simulated particle
    # sim_part_diff = (2*const_Boltz*temp_water/(6*math.pi *visc_water)*1e9) /diameter 
    radius_m = diameter/2 * 1e-9

    sim_part_diff = (const_Boltz*temp_water)/(6*math.pi *visc_water * radius_m)

    # unit sim_part_diff = um^2/s

    # [mum^2/s] x-diffusivity of simulated particle 
    sim_part_sigma_um = np.sqrt(2*sim_part_diff / frames_per_second)
    sim_part_sigma_x = sim_part_sigma_um / microns_per_pixel 
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
    sim_part_y=[]

    drop_rate = RatioDroppedFrames
    
    if drop_rate == 0:
        for sim_part in range(num_particles):
            loop_frame_drop = 0
            for sim_frame in range(frames):
                sim_part_part.append(sim_part)
                sim_part_x.append(np.random.normal(loc=0,scale=sim_part_sigma_x)) #sigma given by sim_part_sigma as above
                sim_part_y.append(np.random.normal(loc=0,scale=sim_part_sigma_x)) #sigma given by sim_part_sigma as above
                # Is that possibly wrong??
     
    else:        
        drop_frame = 1/drop_rate

        if drop_frame > 5:
            print("Drops every %s frame" %(drop_frame))
        else:
            sys.exit("Such high drop rates are probably not right implemented")
     
               
        for sim_part in range(num_particles):
            loop_frame_drop = 0
            for sim_frame in range(frames):
                sim_part_part.append(sim_part)
                
                if loop_frame_drop <= drop_frame:
                    loop_frame_drop += 1
                    lag_frame = 1
                else:
                    loop_frame_drop = 1
                    lag_frame = 2
                
                sim_part_x.append(np.random.normal(loc=0,scale=sim_part_sigma_x * lag_frame)) #sigma given by sim_part_sigma as above
                sim_part_y.append(np.random.normal(loc=0,scale=sim_part_sigma_x * lag_frame)) #sigma given by sim_part_sigma as above
                # Is that possibly wrong??


    # Putting the results into a df and formatting correctly:
    sim_part_tm=pd.DataFrame({'x':sim_part_x, \
                              'y':sim_part_y,  \
                              'mass':mass, \
                              'ep': 0, \
                              'frame':sim_part_frame_list, \
                              'particle':sim_part_part, \
                              "size": 0, \
                              "ecc": 0, \
                              "signal": 0, \
                              "raw_mass":mass,})
    
    
    sim_part_tm.x=sim_part_tm.groupby('particle').x.cumsum()
    sim_part_tm.y=sim_part_tm.groupby('particle').y.cumsum()
    

#    sim_part_tm.index=sim_part_tm.frame # old method RF 190408
#    copies frame to index and thus exists twice. not good
#    bp()


    # here come the localization precision ep on top    
    sim_part_tm.x = sim_part_tm.x + np.random.normal(0,ep,len(sim_part_tm.x))
    sim_part_tm.y = sim_part_tm.y + np.random.normal(0,ep,len(sim_part_tm.x))
    

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



