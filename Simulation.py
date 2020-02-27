# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:17:36 2019

@author: Ronny Förster und Stefan Weidlich
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# False since deprecated from version 3.0
#matplotlib.rcParams['text.usetex'] = False
#matplotlib.rcParams['text.latex.unicode'] = False

import sys
#import matplotlib.pyplot as plt # Libraries for plotting
import NanoObjectDetection as nd

from pdb import set_trace as bp #debugger

from scipy.constants import speed_of_light as c
from numpy import pi
from scipy.constants import Boltzmann as k_b

import time
from joblib import Parallel, delayed
import multiprocessing

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


def RadiationForce(lambda_nm, d_nm, P_W, A_sqm, material = "Gold"):    
#    https://reader.elsevier.com/reader/sd/pii/0030401895007539?token=48F2795599992EB11281DD1C2A50B58FC6C5F2614C90590B9700CD737B0B9C8E94F2BB8A17F74D0E6087FF3B7EF5EF49
    #https://github.com/scottprahl/miepython/blob/master/doc/01_basics.ipynb
    lambda_um = lambda_nm / 1000
    lambda_m  = lambda_nm / 1e9
    k = 2*pi/lambda_m
    
    #particle radius
    r_nm = d_nm / 2
    r_m  = r_nm / 1e9
    
    I = P_W/A_sqm
    
    if material == "Gold":
        au = np.genfromtxt('https://refractiveindex.info/tmp/data/main/Au/McPeak.txt', delimiter='\t')
        #au = np.genfromtxt('https://refractiveindex.info/tmp/data/main/Au/Johnson.txt', delimiter='\t')
        #au = np.genfromtxt('https://refractiveindex.info/tmp/data/main/Au/Werner.txt', delimiter='\t')
    else:
        print("material unknown")
        
    # data is stacked so need to rearrange
    N = len(au)//2
    mylambda = au[1:N,0]
    n_real = au[1:N,1]
    n_imag = au[N+1:,1]
    
    n_part_real = np.interp(lambda_um, mylambda, n_real)
    n_part_imag = np.interp(lambda_um, mylambda, n_imag)
    n_part = n_part_real + 1j * n_part_imag
    
    n_media = 1.333
    
    m = n_part / n_media
    
    print("not sure if this is right")
    m = np.abs(m)
    n_part = np.abs(n_part)
    
    F_scat = 8*pi*n_part*np.power(k,4)*np.power(r_m,6)/(3*c) * ((m**2-1)/(m**2+1)) * I
    
    print(F_scat)

    return F_scat


def MassOfNP(d_nm,rho):
    
    r_m = (d_nm/1E9) / 2
    V = 4/3*pi*np.power(r_m,2)
    
    m = V*rho
    
    
    return m


def E_Kin_Radiation_Force(F,m,t):
    E_kin = np.power(F*t,2)/(2*m)
    
    print(E_kin)
    
    return E_kin





#%%
def ConcentrationVsNN():
    x_size = 100
    y_size = 30    
    num_particles = 50
    # experimental parameters    
    diameter = 100
    frames = 100
    frames_per_second = 100
    microns_per_pixel = 1
    temp_water = 295
    visc_water = 9.5e-16
    max_displacement = 0.8
    resolution = 1
    
    t = SimulateTrajectories(x_size, y_size, num_particles, diameter, frames, frames_per_second, microns_per_pixel, temp_water, visc_water, max_displacement)
    
    eval_t1 = CalcNearestNeighbor(t)
    
    eval_t1["link_possible"] = eval_t1["dr"] < max_displacement
    eval_t1["resolved"] = eval_t1["nn"] > resolution 
    eval_t1["link_same_particle"] = eval_t1["nn"] > eval_t1["dr"]
    
    
    eval_t1["valid"] = eval_t1["link_possible"] & eval_t1["resolved"] & eval_t1["link_same_particle"]

    eval_t2 = SplitTrajectory(eval_t1.copy())

    
    return t, eval_t1, eval_t2
    


def SimulateTrajectories(x_size, y_size, num_particles, diameter, frames, frames_per_second, microns_per_pixel, temp_water, visc_water, max_displacement):

    start_pos = np.random.rand(num_particles,2)
    start_pos[0,:] = 0.5 # particle under investigation is in the middle
    start_pos[:,0] = start_pos[:,0] * x_size
    start_pos[:,1] = start_pos[:,1] * y_size
    
    #central particle
    tm = GenerateRandomWalk(diameter, num_particles, frames, frames_per_second, microns_per_pixel = microns_per_pixel, temp_water = temp_water, visc_water = visc_water)
    
    t = tm[["x","y","frame","particle"]].copy()
    
    # move by starting position
    for loop_particles in range(num_particles):
        t.loc[(t.particle == loop_particles),["x","y"]] = t.loc[(t.particle == loop_particles),["x","y"]] + start_pos[loop_particles,:]
    
    
       
    t["dr"] =  np.sqrt(np.square(t.x.diff(1)) + np.square(t.y.diff(1)))
    
    new_part = t.particle.diff(1) != 0
    t.loc[new_part,"dr"] = np.nan

    return t


# get neareast neighbor
def CalcNearestNeighbor(t, seq = False):
    frames = len(t[t.particle == 0])
    num_particles = len(t.groupby("particle"))
    
    if seq == True:
        print("Do it seriel")
        tic = time.time()
        
        for loop_frame in range(frames):
            eval_t_frame = t[t.frame == loop_frame]
            for loop_particles in range(num_particles):
            #loop_particles = 0    
                pos_part = eval_t_frame[eval_t_frame.particle == loop_particles][["x","y"]].values.tolist()[0]
                test_part = eval_t_frame[eval_t_frame.particle != loop_particles][["x","y"]]
                diff_part = test_part-pos_part
                
                dist_nn = np.min(np.sqrt(diff_part.x**2+diff_part.y**2))

                t.loc[(t.particle == loop_particles) & (t.frame == loop_frame),"nn"] = dist_nn
        #
        toc = time.time()
        print('\nElapsed time computing the average of couple of slices {:.2f} s'.format(toc - tic))



    else:
        print("do it parallel")
        def CalcNearestNeighbour(eval_t_frame, num_particles):
            print(num_particles)
            for loop_particles in range(num_particles):
                print(loop_particles)
                pos_part = eval_t_frame[eval_t_frame.particle == loop_particles][["x","y"]].values.tolist()[0]
                test_part = eval_t_frame[eval_t_frame.particle != loop_particles][["x","y"]]
                diff_part = test_part-pos_part
                
                dist_nn = np.min(np.sqrt(diff_part.x**2+diff_part.y**2))

                eval_t_frame.loc[eval_t_frame.particle == loop_particles,"nn"] = dist_nn
                print(eval_t_frame)
            return eval_t_frame
        
        tic = time.time()
        
        
        #for loop_frame in range(frames):
        ##loop_particles = 0
        #    eval_t = CalcNearestNeighbour(eval_t, num_particles)
        
        num_cores = multiprocessing.cpu_count()
        
        inputs = range(frames)
        print(inputs)
        eval_t1 = Parallel(n_jobs=num_cores)(delayed(CalcNearestNeighbour)(t[t.frame == loop_frame].copy(), num_particles) for loop_frame in inputs)

        eval_t1 = pd.concat(eval_t1)
        eval_t1 = eval_t1[eval_t1.frame > 0]
        eval_t1 = eval_t1.reset_index(drop = True)
    
        #
        toc = time.time()
        print('\nElapsed time computing the average of couple of slices {:.2f} s'.format(toc - tic))   
    
    return eval_t1




def SplitTrajectory(eval_t2):
    num_particles = len(eval_t2.groupby("particle"))    
    #particle id next new particle gets
    free_part_id = num_particles
    
    eval_t2["true_particle"] = eval_t2["particle"]

    
    for loop_particles in range (0,num_particles):
        #select data for current particle
        valid_link = eval_t2[eval_t2.particle == loop_particles]["valid"]
        
        #check where the linking failed because the particle move more than the allowed maximal displacement
        linking_failed = (valid_link == False)
        
        #get frame where linking failed and new trajectory must start
        frames_new_traj = list(np.where(linking_failed))[0]
        
        first_change = True
        
        for frame_split_traj in frames_new_traj:
            if first_change == True:
                eval_t2.loc[(eval_t2.particle == loop_particles) & (eval_t2.frame > frame_split_traj),"particle"] = free_part_id
                first_change = False
            else:
                eval_t2.loc[(eval_t2.particle == free_part_id-1) & (eval_t2.frame > frame_split_traj),"particle"] = free_part_id
                
            free_part_id += 1
        
        #get frame length of trajectory
   
    return eval_t2


#def Blabla():
#    track_length = eval_t2[["particle", "frame"]].groupby("particle").count()
#    
#    track_length.rename(columns={'frame':'traj_length'}, inplace=True)
#    
#    min_traj_length = 50
#    measured_particles = track_length[track_length["traj_length"] > min_traj_length]
#    
#    LUT_particles = eval_t2[["particle", "true_particle"]].drop_duplicates().set_index("particle")


#def CheckSuccessfullLink(eval_t2, max_displacement):
#    num_particles = len(eval_t2.groupby("particle"))    
#    #particle id next new particle gets
#    free_part_id = num_particles
#
#    
#    for loop_particles in range (0,num_particles):
#        #select data for current particle
#        loop_eval_t = eval_t2[eval_t2.particle == loop_particles]
#        
#        #check where the linking failed because the particle move more than the allowed maximal displacement
#        linking_failed = loop_eval_t["dr"] > max_displacement
#        
#        #get frame where linking failed and new trajectory must start
#        frames_new_traj = list(np.where(linking_failed))[0]
#        
#        for frame_split_traj in frames_new_traj:
#            eval_t2.loc[(eval_t2.particle == loop_particles) & (eval_t2.frame > frame_split_traj),"particle"] = free_part_id
#            free_part_id += 1
#        
#        #get frame length of trajectory
#   
#    return eval_t2
#
#
#
#def CheckLinkPossible(eval_t2, max_displacement):
#    eval_t2["link_possible"] = eval_t2["dr"] < max_displacement
#
#    return eval_t2["link_possible"]
#
#
#def CheckUnresolved(eval_t3, resolution):
#    eval_t3["resolved"] = eval_t3["nn"] > resolution  
#    
#    return eval_t3
#
#
#def CheckCorrectLink(eval_t4):
#    eval_t4["link_correct"] = eval_t4["nn"] > eval_t4["dr"]
#    
#    return eval_t4







#result_mean = eval_t.groupby("frame")[["dr","nn"]].mean()
#result_min = eval_t.groupby("frame")[["dr","nn"]].min()
#result_std = eval_t.groupby("frame")[["dr","nn"]].std()
#
#fig = plt.figure()
#frames = result_mean.index
#plt.plot(frames,result_mean.nn)
#plt.plot(frames,result_mean.dr)
#plt.plot(frames,result_min.nn)
#
#fig = plt.figure()
#plt.scatter(t["x"],t["y"])
    