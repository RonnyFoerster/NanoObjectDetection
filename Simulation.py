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

#import matplotlib.pyplot as plt # Libraries for plotting
import NanoObjectDetection as nd

from pdb import set_trace as bp #debugger

from scipy.constants import speed_of_light as c
from numpy import pi
from scipy.constants import Boltzmann as k_b

import time
from joblib import Parallel, delayed
import multiprocessing

import trackpy as tp

# In[]
def PrepareRandomWalk(ParameterJsonFile):
    """ configure the parameters for a randowm walk out of a JSON file, and generate 
    it in a DataFrame
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
    
    if settings["Exp"]["Viscosity_auto"] == 1:
        visc_water = nd.handle_data.GetViscocity(temperature = temp_water, solvent = solvent)
        bp()
    else:
        visc_water = settings["Exp"]["Viscosity"]

    
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
    


def GenerateRandomWalk(diameter, num_particles, frames, frames_per_second, RatioDroppedFrames = 0, ep = 0, mass = 1, microns_per_pixel = 0.477, temp_water = 295, visc_water = 9.5e-16, PrintParameter = True, start_pos = None):
    """ simulate a random walk of Brownian diffusion and return it as Pandas.DataFrame
    object as if it came from real data:
        
    diameter:       particle size in nm
    num_particles:  number of particles to simulate
    frames:         number of frames to simulate
    frames_per_second
    ep = 0:         estimation precision
    mass = 1:       mass of the particle
    microns_per_pixel = 0.477
    temp_water = 295 K
    visc_water = 9.5e-16:
    """
    
    if PrintParameter == True:
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
    
    const_Boltz = k_b

    radius_m = diameter/2 * 1e-9 # in m
    # diffusion constant of the simulated particle (Stokes-Einstein eq.)
    sim_part_diff = (const_Boltz*temp_water)/(6*math.pi *visc_water * radius_m) # [um^2/s]

    # [mum^2/s] x-diffusivity of simulated particle 
    sim_part_sigma_um = np.sqrt(2*sim_part_diff / frames_per_second)
    sim_part_sigma_x = sim_part_sigma_um / microns_per_pixel 
    # [pixel/frame] st.deviation for simulated particle's x-movement (in pixel units!)
    
    # generate list to hold frames:
    sim_part_frame=[]
    for sim_frame in range(frames):
        sim_part_frame.append(sim_frame)
    sim_part_frame_list=sim_part_frame*num_particles
    
       

    #create emply dataframe of correct size
    num_elements = num_particles * frames
    sim_part = pd.DataFrame(columns = ["particle", "dx", "x", "dy", "y"], index = range(0,num_elements))

    #fille the particle row
    sim_part["particle"] = np.repeat(np.arange(num_particles),frames)
    
    # make shift as random number for all particles and frames
    sim_part["dx"] = np.random.normal(loc = 0, scale=sim_part_sigma_x, size = num_elements)
    sim_part["dy"] = np.random.normal(loc = 0, scale=sim_part_sigma_x, size = num_elements)
    
    #first frame of each particle should have dx and dy = 0. It is just disturbing and has no effect 
    sim_part.loc[sim_part.particle.diff(1) != 0, "dx"] = 0
    sim_part.loc[sim_part.particle.diff(1) != 0, "dy"] = 0
    
    # make a cumsum over time for every particle induvidual
    sim_part["x"] = sim_part[["particle", "dx"]].groupby("particle").cumsum()
    sim_part["y"] = sim_part[["particle", "dy"]].groupby("particle").cumsum()
    
    # move every trajectory to starting position
    if (start_pos is None) == False:
        sim_part["x"] = sim_part["x"] + np.repeat(start_pos[:,0],frames)
        sim_part["y"] = sim_part["y"] + np.repeat(start_pos[:,1],frames)
    
    
    sim_part_tm=pd.DataFrame({'x': sim_part["x"], \
                          'dx' :sim_part["dx"], \
                          'y': sim_part["y"],  \
                          'dy': sim_part["dy"],  \
                          'mass': mass,  \
                          'ep': 0,  \
                          'frame': sim_part_frame_list,  \
                          'particle': sim_part["particle"],  \
                          "size": 0, \
                          "ecc": 0, \
                          "signal": 0, \
                          "raw_mass": mass,})
       

    if ep > 0:
        sim_part_tm.x = sim_part_tm.x + np.random.normal(0,ep,len(sim_part_tm.x))
        sim_part_tm.y = sim_part_tm.y + np.random.normal(0,ep,len(sim_part_tm.x))
    
    
#    # OLD METHOD
#        # generate lists to hold particle IDs and
#    # step lengths in x and y direction, coming from a Gaussian distribution
#    sim_part_part=[]
#    sim_part_x=[]
#    sim_part_y=[]
#    drop_rate = RatioDroppedFrames
#    
#    if drop_rate == 0:
#        for sim_part in range(num_particles):
#            loop_frame_drop = 0
#            for sim_frame in range(frames):
#                sim_part_part.append(sim_part)
#                sim_part_x.append(np.random.normal(loc=0,scale=sim_part_sigma_x)) 
#                sim_part_y.append(np.random.normal(loc=0,scale=sim_part_sigma_x)) 
#                # Is that possibly wrong??
#     
#    else:        
#        drop_frame = 1/drop_rate
#
#        if drop_frame > 5:
#            print("Drops every %s frame" %(drop_frame))
#        else:
#            sys.exit("Such high drop rates are probably not correctly implemented")
#     
#               
#        for sim_part in range(num_particles):
#            loop_frame_drop = 0
#            for sim_frame in range(frames):
#                sim_part_part.append(sim_part)
#                
#                if loop_frame_drop <= drop_frame:
#                    loop_frame_drop += 1
#                    lag_frame = 1
#                else:
#                    loop_frame_drop = 1
#                    lag_frame = 2
#                
#                sim_part_x.append(np.random.normal(loc=0,scale=sim_part_sigma_x * lag_frame)) 
#                sim_part_y.append(np.random.normal(loc=0,scale=sim_part_sigma_x * lag_frame)) 
#                # Is that possibly wrong??
#
#
#    # put the results into a df and format them correctly:
#    sim_part_tm=pd.DataFrame({'x':sim_part_x, \
#                              'y':sim_part_y,  \
#                              'mass':mass, \
#                              'ep': 0, \
#                              'frame':sim_part_frame_list, \
#                              'particle':sim_part_part, \
#                              "size": 0, \
#                              "ecc": 0, \
#                              "signal": 0, \
#                              "raw_mass":mass,})
#    
#    # calculate cumulative sums to get position values from x- and y-steps for the full random walk
#    sim_part_tm.x=sim_part_tm.groupby('particle').x.cumsum()
#    sim_part_tm.y=sim_part_tm.groupby('particle').y.cumsum()
    

#    sim_part_tm.index=sim_part_tm.frame # old method RF 190408
#    copies frame to index and thus exists twice. not good
#    bp()

    # here come the localization precision ep on top    
#    sim_part_tm.x = sim_part_tm.x + np.random.normal(0,ep,len(sim_part_tm.x))
#    sim_part_tm.y = sim_part_tm.y + np.random.normal(0,ep,len(sim_part_tm.x))
#    

#    # check if tm is gaussian distributed
#    my_mean = []
#    my_var = []
#    for sim_frame in range(frames):
#        mycheck = sim_part_tm[sim_part_tm.frame == sim_frame].x.values
#        my_mean.append(np.mean(mycheck))
#        my_var.append(np.var(mycheck))
#        
##    plt.plot(my_mean)
##    plt.plot(my_var)
    
    return sim_part_tm


def RadiationForce(lambda_nm, d_nm, P_W, A_sqm, material = "Gold", e_part = None):   
    """ calculate the radiation force onto a scattering sphere
    
    https://reader.elsevier.com/reader/sd/pii/0030401895007539?token=48F2795599992EB11281DD1C2A50B58FC6C5F2614C90590B9700CD737B0B9C8E94F2BB8A17F74D0E6087FF3B7EF5EF49
    https://github.com/scottprahl/miepython/blob/master/doc/01_basics.ipynb
    
    lambda_nm:  wavelength of the incident light
    d_nm:       sphere's diameter
    P_W:        incident power
    A_sqm:      beam/channel cross sectional area
    material:   sphere's material
    """
    
    lambda_um = lambda_nm / 1000
    lambda_m  = lambda_nm / 1e9
    k = 2*pi/lambda_m
    
    #particle radius
    r_nm = d_nm / 2
    r_m  = r_nm / 1e9
    
    I = P_W/A_sqm

    if e_part == None:    
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

        e_part_real = n_part_real**2 - n_part_imag**2
        e_part_imag = 2*n_part_real*n_part_imag
        e_part = e_part_real + 1j * e_part_imag
    
    else:
        if isinstance(e_part, complex) == False:
            raise TypeError("number must be complex, like 1+1j")
        else:
            e_part_real = np.real(e_part)
            e_part_imag = np.imag(e_part)
            n_part = np.sqrt((e_part_real+e_part_real)/2)
    
    n_media = 1.333
    e_media = n_media**2
    
    m = n_part / n_media
    
    print("not sure if this is right")
    m = np.abs(m)
    n_part = np.abs(n_part)
    
    C_scat = 8/3*pi*np.power(k,4)*np.power(r_m,6) * ((m**2-1)/(m**2+1))
    
    V = 4/3*pi*np.power(r_m,3)
    C_abs = k * np.imag(3 * V * (e_part - e_media)/(e_part + 2*e_media))
    
    #Eq 11 in Eq 10
    F_scat = C_scat * n_media/c * I
    F_abs = C_abs * n_media/c * I
    
    print("F_scat = ", F_scat)
    print("F_abs = ", F_abs)

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
def LoopSimulation():
    # Makes several run of the random walk for several particle numbers and frames to get a good statistic
    if 1 == 1:
        frames = 1000
        iterations = 3
        num_particles = [5, 10, 25, 50, 100, 200, 400, 800]
    else:
        frames = 1000
        iterations = 1
        num_particles = [200]                
        
    num_diff_particles = len(num_particles)

    num_eval_particles = np.zeros([iterations, num_diff_particles, frames])
    num_eval_particles_per_frame = np.zeros([iterations, num_diff_particles, frames])
    
    for loop_num, loop_num_particles in enumerate(num_particles):
        print("number of particles", loop_num_particles)
        for loop_iter in range(iterations):
            print("num iteration: ", loop_iter)
            num_eval_particles[loop_iter,loop_num,:], num_eval_particles_per_frame[loop_iter,loop_num,:], volume_nl, t_unconf, t_conf, eval_t1, eval_t2 = ConcentrationVsNN(frames, loop_num_particles)
    
    print("done")
    
    conc_per_nl = list(np.round(np.asarray(num_particles)/volume_nl, 0).astype("int"))
    
    PlotSimulationResults(num_particles, conc_per_nl, frames, num_eval_particles, num_eval_particles_per_frame)
    
    return num_eval_particles, num_eval_particles_per_frame, t_unconf, t_conf, eval_t1, eval_t2



def PlotSimulationResults(num_particles, conc_per_nl, frames, num_eval_particles, num_eval_particles_per_frame, RelDrop = False):
    # Plot it all
    
    PlotNumberParticlesPerFrame(num_particles, conc_per_nl, frames, num_eval_particles_per_frame, RelDrop)
    PlotNumberDifferentParticle(num_particles, conc_per_nl, frames, num_eval_particles, RelDrop)
    
    
def PlotNumberParticlesPerFrame(num_particles, conc_per_nl, frames, num_eval_particles_per_frame, RelDrop = False):
    # Plot the Number of Particles that can be evaluated in average in a frame
    eval_part_mean = np.mean(num_eval_particles_per_frame, axis = 0)
    eval_part_std = np.std(num_eval_particles_per_frame, axis = 0)

    min_traj = np.arange(0,frames)

    rel_error = nd.CalcDiameter.DiffusionError(min_traj, "gain missing", 0, 0, 2)[0]
    
    #get position of rel error
    disp_rel_array = [0.20, 0.10, 0.08, 0.06, 0.05, 0.04]
    
    ax2_value_pos = np.zeros_like(disp_rel_array)
    
    for loop_index, loop_disp_rel_array in enumerate(disp_rel_array):
        ax2_value_pos[loop_index] = np.where(rel_error < loop_disp_rel_array)[0][0]
        
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for loop_num, loop_num_particles in enumerate(num_particles):
#        plt.plot(min_traj, eval_part_mean[loop_num,:])
        show_mean = eval_part_mean[loop_num,:]
        show_std = eval_part_std[loop_num,:]
        show_rel_mean = show_mean / show_mean[0] * 100
        if RelDrop == False:
            ax1.plot(min_traj, show_mean, label= str(loop_num_particles) + " | "+ str(conc_per_nl[loop_num])) 
        else:
            ax1.plot(min_traj, show_rel_mean, label= str(loop_num_particles) + " | "+ str(conc_per_nl[loop_num])) 
#        ax1.errorbar(min_traj, show_mean, show_std, label= str(loop_num_particles) + " | "+ str(conc_per_nl[loop_num]))    


    ax1.set_title("Number of evaluable particles in a frame", fontsize = 16)
    ax1.set_xlim([0,1.4*frames])
    ax1.set_xlabel("Minium Trajectory length", fontsize = 14)
      
    if RelDrop == False:
        ax1.set_ylim([0,np.max(num_particles)*1.1])
        ax1.set_ylabel("Number of evaluated particles", fontsize = 14)
    else:
        ax1.set_ylim([0,105])
        ax1.set_ylabel("Relative number of evaluated particles [%]", fontsize = 14)
    
    ax1.grid()
    
    ax1.legend(title = 'N | c [N/nl]', title_fontsize = 14)#, bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)

    # make second axis with rel. error
    ax2 = ax1.twiny()
    ax2.set_xticks(ax2_value_pos)
    ax2.set_xticklabels(list((np.asarray(disp_rel_array)*100).astype("int")))
    ax2.set_xlim([0,1.4*frames])
    ax2.set_xlabel("Minimum rel. error [%]", fontsize = 14)
    
    
def PlotNumberDifferentParticle(num_particles, conc_per_nl, frames, num_eval_particles, RelDrop = False):
    # Plot the Number of Particles that can be evaluated out of an entire ensemble
    avg_part_mean = np.mean(num_eval_particles, axis = 0)
    avg_part_std = np.std(num_eval_particles, axis = 0)    

    min_traj = np.arange(0,frames)
    rel_error = nd.CalcDiameter.DiffusionError(min_traj, "gain missing", 0, 0, 2)[0]
    
    #get position of rel error
    disp_rel_array = [0.20, 0.10, 0.08, 0.06, 0.05, 0.04]
    
    ax2_value_pos = np.zeros_like(disp_rel_array)
    
    for loop_index, loop_disp_rel_array in enumerate(disp_rel_array):
        ax2_value_pos[loop_index] = np.where(rel_error < loop_disp_rel_array)[0][0]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    for loop_num, loop_num_particles in enumerate(num_particles):        
        show_mean = avg_part_mean[loop_num,:]
        show_std = avg_part_std[loop_num,:]
        show_rel_mean = show_mean / show_mean[0] * 100        
        if RelDrop == False:
            ax1.plot(min_traj, show_mean, label= str(loop_num_particles) + " | "+ str(conc_per_nl[loop_num])) 
        else:
            ax1.plot(min_traj, show_rel_mean, label = str(loop_num_particles) + " | "+ str(conc_per_nl[loop_num]))
        
#        ax1.fill_between(min_traj, show_mean - show_std, show_mean + show_std)
#        ax1.errorbar(min_traj, show_mean, show_std, label= str(loop_num_particles) + " | "+ str(conc_per_nl[loop_num]))
#        plt.loglog(min_traj, show_mean, label= str(loop_num_particles))
    
    ax1.set_title("Number of DIFFERENT evaluable particles", fontsize = 16)
        
    ax1.set_xlim([0,1.4*frames])
    ax1.set_xlabel("Minium Trajectory length", fontsize = 14)
        

          
    if RelDrop == False:
        ax1.set_ylim([0,np.max(num_particles)*1.1])
        ax1.set_ylabel("Number of evaluated particles", fontsize = 14)
    else:
        ax1.set_ylim([0,105])
        ax1.set_ylabel("Relative number of evaluated particles [%]", fontsize = 14)
        
#    ax1.set_ylim([1, 1000])
    

    ax1.grid()

    ax1.legend(title = 'N | c [N/nl]', title_fontsize = 14)


    # make second axis with rel. error
    ax2 = ax1.twiny()
    ax2.set_xticks(ax2_value_pos)
    ax2.set_xticklabels(list((np.asarray(disp_rel_array)*100).astype("int")))
    ax2.set_xlim([0,1.4*frames])
    ax2.set_xlabel("Minimum rel. error [%]", fontsize = 14)
    


def BestConcentration(eval_particles, conc_per_nl, frames):
    #Plot the concentration with the hightes number of evaluated particle
    num_eval = np.mean(eval_particles,0)
    
    best_c = np.zeros_like(num_eval[0,:])
    
    for loop_id,loop_c in enumerate(best_c):
        best_c_pos = np.where(num_eval[:,loop_id] == np.max(num_eval[:,loop_id]))[0][0]
        best_c[loop_id] = conc_per_nl[best_c_pos]
        
    #find edges
    edges = np.where((best_c[1:] - best_c[0:-1]) != 0)[0][:]
        
    min_traj = np.arange(0,frames)

    min_traj = np.arange(0,frames)
    rel_error = nd.CalcDiameter.DiffusionError(min_traj, "gain missing", 0, 0, 2)[0]
    
    #get position of rel error
    disp_rel_array = [0.20, 0.10, 0.08, 0.06, 0.05, 0.04]
    
    ax2_value_pos = np.zeros_like(disp_rel_array)
    
    for loop_index, loop_disp_rel_array in enumerate(disp_rel_array):
        ax2_value_pos[loop_index] = np.where(rel_error < loop_disp_rel_array)[0][0]
    
    min_traj_grid = (edges[1:] + edges[:-1])/2
    best_c_grid = best_c[edges[:-1]+1]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(min_traj_grid,best_c_grid, 'x')
    
    ax1.set_xlabel("Minium Traj", fontsize = 14)
    ax1.set_ylabel("Ideal concentraion [N/nl]", fontsize = 14)
    
    ax1.set_xlim([0, frames])
    ax1.set_ylim([0, 1.1*np.max(conc_per_nl)])
    ax1.grid()
    
    # make second axis with rel. error
    ax2 = ax1.twiny()
    ax2.set_xticks(ax2_value_pos)
    ax2.set_xticklabels(list((np.asarray(disp_rel_array)*100).astype("int")))
    ax2.set_xlim([0,1.4*frames])
    ax2.set_xlabel("Minimum rel. error [%]", fontsize = 14)
    
    return 
    

    
def ConcentrationVsNN(frames, num_particles):
    # Main Function to find out the relation between concentration and evaluatable particles
    import scipy
    
    kb = scipy.constants.k
    pi = scipy.constants.pi
    
    # numerical grid
    x_size = 630
    y_size = 35    
    
    #volume like a cylinder
    volume_nl = pi * ((y_size*1e-5/2)**2) * x_size*1e-5 *1e9
    
    # experimental parameters    
    diameter = 40 #nm    
    frames_per_second = 100
    exposure_time = 1/frames_per_second
    microns_per_pixel = 1
    temp_water = 295
    visc_water_Pas = 1e-3
    visc_water = visc_water_Pas / 1e12
    resolution = 1
    
    if num_particles < 2:
        raise ValueError("Number of particles must be at least 2!")
    
    # calc parameters
    radius_m = diameter/2 * 1e-9 # in m
    diffusion = (kb*temp_water)/(6*pi *visc_water * radius_m) # [um^2/s]
    
    sigma_diffusion = np.sqrt(2*exposure_time*diffusion)
    max_displacement = 5*sigma_diffusion
    min_separation   = 2*max_displacement
    
    max_displacement = 3.15
    min_separation   = 1.89
    
    
    
    # get the trajectories
    t_unconf, t_conf = SimulateTrajectories(x_size, y_size, num_particles, diameter, frames, frames_per_second, microns_per_pixel, temp_water, visc_water, max_displacement, PrintParameter = False)
    
    #nearest neighbor
    nn = t_conf.groupby("frame").apply(tp.static.proximity)
    
    # get trajectory into same format
    t_conf = t_conf.sort_values("frame").set_index(["frame","particle"])
    
    t_conf["nn"]=nn
    
    eval_t1 = t_conf.reset_index()
    
#    # get distance to nearest neighboor
#    print("Calc Nearest Neighbour")
#    eval_t1 = CalcNearestNeighbor(t)
    
    eval_t1["find_possible"] = eval_t1["nn"] > min_separation
    eval_t1["link_possible"] = eval_t1["dr"] < max_displacement
    eval_t1["link_same_particle"] = eval_t1["nn"] > eval_t1["dr"]
    
    #dr and nn are NaN for first frame - so remove the false entries
    eval_t1.loc[np.isnan(eval_t1["dr"]),["link_possible", "link_same_particle"]] = True
    
    
    eval_t1["valid"] = eval_t1["link_possible"] & eval_t1["find_possible"] & eval_t1["link_same_particle"]

#    print("Split Trajectory")
    eval_t2 = eval_t1.sort_values(["particle","frame"])
    eval_t2["true_particle"] = eval_t2["particle"]
    eval_t2["new_traj"] = eval_t2["particle"]
    eval_t2["new_traj"] = 0

    
    #new trajectory, where trajectory is not valied
    eval_t2.loc[eval_t2.valid == False, "new_traj"] = 1

    #new trajectory, where new particle begins
    eval_t2.loc[eval_t2.particle.diff(1) == 1, "new_traj"] = 1
    
    eval_t2["particle"] = np.cumsum(eval_t2["new_traj"])
    
    eval_t2.reset_index()
    
#    eval_t2 = eval_t2.sort_values(["frame","particle"])
    
#    eval_t2 = SplitTrajectory(eval_t1.copy())

    #evaluate how much can be evaluated
    traj_length = eval_t2.groupby("particle").count().x
    
    # get LUT particle to true_particle
    lut_particle_id = eval_t2.groupby("particle").mean().true_particle
    
    # this save how many of the given particles are evaluated at some point of the movie
    num_eval_particles = np.zeros(frames)
    
    # this save how many particles are evaluated in a frame
    num_eval_particles_per_frame = np.zeros(frames)


    for min_traj_length_loop in np.arange(0,frames):
        # index of particles with trajectory of sufficent length
        eval_particles_loc = traj_length[traj_length >= min_traj_length_loop].index
        
        # number of evaluated particles
        try:
            num_eval_particles[min_traj_length_loop] = len(lut_particle_id.loc[eval_particles_loc].unique())
        except:
            bp()
       
        # average number of evaluated particles in a frame
        num_eval_particles_per_frame[min_traj_length_loop] = traj_length.loc[eval_particles_loc].sum() / frames
        
    
    return num_eval_particles, num_eval_particles_per_frame, volume_nl, t_unconf, t_conf, eval_t1, eval_t2



def CheckTrajLeavesConfinement(t_part, x_min, x_max, y_min, y_max):
    # mirror trajectories if confinment is left
    out_left = t_part["x"] < x_min
    out_right = t_part["x"] > x_max
    out_bottom = t_part["y"] < y_min
    out_top = t_part["y"] > y_max
    
    leaves = np.max(out_left | out_right | out_bottom | out_top)
    
    if leaves == True:
        direction = None
        loop_frames = 0
        while direction is None:
            if (out_left.iloc[loop_frames] == True) | ( out_right.iloc[loop_frames] == True):
                direction = "x"
                exit_frame = loop_frames
            if (out_bottom.iloc[loop_frames] == True) | (out_top.iloc[loop_frames] == True): 
                direction = "y"
                exit_frame = loop_frames

            loop_frames = loop_frames + 1
    else:
        direction = None
        exit_frame = None
    
    return leaves, direction, exit_frame    



def SimulateTrajectories(x_size, y_size, num_particles, diameter, frames, frames_per_second, 
                         microns_per_pixel, temp_water, visc_water, max_displacement, PrintParameter = True):
    # simulate the random walk

    start_pos = np.random.rand(num_particles,2)
    start_pos[0,:] = 0.5 # particle under investigation is in the middle
    start_pos[:,0] = start_pos[:,0] * x_size
    start_pos[:,1] = start_pos[:,1] * y_size
    
    #central particle
    tm = GenerateRandomWalk(diameter, num_particles, frames, frames_per_second, 
                            microns_per_pixel = microns_per_pixel, temp_water = temp_water, 
                            visc_water = visc_water, PrintParameter = PrintParameter)

    t = tm[["dx", "x", "dy", "y", "frame", "particle"]].copy()
    
    # move to starting position
    t["x"] = t["x"] + np.repeat(start_pos[:,0],frames)
    t["y"] = t["y"] + np.repeat(start_pos[:,1],frames)

    t_conf = None

    # do confinement
    for loop_particle in range(num_particles):
        #select particle
        t_part = t[t.particle == loop_particle]
        t_part = t_part.set_index("frame")
         
        leaves, direction, exit_frame = nd.Simulation.CheckTrajLeavesConfinement(t_part, 0, x_size, 0, y_size)
        
        while leaves == True:                   
            # do until entire trajectory is inside confinement
            if direction == "x":
                #reflect particle
                t_part.loc[exit_frame,"dx"] = -t_part.loc[exit_frame,"dx"]
                #make new cumsum and shift to starting point again
                t_part["x"] = t_part["dx"].cumsum() + start_pos[loop_particle,0]
                
            if direction == "y":
                t_part.loc[exit_frame,"dy"] = -t_part.loc[exit_frame,"dy"]
                t_part["y"] = t_part["dy"].cumsum() + start_pos[loop_particle,1]
            
            leaves, direction, exit_frame = nd.Simulation.CheckTrajLeavesConfinement(t_part, 0, x_size, 0, y_size)
        
        if t_conf is None:
            t_conf = t_part
        else:
            t_conf = t_conf.append(t_part)

    t_conf = t_conf.reset_index()   
       
    t_conf["dr"] =  np.sqrt(np.square(t_conf.x.diff(1)) + np.square(t_conf.y.diff(1)))
    
    new_part = t_conf.particle.diff(1) != 0
    t_conf.loc[new_part,"dr"] = np.nan

    return t, t_conf


def SplitTrajectory(eval_t2):
    # splits trajectory if linkind failed
    
    num_particles = len(eval_t2.groupby("particle"))    
    #particle id next new particle gets
    free_part_id = num_particles
    
    eval_t2["true_particle"] = eval_t2["particle"]

    
    for loop_particles in range (0,num_particles):
        print(loop_particles)
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







## get nearest neighbor
#def CalcNearestNeighbor(t, seq = False):
#    frames = len(t[t.particle == 0])
#    num_particles = len(t.groupby("particle"))
#    
#    if seq == True:
#        print("Do it seriel")
#        tic = time.time()
#        
#        for loop_frame in range(frames):
#            eval_t_frame = t[t.frame == loop_frame]
#            for loop_particles in range(num_particles):
#            #loop_particles = 0    
#                pos_part = eval_t_frame[eval_t_frame.particle == loop_particles][["x","y"]].values.tolist()[0]
#                test_part = eval_t_frame[eval_t_frame.particle != loop_particles][["x","y"]]
#                diff_part = test_part-pos_part
#                dist_nn = np.min(np.sqrt(diff_part.x**2+diff_part.y**2))
#
#                t.loc[(t.particle == loop_particles) & (t.frame == loop_frame),"nn"] = dist_nn
#        #
#        toc = time.time()
#        print('\nElapsed time computing the average of couple of slices {:.2f} s'.format(toc - tic))
#
#        eval_t1 = t
#
#    else:
#        print("Do it parallel")
#        
#        def CalcNearestNeighbour(eval_t_frame, num_particles):
#            print(num_particles)
#            for loop_particles in range(num_particles):
#                print(loop_particles)
#                pos_part = eval_t_frame[eval_t_frame.particle == loop_particles][["x","y"]].values.tolist()[0]
#                test_part = eval_t_frame[eval_t_frame.particle != loop_particles][["x","y"]]
#                diff_part = test_part-pos_part
#                
#                dist_nn = np.min(np.sqrt(diff_part.x**2+diff_part.y**2))
#
#                eval_t_frame.loc[eval_t_frame.particle == loop_particles,"nn"] = dist_nn
#                print(eval_t_frame)
#            return eval_t_frame
#        
#        tic = time.time()
#        
#        
#        #for loop_frame in range(frames):
#        ##loop_particles = 0
#        #    eval_t = CalcNearestNeighbour(eval_t, num_particles)
#        
#        num_cores = multiprocessing.cpu_count()
#        
#        inputs = range(frames)
#        print(inputs)
#        eval_t1 = Parallel(n_jobs=num_cores)(delayed(CalcNearestNeighbour)(t[t.frame == loop_frame].copy(), num_particles) for loop_frame in inputs)
#
#        eval_t1 = pd.concat(eval_t1)
#        eval_t1 = eval_t1[eval_t1.frame > 0]
#        eval_t1 = eval_t1.reset_index(drop = True)
#    
#        #
#        toc = time.time()
#        print('\nElapsed time computing the average of couple of slices {:.2f} s'.format(toc - tic))   
#    
#    return eval_t1
    
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
    