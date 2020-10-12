# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:17:36 2019

@author: Ronny Förster und Stefan Weidlich and Jisoo
"""
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# False since deprecated from version 3.0
#matplotlib.rcParams['text.usetex'] = False
#matplotlib.rcParams['text.latex.unicode'] = False

#import matplotlib.pyplot as plt # Libraries for plotting
import NanoObjectDetection as nd

from pdb import set_trace as bp #debugger

from scipy.constants import speed_of_light as c
from numpy import pi
import scipy
import scipy.signal
from scipy.constants import Boltzmann as k_b

import time
from joblib import Parallel, delayed
import multiprocessing

import trackpy as tp


def PrepareRandomWalk(ParameterJsonFile = None, diameter = 100, num_particles = 1, frames = 100, RatioDroppedFrames = 0, EstimationPrecision = 0, mass = 0, frames_per_second = 100, microns_per_pixel = 1, temp_water = 293, visc_water = 9.5e-16):
    """ configure the parameters for a randowm walk out of a JSON file, and generate 
    it in a DataFrame
    """
    
    if ParameterJsonFile != None:
        #read para file if existing
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

    
    output = GenerateRandomWalk(diameter, num_particles, frames, frames_per_second, RatioDroppedFrames = RatioDroppedFrames, ep = EstimationPrecision, mass = mass,                                               microns_per_pixel = microns_per_pixel, temp_water = temp_water, visc_water = visc_water)
            
          
    if ParameterJsonFile != None:
        # write if para file is given
        nd.handle_data.WriteJson(ParameterJsonFile, settings) 

    return output
   

def CalcDiffusionCoefficent(radius_m, temp_water = 293, visc_water = 0.001):
    diffusion = (k_b*temp_water)/(6*math.pi *visc_water * radius_m) # [um^2/s]
    
    return diffusion


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

    print("Diffusion coefficent: ", sim_part_diff)

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



def VelocityByExternalForce(F_ext, radius, visc_water):
    #visc_water is the dynamic viscosity (Ns/m^2).
    
    v = F_ext / (6 * pi * radius * visc_water)
    
    return v



def MaximumNAByFocusDepth(dof, lambda_nm, n):
    '''
    dof - depth of focus in nm
    lambda_nm - wavelength in nm
    n - refractive index immersion oil
    '''
    NA = np.sqrt((2*lambda_nm*n) / dof)
    
    return NA
#https://www.microscopyu.com/microscopy-basics/depth-of-field-and-depth-of-focus


def DetectionEfficency(NA,n):
    alpha = np.arcsin(NA/n) #opening angle
    ster = 2*np.pi*(1-np.cos(alpha)) # steradiant
    DE = ster / (4*np.pi)# Detection efficency    
    
    return DE




def MinimalDiameter(d_channel, lambda_nm, P_illu = 0.001, d_nm_min = 1, d_nm_max = 100, n = 1, T = 293, visc_water = 0.001, readout_s = None, N_required = 200):
    '''
    d_channel - diameter of the channel [nm]
    P_illu - power of laser INSIDE the fiber [W]
    n - refractive index immersion oil
    N_required - Intended number of photons of a scatterer per frame
    '''
    NA = MaximumNAByFocusDepth(d_channel, lambda_nm, n) # maximum NA
    
#    DE = DetectionEfficency(NA,n) # maximum 

    lambda_um = lambda_nm / 1000
    res_um = 1.22 * lambda_um / NA # resolution of PSF
    
#    A_max = np.pi * res_um**2 #maximal circle the particle should not leave in order to not blur (=D * t)
    
    # try several particle sizes
    d_m = np.logspace(np.log10(d_nm_min),np.log10(d_nm_max),50) * 1E-9
    

    w_illu = d_channel /1E9 /2
    
    #number of photons
    N_max = np.zeros_like(d_m)
    
    #max framerate
    t_min = np.zeros_like(d_m)
    
    for index, loop_d_m in enumerate(d_m):
        r_m = loop_d_m / 2
    
        # calc corresponding diffusion constant
        D_sqm = (scipy.constants.k * T)/(6*np.pi *visc_water * r_m)
        
        # maximum exposure time before blurr happens
        t_max = (res_um/1e6)**2 / D_sqm
#        t_max = A_max/(1e6**2) / D_sqm
        
        N_maximum = EstimateScatteringIntensity(P_illu, w_illu, lambda_nm, loop_d_m, t_max, NA, n, PrintSteps=False)
    
        if N_maximum > N_required:
            t_min[index] = N_required / N_maximum * t_max
        else:
            t_min[index] = t_max
        N_max[index] = N_maximum
    
    plt.figure()
    plt.loglog(d_m * 1E9, N_max, '.')
    plt.ylabel("Maximum number of photons", fontsize = 14)
    plt.xlabel("Diameter in [nm]", fontsize = 14)
    plt.grid(linewidth = 1, which='major')
    plt.grid(linewidth = 0.2, which='minor')
    plt.show() 
    
    
    plt.figure()
    plt.loglog(d_m * 1E9, t_min, '.')
    plt.ylabel("Minimum exposure time", fontsize = 14)
    plt.xlabel("Diameter in [nm]", fontsize = 14)
    plt.grid(linewidth = 1, which='major')
    plt.grid(linewidth = 0.2, which='minor')
    plt.show() 

    
    if readout_s != None:
        f_max = 1/(t_min+readout_s )
        plt.figure()
        plt.loglog(d_m * 1E9, f_max, '.')
        plt.ylabel("Max Framerate [Hz]", fontsize = 14)
        plt.xlabel("Diameter in [nm]", fontsize = 14)
        plt.grid(linewidth = 1, which='major')
        plt.grid(linewidth = 0.2, which='minor')
        plt.show() 
        
    
    print("\nParameters:")
    print("Channel diamter [um]: ", d_channel/1000)
    print("Maximum NA: ", np.round(NA,3))
    print("Detection efficency: ", np.round(DetectionEfficency(NA,n),3))
    print("Resolution [um]: ", np.round(res_um,2))
    print("Beam radius (waste) [um]: ", w_illu * 1e6)
    print("P_illu [W]: ", P_illu)
    print("Wavelength [nm]: ", lambda_nm)
    print("N required: ", N_required)
    
    return
    
    
    

    


def EstimateScatteringIntensity(P_illu, w_illu, lambda_illu, d, exposure_time, NA, n, PrintSteps = True, Mode = 'Peak'):
    '''
    P_illu - Power of illumination beam [W]
    w_illu - Beam waiste (radius) in m
    lambda_illu - illumination wavelength in nm
    d - diameter of particle in m
    exposure_time - exposure_time in s
    NA - considering air
    n - refractive index of immersion media
    Mode - calculate Intensity at the "Peak" or assume "FlatTop" Profile
    '''
    #Assume gaussian beam

    I_illu = nd.Theory.TheIntensityInFiber(P_illu, w_illu, Mode = Mode)

        
    # assume it is gold
    # crosssection in sq nm
    C_scat = CalcCrossSection(d, lambda_illu)
#    print("Scattering cross-section [sqnm]: ", C_scat)
    
    C_scat = C_scat / 1E9**2 # transform in sqm
    
    P_scat = I_illu * C_scat

    
    E_one_photon = scipy.constants.c * scipy.constants.h / (lambda_illu*1E-9) # In Joule
    
    N_scat_per_s = P_scat / E_one_photon #Number of scattered photons per second
    
    DE = DetectionEfficency(NA,n)# Detection efficency
    
    N_det = np.round(N_scat_per_s * DE * exposure_time)
    
    if PrintSteps == True:
        print("Illumination intensity [W/sqm]: ", I_illu)    
        print("Scattering power [W]: ", P_scat)
        print("Scattered photons [1/s]: ", N_scat_per_s)
        print("Detection efficency: ", DE)  
        print("Number of detected photons: ", N_det)

    
    return N_det



def CalcCrossSection(d, material = "gold", at_lambda_nm = None, do_print = True, n_media = 1.333):
    '''
    d - particle diameter in m
    
    https://miepython.readthedocs.io/en/latest/02_efficiencies.html?highlight=scattering%20cross%20section
    https://opensky.ucar.edu/islandora/object/technotes%3A232/datastream/PDF/view
    '''
    
    import miepython as mp
    import numpy as np
    import matplotlib.pyplot as plt

    if d > 1:
        print("WARNING: d should be given in the unit of meters!")

    # import the parameters from online library
    if material == 'gold':
        data = np.genfromtxt('https://refractiveindex.info/tmp/data/main/Au/Johnson.txt', delimiter='\t')
    if material == "silver":
        data = np.genfromtxt('https://refractiveindex.info/tmp/data/main/Ag/Johnson.txt', delimiter='\t') 
    if material == "polystyrene":
        data = np.genfromtxt('https://refractiveindex.info/tmp/data/organic/C8H8%20-%20styrene/Sultanova.txt', delimiter='\t') 
    if material == "DNA":
        data = np.genfromtxt('https://refractiveindex.info/tmp/data/other/human%20body/DNA/Inagaki.txt', delimiter='\t') 
    if material == "silica":
        data = np.genfromtxt('https://refractiveindex.info/tmp/data/main/SiO2/Malitson.txt', delimiter='\t') 


    # data is stacked so need to rearrange
    N = len(data)//2
    lambda_um = data[1:N,0]
#    lambda_um = lambda_um / n_media

    m_real = data[1:N,1]
    
    # some of the data do not have any complex part. than the reading in fails
    
    num_nan = len(data[np.isnan(data)])
    if num_nan == 4:
        m_imag = data[N+1:,1]
    else:
        m_imag = np.zeros_like(m_real)
    
    #lambda_um = lambda_um[30:35]
    # m_real = m_real[30:35]
    # m_imag = m_imag[30:35]
    
#    print("lambda_um: ", lambda_um)
#    print("m_real: ", m_real)
#    print("m_imag: ", m_imag)
    

    r_um = d*1e6/2 #radius in microns
    r_nm = r_um * 1000
    
#    print("particle radius in nm: ", r_nm)
    
    x = n_media*2*np.pi*r_um/lambda_um;
    m = (m_real - 1.0j * m_imag) / n_media
   
    qext, qsca, qback, g = mp.mie(m,x)

    # print("lambda_um: ", lambda_um[34])
    # print("m_re: ", m_real[34]) 
    # print("m_im: ", m_imag[34]) 
    # print("x: ", x[34]) 
    # print("m: ", m[34]) 
    # print(mp.mie(m[34],x[34])) 
    
    qabs = (qext - qsca)
    absorb  = qabs * np.pi * r_nm**2
    scat   = qsca * np.pi * r_nm**2
    extinct = qext* np.pi * r_nm**2
    
    lambda_nm = 1000 * lambda_um

    if at_lambda_nm  == None:
#        plt.plot(lambda_nm,qext,'g.-')   
#        plt.plot(lambda_nm,qext - qsca,'r.-')   
        plt.plot(lambda_nm,qsca,'k.-')   
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Efficency")
        plt.title("Efficency for %.1f nm Spheres" % (r_nm*2))
        plt.xlim([400, 800])
        
        plt.figure()
        plt.plot(lambda_nm,scat,'r.-')   
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Cross Section (1/$nm^2$)")
        plt.title("Cross Sections for %.1f nm Spheres" % (r_nm*2))
        
        plt.xlim(300,800)
        plt.show()
        
        C_Abs = absorb
        C_Scat = scat
        
    else:
        C_Scat = np.interp(at_lambda_nm, lambda_nm, scat)
        C_Abs   = np.interp(at_lambda_nm, lambda_nm, absorb)
        if do_print == True:
            print("Size parameter: ", np.interp(at_lambda_nm, lambda_nm, x))
            print("Scatterung efficency: ", np.interp(at_lambda_nm, lambda_nm, qsca))
            print("Scattering cross-section [nm²]: ", C_Scat)
    
    return C_Scat, C_Abs


def CalcCrossSection_OLD(lambda_nm, d_nm, material = "Gold", e_part = None):
    """ calculate the scattering crosssections of a scattering sphere
    
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
    
    C_scat = 8/3*pi*np.power(k,4)*np.power(r_m,6) * ((m**2-1)/(m**2+1))**2
    
    V = 4/3*pi*np.power(r_m,3)
    C_abs = k * np.imag(3 * V * (e_part - e_media)/(e_part + 2*e_media))
    
    print("\nC_scat [sqm]: ", C_scat)
    print("C_scat [sq nm]: ", C_scat / (1e-9**2))
    print("\nC_abs [sqm]: ", C_abs)
    print("C_abs [sq nm]: ", C_abs / (1e-9**2))
    
    return C_scat, C_abs




def MassOfNP(d_nm,rho):
    
    r_m = (d_nm/1E9) / 2
    V = 4/3*pi*np.power(r_m,2)
    
    m = V*rho
    
    
    return m


def E_Kin_Radiation_Force(F,m,t):
    E_kin = np.power(F*t,2)/(2*m)
    
    print(E_kin)
    
    return E_kin



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







def LocalAccuracyVsObjective():

    objective = np.array([
                 [0.1, 1],
                 [0.25, 1],
                 [0.5, 1],
                 [0.75, 1],
                 [1.3, 1.46]
                 ])
    
    num_z = 5
#    my_z = np.arange(0,num_z)
    my_z = np.linspace(0,8,num_z)
    
    loc_acu = np.zeros([num_z, np.shape(objective)[0]])
    signal  = np.zeros_like(loc_acu)
    
    for ii,use_objective in enumerate(objective):
        print("Iterate new objective")
        
        my_NA = use_objective[0]   
        my_n = use_objective[1]   
        for zz, loop_z in enumerate(my_z):
            print("Iterate new focus")
            mypsf, image, correl, loc_acu[zz, ii], signal[zz, ii] = DefocusCrossCorrelation(NA = my_NA, n=my_n, sampling_z = 1000, shape_z = 50, use_z = zz, ShowPlot = False)
    
    
    plt.figure()
    plt.subplot(211)
    plt.semilogy(my_z, loc_acu, ':x')
    plt.ylabel("loc accuracy [nm]", fontsize = 14)
    plt.xlabel("z [um]", fontsize = 14)
    plt.xlim([0, num_z+1])
    plt.legend(objective[:,0], title = "NA", fontsize = 14, title_fontsize = 14)
    
    plt.subplot(212)
    plt.semilogy(my_z, signal, ':x')
    plt.ylabel("Signal [a. u.]", fontsize = 14)
    plt.xlabel("z [um]", fontsize = 14)
    plt.xlim([0, num_z+1])
    plt.legend(objective[:,0], title = "NA", fontsize = 14, title_fontsize = 14)
    
    
    return loc_acu
        


def DefocusCrossCorrelation(NA = 0.25, n=1, sampling_z = None, shape_z = None, use_z = 0, ShowPlot = True):
    mpl.rc('image', cmap = 'gray')
    
    total_photons = 20000
    
    empsf, arg_psf = nd.Theory.PSF(NA,n, sampling_z, shape_z)


    mypsf = nd.Theory.RFT2FT(empsf.slice(use_z))
    mypsf = mypsf / np.sum(mypsf)
    
    num_det_photons = total_photons * nd.Simulation.DetectionEfficency(arg_psf["num_aperture"], arg_psf["refr_index"])
    
    obj = np.zeros_like(mypsf)
    obj[np.int(0.8*arg_psf["shape"][1]),np.int(1.1*arg_psf["shape"][1])] = 1
    
    
    #make image
    import scipy
    image_no_noise = np.real(scipy.signal.fftconvolve(obj, mypsf, mode = 'same'))
    image_no_noise[image_no_noise<0] = 1E-10
    image_no_noise = image_no_noise / np.sum(image_no_noise) * num_det_photons
    
    #background
    bg_no_noise = np.zeros_like(mypsf)
    bg_no_noise[:,:] = 1
    sigma_bg = 2.4 #Basler cam
    
    # simulated several noisy images to find out the localization accuracy of the center of mass approach
    num_runs = 5
    com = np.zeros([num_runs,2])
    signal = np.zeros(num_runs)
    
    for ii in range (num_runs):
#        print("argument PSF: ", arg_psf)
#        print("Number of detected photons: ", num_det_photons)
        
        image = np.random.poisson(image_no_noise)    
        bg    = np.random.normal(0, sigma_bg, bg_no_noise.shape)
        bg[bg<0] = 0
        
        image = np.round(image + bg)
        image[image < 0] = 0
        
        psf_zn = mypsf -np.mean(mypsf) #zero normalized psf
        correl = scipy.signal.correlate(image, mypsf, mode = 'same', method = 'fft')
        correl_zn = scipy.signal.correlate(image - np.mean(image), psf_zn, mode = 'same', method = 'fft')
        
        # find the particle
        my_diam = 7
        mylocate = tp.locate(correl, diameter = my_diam, separation = 50, preprocess = False, topn = 1)

        com[ii,:] = mylocate[["y", "x"]].values
        
        #select brightes spot only
#        print("numer of found particles: ", len(mylocate))
        mylocate = mylocate.sort_values("mass").iloc[0]
        
        com[ii,:] = mylocate[["y", "x"]].values

        signal[ii] = mylocate["mass"] / (np.mean(correl) * np.pi * (my_diam/2)**2)
            

    #localization accuracy
    loc_acu = np.std(com)
    signal = np.mean(signal)
    
    #into nm

    loc_acu = loc_acu * arg_psf["sampling"][1]

    if ShowPlot == True:
        print("mean center of mass = ", np.mean(com,axis = 0))
        print("std center of mass = ", np.std(com,axis = 0))
        print("std center of mass total = ", loc_acu )
        
        import matplotlib.pyplot as plt
        # plt.rcParams['figure.figsize'] = [100, 5]
        plt.figure(figsize=(15, 10))
        # plt.rcParams['figure.figsize'] = [100, 5]

    
        fsize = 20
        plt.subplot(231)
        plt.imshow(obj)
        plt.title("object", fontsize = fsize)

        
        plt.subplot(232)
        plt.imshow(image_no_noise)
        plt.title("image - without noise", fontsize = fsize)
        
        plt.subplot(233)
        plt.imshow(image)
        plt.title("image - with noise", fontsize = fsize)
        
        # plt.subplot(334)
        # plt.imshow(mypsf)
        # plt.title("correl - PSF")

        plt.subplot(234)
        plt.imshow(correl)
        plt.title("image - SNR improved", fontsize = fsize)
        plt.xlabel("[Px]", fontsize = fsize)
        plt.ylabel("[Px]", fontsize = fsize)
        
        # plt.subplot(234)
        # plt.imshow(psf_zn)
        # plt.title("correl - PSF zn")
        
        plt.subplot(235)
        plt.imshow(correl_zn)
        plt.title("image - SNR improved 2", fontsize = fsize)
        
        plt.subplot(236)
        rl_decon = nd.Theory.DeconRLTV(image, mypsf, 0.01, num_iterations = 100)
        # rl_decon = nd.Theory.DeconRLTV(image, mypsf, 0.0, num_iterations = 20)
        plt.imshow(rl_decon)
        plt.title("Richardson-Lucy", fontsize = fsize)
        
        plt.show()
#        rawframes_filtered[loop_frames,:,:] = np.real(np.fft.ifft2(ndimage.fourier_gaussian(np.fft.fft2(rawframes_np[loop_frames,:,:]), sigma=[gauss_kernel_rad  ,gauss_kernel_rad])))


    return mypsf, image, correl, loc_acu, signal, empsf





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
    