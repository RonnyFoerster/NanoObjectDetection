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

#import matplotlib.pyplot as plt # Libraries for plotting
import NanoObjectDetection as nd

from pdb import set_trace as bp #debugger

from numpy import pi
import scipy
import scipy.signal
from scipy.constants import Boltzmann as k_b

# import miepython as mp

import trackpy as tp

"""
Simulation

In case the data is simulated and not acquired physically the parameters can be found here.

| key: SimulateData
| description: Boolean if the data shall be simulated
| example: 1
| unit: boolean

| key: DiameterOfParticles
| description: Diameter(s) of the simulated particles (float or list of float)
| example: [50, 80]
| unit: nm

| key: NumberOfParticles
| description: Number(s) of created particles (int or list of int)
| example: 42
| unit: 

| key: NumberOfFrames
| description: Number of simulated frames
| example: 420
| unit: frames

| key: NumMicrosteps
| description: Number of microsteps per exposure time (NOT SWITCHED ON YET)
| example: 5
| unit: None

| key: mass
| description: Mass of the particles (float or list of float)
| example: 100
| unit: None

| key: Photons
| description: Number of photons 
| example: 100
| unit: None
    
| key: EstimationPrecision
| description: Estimation precision 
| example: !!! TODO !!!
| unit: !!! TODO !!!

| key: Max_traj_length
| description: Maximum trajectory length. Longer trajectories are cut and treated as independent particles at different time points.
| example: 300
| unit: frames
"""

## In[Functions]
def PrepareRandomWalk(ParameterJsonFile = None, diameter = 100, num_particles = 1, frames = 100, EstimationPrecision = 0, mass = 0, frames_per_second = 100, microns_per_pixel = 1, temp_water = 293, visc_water = 9.5e-16, seed_startpos=None, oldSim=False):
    """ configure the parameters for a randowm walk out of a JSON file, and generate
    it in a DataFrame

    if seed_startpos is not None, random trajectory starting points are generated,
    otherwise the starting position for every particle is (0,0)
    """

    # print error, if filepath was not entered - error is thrown later anyways
    try:
        settings = nd.handle_data.ReadJson(ParameterJsonFile)
    except TypeError:
        print('You did not enter a filepath to the parameter json file. \nPlease do so and repeat.')

    # keys that may contain lists of values
    diameter            = settings["Simulation"]["DiameterOfParticles"]
    num_particles       = settings["Simulation"]["NumberOfParticles"]
    mass                = settings["Simulation"]["mass"]

    frames              = settings["Simulation"]["NumberOfFrames"]
    EstimationPrecision = settings["Simulation"]["EstimationPrecision"]

    # try:
    Photons = settings["Simulation"]["Photons"]
    # except KeyError:
    #     print('Photons not found in json parameter file. Take default value: 1000')
    #     Photons = 1000

    frames_per_second   = settings["Exp"]["fps"]
    microns_per_pixel   = settings["Exp"]["Microns_per_pixel"]
    temp_water          = settings["Exp"]["Temperature"]

    # NA = settings["Exp"]["NA"]
    # my_lambda = settings["Exp"]["lambda"]

    # #sigma PSF in nm
    # sigma_PSF = nd.Theory.SigmaPSF(NA, my_lambda)

    # diffusion = nd.Theory.StokesEinsteinEquation(diam = None, temp_water = 295, visc_water = 9.5E-16)

    # ep = nd.theory.LocalError(sigma_PSF, diffusion, t_exp, photons)

    solvent = settings["Exp"]["solvent"]

    # define a field of view (FoV) for the simulation
    FoVheight = settings["Fiber"]["TubeDiameter_nm"]*0.001 /microns_per_pixel # px
    try:
        FoVlength = settings["Simulation"]["FoVlength"] # px
    except KeyError:
        nd.logging.warning('FoV length not found in json parameter file. Take default value: 1000px')
        FoVlength = 1000


    if settings["Exp"]["Viscosity_auto"] == 1:
        visc_water = nd.handle_data.GetViscocity(temperature = temp_water, solvent = solvent)
    else:
        visc_water = settings["Exp"]["Viscosity"]

    # ========== up to here, parameters should be completely defined ============

    if not(type(diameter)==list):
        diameter = [diameter]
    if not(type(num_particles)==list):
        num_particles = [num_particles] * len(diameter) # adjust list lengths
    else:
        if not(len(num_particles)==len(diameter)):
            nd.logging.warning('Given diameters and number of particles are not equal. Please adjust.')
            # this provides the info... an error will be thrown later in the loop automatically
    if not(type(mass)==list):
        mass = [mass] * len(diameter) # adjust list lengths
    else:
        if not(len(mass)==len(diameter)):
            nd.logging.warning('Given diameters and mass values are not equal. Please adjust.')
            # this provides the info... an error will be thrown later in the loop automatically

    output = pd.DataFrame()
    # loop over all given diameters
    for n_d in range(len(diameter)):

        start_pos=None
        if not(seed_startpos is None):
            # initialize a random generator with fixed seed
            rng=np.random.default_rng(seed_startpos + n_d)
            # create random trajectory starting points, uniformly distributed within the FoV
            start_pos_x = FoVlength * rng.random(num_particles[n_d])
            start_pos_y = FoVheight * rng.random(num_particles[n_d])
            start_pos = np.column_stack((start_pos_x, start_pos_y)) # px


        if oldSim==True:
            objall = GenerateRandomWalk_old(diameter[n_d], num_particles[n_d], frames, frames_per_second,
                                            ep = EstimationPrecision,
                                            mass = mass[n_d], microns_per_pixel = microns_per_pixel,
                                            temp_water = temp_water, visc_water = visc_water)
        else:
            objall = GenerateRandomWalk(diameter[n_d], num_particles[n_d], frames,
                                        frames_per_second, ep = EstimationPrecision,
                                        mass = mass[n_d], microns_per_pixel = microns_per_pixel,
                                        temp_water = temp_water, visc_water = visc_water,
                                        start_pos = start_pos)

        # adjust the particle IDs so that they don't appear twice (or more)
        if any(output):
            objall.particle = objall.particle + output.particle.max() + 1
        output = pd.concat([output,objall])
        output = output.reset_index() # create a continuous index for the complete DataFrame
        output = output.drop(['index'],axis=1) # delete the old one (which contains duplicates for n_d>0)


    if ParameterJsonFile != None:
        # write if para file is given
        nd.handle_data.WriteJson(ParameterJsonFile, settings)

    return output


# def CalcDiffusionCoefficent(radius_m, temp_water = 293, visc_water = 0.001):
#     diffusion = (k_b*temp_water)/(6*math.pi *visc_water * radius_m) # [um^2/s]
#     return diffusion


def GenerateRandomWalk(diameter, num_particles, frames, frames_per_second, t_exp = 0,
                       num_microsteps = 1, ep = 0, mass = 1, microns_per_pixel = 0.477,
                       temp_water = 295, visc_water = 9.5e-16, PrintParameter = True,
                       start_pos = None, NumDims = 2):
    """ simulate a random walk of Brownian diffusion and return it as Pandas.DataFrame
    object as if it came from real data:

    diameter:       particle size in nm
    num_particles:  number of particles to simulate
    frames:         number of frames to simulate
    frames_per_second
    t_exp           exposure time of one frame
    num_microsteps  number of microsteps in each frame
    ep = 0:         estimation precision
    mass = 1:       mass of the particle
    microns_per_pixel = 0.477
    temp_water = 295 K
    visc_water = 9.5e-16:
    """


    if PrintParameter == True:
        print("Random walk parameters: \
              \n diameter = {} \
              \n num_particles = {} \
              \n frames = {} \
              \n frames_per_second = {} \
              \n exposure time = {} \
              \n number of microsteps = {} \
              \n ep = {} \
              \n mass = {} \
              \n microns_per_pixel = {} \
              \n temp_water = {} \
              \n visc_water = {}" \
              .format(diameter, num_particles, frames, frames_per_second, t_exp,  num_microsteps, ep, mass, microns_per_pixel, temp_water, visc_water))

    radius_m = diameter/2 * 1e-9 # in m

    # diffusion constant of the simulated particle (Stokes-Einstein eq.)
    sim_part_diff = (k_b*temp_water)/(6*pi *visc_water * radius_m) # [um^2/s]

    print("Diffusion coefficent: ", sim_part_diff)

    # [mum^2/s] x-diffusivity of simulated particle
    # exposure time of a microstep
    t_exp_step = t_exp / num_microsteps

    # time one frame is long
    t_frame = 1/frames_per_second

    #readout time
    t_readout = t_frame - t_exp

    # sigma of diffusion in a microstep during exposure in um
    sim_part_sigma_um_step = np.sqrt(2*sim_part_diff * t_exp_step)

    # convert to px
    sim_part_sigma_x_step = sim_part_sigma_um_step / microns_per_pixel

    # sigma of diffusion during readout in um
    sim_part_sigma_um_readout = np.sqrt(2*sim_part_diff * t_readout)

    # convert to px
    sim_part_sigma_x_readout = sim_part_sigma_um_readout / microns_per_pixel

    # std of random walk for one frame. it consists of num_microsteps diffusion steps during the exposure with a std of sim_part_sigma_x_step and one diffusion step during readout
    sim_part_sigma_x_microstep = np.append(np.repeat(sim_part_sigma_x_step, num_microsteps), sim_part_sigma_x_readout)

    # for later cumsum save if a step is exposed (visible) or readout (invisible)
    step_mode = np.append(np.repeat("exp", num_microsteps), "readout")

    # copy the diffusion std and mode for all frames and particles
    sim_part_sigma_x = np.tile(sim_part_sigma_x_microstep, frames * num_particles)
    step_mode  = np.tile(step_mode , frames * num_particles)

    # number of required random walks per frame (including the readout walk)
    steps_per_frame = num_microsteps+1

    # number of random walks
    num_elements = steps_per_frame * frames * num_particles

    # here we save the random walks into
    if NumDims == 1:
        sim_part = pd.DataFrame(columns = ["frame", "particle", "step", "dx", "x"], index = range(0,num_elements))

    elif NumDims == 2:
        sim_part = pd.DataFrame(columns = ["frame", "particle", "step", "dx", "x", "dy", "y"], index = range(0,num_elements))

    # create frame number
    frame_numbers = np.arange(0,frames)

    # each frame number is repeated steps_per_frame due to microsteps and readout
    frame_microsteps = np.repeat(frame_numbers, steps_per_frame)

    # procedure is repeated for all particles
    frame_tot = np.tile(frame_microsteps, num_particles)

    sim_part["frame"] = frame_tot

    #fill the particle row
    sim_part["particle"] = np.repeat(np.arange(num_particles),steps_per_frame*frames)

    # make shift as random number for all particles and frames
    sim_part["dx"] = np.random.normal(loc = 0, scale=sim_part_sigma_x, size = num_elements)

    #first frame of each particle should have dx and dy = 0. It is just disturbing and has no effect
    sim_part.loc[sim_part.particle.diff(1) != 0, "dx"] = 0

    if NumDims == 2:
         sim_part["dy"] = np.random.normal(loc = 0, scale=sim_part_sigma_x, size = num_elements)
         sim_part.loc[sim_part.particle.diff(1) != 0, "dy"] = 0

    #save if exposed or read out
    sim_part["step"] = step_mode

    # move every trajectory to starting position, if provided
    if start_pos != None:
        # sim_part["x"] = sim_part["x"] + np.repeat(start_pos[:,0],frames)

        print("RF: Guess that needs some debugging when someone is using it.")

        # RF210114 - set dx position in first frame for every particle to starting position
        sim_part.loc[(sim_part.frame == 0) & (sim_part.step == 'exp') & (sim_part.dx == 0), "dx"] = start_pos

        if NumDims == 2:
            print("COPY HERE FROM ABOVE!")
        # sim_part["y"] = sim_part["y"] + np.repeat(start_pos[:,1],frames)

    # sum up the individual steps over time via a cumsum to get the particle position over time
    sim_part["x"] = sim_part[["particle", "dx"]].groupby("particle").cumsum()

    if NumDims == 2:
        sim_part["y"] = sim_part[["particle", "dy"]].groupby("particle").cumsum()


    # average of microsteps position in each frame and particle. this is where the center of mass of the localization is
    # pos_avg = sim_part[sim_part.step == "exp"].groupby(["particle", "frame"]).mean()[["x","y"]]

    if NumDims == 1:
        pos_avg = sim_part[sim_part.step == "exp"].groupby(["particle", "frame"]).mean()["x"]
    elif NumDims == 2:
        pos_avg = sim_part[sim_part.step == "exp"].groupby(["particle", "frame"]).mean()[["x","y"]]

    ep = ep*1E6 / microns_per_pixel
    # only one var for x and y - not sure here - maybe no y would be better
    if num_microsteps > 1:
        # variance of the localization
        # pos_var = sim_part[sim_part.step == "exp"].groupby(["particle", "frame"]).var()[["x","y"]]

        if NumDims == 1:
            pos_var = sim_part[sim_part.step == "exp"].groupby(["particle", "frame"]).var()["x"]
        elif NumDims == 2:
            pos_var = sim_part[sim_part.step == "exp"].groupby(["particle", "frame"]).var()[["x","y"]]
            #make geometric mean
            pos_var = np.sqrt(pos_var["x"]*pos_var["y"])

        #average of mean is 1/4 and 1/4 of variances
        #pos_var = (pos_var["x"] + pos_var["y"]) / 4
        # pos_var  = pos_var["x"]
        # the uncertainty of the movement and the photon limited localization noise is convolved. the convolution of two gaussians is a gaussian with var_prod = var1 + var2
        motion_ep = np.sqrt(ep**2 + pos_var)
        motion_ep = motion_ep.values

    else:
        motion_ep = ep


    sim_part_tm = pos_avg.reset_index()

    sim_part_tm["mass"] =mass
    #sim_part_tm['ep'] = ep

    sim_part_tm['ep'] = motion_ep

    sim_part_tm["size"] = 0
    sim_part_tm["ecc"] = 0
    sim_part_tm["signal"] = 0
    sim_part_tm["raw_mass"] = mass
    sim_part_tm["rel_step"] = mass
    sim_part_tm["abstime"] = sim_part_tm["frame"] * t_frame


    #insert localization accuracy to theoretical know position
    if np.max(motion_ep) > 0:
        sim_part_tm.x = sim_part_tm.x + np.random.normal(0, sim_part_tm.ep)

        if NumDims == 2:
            sim_part_tm.y = sim_part_tm.y + np.random.normal(0, sim_part_tm.ep)


    return sim_part_tm


def GenerateRandomWalk_old(diameter, num_particles, frames, frames_per_second,
                           RatioDroppedFrames=0, ep=0, mass=1, microns_per_pixel=0.477,
                           temp_water=295, visc_water=9.5e-16):
    """ previous function - restored for comparison and re-running of older scripts

    simulate a random walk of Brownian diffusion and return it as Pandas.DataFrame as
    if it came from real data

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

    print("Random walk parameters: \
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
    sim_part_diff = (const_Boltz*temp_water)/(6*math.pi *visc_water * radius_m)
    # unit sim_part_diff = um^2/s

    print("Diffusion coefficent: ", sim_part_diff)

    # [mum^2/s] x-diffusivity of simulated particle
    sim_part_sigma_um = np.sqrt(2*sim_part_diff / frames_per_second)
    sim_part_sigma_x = sim_part_sigma_um / microns_per_pixel
    # [pixel/frame] st.deviation for simulated particle's x-movement (in pixel units!)

    # generating list to hold frames:
    sim_part_frame=[]
    for sim_frame in range(frames):
        sim_part_frame.append(sim_frame)
    sim_part_frame_list=sim_part_frame*num_particles

    # generating lists to hold particle IDs and
    # step lengths in x and y direction, coming from a Gaussian distribution
    sim_part_part=[]
    sim_part_x=[]
    sim_part_y=[]

    drop_rate = RatioDroppedFrames


    for sim_part in range(num_particles):
        loop_frame_drop = 0
        for sim_frame in range(frames):
            sim_part_part.append(sim_part)
            sim_part_x.append(np.random.normal(loc=0,scale=sim_part_sigma_x))
            sim_part_y.append(np.random.normal(loc=0,scale=sim_part_sigma_x))
            # Is that possibly wrong??



    # Putting the results into a df and formatting correctly:
    sim_part_tm=pd.DataFrame({'x':sim_part_x, \
                              'y':sim_part_y,  \
                              'mass':mass, \
                              'ep': ep, \
                              'frame':sim_part_frame_list, \
                              'particle':sim_part_part, \
                              "size": 0, \
                              "ecc": 0, \
                              "signal": 0, \
                              "raw_mass": mass, \
                              "rel_step": mass})

    # calculate cumulative sums to get position values from x- and y-steps for the full random walk
    sim_part_tm.x=sim_part_tm.groupby('particle').x.cumsum()
    sim_part_tm.y=sim_part_tm.groupby('particle').y.cumsum()


#    sim_part_tm.index=sim_part_tm.frame # old method RF 190408
#    copies frame to index and thus exists twice. not good
#    bp()

    # here come the localization precision ep on top
    if ep>0:
        sim_part_tm.x = sim_part_tm.x + np.random.normal(0,ep,len(sim_part_tm.x))
        sim_part_tm.y = sim_part_tm.y + np.random.normal(0,ep,len(sim_part_tm.x))


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


def DepthOfField(NA, n, my_lambda):
    # https://doi.org/10.1111/j.1365-2818.1988.tb04563.x
    # Sheppard Depth of field in optical microscopy
    # Eq 8
    alpha = np.arcsin(NA/n)
    dz = 1.77 * my_lambda / (4 * (np.sin(alpha/2))**2 * (1-1/3*(np.tan(alpha/2))**4))

    return dz



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



def n_water20(wl):
    """ calculates refractive index of water at a given wavelength in um

    Daimon, Masumura 2007
    https://refractiveindex.info/tmp/data/main/H2O/Daimon-20.0C.html
    """
    # analytical expression for the refr. index at 20°C
    n_water = ( 1 + 5.684027565E-1/(1-5.101829712E-3/wl**2) +
                1.726177391E-1/(1-1.821153936E-2/wl**2) +
                2.086189578E-2/(1-2.620722293E-2/wl**2) +
                1.130748688E-1/(1-1.069792721E1/wl**2) )**.5
    return n_water



# def CalcCrossSection(d, material = "gold", at_lambda_nm = None, do_print = True, n_medium = 1.333):
#     """ calculate the scattering and absorption crosssection of a spherical particle

#     https://miepython.readthedocs.io/en/latest/02_efficiencies.html?highlight=scattering%20cross%20section
#     https://opensky.ucar.edu/islandora/object/technotes%3A232/datastream/PDF/view

#     Parameters
#     ----------
#     d :             particle diameter in nm
#     material :      particle material (implemented: gold, silver, polystyrene, DNA, silica)
#     at_lambda_nm :  wavelength of the incident light, "None" creates a plot for the full VIS range
#     do_print :      print the results if TRUE
#     n_medium :      refr. index of surrounding medium, "None" uses Daimon2007 data for water at 20°C

#     Returns
#     -------
#     C_Scat, C_Abs : scattering and absorption cross section in nm^2
#     """


#     if d > 1:
#         print("WARNING: d should be given in the unit of meters!")

#     # import the parameters from online library
#     if material == 'gold':
#         data = np.genfromtxt('https://refractiveindex.info/tmp/data/main/Au/McPeak.txt', delimiter='\t')
#     elif material == "silver":
#         data = np.genfromtxt('https://refractiveindex.info/tmp/data/main/Ag/Johnson.txt', delimiter='\t')
#     elif material == "polystyrene":
#         data = np.genfromtxt('https://refractiveindex.info/tmp/data/organic/C8H8%20-%20styrene/Sultanova.txt', delimiter='\t')
#     elif material == "DNA":
#         data = np.genfromtxt('https://refractiveindex.info/tmp/data/other/human%20body/DNA/Inagaki.txt', delimiter='\t')
#     elif material == "silica":
#         data = np.genfromtxt('https://refractiveindex.info/tmp/data/main/SiO2/Malitson.txt', delimiter='\t')

#     # data is stacked so need to rearrange
#     N = len(data)//2
#     lambda_um = data[1:N,0]

#     if n_medium == None:
#         if at_lambda_nm == None:
#             n_medium = np.array([n_water20(lam) for lam in lambda_um])
#         else:
#             n_medium = n_water20(at_lambda_nm)

# #    lambda_um = lambda_um / n_medium

#     m_real = data[1:N,1]

#     # some of the data do not have any complex part. than the reading in fails

#     num_nan = len(data[np.isnan(data)])
#     if num_nan == 4:
#         m_imag = data[N+1:,1]
#     else:
#         m_imag = np.zeros_like(m_real)

#     #lambda_um = lambda_um[30:35]
#     # m_real = m_real[30:35]
#     # m_imag = m_imag[30:35]

# #    print("lambda_um: ", lambda_um)
# #    print("m_real: ", m_real)
# #    print("m_imag: ", m_imag)


#     r_um = d*1e6/2 #radius in microns
#     r_nm = r_um * 1000

# #    print("particle radius in nm: ", r_nm)

#     x = n_medium*2*np.pi*r_um/lambda_um;
#     m = (m_real - 1.0j * m_imag) / n_medium

#     qext, qsca, qback, g = mp.mie(m,x)

#     # print("lambda_um: ", lambda_um[34])
#     # print("m_re: ", m_real[34])
#     # print("m_im: ", m_imag[34])
#     # print("x: ", x[34])
#     # print("m: ", m[34])
#     # print(mp.mie(m[34],x[34]))

#     qabs = (qext - qsca)
#     absorb  = qabs * np.pi * r_nm**2
#     scat   = qsca * np.pi * r_nm**2
#     extinct = qext* np.pi * r_nm**2

#     lambda_nm = 1000 * lambda_um

#     if at_lambda_nm  == None:
# #        plt.plot(lambda_nm,qext,'g.-')
# #        plt.plot(lambda_nm,qext - qsca,'r.-')
#         plt.plot(lambda_nm,qsca,'k.-')
#         plt.xlabel("Wavelength (nm)")
#         plt.ylabel("Efficency")
#         plt.title("Scattering efficency for %.1f nm %s spheres" % (r_nm*2,material))
#         plt.xlim([400, 800])

#         plt.figure()
#         plt.plot(lambda_nm,scat,'r.-')
#         plt.xlabel("Wavelength (nm)")
#         plt.ylabel("Cross section ($nm^2$)")
#         plt.title("Scattering cross section for %.1f nm %s spheres" % (r_nm*2,material))

#         plt.xlim(300,800)
#         plt.show()

#         C_Abs = absorb
#         C_Scat = scat

#     else:
#         C_Scat = np.interp(at_lambda_nm, lambda_nm, scat)
#         C_Abs   = np.interp(at_lambda_nm, lambda_nm, absorb)
#         if do_print == True:
#             print("Size parameter: ", np.interp(at_lambda_nm, lambda_nm, x))
#             print("Scattering efficency: ", np.interp(at_lambda_nm, lambda_nm, qsca))
#             print("Scattering cross-section [nm²]: ", C_Scat)

#     return C_Scat, C_Abs



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
    """ simulate the random walk

    MN2011 obsolete function?
    """

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



def RandomSamplesFromDistribution(N,mean,CV,seed=None):
    """ generate N randomly chosen sizes from a Gaussian distribution
    with given mean and CV (coefficient of variation = std/mean)
    """
    # use Generator(PCG64) from numpy
    if seed == None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed=seed)

    sigma = CV*mean
    sample = rng.normal(mean, sigma, N)

    return sample



def RandomWalkCrossSection(settings = None, D = None, traj_length = None, dt = None, r_max = None, num_particles = 10, ShowHist = False, ShowTraj = False, ShowReflection = False):
    """
    nd.Simulation.RandomWalkCrossSection(D = 13, traj_length=10000, dt=1/500, r_max = 8, ShowTraj = True, num_particles = 10, ShowReflection = True)
    """

    if settings != None:
        r_max = settings["Fiber"]["TubeDiameter_nm"] /2 /1000
        dt = 1/settings["Exp"]["fps"]

    num_elements = traj_length * num_particles

    sigma_step = np.sqrt(2*D*dt)


    sim_part = pd.DataFrame(columns = ["particle", "frame", "step", "dx", "x", "dy", "y", "r", "I"], index = range(0,num_elements))

    # create frame number
    frame_numbers = np.arange(0,traj_length)

    # procedure is repeated for all particles
    sim_part["frame"] =  np.tile(frame_numbers, num_particles)


    #fill the particle row
    sim_part["particle"] = np.repeat(np.arange(num_particles),traj_length)

    # make shift as random number for all particles and frames
    sim_part["dx"] = np.random.normal(loc = 0, scale=sigma_step, size = [num_elements])

    sim_part["dy"] = np.random.normal(loc = 0, scale=sigma_step, size = [num_elements])


    #first frame of each particle should have dx and dy = 0. It is just disturbing and has no effect
    # sim_part.loc[sim_part.particle.diff(1) != 0, "dx"] = 0
    # sim_part.loc[sim_part.particle.diff(1) != 0, "dy"] = 0

    #make starting position
    for ii in range(num_particles):
        valid_r = False
        while valid_r == False:
            try_new_x = np.random.uniform(low = 0, high = r_max)
            try_new_y = np.random.uniform(low = 0, high = r_max)
            try_new_r = np.hypot(try_new_x, try_new_y)

            if try_new_r <= r_max:
                valid_r = True
                if ii == 0:
                    x_start = try_new_x
                    y_start = try_new_y
                else:
                    x_start = np.append(x_start, try_new_x)
                    y_start = np.append(y_start, try_new_y)


    sim_part.loc[sim_part.particle.diff(1) != 0, "dx"] = x_start
    sim_part.loc[sim_part.particle.diff(1) != 0, "dy"] = y_start

    for ii_part in range(num_particles):
        print(ii_part)
        dx = sim_part[sim_part.particle == ii_part]["dx"].values
        dy = sim_part[sim_part.particle == ii_part]["dy"].values


        [x,y, num_hits] = ConfineTraj(dx, dy, r_max, ShowReflection = ShowReflection)

        nd.logger.debug("number of hits: ", num_hits)

        sim_part.loc[sim_part.particle == ii_part, "x"] = x
        sim_part.loc[sim_part.particle == ii_part, "y"] = y

    sim_part["x"] = sim_part["x"].astype(float)
    sim_part["y"] = sim_part["y"].astype(float)

    sim_part.loc[sim_part.particle.diff(1) != 0, "dx"] = 0
    sim_part.loc[sim_part.particle.diff(1) != 0, "dy"] = 0


    sim_part["r"] = np.hypot(sim_part["x"], sim_part["y"])

    # import trackpy as tp
    # im = tp.imsd(sim_part, 1, 1/dt, max_lagtime = int(traj_length/5))
    # plt.figure()
    # plt.plot(im.index,im, ':x')

    #here comes the Mode
    def GaussInt(r, waiste):
        return np.exp(-(r/waiste)**2)

    waiste = r_max / 2

    sim_part["I"] = GaussInt(sim_part["r"], waiste)

    I_mean = sim_part[["particle", "I"]].groupby("particle").mean()
    I_mean = np.asarray(I_mean["I"])

    print("I_mean: ", I_mean)

    # plt.figure()
    # # plt.hist(sim_part["r"], bins = 100)
    # plt.hist(sim_part["I"], bins = 100)

    I_mean_mean = np.mean(I_mean)
    I_mean_std = np.std(I_mean)

    # print("mean: %.3f" % I_mean_mean)
    # print("std: %.3f" % I_mean_std)

    CI68_low = np.percentile(I_mean, q = 16)
    CI68_high = np.percentile(I_mean, q = 84)

    # print("10%% percentile: %.3f" %I10)
    # print("90%% percentile: %.3f" %I90)

    if ShowHist == True:
        plt.figure()
        print(int(num_particles / 10))
        plt.hist(I_mean, bins = int(num_particles / 10))


    if ShowTraj == True:
        plt.figure()
        tp.plot_traj(sim_part[sim_part.particle == 0], colorby='frame')
        ax = plt.gca()
        ax.set_xlim([-r_max, r_max])
        ax.set_ylim([-r_max, r_max])

    return CI68_low, I_mean_mean, CI68_high, sim_part


def ConfineTraj(dx, dy, r_max, ShowReflection = False):
    "Confines a trajectory inside a circle of radius r"
    x = dx.copy()
    y = dy.copy()

    #starting position
    x[0] = dx[0]
    y[0] = dy[0]

    frames = len(x)

    num_hits = 0

    for ii in range(1,frames):
        x[ii] = x[ii-1] + dx[ii]
        y[ii] = y[ii-1] + dy[ii]

        r_ii = np.hypot(x[ii],y[ii])

        if r_ii > r_max:
            num_hits = num_hits + 1
            [x[ii], y[ii]] = ReflectTraj(x[ii-1], y[ii-1], dx[ii], dy[ii], r_max, ShowReflection = ShowReflection)

    return x,y, num_hits


def Test_ReflectTraj():
    nd.Simulation.ReflectTraj(0, 0, np.sqrt(50), 0, 6)
    nd.Simulation.ReflectTraj(-2, 0, -6, 0, 6)
    nd.Simulation.ReflectTraj(0, 2, 0, 6, 6)
    nd.Simulation.ReflectTraj(0, -2, 0, -6, 6)
    nd.Simulation.ReflectTraj(0, 0, 5, 5, 6)
    nd.Simulation.ReflectTraj(-2, -2, -5, -5, 6)
    nd.Simulation.ReflectTraj(-4.24, 3, 0, +2, 6)
    nd.Simulation.ReflectTraj(-5.5, 0 , 0.5, -4, 6)
    nd.Simulation.ReflectTraj(+5.5, 0 , 0.5, -4, 6)
    nd.Simulation.ReflectTraj(+5.5, 0 , 0.5, +4, 6)
    nd.Simulation.ReflectTraj(+1, 5 , -0, +5, 6)

def ReflectTraj(x1, y1, dx, dy, r, ShowReflection = False):
    """reflects particle trajectory it leaves a given circle"""

    # calc position outside boundary
    x_out = x1 + dx
    y_out = y1 + dy

    # get the p-value
    # p is in [0,1] and says after which amount of the movement, the particle his the wall

    # solve quadratic function sothat x and y are on the circle
    a = (2*dx*x1 + 2*dy*y1)/(dx**2 + dy**2)
    b = (x1**2 + y1**2 - r**2)/(dx**2 + dy**2)

    p_1 = -a/2 + np.sqrt((a/2)**2 - b)
    p_2 = -a/2 - np.sqrt((a/2)**2 - b)

    # p_2 is laselect p
    if p_2 > 0:
        p = p_2
    else:
        p = p_1


    #this is the point where the particle hits the wall
    x_c = x1 + p *dx
    y_c = y1 + p *dy

    # angle in polar coordinates where particle hits the wall
    if y_c >= 0:
        phi = np.arccos(x_c / r)
    else:
        phi = -np.arccos(x_c / r)

    # elemenatry surface vector
    # https://math.stackexchange.com/questions/13261/how-to-get-a-reflection-vector
    e_n = np.array([-np.cos(phi), -np.sin(phi)])

    # incoming remaining missplacement vector
    d_i = (1-p) * np.array([dx, dy])

    # reflected remaining misplacement vector
    [dx_r, dy_r] = d_i - (2*np.dot(d_i, e_n) * e_n)

    x2 = x_c + dx_r
    y2 = y_c + dy_r

    new_r = np.hypot(x2,y2)

    if ShowReflection == True:
        plt.plot([x1, x_out],[y1, y_out], ':x')
        plt.plot([x1, x_c, x2],[y1, y_c, y2], '-x')

        circle = plt.Circle((0,0), radius = r, fill = False)
        ax = plt.gca()
        ax.add_patch(circle)

        # print("p= ", p)
        # print("Phi= ",phi/np.pi*180)
        # print("[x1_parr, x1_perp]: ", [x1_parr, x1_perp])
        # print("[dx1_parr, dx1_perp]: ", [dx1_parr, dx1_perp])
        # print("[x2_parr, x2_perp]: ", [x2_parr, x2_perp])
        # print( "[x_c_back, y_c_back]", [x_c_back, y_c_back])


    if new_r > r:
        x2, y2 = ReflectTraj(x_c, y_c, dx_r, dy_r, r)

    return x2, y2


def ChangeCordSystem(theta, dx_c, dy_c, x_in, y_in):
    """
    Translate Coordinate System to new center dx_c and dx_y
    Rotate by angle theta
    Return transformed coordinates (x2,y2) of input system (x1,y1)
    """

    # https://moonbooks.org/Articles/How-to-create-and-apply-a-rotation-matrix-using-python-/

    # translate to new system
    x_out = x_in - dx_c
    y_out = y_in - dy_c

    # rot array
    R = np.array(( (np.cos(theta), -np.sin(theta)),
               (np.sin(theta),  np.cos(theta)) ))

    # rotate it
    [x_out, y_out] = R.dot([x_out, y_out])

    return x_out, y_out

def HistDistanceBetweenParticles():
    num_part = 200
    
    pos = np.sort(np.random.random(num_part)*4000)
    
    diff = pos[1:]-pos[:-1]
    
    min_dist = np.min([diff[:-1], diff[1:]], axis = 0)
    
    plt.hist(min_dist, bins = int(num_part/5))