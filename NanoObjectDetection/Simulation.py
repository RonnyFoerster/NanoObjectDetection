# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:17:36 2019

@author: Ronny Förster und Stefan Weidlich and Jisoo
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import pi
import scipy
from scipy.constants import Boltzmann as k_b
import trackpy as tp


#import matplotlib.pyplot as plt # Libraries for plotting
import NanoObjectDetection as nd




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

    if ParameterJsonFile != None:
        nd.logger.info('Get simulation parameters from json')
        
        settings = nd.handle_data.ReadJson(ParameterJsonFile)


        # keys that may contain lists of values
        diameter            = settings["Simulation"]["DiameterOfParticles"]
        num_particles       = settings["Simulation"]["NumberOfParticles"]
        mass                = settings["Simulation"]["mass"]
        frames              = settings["Simulation"]["NumberOfFrames"]
        EstimationPrecision = settings["Simulation"]["EstimationPrecision"]
        FoVlength           = settings["Simulation"]["FoVlength"] # px

        frames_per_second   = settings["Exp"]["fps"]
        microns_per_pixel   = settings["Exp"]["Microns_per_pixel"]
        temp_water          = settings["Exp"]["Temperature"]
        solvent             = settings["Exp"]["solvent"]
        
        FoVheight = settings["Fiber"]["TubeDiameter_nm"]*0.001 /microns_per_pixel # px

    else:
        nd.logger.info('No json file given, so use the parameters given to the function')




    if settings["Exp"]["Viscosity_auto"] == 1:
        visc_water = nd.Experiment.GetViscosity(temperature = temp_water, solvent = solvent)
        
    else:
        visc_water = settings["Exp"]["Viscosity"]

    # ========== up to here, parameters should be completely defined ============

    # ensure that diameter is a list
    if not(type(diameter)==list):
        diameter = [diameter]
    
    # ensure that num_particles is a list
    if not(type(num_particles)==list):
        num_particles = [num_particles] * len(diameter) # adjust list lengths
    else:
        if not(len(num_particles)==len(diameter)):
            nd.logging.warning('Given diameters and number of particles are not equal. Please adjust.')

           
    # ensure that mass is a list
    if not(type(mass)==list):
        mass = [mass] * len(diameter) # adjust list lengths
    else:
        if not(len(mass)==len(diameter)):
            nd.logging.warning('Given diameters and mass values are not equal. Please adjust.')


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


        #here comes the true random walk
        objall = GenerateRandomWalk(diameter[n_d], num_particles[n_d], frames, frames_per_second, ep = EstimationPrecision, mass = mass[n_d], microns_per_pixel = microns_per_pixel, temp_water = temp_water, visc_water = visc_water, start_pos = start_pos)


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



def RandomWalkCrossSection(settings = None, D = None, traj_length = None, dt = None, r_max = None, num_particles = 10, ShowHist = False, ShowTraj = False, ShowReflection = False):
    """
    THIS IS A RANDOM WALK IN THE CROSS-SECTION OF THE FIBER CHANNEL
    
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


