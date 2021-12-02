# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:37:39 2021

Store old function that are not used anymore, but maybe are of use later, for debugging, version rollback, etc

@author: foersterronny
"""

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


def HistDistanceBetweenParticles():
    num_part = 200
    
    pos = np.sort(np.random.random(num_part)*4000)
    
    diff = pos[1:]-pos[:-1]
    
    min_dist = np.min([diff[:-1], diff[1:]], axis = 0)
    
    plt.hist(min_dist, bins = int(num_part/5))




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
            nd.logger.warning("Something went wrong!")

        # average number of evaluated particles in a frame
        num_eval_particles_per_frame[min_traj_length_loop] = traj_length.loc[eval_particles_loc].sum() / frames


    return num_eval_particles, num_eval_particles_per_frame, volume_nl, t_unconf, t_conf, eval_t1, eval_t2




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



def E_Kin_Radiation_Force(F,m,t):
    E_kin = np.power(F*t,2)/(2*m)

    print(E_kin)

    return E_kin

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




def PhotonForceOverDiameterAndN():
    nd.logger.error("This function is switched off because the required MiePython module makes problems")
#     d = np.linspace(1,100,10) * 1E-9
#     use_material = ["gold", "polystyrene", "DNA"]
#     density_material = [19320, 1000, 1700]

#     my_lambda = 532
#     P = 10E-3
#     radius = 25E-6
    
#     I = nd.Theory.IntensityInFiber(P, radius, Mode = 'Peak')
    
#     scatt_crosssection_sqnm = np.zeros([d.shape[0], len(use_material)])
#     abs_crosssection_sqnm = np.zeros_like(scatt_crosssection_sqnm)
#     photon_pressure = np.zeros_like(scatt_crosssection_sqnm)
#     acceleration = np.zeros_like(scatt_crosssection_sqnm)
    
    
    
#     for ii, loop_d in enumerate(d):
#         for jj, loop_material in enumerate(use_material):
#             C_scat, C_abs = nd.Simulation.CalcCrossSection(loop_d, material = loop_material , at_lambda_nm = my_lambda, do_print = False)
#             scatt_crosssection_sqnm[ii,jj] = C_scat
#             abs_crosssection_sqnm[ii,jj]   = C_abs
            
#             photon_pressure[ii,jj] = nd.Theory.RadiationForce(I, C_scat*1E-18, C_abs*1E-18, n_media = 1.333)
           
#     #get acceleration
#     V = 4/3*np.pi * (d/2)**3
    
#     for ii, loop_density in enumerate(density_material):
#         m = V * loop_density
#         acceleration[:,ii] = photon_pressure[:, ii] / m
   
         
#     fig, (ax_c_scat, ax_c_abs, ax_force, ax_accel) = plt.subplots(4, sharex=True, gridspec_kw={'hspace': 0.1}, figsize=(9,12))
    
#     d_nm = d * 1E9
        
#     my_fontsize = 18
    
#     ax_c_scat.loglog(d_nm, scatt_crosssection_sqnm, '.-')
#     ax_c_scat.set_ylabel(r"$\sigma_{scat}\ [nm^2]$", fontsize = my_fontsize)
#     ax_c_scat.legend(use_material)
#     ax_c_scat.grid(which = 'both', axis = 'x')
#     ax_c_scat.grid(which = 'major', axis = 'y')

#     ax_c_abs.loglog(d_nm, abs_crosssection_sqnm, '.-')
#     ax_c_abs.set_ylabel(r"$\sigma_{abs}\ [nm^2]$", fontsize = my_fontsize)
#     ax_c_abs.grid(which = 'both', axis = 'x')
#     ax_c_abs.grid(which = 'major', axis = 'y')
    
#     ax_force.loglog(d_nm, photon_pressure, '.-')
#     ax_force.set_ylabel(r"$F_{photon}\ [N]$", fontsize = my_fontsize)
#     ax_force.grid(which = 'both', axis = 'x')
#     ax_force.grid(which = 'major', axis = 'y')

#     ax_accel.loglog(d_nm, acceleration, '.-')
#     ax_accel.set_xlabel("d [nm]", fontsize = my_fontsize)
#     ax_accel.set_ylabel(r"$a_{photon}\ [\frac{m}{s^2}]$", fontsize = my_fontsize)
#     ax_accel.set_xlim([np.min(d_nm),np.max(d_nm)])
#     ax_accel.grid(which = 'both', axis = 'x')
#     ax_accel.grid(which = 'major', axis = 'y')

#     fig.suptitle("P={:.3f}W, lambda={:.0f}nm, waiste={:.1f}um".format(P, my_lambda, radius*1E6), fontsize = my_fontsize)



def DepthOfField(NA, n, my_lambda):
    # https://doi.org/10.1111/j.1365-2818.1988.tb04563.x
    # Sheppard Depth of field in optical microscopy
    # Eq 8
    alpha = np.arcsin(NA/n)
    dz = 1.77 * my_lambda / (4 * (np.sin(alpha/2))**2 * (1-1/3*(np.tan(alpha/2))**4))

    return dz


def GetVarOfSettings(settings, key, entry):
    """ read the variable inside a dictonary
    
    settings: dict
    key: type of setting
    entry: variable name
    old function - should no be in use anymore
    """
    
    print("nd.handle_data.GetVarOfSettings is an old function, which should not be used anymore. Consider replacing it by settings[key][entry].")
    
    if entry in list(settings[key].keys()):
        # get value
        value = settings[key][entry]
        
    else:
        print("!!! Parameter settings['%s']['%s'] not found !!!" %(key, entry))
        # see if defaul file is available
        if "DefaultParameterJsonFile" in list(settings["Exp"].keys()):
            print('!!! Default File found !!!')
            path_default = settings["Exp"]["DefaultParameterJsonFile"]
            
            settings_default = ReadJson(path_default)
            default_value = settings_default[key][entry]
            
            ReadDefault = 'invalid'
            while (ReadDefault in ["y", "n"]) == False:
                # Do until input is useful
                ReadDefault = nd.handle_data.GetInput("Shall default value %s been used?" %default_value, ["y", "n"])
            
            if ReadDefault == "y":
                value = default_value

            else:
                SetYourself = 'invalid'
                while (SetYourself in ["y", "n"]) == False:
                    # Do until input is useful
                    ReadDefault = nd.handle_data.GetInput("Do you wanna set the value yourself?", ["y", "n"])

                 
                if ReadDefault == "y":
                    value = input("Ok. Set the value of settings['%s']['%s']: " %(key, entry))
                else:
                    sys.exit("Well... you had your chances!")
                                               
    return value


def RotImages(rawframes_np, ParameterJsonFile, Do_rotation = None, rot_angle = None):
    """ rotate the rawimage by rot_angle """
    import scipy # it's not necessary to import the full library here => to be shortened
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    Do_rotation = settings["PreProcessing"]["Do_or_apply_data_rotation"]
    
    if Do_rotation == True:
        nd.logger.info('Rotation of rawdata: start removing')
        rot_angle = settings["PreProcessing"]["rot_angle"]

    
        if rawframes_np.ndim == 2:
            im_out = scipy.ndimage.interpolation.rotate(rawframes_np, angle = rot_angle, axes=(1, 0), reshape=True, output=None, order=1, mode='constant', cval=0.0, prefilter=True)
        else:
            im_out = scipy.ndimage.interpolation.rotate(rawframes_np, angle = rot_angle, axes=(1, 2), reshape=True, output=None, order=1, mode='constant', cval=0.0, prefilter=True)

        nd.logger.info("Rotation of rawdata: Applied with an angle of %d" %rot_angle)
        
    else:
        im_out = rawframes_np
        nd.logger.info("Rotation of rawdata: Not Applied")

    nd.handle_data.WriteJson(ParameterJsonFile, settings)

    return im_out


def GetSettingsParameters(settings):
    """
    Get required parameters out of the settings

    Parameters
    ----------
    settings : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.
    Exception
        DESCRIPTION.

    Returns
    -------
    temp_water : TYPE
        DESCRIPTION.
    amount_lagtimes_auto : Boolean
        Defines if the number of lagtimes is set automatically.
    MSD_fit_Show : TYPE
        Show MSD curve.
    MSD_fit_Save : TYPE
        Save MSD curve.
    do_rolling : TYPE
        Rolling means time dependent evaluations.

    """


    temp_water = settings["Exp"]["Temperature"]
    amount_lagtimes_auto = settings["MSD"]["Amount lagtimes auto"]
    MSD_fit_Show = settings["Plot"]['MSD_fit_Show']
    MSD_fit_Save = settings["Plot"]['MSD_fit_Save']
    do_rolling = settings["Time"]["DoRolling"]
 
    # sanity checks    
    if MSD_fit_Save == True:
        MSD_fit_Show = True
        
    if amount_lagtimes_auto == 1:
        if settings["Exp"]["gain"] == "unknown":
            raise ValueError("Number of considered lagtimes can't be estimated "
                             "automatically, if gain is unknown. Measure gain, or change 'Amount lagtime auto' values to 0.")
    if temp_water < 250:
        raise Exception("Temperature is below 250K! Check if the temperature is inserted in K not in C!")
   
    return temp_water, amount_lagtimes_auto, MSD_fit_Show, MSD_fit_Save, do_rolling



def CheckIfTrajectoryHasError(nan_tm, traj_length, MinSignificance = 0.1, PlotErrorIfTestFails = False, PlotAlways = False, ID='unknown', processOutput = False):
    nd.logger.error("Function name is old. Use KolmogorowSmirnowTest instead")
    
    return KolmogorowSmirnowTest(nan_tm, traj_length, MinSignificance = 0.1, PlotErrorIfTestFails = False, PlotAlways = False, ID='unknown', processOutput = False)



def ConcludeResultsRolling(sizes_df_lin_rolling, diff_direct_lin_rolling, diff_std_rolling, diameter_rolling, 
                    particleid, traj_length, amount_frames_lagt1, start_frame,
                    mean_mass, mean_size, mean_ecc, mean_signal, mean_raw_mass, mean_ep,
                    red_x, max_step):
    
    # Storing results in df:
    new_panda = pd.DataFrame(data={'particle': particleid,
                                  'frame': np.arange(start_frame,start_frame + len(diff_direct_lin_rolling), dtype = int),
                                  'diffusion': diff_direct_lin_rolling,
                                  'diffusion std': diff_std_rolling,
                                  'diameter': diameter_rolling,
                                  'ep': mean_ep,
                                  'red_x' : red_x, 
                                  'signal': mean_signal,
                                  'mass': mean_mass,
                                  'rawmass': mean_raw_mass,
                                  'max step': max_step,
                                  'first frame': start_frame,
                                  'traj length': traj_length,
                                  'valid frames':amount_frames_lagt1,
                                  'size': mean_size,
                                  'ecc': mean_ecc})

    sizes_df_lin_rolling = sizes_df_lin_rolling.append(new_panda, sort=False)
    
    return sizes_df_lin_rolling



def CreateNewMSDPlot(any_successful_check, settings):
    """ for first time call, open a new plot window for the MSD ensemble fit """
    if any_successful_check == False:
        any_successful_check = True
        
        MSD_fit_Show = settings["Plot"]["MSD_fit_Show"]
        MSD_fit_Save = settings["Plot"]["MSD_fit_Save"]
        
        if (MSD_fit_Show == True) or (MSD_fit_Save == True):
            plt.figure("MSD-Plot", clear = True)

    return any_successful_check



def ContinousIndexingTrajectory(t):
    min_frame = t.index.values.min() # frame that particle's trajectory begins
    max_frame = t.index.values.max() # frame that particle's trajectory ends
    
    new_index = np.linspace(min_frame,max_frame, (max_frame - min_frame + 1))
    
    t = t.reindex(index=new_index)
    
    return t


def OptimizeTrajLenght(t6_final, ParameterJsonFile, obj_all, microns_per_pixel = None, frames_per_second = None, amount_summands = None, amount_lagtimes = None,  Histogramm_Show = True, EvalOnlyLongestTraj = 0):
    """ ... """
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    Max_traj_length = settings["Simulation"]["Max_traj_length"]
    NumberOfFrames  = settings["Simulation"]["NumberOfFrames"]
    
    max_ind_particles = int(NumberOfFrames / Max_traj_length)
    cur_ind_particles = 1
    
    sizes_df_lin = []
    any_successful_check = False
    
    while cur_ind_particles <= max_ind_particles:   
        print(cur_ind_particles)
        current_max_traj_length = int(np.floor(NumberOfFrames / cur_ind_particles))   
    
        # calculate the msd and process to diffusion and diameter
        sizes_df_lin_new, any_successful_check = nd.CalcDiameter.Main(t6_final, ParameterJsonFile, obj_all, Max_traj_length = current_max_traj_length)


        if cur_ind_particles == 1:
            sizes_df_lin = sizes_df_lin_new
        else:
            
            sizes_df_lin = sizes_df_lin.append(sizes_df_lin_new, sort=False)
            
        cur_ind_particles = cur_ind_particles * 2


    return sizes_df_lin, any_successful_check



def RollingPercentilFilter(rawframes_np, settings, PlotIt = True):
    """
    Old function that removes a percentile/median generates background image from the raw data.
    The background is calculated time-dependent.
    """
    nd.logger.warning('THIS IS AN OLD FUNCTION! SURE YOU WANNA USE IT?')

    nd.logger.info('Remove background by rolling percentile filter: starting...')
    
    rolling_length = ["PreProcessing"]["RollingPercentilFilter_rolling_length"]
    rolling_step = ["PreProcessing"]["RollingPercentilFilter_rolling_step"]
    percentile_filter = ["PreProcessing"]["RollingPercentilFilter_percentile_filter"]   
    
    for i in range(0,len(rawframes_np)-rolling_length,rolling_step):
        my_percentil_value = np.percentile(rawframes_np[i:i+rolling_length], percentile_filter, axis=0)
        rawframes_np[i:i+rolling_step] = rawframes_np[i:i+rolling_step] - my_percentil_value


    if PlotIt == True:
        nd.visualize.Plot2DImage(rawframes_np[0,:,0:500], title = "Raw Image (rolling percentilce subtracted) (x=[0:500])", xlabel = "x [Px]", ylabel = "y [Px]", ShowColorBar = False)

    nd.logger.info('Remove background by rolling percentile filter: ...finished')

    return rawframes_np



def UseSuperSampling(image_in, ParameterJsonFile, fac_xy = None, fac_frame = None):
    """ supersamples the data in x, y and frames by integer numbers
    
    e.g.: fac_frame = 5 means that only every fifth frame is kept
    """
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    DoSimulation = settings["Simulation"]["SimulateData"]
    
    if DoSimulation == 1:
        nd.logger.info("No data. Do a simulation later on")
        image_super = 0
                
    else:
        # supersampling  
        if settings["Subsampling"]["Apply"] == 0:
            nd.logger.info("Supersampling NOT applied")
            fac_xy = 1
            fac_frame = 1
            
        else:
            fac_xy = settings["Subsampling"]['fac_xy']
            fac_frame = settings["Subsampling"]['fac_frame']           
            nd.logger.info("Supersampling IS applied. With factors %s in xy and %s in frame ", fac_xy, fac_frame)
            
        image_super = image_in[::fac_frame, ::fac_xy, ::fac_xy]
            
            
        settings["MSD"]["effective_fps"] = round(settings["Exp"]["fps"] / fac_frame,2)
        settings["MSD"]["effective_Microns_per_pixel"] = round(settings["Exp"]["Microns_per_pixel"] / fac_xy,5)
        
        
        if (settings["Subsampling"]["Save"] == 1) and (settings["Subsampling"]["Apply"] == 1):
            data_folder_name = settings["File"]["data_folder_name"]
            SaveROIToTiffStack(image_super, data_folder_name)
        
        
        WriteJson(ParameterJsonFile, settings)
    
    return image_super



def Correlation(ParameterJsonFile, sizes_df_lin, show_plot = None, save_plot = None):
    """ produces grid plot for the investigation of particle data correlation
    
    for more information, see
    https://stackoverflow.com/questions/30942577/seaborn-correlation-coefficient-on-pairgrid
    """
    import NanoObjectDetection as nd
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    sns.set(style="white")
#    mygraph = sns.pairplot(sizes_df_lin[["diameter", "mass","signal"]])
    
    mycol = list(sizes_df_lin.columns)
    #remove diamter std because it is note done yet
#    mycol = [x for x in mycol if x!= "diameter std"]

    mygraph = sns.pairplot(sizes_df_lin[mycol])
    
    mygraph.map_lower(corrfunc)
    mygraph.map_upper(corrfunc)

    if save_plot == True:
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "Correlation",
                                       settings, data = sizes_df_lin)
        
    plt.show()



def Pearson(ParameterJsonFile, sizes_df_lin, show_plot = None, save_plot = None):
    import NanoObjectDetection as nd
    from NanoObjectDetection.PlotProperties import axis_font
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    sns.set_context("paper", rc={"font.size":int(axis_font["size"]),
                                 "axes.titlesize":8,
                                 "axes.labelsize":100})
                            
    mycorr = sizes_df_lin.corr()
    
    plt.figure()
    sns.set(font_scale=0.6)
#    mygraph = sns.heatmap(np.abs(mycorr), annot=True, vmin=0, vmax=1)
    mygraph = sns.heatmap(mycorr, annot=True, vmin=-1, vmax=1, cmap="bwr", cbar_kws={'label': 'Pearson Coefficent'})
    
#    sns.set(font_scale=1.2)
#    mygraph = sns.clustermap(np.abs(mycorr), annot=True)
    

    if save_plot == True:
        print("Saving the Pearson heat map take a while. No idea why =/")
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "PearsonCoefficent",
                                       settings, data = sizes_df_lin)

    plt.show()
    
    
    
def split_traj(t2_long, t3_gapless, ParameterJsonFile):
    """ wrapper function for 'split_traj_at_high_steps' , returns both the original
    output and one without missing time points
    """

    nd.logger.warning("split_traj is an old function which is not used anymore. Use split_traj_at_high_steps instead.")


    t4_cutted, t4_cutted_no_gaps = split_traj_at_high_steps(t2_long, t3_gapless, ParameterJsonFile)


    # close gaps to have a continous trajectory
    # t4_cutted_no_gaps = nd.get_trajectorie.close_gaps(t4_cutted)

    return t4_cutted, t4_cutted_no_gaps


def Plot2DPlot(x_np, y_np, title = None, xlabel = None, ylabel = None, myalpha = 1, mymarker = 'x', mylinestyle  = ':', x_lim = None, y_lim = None, y_ticks = None, semilogx = False, FillArea = False, Color = None):
    """ plot 2D-data in standardized format as line plot """
        
    plt.style.use(params)
    
    plt.figure()
    if semilogx == False:
        if Color == None:
            plt.plot(x_np,y_np, marker = mymarker, linestyle  = mylinestyle, alpha = myalpha)
        else:
            plt.plot(x_np,y_np, marker = mymarker, linestyle  = mylinestyle, alpha = myalpha, color = Color)
            
        if FillArea == True:
            plt.fill_between(x_np,y_np, y2 = 0, color = Color, alpha=0.5)
    else:
        plt.semilogx(x_np,y_np, marker = mymarker, linestyle  = mylinestyle, alpha = myalpha)
        import matplotlib.ticker

        ax = plt.gca()
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
        
    plt.title(title, **title_font)
    plt.xlabel(xlabel, **axis_font)
    plt.ylabel(ylabel, **axis_font)

    if x_lim != None:
        plt.xlim(x_lim)
        
    if y_lim != None:
        plt.ylim(y_lim)
                
    if y_ticks != None:
        frame1 = plt.gca() 
        frame1.axes.yaxis.set_ticks(y_ticks)


    nd.PlotProperties
    plt.style.use(params)
    
    ax = plt.gca()
    return ax

def corrfunc(x, y, **kws):
    """ auxiliary function for applying Pearson statistics on (diameter) data """
    
    pearson, _ = scipy.stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("p = {:.2f}".format(pearson),
                xy=(.1, .9), xycoords=ax.transAxes)
    
    
def NumberOfBinsAuto(mydata, average_height = 4):
    """
    Estiamtes how many bins are requried to get a good histogram
    """
    number_of_points = len(mydata)
    
    bins = int(np.ceil(number_of_points / average_height))
    
    return bins


# def PlotGaussNSizes(ax, diam_grid, max_y, sizes, num_dist_max=2, weighting=False, useAIC=False, showICplot=False):
#     """ plot Gaussian fits on top of a histogram/PDF
    
#     probably a bit redundant... but at least it's working reliably
#     """    
#     # use Gaussian mixture model (GMM) fitting to get best parameters
#     diam_means, diam_stds, weights = \
#         nd.statistics.StatisticDistribution(sizes, weighting=weighting,
#                                             num_dist_max=num_dist_max,
#                                             showICplot=showICplot, useAIC=useAIC)
    
#     # compute individual Gaussian functions from GMM fitted parameters 
#     dist = np.array([weights[n]*myGauss(diam_grid,diam_means[n],diam_stds[n]) 
#                      for n in range(weights.size)])
    
#     # calculate sum of all distributions
#     dsum = dist.sum(axis=0)
#     # normalize dsum to histogram/PDF max. value...
#     normFactor = max_y / dsum.max()
#     dsum = normFactor * dsum
#     dist = normFactor * dist # and the individual distributions accordingly
    
#     ax.plot(diam_grid,dist.transpose(),ls='--')
#     ax.plot(diam_grid,dist.sum(axis=0),color='k')
    
#     return diam_means, diam_stds, weights