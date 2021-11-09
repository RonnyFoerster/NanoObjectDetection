# -*- coding: utf-8 -*-
"""
Localize particles and forms trajectories.

Created on Tue Feb  5 12:23:39 2019

@author: Ronny Förster und Stefan Weidlich

"""

import numpy as np
import pandas as pd
import trackpy as tp
import sys
import multiprocessing
from joblib import Parallel, delayed

import NanoObjectDetection as nd
import matplotlib.pyplot as plt # Libraries for plotting



def FindSpots(rawframes_np, rawframes_pre, ParameterJsonFile, max_iterations = 10, SaveFig = False, gamma = 0.8, DoParallel = True):
    """
    wrapper for trackpy routine tp.batch, which spots particles

    Parameters
    ----------
    rawframes_np : numpy array
        unprocessed RAW image.
    rawframes_pre : numpy array
        preprocessed image (background, filter, etc.).
    ParameterJsonFile : TYPE
        DESCRIPTION.
    max_iterations : int, optional
        parameter for trackpy. The default is 10.
    SaveFig : Boolean, optional
        DESCRIPTION. The default is False.
    gamma : float, optional
        gamma for plotting. The default is 0.8.
    DoParallel : Boolean, optional
        Boolean to switch parallel programming on/off. The default is True.

    Returns
    -------
    obj_all : TYPE
        DESCRIPTION.
    """

    settings = nd.handle_data.ReadJson(ParameterJsonFile)    

    # get the parameters
    diameter = settings["Find"]["tp_diameter"]
    separation = settings["Find"]["tp_separation"]
    minmass    = settings["Find"]["tp_minmass"]
    percentile = settings["Find"]["tp_percentile"]    
    
    # convolve the raw image to enhance the SNR
    ImgConvolvedWithPSF = settings["PreProcessing"]["EnhanceSNR"]
    
    # trackpy needs to know the the images are already (lowpass) filted
    DoPreProcessing = (ImgConvolvedWithPSF == False)


    nd.logger.debug("If you have problems with IOPUB data rate exceeded, take a look here: https://www.youtube.com/watch?v=B_YlLf6fa5A")

    nd.logger.info("Minmass = %s", minmass)
    nd.logger.info("Separation = %s", separation)
    nd.logger.info("Diameter = %s", diameter)
    nd.logger.info("Max iterations = %s", max_iterations)
    nd.logger.info("PreProcessing of Trackpy = %s", DoPreProcessing)
    nd.logger.info("Percentile = %s", percentile)


    # convert image to int16 otherwise trackpy performs a min-max-stretch of the data in tp.preprocessing.convert_to_int - that is horrible.
    rawframes_pre = nd.PreProcessing.MakeInt16(rawframes_pre, AllowNonPosValues = False)
    
    if isinstance(rawframes_pre, np.float) == True:
        np.logger.warning("Given image is of datatype float. It is converted to int16. That is prone to errors for poor SNR; slow and memory waisting.")
        rawframes_pre = np.int16(rawframes_pre)


    # HERE HAPPENS THE LOCALIZATION OF THE PARTICLES
    obj_all = FindSpots_tp(rawframes_pre, diameter, minmass, separation, max_iterations, DoPreProcessing, percentile, DoParallel = DoParallel)

    # check if any particle is found. If not reduce minmass
    if obj_all.empty:
        nd.logger.error("Image is empty - reduce Minimal bead brightness")

    
    # set all nan in estimation precision to 0
    obj_all.loc[np.isnan(obj_all.ep), "ep"] = 0
    
    # insert absolute time as a columns to pandas
    obj_all['abstime'] = obj_all['frame'] / settings["Exp"]["fps"]


    nd.handle_data.WriteJson(ParameterJsonFile, settings)

    # insert addition columns defining where a found feature is overexposed or not. Overexposure features should no be used for MSD but are of value to track the particle
    obj_all = LabelOverexposedObjects(ParameterJsonFile, obj_all, rawframes_np)

    # plot it
    if SaveFig == True:
        FindSpots_plotting(rawframes_pre, obj_all, settings, gamma)

    # save output
    if settings["Plot"]["save_data2csv"] == 1:
        nd.handle_data.pandas2csv(obj_all, settings["Plot"]["SaveFolder"], "obj_all")

    objects_per_frame = nd.particleStats.ParticleCount(obj_all, rawframes_pre.shape[0])
    print(objects_per_frame.describe())

    return obj_all



def FindSpots_plotting(frames_np, output, settings, gamma):
    """
    Plots the first frame and the localized particles

    Parameters
    ----------
    frames_np : TYPE
        DESCRIPTION.
    output : TYPE
        DESCRIPTION.
    settings : TYPE
        DESCRIPTION.
    gamma : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    from NanoObjectDetection.PlotProperties import axis_font, title_font


    nd.logger.warning("Warning, RF trys something here...")
    fig = plt.figure()
    if frames_np.ndim == 3:
        plt.imshow(frames_np[0,:,:])
    else:
        plt.imshow(frames_np[:,:])


    plt.scatter(output["x"],output["y"], s = 500, facecolors='none', edgecolors='r', linewidths=2)

    plt.title("Identified Particles in first frame", **title_font)
    plt.xlabel("long. Position [Px]", **axis_font)
    plt.ylabel("trans. Position [Px]", **axis_font)
    save_folder_name = settings["Plot"]["SaveFolder"]
    save_image_name = 'Optimize_First_Frame'


    settings = nd.visualize.export(save_folder_name, save_image_name, settings, use_dpi = 300)

    nd.logger.warning("Ronny changed something here and did not had the time to check it")
        #plt.close(fig)



def FindSpots_tp(frames_np, diameter, minmass, separation, max_iterations, DoPreProcessing, percentile, DoParallel = True):
    """
    run trackpy with the given parameters seriell or parallel (faster after "long" loading time)

    Parameters
    ----------
    frames_np : numpy array
        image in.
    diameter : TYPE
        diameter of the particles to find.
    minmass : TYPE
        minimal brightness of the particles.
    separation : TYPE
        minimal distance between to particles (in pixel).
    max_iterations : TYPE
        DESCRIPTION.
    DoPreProcessing : Boolean
        says if image was already preprocessed (e.g. lowpass (SNR) filter).
    percentile : TYPE
        minimal percentile a particle brightness must be in order to be detected - shall avoid noizy salt and pepper artefacts.
    DoParallel : TYPE, optional
        defines if the parallel algorithm is switched on. The default is True.

    Returns
    -------
    output : pandas
        particle localization (x, y, frame, mass, etc.).

    """
    
    
    # get the number of frames
    num_frames = frames_np.shape[0]

    nd.logger.info("Find the particles - starting...")

    # for less than 100 frames parallel finding is not faster
    if num_frames < 100:
        DoParallel = False

    if DoParallel == False:
        nd.logger.debug("Particle finding is seriell")
        output = tp.batch(frames_np, diameter, minmass = minmass, separation = separation, max_iterations = max_iterations, preprocess = DoPreProcessing, engine = 'auto', percentile = percentile)

    else:
        # make it parallel accoring to the trackpy version
        tp_version = nd.Tools.GetTpVersion()
        if tp_version == 5:
            nd.logger.warning("Trackpy 5 has problems running in parallel on windows =/. Use tracpy 4 instead. (0.4.2 is good)")

            # output = tp.batch(frames_np, diameter, minmass = minmass, separation = separation, max_iterations = max_iterations, preprocess = DoPreProcessing, engine = 'auto', percentile = percentile)

        if tp_version == 4:
            nd.logger.debug("Find the particles - Trackpy 4")
            
            # prepare parallel processing
            num_cores = multiprocessing.cpu_count() # number of cores
            inputs = range(num_frames) #loop for the frames
            num_verbose = nd.handle_data.GetNumberVerbose() #defines how much is logged 
            
            nd.logger.info("Find the particles - parallel (Number of cores: %s): starting....", num_cores)
        
            # here comes the parallelization. the result is a list of pandas
            output_list = Parallel(n_jobs=num_cores, verbose=num_verbose)(delayed(tp.batch)(frames_np[loop_frame:loop_frame+1,:,:].copy(), diameter, minmass = minmass, separation = separation, max_iterations = max_iterations, preprocess = DoPreProcessing, engine = 'auto', percentile = percentile) for loop_frame in inputs)


            # parallel looses frame number, so we add it again
            # list of all the frames which are empty
            empty_frame = []
            for frame_id,_ in enumerate(output_list):
                # loop though the list of pandas and change the frame from 0 to the loop variable
                output_list[frame_id].frame = frame_id
                
                # if frame is empty, add it to the list
                if len(output_list[frame_id]) == 0:
                    empty_frame.append(frame_id)

            # remove all empty list elements, so that are PURE list of pandas remains            
            # go through empty frames (start from the back deleting otherwise indexing fails)
            for frame_id in (np.flip(empty_frame)):
                del output_list[frame_id]

            # make list of pandas to one big pandas
            output = pd.concat(output_list)

            # reset the index which starts at every frame again
            output = output.reset_index(drop=True)

    nd.logger.info("Find the particles - finished")

    return output



def Link(obj, ParameterJsonFile, SearchFixedParticles = False):
    """  
    wrapper for the trackpy routine tp.link, which forms trajectories out of particle positions, out of the json file
    

    Parameters
    ----------
    obj : pandas
        Localized Objects.
    ParameterJsonFile : TYPE
        DESCRIPTION.
    SearchFixedParticles : Boolean, optional
        Defines if it links fixed/stationary or moving particles. The default is False.


    Returns
    -------
    traj : pandas
        Particle Trajectories.
    traj_cutted : pandas
        Particle Trajectories cutted at high intensity steps.
        
    """
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)

    dark_time = settings["Link"]["Dark time"]

    # depending  if moving or fixed particles are searched, the maximum allowed movement is set
    if SearchFixedParticles == False:
        max_displacement = settings["Link"]["Max displacement"]
    else:
        max_displacement = settings["Link"]["Max displacement fix"]

    # here comes the linking
    nd.logger.info("Linking particles to trajectories: staring...")

    if nd.logger.getEffectiveLevel() >= 20:
        # Switch the logging of for the moment
        tp.quiet(suppress=True)

    # that is the linking itself
    traj_all = tp.link_df(obj, max_displacement, memory=dark_time)

    nd.logger.info("Linking particles to trajectories: ...finished")
    if nd.logger.getEffectiveLevel() >= 20:
        # Switch the logging of for the moment
        tp.quiet(suppress=False)

    
    # delete trajectories which are not long enough. Stationary objects have long trajcetories and survive the test. Avoid very short trajectories which probably not come from real particles, but from noise
    
    if SearchFixedParticles == False:
        Mode = "Moving Before Drift"
    else:
        Mode = "Fixed"
    
    traj = filter_stubs(traj_all, ParameterJsonFile, Mode = Mode)


    #cut intensity steps for moving particles if switched on
    if SearchFixedParticles == False:
        SplitIntensityJumps = settings["Split"]["IntensityJump"]
        if SplitIntensityJumps  == 1:
            traj, _ = CutTrajAtIntensityJump_Main(ParameterJsonFile, traj)
            
    # save the csv if required
    if settings["Plot"]["save_data2csv"] == 1:
        if SearchFixedParticles == False:
            nd.handle_data.pandas2csv(traj, settings["Plot"]["SaveFolder"], "traj_fixed")
        else:
            nd.handle_data.pandas2csv(traj, settings["Plot"]["SaveFolder"], "traj_moving")            
    
    nd.handle_data.WriteJson(ParameterJsonFile, settings)


    
    return traj



# Old Function Name
link_df = Link



def filter_stubs(traj_all, ParameterJsonFile, Mode = None, FixedParticles = False, BeforeDriftCorrection = False):
    """
    wrapper for tp.filter_stubs, which filters out too short trajectories, including a check whether a trajectory is close enough to random Brownian motion

    Parameters
    ----------
    traj_all : TYPE
        DESCRIPTION.
    ParameterJsonFile : TYPE
        DESCRIPTION.
    Mode : STRING, optional
        Describes the type of trajectory. Options: 'Fixed' (link stationary particles); 'Moving Before Drift' (Link Particles before the drift correction) and  'Moving After Drift' (Link Particles after the drift correction)
    FixedParticles : Boolean, optional
        OLD: describes if the particle are fixed or not. The default is False.
    BeforeDriftCorrection : TYPE, optional
        OLD: describes if the trajectories are drift corrected. The default is False.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    traj_min_length : pandas
        Trajectory who are longer than minimum number of required frames


    """
        
    settings = nd.handle_data.ReadJson(ParameterJsonFile)

    nd.logger.info("Remove to short trajectories.")

    if Mode == "Fixed":
        # STATIONARY particles must have a specfic minimum length - otherwise it is noise
        nd.logger.info("Apply to stationary particle")
        min_tracking_frames = settings["Link"]["Dwell time stationary objects"]

        # only keep the trajectories with the given minimum length
        traj_min_length = tp.filter_stubs(traj_all, min_tracking_frames)

    elif Mode == "Moving Before Drift":
        # MOVING particles BEFORE DRIFT CORRECTION
        # moving particle must have a minimum length. However, for the drift correction, the too short trajectories are still full of information about flow and drift. Thus keep them for the moment but they wont make it to the final MSD evaluation
        nd.logger.info("Apply to diffusing particles - BEFORE drift correction")
        min_tracking_frames = settings["Link"]["Min tracking frames before drift"]

        # only keep the trajectories with the given minimum length
        traj_min_length = tp.filter_stubs(traj_all, min_tracking_frames)

    elif Mode == "Moving After Drift":
        # MOVING particles AFTER DRIFT CORRECTION
        nd.logger.info("Apply to diffusing particles - AFTER drift correction")
        min_tracking_frames = settings["Link"]["Min_tracking_frames"]

        # only keep the trajectories with the given minimum length
        traj_min_length = tp.filter_stubs(traj_all[traj_all.saturated == False], min_tracking_frames)
        
    else:
        nd.logger.error("Mode parameter not given! Use: \
                        \n 'Fixed' (FixedParticles = True) \
                        \n 'Moving Before Drift' (FixedParticles = False and BeforeDriftCorrection = True) \
                        \n 'Moving After Drift' (FixedParticles = False and BeforeDriftCorrection = False) ")


    nd.logger.info("Minimum trajectorie length: %s", min_tracking_frames)



    # RF 190408 remove Frames because the are doupled and panda does not like it
    index_unequal_frame = traj_min_length[traj_min_length.index != traj_min_length.frame]
    if len(index_unequal_frame) > 0:
        nd.logger.error("TrackPy does something stupid. The index value (which should be the frame) is unequal to the frame")
        raise ValueError()
    else:
        traj_min_length = traj_min_length.drop(columns="frame")
        traj_min_length = traj_min_length.reset_index()


    # plot properties for the user
    DisplayNumDroppedTraj(traj_all, traj_min_length, Mode)

    #save the pandas
    if settings["Plot"]["save_data2csv"] == 1:
        if Mode == "Fixed":
            file_name = "traj_fixed"
        elif Mode == "Moving Before Drift":
            file_name = "traj_all"
        elif Mode == "Moving After Drift":
            file_name = "traj_final"
        
        nd.handle_data.pandas2csv(traj_min_length, settings["Plot"]["SaveFolder"], file_name)


    return traj_min_length # trajectory DataFrame



def DisplayNumDroppedTraj(traj_all, traj_min_length, Mode):
    """
    gives the user feed back how many trajectories are discared because they are to short
    """

     #ID of particles that are INSERTED
    particle_number = traj_all['particle'].unique();

    #total number of INSERTED particles
    amount_particles = len(particle_number);

    #ID of VALID particles that have sufficent trajectory length
    valid_particle_number = traj_min_length['particle'].unique();

    #total number of VALID particles
    amount_valid_particles = len(valid_particle_number);

    # calculate ratio of dropped trajectories
    amount_removed_traj = amount_particles - amount_valid_particles
    ratio_removed_traj = amount_removed_traj/amount_particles * 100

    # print some properties for the user
    if Mode == "Fixed":
        # STATIONARY particles
        nd.logger.info('Number of stationary objects (might be detected multiple times after being dark): %s', amount_valid_particles)

    elif (Mode == "Moving Before Drift") or (Mode == "Moving After Drift"):
        # MOVING particles
        nd.logger.info("Too short trajectories removed!")
        nd.logger.info("Before: %s, After: %s, Removed: %s (%.2f%%) ", amount_particles, amount_valid_particles, amount_removed_traj, ratio_removed_traj)


        if amount_valid_particles == 0:
            nd.logger.error("All particles removed!")
            raise ValueError


def CheckForPureBrownianMotion(traj, PlotErrorIfTestFails, yEval = False):
    """
    Check each Trajectory if its pure Brownian motion by a Kolmogorow-Smirnow test
    """

    if yEval == False:
        nd.logger.info("Remove non-gaussian trajectories - x: starting...")
    else:
        nd.logger.info("Remove non-gaussian trajectories - y: starting...")

    # get all particle IDs
    valid_particle_number = traj.particle.unique()

    # add a new column to the final df
    traj['stat_sign'] = 0

    # loop through all valid particle numbers
    for i,particleid in enumerate(valid_particle_number):
        nd.logger.debug("particleid: %.0f", particleid)

        # select traj to analyze in loop
        eval_tm = traj[traj.particle == particleid]

        #HANNA WAS HERE!
        if len(eval_tm) == 1:
            # huge error source
            traj_has_error = True
            nd.logger.warning("Particle is just 1 frame long - kick it out.")

        else:
            # get the misplacement value of the first lagtime for the Kolmogorov test out of the MSD analysis (discard the rest of the outputs)
            _, _, _, traj_length, nan_tm = \
            nd.CalcDiameter.CalcMSD(eval_tm, lagtimes_min = 1, lagtimes_max = 1, yEval = yEval)

            if type(nan_tm) == type(0):
                # make this more elegant later (maybe)
                traj_has_error = True
                nd.logger.warning("Trajectory has only gaps - kick it out.")

            # Kolmogorow-Smirnow significance test
            else:
                traj_has_error, stat_sign, dx = nd.CalcDiameter.KolmogorowSmirnowTest(nan_tm, traj_length, MinSignificance = 0.01,  PlotErrorIfTestFails = PlotErrorIfTestFails, ID=particleid)

        if traj_has_error == True:
            #remove if traj has error
            nd.logger.debug("Drop particleID: %s (Significance = %.6f)", particleid, stat_sign)

            traj = traj.drop(traj[traj.particle == particleid].index)
            
        else:
            #insert statistical significance to trajectory as property
            traj.loc[traj.particle==particleid,'stat_sign'] = stat_sign


    # number of particle before and after particle test
    num_before = len(valid_particle_number)
    num_after = len(traj['particle'].unique());
    num_lost = num_before - num_after
    dropped_ratio = np.ceil(num_lost / num_before * 100)

    nd.logger.info("Remove non-gaussian trajectories: ...finished")
    nd.logger.info("Before: %d, After: %d, Removed: %d (%d%%) ", num_before, num_after , num_lost, dropped_ratio)

    return traj


def RemoveSpotsInNoGoAreas(obj_all, traj_fixed, ParameterJsonFile, min_distance = None):
    """
    Delete objects that were found in the close neighborhood of fixed particles

    In case of stationary/fixed objects a moving particle should not come too close.
    This is because stationary objects might be very bright clusters which overshine
    the image of the dim moving particle. Thus a 'not-go-area' is defined.
    """
    settings = nd.handle_data.ReadJson(ParameterJsonFile)

    if settings["StationaryObjects"]["Analyze fixed spots"] == 1:
        nd.logger.info("Remove trajectorie close to stationary objects")

        # required minimum distance in pixels between moving and stationary particles
        min_distance = settings["StationaryObjects"]["Min distance to stationary object"]
        nd.logger.info("Minimal distance to stationary object: %s", min_distance)

        # average data of each particle
        stationary_particles = traj_fixed.groupby("particle").mean()

        # loop through all stationary objects (contains position (x,y) and time of existence (frame))
        num_loop_elements = len(stationary_particles)
        nd.logger.info("Number of stationary particles: %s", num_loop_elements)

        # num_cores = multiprocessing.cpu_count()
        # print("Remove spots in no go areas - parallel. Number of cores: ", num_cores)
        # inputs = range(num_frames)

        # output_list = Parallel(n_jobs = num_cores, verbose=5)(delayed(RemoveSpotsInNoGoAreas_loop)(frames_np[loop_frame:loop_frame+1,:,:].copy()) for loop_frame in inputs)



        for loop_t2_long_fix in range(0,num_loop_elements):
            nd.logger.warning("RF: THIS IS A GOOD MOMENT TO PROGRAMM THIS INTO PARALLEL!")

            nd.visualize.update_progress("Remove spots in no-go-areas", (loop_t2_long_fix+1)/num_loop_elements)

#            print(loop_t2_long_fix)
            # stationary object to check if it disturbs other particles
            my_check = stationary_particles.iloc[loop_t2_long_fix]

            # SEMI EXPENSIVE STEP: calculate the position and time mismatch between all objects
            # and stationary object under investigation
            mydiff = obj_all[['x','y']] - my_check[['x','y']]

            # get the norm
            # THIS ALSO ACCOUNT FOR THE TIME DIMENSION!
            # --> IF THE STATIONARY OBJECT VANISHED ITS "SHADOW" CAN STILL DELETE A MOVING PARTICLE
            # mynorm = np.linalg.norm(mydiff.values,axis=1)
            mydiff["r"] = np.sqrt(np.square(mydiff).sum(axis = 1))

            # check for which particles the criteria of minimum distance is fulfilled
            # valid_distance = mynorm > min_distance
            valid_distance = mydiff["r"] > min_distance

            # keep only the good ones
            obj_all = obj_all[valid_distance]
            
        obj_moving = obj_all.copy()

    else:
        nd.logger.warning("Trajectories not removed if too close to stationary objects")
        
        obj_moving = obj_all.copy()

    nd.handle_data.WriteJson(ParameterJsonFile, settings)

    #save the pandas
    if settings["Plot"]["save_data2csv"] == 1:
        nd.handle_data.pandas2csv(obj_moving, settings["Plot"]["SaveFolder"], "obj_moving_no_go_area")

    return obj_moving



def RemoveSpotsInNoGoAreas_loop(stationary_particles, loop_t2_long_fix, obj, min_distance):
    """
    Central loop in RemoveSpotsInNoGoAreas
    make a separate function for it for better parallelisazion
    """

    nd.logging.error("RF 211011: THIS IS NOT WORKING YET. COPMARE TO RemoveSpotsInNoGoAreas")
    
    # stationary object to check if it disturbs other particles
    my_check = stationary_particles.iloc[loop_t2_long_fix]

    # SEMI EXPENSIVE STEP: calculate the position and time mismatch between all objects
    # and stationary object under investigation
    mydiff = obj[['x','y']] - my_check[['x','y']]

    # get the norm
    # THIS ALSO ACCOUNT FOR THE TIME DIMENSION!
    # --> IF THE STATIONARY OBJECT VANISHED ITS "SHADOW" CAN STILL DELETE A MOVING PARTICLE
    mynorm = np.linalg.norm(mydiff.values,axis=1)

    # check for which particles the criteria of minimum distance is fulfilled
    valid_distance = mynorm > min_distance

    # keep only the good ones
    obj = obj[valid_distance]



def LabelOverexposedObjects(ParameterJsonFile, obj_moving, rawframes_np):
    """ label objects where the camera sensor was (over)saturated
    """
    settings = nd.handle_data.ReadJson(ParameterJsonFile)

    SaturatedPixelValue = settings["Find"]["SaturatedPixelValue"]

    nd.logger.debug("Saturated pixel value: %.0f", SaturatedPixelValue)

    obj_moving["saturated"] = False # create new column to mark the objects

    # get index of saturated column
    ix_saturated = obj_moving.columns.get_loc("saturated")

    if SaturatedPixelValue == 'No Saturation':
        nd.logger.info("No saturated pixel")

    else:
        nd.logger.info("Mark objects that have saturated pixels")

        # bring objects in order of ascending intensity values ("mass")
        sort_obj_moving = obj_moving.sort_values("raw_mass", ascending = False)

        total = len(obj_moving)
        counter = 0
        framecount = 0
        framelist = []

        saturated_psf = True
        while saturated_psf:
            # get pos and frame of spot with highest mass
            pos_x = np.int(sort_obj_moving.iloc[counter]["x"])
            pos_y = np.int(sort_obj_moving.iloc[counter]["y"])
            frame = np.int(sort_obj_moving.iloc[counter]["frame"])

            # get signal at maximum
            # search in surrounding pixels
            try:
                signal_at_max = np.max(rawframes_np[frame, pos_y-1:pos_y+2, pos_x-1:pos_x+2])
            except:
                #particle is close to an edge - just use center value
                signal_at_max = rawframes_np[frame,pos_y,pos_x]

            if signal_at_max >= SaturatedPixelValue:
                nd.logger.debug("Label overexposed particles at (frame,x,y) = (%i, %i, %i)", frame, pos_x, pos_y)

                sort_obj_moving.iloc[counter, ix_saturated] = True
                # former version: kick the overexposed object out
                # sort_obj_moving = sort_obj_moving.iloc[:-1]
                counter += 1

                if not(frame in framelist):
                    framecount += 1
                    framelist.append(frame)
            else:
                saturated_psf = False

        nd.logger.info("Detected and marked {} overexposed particles ({:.3f} %) in {} frames".format(counter, 100*counter/total, framecount))

        #undo the sorting
        obj_moving = sort_obj_moving.sort_values(["frame", "x"])

    return obj_moving



def close_gaps(traj):
    """ fill gaps in the trajectory by nearest neighbour

    necessary for following filtering where again a nearest neighbour is applied,
    otherwise the median fill will just ignore missing time points

    note: "RealData" column is introduced here (to distinguish between real and artificial data points)
    """
    valid_particle_number = traj['particle'].unique(); 
    
    #particlue numbers that fulfill all previous requirements
    amount_valid_particles = len(valid_particle_number);

    # use a copy of traj
    traj_before = traj.copy();

    # SORT BY PARTICLE AND THEN BY FRAME
    traj_search_gap = traj_before.sort_values(by = ['particle', 'frame'])
    traj_search_gap["RealData"] = True

    if amount_valid_particles < 100:
        nd.logger.info("Close the gaps in trajectory - sequential.")

        for i, loop_particle in enumerate(valid_particle_number):
            # select trajectory and close its gaps
            eval_traj = traj_search_gap[traj_search_gap.particle == loop_particle]
            traj_loop = close_gaps_loop(eval_traj)

            #depending if its the first result or not make a new dataframe or concat it
            if i == 0:
                traj_gapless = traj_loop
            else:
                traj_gapless = pd.concat([traj_gapless, traj_loop])

        # reset the indexing because this is lost in concat
        traj_gapless = traj_gapless.reset_index(drop = True)

    else:
        # prepare the multiprocessing
        num_cores = multiprocessing.cpu_count()
        nd.logger.info("Close the gaps in trajectory - parallel. Number of cores: %s", num_cores)

        num_verbose = nd.handle_data.GetNumberVerbose()
        output_list = Parallel(n_jobs=num_cores, verbose = num_verbose)(delayed(close_gaps_loop)(traj_search_gap[traj_search_gap.particle == loop_particle]) for loop_particle in valid_particle_number)

        #make list on panas to one big panda
        traj_gapless = pd.concat(output_list)

        #reset index
        traj_gapless = traj_gapless.sort_values(["particle", "frame"])
        traj_gapless = traj_gapless.reset_index(drop = True)


    # do some logging for the user
    total_point = len(traj_gapless)
    num_gaps = len(traj_gapless[traj_gapless.RealData==False])

    percentage = num_gaps / total_point * 100

    nd.logger.info("Total trajectory points: %d, Closed gaps: %d (%.2f%%)", total_point, num_gaps, percentage)


    return traj_gapless


def close_gaps_loop(t1_loop):
    """
    Function that closed gaps in a trajectory by interpolating missing time points

    Parameters
    ----------
    t1_loop : pandas
        traj of a singe particle.

    Returns
    -------
    t1_loop : pandas
        traj of a singe particle - with no holes.

    """
    

    # number of detected frames
    num_catched_frames = t1_loop.shape[0]

    # check if particle is not empty
    if num_catched_frames > 0:
        # calculate possible number of frames
        start_frame = t1_loop.frame.min()
        end_frame = t1_loop.frame.max()

        num_frames = end_frame - start_frame + 1

        # check if frames are missing
        if num_catched_frames < num_frames:
            # insert missing indicies by nearest neighbour
            index_frames = np.linspace(start_frame, end_frame, num_frames, dtype='int')

            t1_loop = t1_loop.set_index("frame")
            t1_loop = t1_loop.reindex(index_frames)

            # set RealData entry to False for all new inserted data points
            t1_loop.loc[t1_loop["RealData"] != True, "RealData"] = False

            t1_loop = t1_loop.interpolate('nearest')

            # make integeter particle id again (lost by filling with nan)
            t1_loop.particle = t1_loop.particle.astype("int")

            t1_loop = t1_loop.reset_index()

    return t1_loop



def CutTrajAtIntensityJump_Main(ParameterJsonFile, traj):
    """
    This function cuts the trajectory at to high intensity jumps
    """

    settings = nd.handle_data.ReadJson(ParameterJsonFile)

    if settings["Split"]["IntensityJump"] == 1:
        nd.logger.info("Cut trajectories at intensity jumps.")
        # close gaps to calculate intensity derivative
        traj_no_gaps = nd.get_trajectorie.close_gaps(traj)


        #calculate intensity fluctuations as a sign of wrong assignment
        traj_calc_fluc = nd.get_trajectorie.calc_intensity_fluctuations(traj_no_gaps, ParameterJsonFile)

        # split trajectories if necessary (e.g. too large intensity jumps)
        traj_cutted, _ = nd.get_trajectorie.split_traj_at_high_steps(traj, traj_calc_fluc, ParameterJsonFile)

    else:
        nd.logger.info("Dont cut trajectories at intensity jumps.")
        traj_cutted = traj.copy()

    return traj_cutted



def calc_intensity_fluctuations(traj, ParameterJsonFile, PlotIntMedianFit = False, PlotIntFlucPerBead = False):
    """ calculate the intensity fluctuation of a particle along its trajectory

    note: "mass_smooth" and "rel_step" columns are introduced here
    """

    #Calc the derivative. If that one is to large (intensity jump) that means that the particle linking did something stupid.

    nd.logger.info("Calculate the intensities derivative.")

    settings = nd.handle_data.ReadJson(ParameterJsonFile)

    # MEDIAN FILTER OF MASS
    # CALCULATE RELATIVE STEP HEIGHTS OVER TIME

    see_speckle = settings["Fiber"]["Mode"] == "Speckle"
    if see_speckle == 1:
        # FIBER WITH SPECKLES!
        # apply rolling median filter on data sorted by particleID
        # NOT VERY ACCURATE BUT DOES IT FOR THE MOMENT.

        nd.logger.info("Assume SPECKLES in fiber mode.")

        # average time of a fluctuation
        filter_time = 2*settings["Link"]["Dark time"] + 1

        # median mass filter
        rolling_median_filter = traj.groupby('particle')['mass'].rolling(2*filter_time, center=True).median()

        # get it back to old format
        rolling_median_filter = rolling_median_filter.to_frame() # convert to DataFrame
        rolling_median_filter = rolling_median_filter.reset_index(level='particle')

        # insert median filtered mass in original data frame
        traj['mass_smooth'] = rolling_median_filter['mass'].values

        # CALC DIFFERENCES in mass and particleID
        my_diff = traj[['particle','mass_smooth']].diff()

        # remove gap if NaN
        my_diff.loc[pd.isna(my_diff['particle']),'mass_smooth'] = 0 # RF 180906

        # remove gap if new particle occurs
        my_diff.loc[my_diff['particle'] > 0 ,'mass_smooth'] = 0 # RF 180906

        # remove NaN if median filter is too close to the edge defined by dark time in the median filter
        my_diff.loc[pd.isna(my_diff['mass_smooth']),'mass_smooth'] = 0 # RF 180906

        # relative step is median smoothed difference over its value
        # step height
        my_step_height = abs(my_diff['mass_smooth'])
        # average step offset (top + bottom )/2
        my_step_offset = traj.groupby('particle')['mass_smooth'].rolling(2).mean()
        my_step_offset = my_step_offset.to_frame().reset_index(level='particle')
        # relative step
        traj['rel_step'] = np.array(my_step_height) / np.array(my_step_offset.mass_smooth)


    elif see_speckle == 0:
        nd.logger.info("Assume NO SPECKLES in fiber mode.")
        # FIBER WITHOUT SPECKLES!

        # CALC DIFFERENCES in mass and particleID
        my_diff = traj[['particle','mass']].diff(1)

        # remove gap if NaN
        my_diff.loc[pd.isna(my_diff['particle']),'mass'] = 0

        # remove gap if new particle occurs
        my_diff.loc[my_diff['particle'] > 0 ,'mass'] = 0

        # remove NaN if median filter is too close to the edge defined by dark time in the median filter
        my_diff.loc[pd.isna(my_diff['mass']),'mass'] = 0 # RF 180906

        # step height
        my_step_height = abs(my_diff['mass'])
        
        # average step offset (top + bottom )/2
        my_step_offset = traj.groupby('particle')['mass'].rolling(2).mean()
        my_step_offset = my_step_offset.to_frame().reset_index(level='particle')

        # relative step
        traj['rel_step'] = np.array(my_step_height) / np.array(my_step_offset.mass_smooth)



    if PlotIntMedianFit == True:
        nd.visualize.IntMedianFit(traj)

    if PlotIntFlucPerBead == True:
        nd.visualize.MaxIntFluctuationPerBead(traj)

    nd.handle_data.WriteJson(ParameterJsonFile, settings)

    #save the pandas
    if settings["Plot"]["save_data2csv"] == 1:
        nd.handle_data.pandas2csv(traj, settings["Plot"]["SaveFolder"], "traj")

    return traj



def split_traj_at_high_steps(traj, traj_calc_fluc, ParameterJsonFile, PlotTrajWhichNeedACut = False, NumParticles2Plot = 3, PlotAnimationFiltering = False, rawframes_ROI = -1):
    """
    split trajectories at high intensity jumps.
    Assumption: An intensity jump is more likely to happen because of a wrong assignment in the trajectory building routine, than a real intensity jump due to radius change (I_scatterung ~ R^6) or heavy substructures/intensity holes in the laser mode.

    Parameters
    ----------
    traj : pandas
        trajectory.
    traj_calc_fluc : pandas
        trajectory, with filled gaps by interpolation.
    ParameterJsonFile : TYPE
        DESCRIPTION.
    PlotTrajWhichNeedACut : Boolean, optional
        Plots the mass of trajectories that need to be cutted. The default is False.
    NumParticles2Plot : Int, optional
        Maximum number of shown plots. The default is 3.
    PlotAnimationFiltering : Boolean, optional
        Do Animation (not sure if this is working). The default is False.
    rawframes_ROI : TYPE, optional
        Rawdate for animation. The default is -1.

    Returns
    -------
    None.

    """

    nd.logger.info("Split particles trajectory at too high intensity jumps.")

    # get parameters
    settings = nd.handle_data.ReadJson(ParameterJsonFile)

    max_rel_median_intensity_step = settings["Split"]["Max rel median intensity step"]
    min_tracking_frames_before_drift = settings["Link"]["Min tracking frames before drift"]
    
    # this will be the result later
    traj_cutted = traj_calc_fluc.copy()

    # check where intensity change is above threshold
    split_particle_at = traj_calc_fluc[traj_calc_fluc['rel_step'] > max_rel_median_intensity_step]

    # get frame and particle number where threshold is broken
    split_particle_at = split_particle_at[['frame','particle']]
    
    # how many splits of trajectories are needed
    num_splits = split_particle_at.shape[0]
    
    # how many trajectories are concerned
    num_split_traj = split_particle_at['particle'].nunique()
    
    # calculate the ratio of affected particles
    num_particles = traj_calc_fluc['particle'].nunique()
    ratio_split_traj = num_split_traj / num_particles * 100

    nd.logger.info('Number of trajectories with problems: %d out of %d (%d%%)', num_split_traj, num_particles, ratio_split_traj)

    # get highest particle ID.
    num_last_particle = np.max(traj_calc_fluc['particle'])

    # free particle ID
    free_particle_id = num_last_particle + 1

    # loop variable in case of plotting
    if PlotTrajWhichNeedACut == True:
        counter_display = 1

    
# now split the track at the jump into two with distinct IDs
    for x in range(0,num_splits):
        nd.visualize.update_progress("Split trajectories at high intensity jumps", (x+1) / num_splits)
        nd.logger.debug('split trajectorie %s out of %s', x, num_splits)

        # get index, particle id and frame the split has to be done
        split_particle = split_particle_at.iloc[x]
        particle_to_split = split_particle['particle']
        first_new_frame = split_particle['frame']

        # give any frame after the <first_new_frame> a new particle id
        traj_cutted.loc[((traj_calc_fluc.particle == particle_to_split) & (traj_cutted.frame >= first_new_frame)),'particle'] = free_particle_id

        # update free particle id
        free_particle_id = free_particle_id + 1;

        # set the calculated intensity step at the EDGE of the new trajectory to 0
        traj_cutted.loc[(traj_cutted.particle == free_particle_id) & (traj_cutted.frame == first_new_frame),"rel_step"] = 0


        # just for visualization
        if PlotTrajWhichNeedACut == True:
            if counter_display <= NumParticles2Plot:
                counter_display = counter_display + 1
                nd.visualize.CutTrajectorieAtStep(traj_calc_fluc, particle_to_split, max_rel_median_intensity_step)


    # remove to short trajectories
    traj_cutted = tp.filter_stubs(traj_cutted, min_tracking_frames_before_drift) 

    # number of trajectories / partilces after and before the splitting, and after removing to short trajectories
    num_after_split = traj_cutted['particle'].nunique()
    num_before = traj_calc_fluc['particle'].nunique()
    num_after_stub = traj_cutted['particle'].nunique()
    
    # number of to short trajectories
    num_stubbed = num_after_split - num_after_stub


    nd.logger.info('Number of trajectories: before: %s; after: %s', num_before, num_after_stub)

    nd.logger.debug('Number of trajectories (before filter stubs): before: %s; after: %s', num_before, num_after_split)
    nd.logger.debug('Number of performed splits: %s', num_splits)
    nd.logger.debug('Number of trajectories that became too short and were filtered out: %s', num_stubbed)



    if PlotAnimationFiltering == True:
        if rawframes_ROI == -1:
            sys.exit("Insert the rawdata (rawframes_ROI)")

        else:
            #  Video of particles: Showing movement and effects of filtering
            #------------------------------------------------------------------------------
            #rawframes_before_filtering=rawframes
            nd.visualize.AnimateProcessedRawDate(rawframes_ROI, traj)

    # prepare the return values. one with, one without the gaps
    traj_cutted_no_gaps = traj_cutted.copy()

    # In other words: remove the interpolated data points again, which have been introduced to make a proper intensity jump analyis. However, the interpolated x and y points would disturb the MSD analysis, because they are not measured
    traj_cutted = traj_cutted.loc[traj_cutted["RealData"] == True ]

    # remove the not RealData, because it is not needed anymore
    traj_cutted = traj_cutted.drop(columns="RealData")
    traj_cutted_no_gaps = traj_cutted_no_gaps.drop(columns="RealData")

    # resets the index after all this concating and deleting
    traj_cutted = traj_cutted.reset_index(drop = True)
    traj_cutted_no_gaps = traj_cutted_no_gaps.reset_index(drop = True)


    nd.handle_data.WriteJson(ParameterJsonFile, settings)

    #save the pandas
    if settings["Plot"]["save_data2csv"] == 1:
        nd.handle_data.pandas2csv(traj_cutted, settings["Plot"]["SaveFolder"], "traj_cutted")
        nd.handle_data.pandas2csv(traj_cutted_no_gaps, settings["Plot"]["SaveFolder"], "traj_cutted_no_gaps")

    return traj_cutted, traj_cutted_no_gaps



def split_traj_at_long_trajectory(traj, settings, Min_traj_length = None, Max_traj_length = None):
    """  
    split trajectories if they are longer than a given value

    This might be usefull, to have sufficently long trajectories all at the same length.
    Otherwise they have very different confidence intervalls
    E.g: 2 particles: 1: 500 frames, 2: 2000 frames
    particle 2 is splitted into 4 500frames

    Splitting of a particle is fine, because a particle can go out of focus and return later
    and is assigned as new particle too.

    Important is to look at the temporal component, thus particle 2 never exists twice
    """
    
    # kepp tail means that overhanging frames, which are not long enough for an independent trajectory are kept or discarded
    
    keep_tail = settings["Split"]["Max_traj_length_keep_tail"]

    if Max_traj_length is None:
        Max_traj_length = settings["Split"]["Max_traj_length"]

    #check if int
    if isinstance(Max_traj_length, int) == False:
        if (Max_traj_length - int(Max_traj_length)) == 0:
            Max_traj_length = int(Max_traj_length)
        else:
            sys.exit("Max_traj_length must be integer")

    if Min_traj_length is None:
        Min_traj_length = settings["Link"]["Min_tracking_frames"]

    #check if int
    if isinstance(Min_traj_length, int) == False:
        if (Min_traj_length - int(Min_traj_length)) == 0:
            Min_traj_length = int(Min_traj_length)
        else:
            sys.exit("Min_traj_length must be integer")

    # generate free particle ID
    free_particle_id = np.max(traj["particle"]) + 1

    # new panda header that saves the original particle ID, in case a trajectory is split
    # that can be used for time-resolved evaluations of the same particles or to meausure the std of the retrieved diameter in case the particle does not change
    traj["true_particle"] = traj["particle"]

    #traj length of each (true) particle
    traj_length = traj.groupby(["particle"]).frame.max() - traj.groupby(["particle"]).frame.min()

    # split if trajectory is longer than max value
    split_particles = traj_length > Max_traj_length

    # list of particleIDs that need splitting
    particle_list = split_particles.index[split_particles] 

    particle_list = np.asarray(particle_list.values,dtype='int')

    # number of trajectories that need a split
    num_particle_list = len(particle_list)

    # loop over all particleIDs
    for count, test_particle in enumerate(particle_list):
        nd.visualize.update_progress("Split too long trajectories", (count+1) / num_particle_list)

        traj_length = len(traj[traj["particle"] == test_particle])

        # cut trajectory until it is not longer than Max_traj_length
        while traj_length > Max_traj_length:

            # two independent trajectories, each beeing not shorter than Max_traj_length (traj_length > 2*Max_traj_length)
            # overhanging frames, which are not long enough for induvidual trajectory, get an independent trajectory for the momement and are filtered out later for beeing to short (keep tail == 0)
            if (traj_length > 2*Max_traj_length) or (keep_tail == 0):
                # start_frame for new particle id
                start_frame = traj[traj["particle"] == test_particle].iloc[Max_traj_length]["frame"]

                # every trajectory point above the start frame gets the new id
                traj.loc[(traj["particle"] == test_particle) & (traj["frame"] >= start_frame), "particle"] = free_particle_id

                #next particle to test is the new generated one (the tail)
                test_particle = free_particle_id

                # new free particle id
                free_particle_id = free_particle_id + 1

                #traj length of particle under investigation
                traj_length = len(traj[traj["particle"] == test_particle])
            else:
                #leave while loop
                break

    # remove the tails
    if keep_tail == 0:
        traj = tp.filter_stubs(traj, Min_traj_length)

    return traj
