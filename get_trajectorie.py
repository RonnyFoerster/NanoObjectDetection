# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 12:23:39 2019

@author: Ronny Förster und Stefan Weidlich

"""

# coding: utf-8
#from __future__ import division, unicode_literals, print_function # for compatibility with Python 2 and 3
import numpy as np
import pandas as pd
import trackpy as tp
import warnings
import sys
import multiprocessing
from joblib import Parallel, delayed

import NanoObjectDetection as nd
import matplotlib.pyplot as plt # Libraries for plotting
from tqdm import tqdm# progress bar
# from pdb import set_trace as bp


def FindSpots(frames_np, ParameterJsonFile, UseLog = False, diameter = None,
              minmass=None, maxsize=None, separation=None, max_iterations = 10,
              SaveFig = False, gamma = 0.8, ExternalSlider = False, oldSim=False):
    """ wrapper for trackpy routine tp.batch, which spots particles

    important parameters:
    separation = settings["Find"]["Separation data"] ... minimum distance of spotes objects
    minmass    = settings["Find"]["Minimal bead brightness"]   ... minimum brightness of an object in order to be saved
    """

    settings = nd.handle_data.ReadJson(ParameterJsonFile)

    DoSimulation = settings["Simulation"]["SimulateData"]
    if DoSimulation == 1:
        nd.logger.info("A SIMULATION IS PERFORMED!")
        output = nd.Simulation.PrepareRandomWalk(ParameterJsonFile,oldSim=oldSim)

    else:

        ImgConvolvedWithPSF = settings["PreProcessing"]["EnhanceSNR"]
        DoPreProcessing = (ImgConvolvedWithPSF == False)

        separation = settings["Find"]["Separation data"]
        minmass    = settings["Find"]["Minimal bead brightness"]
        percentile = settings["Find"]["PercentileThreshold"]


        if diameter == None:
            diameter = settings["Find"]["Estimated particle size"]

        output_empty = True

        while output_empty == True:
            nd.logger.info("If you have problems with IOPUB data rate exceeded, take a look here: https://www.youtube.com/watch?v=B_YlLf6fa5A")

            nd.logger.info("Minmass = %s", minmass)
            nd.logger.info("Separation = %s", separation)
            nd.logger.info("Diameter = %s", diameter)
            nd.logger.info("Max iterations = %s", max_iterations)
            nd.logger.info("PreProcessing of Trackpy = %s", DoPreProcessing)


            # convert image to uint16 otherwise trackpy performs a min-max-stretch of the data in tp.preprocessing.convert_to_int - that is horrible.
            frames_np[frames_np < 0] = 0
            frames_np = np.uint16(frames_np)

            if ExternalSlider == False:
                # HERE HAPPENS THE LOCALIZATION OF THE PARTICLES
                num_frames = frames_np.shape[0]

                if num_frames < 100:
                    nd.logger.info("Find the particles - seriell")
                    output = tp.batch(frames_np, diameter, minmass = minmass, separation = separation, max_iterations = max_iterations, preprocess = DoPreProcessing, engine = 'auto', percentile = percentile)

                else:
                    num_cores = multiprocessing.cpu_count()
                    nd.logger.info("Find the particles - parallel. Number of cores: %s", num_cores)
                    inputs = range(num_frames)

                    num_verbose = nd.handle_data.GetNumberVerbose()
                    output_list = Parallel(n_jobs=num_cores, verbose=num_verbose)(delayed(tp.batch)(frames_np[loop_frame:loop_frame+1,:,:].copy(), diameter, minmass = minmass, separation = separation, max_iterations = max_iterations, preprocess = DoPreProcessing, engine = 'auto', percentile = percentile) for loop_frame in inputs)

                    empty_frame = []
                    #parallel looses frame number, so we add it again
                    for frame_id,_ in enumerate(output_list):
                        output_list[frame_id].frame = frame_id
                        if len(output_list[frame_id]) == 0:
                            empty_frame.append(frame_id)

                    # go through empty frames (start from the back deleting otherwise indexing fails)
                    for frame_id in (np.flip(empty_frame)):
                        del output_list[frame_id]

                    # make list of pandas to one big pandas
                    output = pd.concat(output_list)

                    nd.logger.info("Find the particles - finished")

            else:
#               print("WARNING UPDATE THIS!")
                output = tp.batch(frames_np, diameter, minmass = minmass, separation = (diameter, separation), max_iterations = max_iterations, preprocess = DoPreProcessing, percentile = percentile)


            if ExternalSlider == True:
                # leave without iteration, this is done outside by a slider
                output_empty = False

            else:
                # check if any particle is found. If not reduce minmass
                if output.empty:
                    nd.logger.warning("Image is empty - reduce Minimal bead brightness")
                    minmass = minmass / 10
                    settings["Find"]["Minimal bead brightness"] = minmass
                else:
                    output_empty = False
                    nd.logger.info("Set all NaN in estimation precision to 0")
                    output.loc[np.isnan(output.ep), "ep"] = 0
                    output['abstime'] = output['frame'] / settings["MSD"]["effective_fps"]


        nd.handle_data.WriteJson(ParameterJsonFile, settings)


        if SaveFig == True:
            from NanoObjectDetection.PlotProperties import axis_font, title_font

#            params, title_font, axis_font = nd.visualize.GetPlotParameters(settings)
            if ExternalSlider == False:
                nd.logger.warning("Warning, RF trys something here...")
                fig = plt.figure()
                if frames_np.ndim == 3:
                    plt.imshow(frames_np[0,:,:])
                else:
                    plt.imshow(frames_np[:,:])
            else:
                fig = plt.figure()
                plt.figure(figsize = [20,20])
                plt.imshow(nd.handle_data.DispWithGamma(frames_np[0,:,:] ,gamma = gamma), cmap = "gray")

            if ExternalSlider == False:
                plt.scatter(output["x"],output["y"], s = 20, facecolors='none', edgecolors='r', linewidths=0.3)
            else:
                plt.scatter(output["x"],output["y"], s = 500, facecolors='none', edgecolors='r', linewidths=2)

            plt.title("Identified Particles in first frame", **title_font)
            plt.xlabel("long. Position [Px]", **axis_font)
#            plt.ylabel("trans. Position [Px]", **axis_font)
            save_folder_name = settings["Plot"]["SaveFolder"]
            save_image_name = 'Optimize_First_Frame'


            if ExternalSlider == False:
                settings = nd.visualize.export(save_folder_name, save_image_name, settings, use_dpi = 300)

                nd.logger.warning("Ronny changed something here and did not had the time to check it")
                #plt.close(fig)

    return output # usually pd.DataFrame with feature position data



def QGridPandas(my_pandas):
    #https://medium.com/@tobiaskrabel/how-to-fix-qgrid-in-jupyter-lab-error-displaying-widget-model-not-found-55a948b183a1
    import qgrid
    import time

    # 3 seconds sleep of sometimes retarted JupyterLab that does not plot anything for 3 seconds
    time.sleep(3)

    qgrid_widget = qgrid.show_grid(my_pandas, show_toolbar=True)


    return qgrid_widget



def AnalyzeMovingSpots(frames_np, ParameterJsonFile):
    """ find moving spots by using a much larger diameter than for the slow moving (unsmeared) particles
    """
    settings = nd.handle_data.ReadJson(ParameterJsonFile)

    diameter = settings["Find"]["Estimated moving particle size"]
    if diameter == "auto":
        diameter_fixed = settings["Find"]["Estimated particle size"]

        # use the double size for the smeared particles
        diameter = (np.asarray(diameter_fixed)*2+1).tolist()

    output = nd.get_trajectorie.FindSpots(frames_np, ParameterJsonFile, diameter = diameter)

    return output



def link_df(obj, ParameterJsonFile, SearchFixedParticles = False, max_displacement = None, dark_time = None):
    """ wrapper for the trackpy routine tp.link, which forms trajectories
    out of particle positions, out of the json file

    important parameters:
    SearchFixedParticles = defines whether fixed or moving particles are under current investigation
    dark_time            = settings["Link"]["Dark time"] ... maximum number of frames a particle can disappear
    max_displacement     = ["Link"]["Max displacement"]   ...maximum displacement between two frames

    """
    settings = nd.handle_data.ReadJson(ParameterJsonFile)

    dark_time = settings["Link"]["Dark time"]

    if SearchFixedParticles == False:
        max_displacement = settings["Link"]["Max displacement"]
    else:
        max_displacement = settings["Link"]["Max displacement fix"]

    # here comes the linking
    nd.logger.info("Linking particles to trajectories: staring...")

    if nd.logger.getEffectiveLevel() >= 20:
        # Switch the logging of for the moment
        tp.quiet(suppress=True)
    
    t1_orig = tp.link_df(obj, max_displacement, memory=dark_time)

    nd.logger.info("Linking particles to trajectories: ...finished")
    if nd.logger.getEffectiveLevel() >= 20:
        # Switch the logging of for the moment
        tp.quiet(suppress=False)


    nd.handle_data.WriteJson(ParameterJsonFile, settings)

    return t1_orig



def filter_stubs(traj_all, ParameterJsonFile, FixedParticles = False,
                 BeforeDriftCorrection = False, min_tracking_frames = None,
                 ErrorCheck = True, PlotErrorCheck = True):
    """ wrapper for tp.filter_stubs, which filters out too short trajectories,
    including a check whether a trajectory is close enough to random Brownian motion

    important parameters:
    FixedParticles        = defines whether fixed or moving particles are under current investigation
    Fixed particles must have long trajectories to ensure that they are really stationary.

    BeforeDriftCorrection = defines if the particles have been already drift corrected
    Before drift correction short trajectories are okay.
    """
    settings = nd.handle_data.ReadJson(ParameterJsonFile)

    nd.logger.info("Remove to short trajectories.")

    if (FixedParticles == True) and (BeforeDriftCorrection == True):
        # STATIONARY particles must have a specfic minimum length - otherwise it is noise
        nd.logger.info("Apply to stationary particle")
        min_tracking_frames = settings["Link"]["Min tracking frames before drift"]

    elif (FixedParticles == False) and (BeforeDriftCorrection == True):
        # MOVING particles BEFORE DRIFT CORRECTION
        # moving particle must have a minimum length. However, for the drift correction, the too short trajectories are still full of information about flow and drift. Thus keep them for the moment but they wont make it to the final MSD evaluation
        nd.logger.info("Apply to diffusing particles - BEFORE drift correction")
        min_tracking_frames = settings["Link"]["Min tracking frames before drift"]

    else:
        # MOVING particles AFTER DRIFT CORRECTION
        nd.logger.info("Apply to diffusing particles - AFTER drift correction")
        min_tracking_frames = settings["Link"]["Min_tracking_frames"]


    nd.logger.info("Minimum trajectorie length: %s", min_tracking_frames)

    # only keep the trajectories with the given minimum length
    traj_min_length = tp.filter_stubs(traj_all, min_tracking_frames)

    # RF 190408 remove Frames because the are doupled and panda does not like it
    index_unequal_frame = traj_min_length[traj_min_length.index != traj_min_length.frame]
    if len(index_unequal_frame) > 0:
        nd.logger.error("TrackPy does something stupid. The index value (which should be the frame) is unequal to the frame")
        raise ValueError()
    else:
        traj_min_length = traj_min_length.drop(columns="frame")
        traj_min_length = traj_min_length.reset_index()


    # plot properties for the user
    DisplayNumDroppedTraj(traj_all, traj_min_length, FixedParticles, BeforeDriftCorrection)

        #ID of VALID particles that have sufficent trajectory length
    valid_particle_number = traj_min_length['particle'].unique();

    # check if the trajectories follow brownian motion after the drift correction.
    if (FixedParticles==False) and (BeforeDriftCorrection==False) and (ErrorCheck==True):
        #check if the histogram of each particle displacement is Gaussian shaped
        traj_min_length = CheckForPureBrownianMotion(valid_particle_number, traj_min_length, PlotErrorCheck)
    elif (FixedParticles==False) and (BeforeDriftCorrection==False) and (ErrorCheck==False):
        nd.logger.warning("Test for unbrownian motion is skipped.")


    return traj_min_length # trajectory DataFrame


def DisplayNumDroppedTraj(traj_all, traj_min_length, FixedParticles, BeforeDriftCorrection):
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
    if (FixedParticles == True) and (BeforeDriftCorrection == True):
        # STATIONARY particles
        nd.logger.info('Number of stationary objects (might be detected multiple times after being dark): %s', amount_valid_particles)

    else:
        # MOVING particles
        nd.logger.info("Too short trajectories removed!")
        nd.logger.info("Before: %s, After: %s, Removed: %s (%.2f%%) ", amount_particles, amount_valid_particles, amount_removed_traj, ratio_removed_traj)


        if amount_valid_particles == 0:
            nd.logger.error("All particles removed!")
            raise ValueError


def CheckForPureBrownianMotion(valid_particle_number, traj_min_length, PlotErrorCheck):
    """
    Check each Trajectory if its pure Brownian motion by a Kolmogorow-Smirnow test
    """
    
    nd.logger.info("Remove non-gaussian trajectories: starting...")
    
    # add a new column to the final df
    traj_min_length['stat_sign'] = 0    
    
    for i,particleid in enumerate(valid_particle_number):
        # print("particleid: ", particleid)
        # select traj to analyze in loop
        eval_tm = traj_min_length[traj_min_length.particle == particleid]

        # get the misplacement value of the first lagtime for the Kolmogorov test out of the MSD analysis (discard the rest of the outputs)
        nan_tm_sq, amount_frames_lagt1, enough_values, traj_length, nan_tm = \
        nd.CalcDiameter.CalcMSD(eval_tm, lagtimes_min = 1, lagtimes_max = 1)

        # put the misplacement vector into the Kolmogorow-Smirnow significance test
        traj_has_error, stat_sign, dx = \
        nd.CalcDiameter.CheckIfTrajectoryHasError(nan_tm, traj_length, MinSignificance = 0.01,  PlotErrorIfTestFails = PlotErrorCheck, ID=particleid)

        if traj_has_error == True:
            #remove if traj has error
            nd.logger.info("Drop particleID: %s (Significance = %s)", particleid, stat_sign)
            
            #drop particles with unbrownian trajectory
            # traj_min_length = traj_min_length[traj_min_length.particle!=particleid]
            
            # RF210127
            traj_min_length = traj_min_length.drop(traj_min_length[traj_min_length.particle == particleid].index)
        else:
            #insert statistical significance to trajectory as property
            traj_min_length.loc[traj_min_length.particle==particleid,'stat_sign'] = stat_sign

        

    # number of particle before and after particle test
    num_before = len(valid_particle_number)
    num_after = len(traj_min_length['particle'].unique());
    num_lost = num_before - num_after
    dropped_ratio = num_lost / num_before * 100

    nd.logger.info("Remove non-gaussian trajectories: ...finished")
    nd.logger.info("Before: %d, After: %d, Removed: %d (%d%%) ", (num_before, num_after , num_lost, dropped_ratio))

    return traj_min_length


def RemoveSpotsInNoGoAreas(obj, t2_long_fix, ParameterJsonFile, min_distance = None):
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
        stationary_particles = t2_long_fix.groupby("particle").mean() 

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
            mydiff = obj[['x','y']] - my_check[['x','y']]

            # get the norm
            # THIS ALSO ACCOUNT FOR THE TIME DIMENSION!
            # --> IF THE STATIONARY OBJECT VANISHED ITS "SHADOW" CAN STILL DELETE A MOVING PARTICLE
            mynorm = np.linalg.norm(mydiff.values,axis=1)

            # check for which particles the criteria of minimum distance is fulfilled
            valid_distance = mynorm > min_distance

            # keep only the good ones
            obj = obj[valid_distance]

    else:
        nd.logger.warning("Trajectories not removed if too close to stationary objects")

    nd.handle_data.WriteJson(ParameterJsonFile, settings)

    return obj



def RemoveSpotsInNoGoAreas_loop(stationary_particles, loop_t2_long_fix, obj, min_distance):
    """
    Central loop in RemoveSpotsInNoGoAreas
    make a separate function for it for better parallelisazion
    """

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
            
            

def RemoveOverexposedObjects(ParameterJsonFile, obj_moving, rawframes_rot):
    """ delete objects where the camera sensor was (over)saturated
    """
    settings = nd.handle_data.ReadJson(ParameterJsonFile)

    SaturatedPixelValue = settings["Find"]["SaturatedPixelValue"]

    # bring objects in order of ascending intensity values ("mass")
    sort_obj_moving = obj_moving.sort_values("raw_mass")

    counter = 0
    framecount = 0
    framelist = []

    saturated_psf = True
    while saturated_psf:
        # get pos and frame of spot with highest mass
        pos_x = np.int(sort_obj_moving.iloc[-1]["x"])
        pos_y = np.int(sort_obj_moving.iloc[-1]["y"])
        frame = np.int(sort_obj_moving.iloc[-1]["frame"])

        # get signal at maximum
        signal_at_max = rawframes_rot[frame,pos_y,pos_x]
        if signal_at_max >= SaturatedPixelValue:
            nd.logger.warning("Check if this is correct and insert more logging whats happening")
            sort_obj_moving = sort_obj_moving.iloc[:-1] # kick the overexposed object out
            counter += 1

            if not(frame in framelist):
                framecount += 1
                framelist.append(frame)
        else:
            saturated_psf = False

    
    

    obj_moving = sort_obj_moving

    return obj_moving



def close_gaps(t1):
    """ fill gaps in the trajectory by nearest neighbour

    necessary for following filtering where again a nearest neighbour is applied,
    otherwise the median fill will just ignore missing time points

    note: "RealData" column is introduced here (to distinguish between real and artificial data points)
    """
    valid_particle_number = t1['particle'].unique(); #particlue numbers that fulfill all previous requirements
    amount_valid_particles = len(valid_particle_number);

    # use a copy of t1
    t1_before = t1.copy();

    # SORT BY PARTICLE AND THEN BY FRAME
    t1_search_gap = t1_before.sort_values(by = ['particle', 'frame'])
    t1_search_gap["RealData"] = True
    #t1_search_gap.head()

    if amount_valid_particles < 100:
        nd.logger.info("Close the gaps in trajectory - sequential.")
        
        for i, loop_particle in enumerate(valid_particle_number):
            # select trajectory and close its gaps
            eval_traj = t1_search_gap[t1_search_gap.particle == loop_particle]
            t1_loop = clopse_gaps_loop(eval_traj )
        
            #depending if its the first result or not make a new dataframe or concat it
            if i == 0:
                t1_gapless = t1_loop
            else:
                t1_gapless = pd.concat([t1_gapless, t1_loop])
        
    else:
        num_cores = multiprocessing.cpu_count()
        nd.logger.info("Close the gaps in trajectory - parallel. Number of cores: %s", num_cores)
    
        num_verbose = nd.handle_data.GetNumberVerbose()
        output_list = Parallel(n_jobs=num_cores, verbose = num_verbose)(delayed(clopse_gaps_loop)(t1_search_gap[t1_search_gap.particle == loop_particle]) for loop_particle in valid_particle_number)

        #make list on panas to one big panda
        t1_gapless = pd.concat(output_list)


    # do some plotting for the user
    traj_total_data_points = len(t1_gapless[t1_gapless.RealData==True])
    traj_filled_data_points = len(t1_gapless[t1_gapless.RealData==False])
    
    percentage = traj_filled_data_points / traj_total_data_points * 100
    
    nd.logger.info("Total trajectory points: %d, Closed gaps: %d (%.2f%%)", traj_total_data_points, traj_filled_data_points, percentage)
    


    return t1_gapless


def clopse_gaps_loop(t1_loop):
    # main loop inside close gaps    

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


def calc_intensity_fluctuations(t1_gapless, ParameterJsonFile, dark_time = None, PlotIntMedianFit = False, PlotIntFlucPerBead = False):
    """ calculate the intensity fluctuation of a particle along its trajectory

    note: "mass_smooth" and "rel_step" columns are introduced here
    """

    settings = nd.handle_data.ReadJson(ParameterJsonFile)

    filter_time = 2*settings["Link"]["Dark time"] + 1

    # MEDIAN FILTER OF MASS
    # CALCULATE RELATIVE STEP HEIGHTS OVER TIME

    # apply rolling median filter on data sorted by particleID
    # NOT VERY ACCURATE BUT DOES IT FOR THE MOMENT.
    rolling_median_filter = t1_gapless.groupby('particle')['mass'].rolling(2*filter_time, center=True).median()

    # get it back to old format
    rolling_median_filter = rolling_median_filter.to_frame() # convert to DataFrame
    rolling_median_filter = rolling_median_filter.reset_index(level='particle')

    # insert median filtered mass in original data frame
    t1_gapless['mass_smooth'] = rolling_median_filter['mass'].values

    # CALC DIFFERENCES in mass and particleID
    my_diff = t1_gapless[['particle','mass_smooth']].diff()


    # remove gap if NaN
    my_diff.loc[pd.isna(my_diff['particle']),'mass_smooth'] = 0 # RF 180906

    # remove gap if new particle occurs
    my_diff.loc[my_diff['particle'] > 0 ,'mass_smooth'] = 0 # RF 180906

    # remove NaN if median filter is too close to the edge defined by dark time in the median filter
    my_diff.loc[pd.isna(my_diff['mass_smooth']),'mass_smooth'] = 0 # RF 180906


    # relative step is median smoothed difference over its value
    #t1_search_gap_filled['rel_step'] = abs(my_diff['mass_smooth']) / t1_search_gap_filled['mass_smooth']
    # step height
    my_step_height = abs(my_diff['mass_smooth'])
    # average step offset (top + bottom )/2
    my_step_offset = t1_gapless.groupby('particle')['mass_smooth'].rolling(2).mean()
    my_step_offset = my_step_offset.to_frame().reset_index(level='particle')
    # relative step
    #t1_search_gap_filled['rel_step'] = my_step_height / my_step_offest.mass_smooth
    t1_gapless['rel_step'] = np.array(my_step_height) / np.array(my_step_offset.mass_smooth)

    if PlotIntMedianFit == True:
        nd.visualize.IntMedianFit(t1_gapless)

    if PlotIntFlucPerBead == True:
        nd.visualize.MaxIntFluctuationPerBead(t1_gapless)

    nd.handle_data.WriteJson(ParameterJsonFile, settings)

    return t1_gapless



def split_traj(t2_long, t3_gapless, ParameterJsonFile):
    """ wrapper function for 'split_traj_at_high_steps' , returns both the original
    output and one without missing time points
    """

    settings = nd.handle_data.ReadJson(ParameterJsonFile)

    t4_cutted, settings = split_traj_at_high_steps(t2_long, t3_gapless, settings)
    nd.handle_data.WriteJson(ParameterJsonFile, settings)

    # close gaps to have a continous trajectory
    t4_cutted_no_gaps = nd.get_trajectorie.close_gaps(t4_cutted)


    return t4_cutted, t4_cutted_no_gaps



def split_traj_at_high_steps(t2_long, t3_gapless, settings, max_rel_median_intensity_step = None, min_tracking_frames_before_drift = None, PlotTrajWhichNeedACut = False, NumParticles2Plot = 3, PlotAnimationFiltering = False, rawframes_ROI = -1):
    """ split trajectories at high intensity jumps

    Assumption: An intensity jump is more likely to happen because of a wrong assignment in
    the trajectory building routine, than a real intensity jump due to radius change (I_scatterung ~ R^6)
    or heavy substructures/intensity holes in the laser mode.
    """
    t4_cutted = t3_gapless.copy()

    max_rel_median_intensity_step = settings["Split"]["Max rel median intensity step"]

    min_tracking_frames_before_drift = settings["Link"]["Min tracking frames before drift"]


    # check if threshold is exceeded
    split_particle_at = t3_gapless[t3_gapless['rel_step'] > max_rel_median_intensity_step]

    # get frame and particle number where threshold is broken
    split_particle_at = split_particle_at[['frame','particle']]
    #split_particle_at = split_particle_at[['frame_as_column','particle']]

    # how many splits of trajectories are needed
    num_split_particles = split_particle_at.shape[0]
    # how many trajectories are concerned
    num_split_traj = split_particle_at['particle'].nunique()
    ratio_split_traj = num_split_traj / t3_gapless['particle'].nunique() * 100

    # currently last bead (so the number of the new bead is defined and unique)
    num_last_particle = np.max(t3_gapless['particle'])

    # loop variable in case of plotting
    if PlotTrajWhichNeedACut == True:
        counter_display = 1

    # now split the beads at the gap into two beads

    for x in range(0,num_split_particles):
        nd.visualize.update_progress("Split trajectories at high intensity jumps", (x+1) / num_split_particles)
        #print('split trajectorie',x ,'out of ',num_split_particles)

        # NOW DO A PRECISE CHECK OF THE MASS

        # get index, number of particle which need splitting
        # and at which frame the split has to be done
        split_particle = split_particle_at.iloc[x]
        particle_to_split = split_particle['particle']
        first_new_frame = split_particle['frame']
    #    first_new_frame = split_particle['frame_as_column']


        # to see if there is a gap --> t1_search_gap[t1_search_gap['particle']==particle_to_split]['mass']
        # select right particle with following frames
        # the ".at" is necessary to change the value not on a copy but on t1 itself
        # https://stackoverflow.com/questions/13842088/set-value-for-particular-cell-in-pandas-dataframe-using-index

        # old RF removed it 210218, because index should be frame
        # t4_cutted.at[((t4_cutted.particle == particle_to_split) & (t4_cutted.index < first_new_frame)),'particle'] = num_last_particle + 1
        
        # t4_cutted.at[((t4_cutted.particle == particle_to_split) & (t4_cutted.frame < first_new_frame)),'particle'] = num_last_particle + 1

        # RF put that back to working
        t4_cutted.loc[((t4_cutted.particle == particle_to_split) & (t4_cutted.frame >= first_new_frame)),'particle'] = num_last_particle + 1

        # to remove the step which is now not here anymore
        t4_cutted.loc[(t4_cutted.particle == particle_to_split) & (t4_cutted.frame == first_new_frame),"rel_step"] = 0
        # old: when frame was still a column and not the index
#        t4_cutted.loc[(t4_cutted.particle == particle_to_split) & (t4_cutted.frame == first_new_frame),"rel_step"] = 0

        # num_last_particle ++
        num_last_particle = num_last_particle + 1;

        # just for visualization
        if PlotTrajWhichNeedACut == True:
            if counter_display <= NumParticles2Plot:
                counter_display = counter_display + 1
                nd.visualize.CutTrajectorieAtStep(t3_gapless, particle_to_split, max_rel_median_intensity_step)


    num_part_after_split = t4_cutted['particle'].nunique()

    # get rid of too short tracks - again because some have been splitted
    #t1=t1.rename(columns={'frame_as_column':'frame'})

    t4_cutted = tp.filter_stubs(t4_cutted, min_tracking_frames_before_drift) # filtering out of artifacts that are seen for a short time only
    t4_cutted = t4_cutted.drop(columns="frame")
    t4_cutted = t4_cutted.reset_index()
    # the second argument is the maximum amount of frames that a particle is supposed not to be seen in order
    # not to be filtered out.
    #t1=t1.rename(columns={'frame':'frame_as_column'})
    
    num_before = t3_gapless['particle'].nunique()
    num_after = t4_cutted['particle'].nunique()
    
    nd.logger.info('Trajectories with risk of wrong assignments (i.e. before splitting): %s', num_before)
    nd.logger.info('Trajectories with reduced risk of wrong assignments (i.e. after splitting): %s', num_after)
    # Compare the number of particles in the unfiltered and filtered data. (Mona: what does "filter" mean here??)

    nd.logger.info('Number of performed trajectory splits: %s', num_split_particles)
    nd.logger.info('Number of concerned trajectories: %d (%d%%)', num_split_traj, ratio_split_traj)
    nd.logger.info('Number of trajectories that became too short and were filtered out: %s', num_part_after_split-t4_cutted['particle'].nunique())

    #'''
    #wrong_particles=beads_property[beads_property['max_step'] > max_rel_median_intensity_step].index.get_value


    if PlotAnimationFiltering == True:
        if rawframes_ROI == -1:
            sys.exit("Insert the rawdata (rawframes_ROI)")

        else:
            #  Video of particles: Showing movement and effects of filtering
            #------------------------------------------------------------------------------
            #rawframes_before_filtering=rawframes
            nd.visualize.AnimateProcessedRawDate(rawframes_ROI, t2_long)


    t4_cutted = t4_cutted.loc[t4_cutted["RealData"] == True ]

    t4_cutted = t4_cutted.drop(columns="RealData")

    return t4_cutted, settings



def split_traj_at_long_trajectory(t4_cutted, settings, Min_traj_length = None, Max_traj_length = None):
    """ split trajectories if they are longer than a given value

    This might be usefull, to have sufficently long trajectories all at the same length.
    Otherwise they have very different confidence intervalls
    E.g: 2 particles: 1: 500 frames, 2: 2000 frames
    particle 2 is splitted into 4 500frames

    Splitting of a particle is fine, because a particle can go out of focus and return later
    and is assigned as new particle too.

    Important is to look at the temporal component, thus particle 2 never exists twice
    """
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


    free_particle_id = np.max(t4_cutted["particle"]) + 1

    t4_cutted["true_particle"] = t4_cutted["particle"]

    #traj length of each (true) particle
    traj_length = t4_cutted.groupby(["particle"]).frame.max() - t4_cutted.groupby(["particle"]).frame.min()

    # split if trajectory is longer than max value
    split_particles = traj_length > Max_traj_length # pd.Series of boolean

    particle_list = split_particles.index[split_particles] # index list of particleIDs

    particle_list = np.asarray(particle_list.values,dtype='int')

    num_particle_list = len(particle_list)

    # loop over all particleIDs
    for count, test_particle in enumerate(particle_list):
        nd.visualize.update_progress("Split too long trajectories", (count+1) / num_particle_list)

#        start_frame = t4_cutted[t4_cutted["particle"] == test_particle]["frame"].iloc[0]
#        end_frame   = t4_cutted[t4_cutted["particle"] == test_particle]["frame"].iloc[-1]

        # start_frame = t4_cutted[t4_cutted["particle"] == test_particle]["frame"].iloc[0]
#        end_frame   = t4_cutted[t4_cutted["particle"] == test_particle]["frame"].iloc[-1]

        traj_length = len(t4_cutted[t4_cutted["particle"] == test_particle])
        # print("traj_length", traj_length)

        # cut trajectory until it is not longer than Max_traj_length
        while traj_length > Max_traj_length:

            if (traj_length > 2*Max_traj_length) or (keep_tail == 0):
                # start_frame for new particle id
                start_frame = t4_cutted[t4_cutted["particle"] == test_particle].iloc[Max_traj_length]["frame"]

                # every trajectory point above the start frame gets the new id
                t4_cutted.loc[(t4_cutted["particle"] == test_particle) & (t4_cutted["frame"] >= start_frame), "particle"] = free_particle_id

                #next particle to test is the new generated one (the tail)
                test_particle = free_particle_id

                # new free particle id
                free_particle_id = free_particle_id + 1

                #traj length of particle under investigation
                traj_length = len(t4_cutted[t4_cutted["particle"] == test_particle])
            else:
                break

    if keep_tail == 0:
        t4_cutted = tp.filter_stubs(t4_cutted, Min_traj_length)

    return t4_cutted



#def OptimizeParamFindSpots(rawframes_ROI, ParameterJsonFile, SaveFig, gamma = 0.8, diameter=None , minmass=None, separation=None):
#
#    settings = nd.handle_data.ReadJson(ParameterJsonFile)
#
#    diameter = settings["Find"]["Estimated particle size"]
#    minmass = settings["Find"]["Minimal bead brightness"]
#    separation = settings["Find"]["Separation data"]
#
#    obj = nd.get_trajectorie.batch_np(rawframes_ROI[0:1,:,:], ParameterJsonFile, diameter = diameter,
#                                      minmass = minmass, separation = separation)
#
#    params, title_font, axis_font = nd.visualize.GetPlotParameters(settings)
#
##    mpl.rcParams.update(params)
#
#
#
#    fig = plt.figure()
#
#    plt.imshow(nd.handle_data.DispWithGamma(rawframes_ROI[0,:,:] ,gamma = gamma))
#
#
#    plt.scatter(obj["x"],obj["y"], s = 20, facecolors='none', edgecolors='r', linewidths=0.3)
#
##    my_s = rawframes_ROI.shape[2] / 10
##    my_linewidths = rawframes_ROI.shape[2] / 1000
##    plt.scatter(obj["x"],obj["y"], s = my_s, facecolors='none', edgecolors='r', linewidths = my_linewidths)
#
#
#    plt.title("Identified Particles in first frame", **title_font)
#    plt.xlabel("long. Position [Px]")
#    plt.ylabel("trans. Position [Px]", **axis_font)
#
#
#    if SaveFig == True:
#        save_folder_name = settings["Plot"]["SaveFolder"]
#        save_image_name = 'Optimize_First_Frame'
#
#        settings = nd.visualize.export(save_folder_name, save_image_name, settings, use_dpi = 300)
#
#        # close it because its reopened anyway
##        plt.close(fig)
#
#
#    return obj, settings


# # not really done
# def RemoveNoGoAreasAroundOverexposedAreas():
#     # check for saturated/overexposed areas on the chip
#     my_max = 65520

#     import numpy as np

#     sat_px = np.argwhere(rawframes_rot >= my_max)

#     import numpy as np
#     import pandas as pd
#     import time

#     num_sat_px = sat_px.shape[0]
#     # minimum allowed distance to saturated area
#     min_distance = settings["Find"]["Separation data"]

#     min_distance = 10

#     # first frame. variable is used to check if a new frames is reached
#     frame_sat_px_old = -1

#     #previous position
#     loop_sat_pos_old = np.zeros(2)
#     loop_sat_pos = np.zeros(2)

#     for loop_counter, loop_sat_px in enumerate(sat_px):
#     #    print(loop_counter)
#         t = time.time()
#     #    print("avoid saturaed px = ", loop_sat_px)

#         nd.visualize.update_progress("Remove Spots In Overexposed Areas", (loop_counter+1)/num_sat_px)

#         # position of the overexposition
#         loop_sat_pos[:] = [loop_sat_px[1], loop_sat_px[2]]


#         # check the difference to the precious saturated pixel

#         diff_px = np.linalg.norm(loop_sat_pos_old - loop_sat_pos)

#         # if they are neighbouring than ignore it to speed it up
#         if diff_px > 5:
#             # frame of the overexposition
#             frame_sat_px = loop_sat_px[0]

#     #        print("t1 = ", time.time() - t)

#             # only do that if a new frame starts
#     #        if frame_sat_px != frame_sat_px_old:
#     #            print("frame:", frame_sat_px)
#     ##            obj_moving_frame = obj_moving[obj_moving.frame == frame_sat_px]
#     #            frame_sat_px_old = frame_sat_px

#     #        print("t2 = ", time.time() - t)

#             # SEMI EXPENSIVE STEP: calculate the position and time mismatch between all objects
#             # and stationary object under investigation
#     #        mydiff = obj_moving_frame[['x','y']] - [pos_sat_px_x, pos_sat_px_y]
#             mydiff = obj_moving[obj_moving.frame == frame_sat_px][['y','x']] - loop_sat_pos
#     #        print("t3 = ", time.time() - t)

#             # get the norm
#     #        mynorm = np.linalg.norm(mydiff.values,axis=1)
#             mynorm = np.sqrt(mydiff["y"]**2 + mydiff["x"]**2)
#     #        print("t4 = ", time.time() - t)
#             # check for which particles the criteria of minimum distance is fulfilled

#             remove_close_object = mynorm < min_distance

#             remove_loc = remove_close_object.index[remove_close_object].tolist()

#             # keep only the good ones
#             obj_moving = obj_moving.drop(remove_loc)

#     #        obj_moving = pd.concat([obj_moving_frame[valid_distance], obj_moving[obj_moving.frame != frame_sat_px]])
#     #        print("t5 = ", time.time() - t)

#             frame_sat_px_old = frame_sat_px

#             loop_sat_pos_old = loop_sat_pos.copy()
