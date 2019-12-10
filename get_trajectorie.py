# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 12:23:39 2019

@author: Ronny Förster und Stefan Weidlich

"""


# In[0]:
# coding: utf-8
"""
Analyzis of Gold-Particle data for ARHCF paper

Created on 20181001 by Stefan Weidlich (SW)
Based on previous skript of Ronny Förster (RF) and SW

Target here: Implement only routines as they'll be used in paper

Modifications:
181020, SW: Cleaning up of code
Amongst others: Python v3.5 implementation deleted. Now working with 3.6 and above only
181025, SW: Adjustment of tracking-parameters and structuring of header
--> Realized that 32-bit parameters for tracking lead to unsufficient pixel-precision for 64 bit-version
181026, SW: Taking out of log-tracking. --> Not needed

******************************************************************************
Importing neccessary libraries
"""
#from __future__ import division, unicode_literals, print_function # For compatibility with Python 2 and 3
import numpy as np # Library for array-manipulation
import pandas as pd # Library for DataFrame Handling
import trackpy as tp # trackpy offers all tools needed for the analysis of diffusing particles
import warnings
import sys

import NanoObjectDetection as nd
import matplotlib.pyplot as plt # Libraries for plotting
#import matplotlib as mpl # I import this entirely here, as it's needed to change colormaps
from tqdm import tqdm# progress bar

from pdb import set_trace as bp #debugger

# In[]
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



def FindSpots(frames_np, ParameterJsonFile, UseLog = False, diameter = None, minmass=None, maxsize=None, separation=None,
                    max_iterations=10, SaveFig = False, gamma = 0.8):
    """
    Defines the parameters for the trackpy routine tp.batch, which spots particles, out of the json file
    
    important parameters:
    separation = settings["Find"]["Separation data"] ... minimum distance of spotes objects
    minmass    = settings["Find"]["Minimal bead brightness"]   ... minimum brightness of an object in order to be saved
    """
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    DoSimulation = settings["Simulation"]["SimulateData"]
    if DoSimulation == 1:
        print("No data. A simulation is done instead")        
        output = nd.Simulation.PrepareRandomWalk(ParameterJsonFile)       
        
    else:
#        UseLog = settings["Find"]["Analyze in log"]
#        
#        if UseLog == True:
#            frames_np = nd.handle_data.LogData(frames_np)
        
        
        separation = settings["Find"]["Separation data"]
        minmass    = settings["Find"]["Minimal bead brightness"]
        
        if UseLog == False:
            diameter = settings["Find"]["Estimated particle size"]
        else:
            diameter = settings["Find"]["Estimated particle size (log-scale)"]
    
    
        output_empty = True
        
        while output_empty == True:
            output = tp.batch(frames_np, diameter, minmass = minmass, separation = separation, max_iterations = max_iterations)
                   
            if output.empty:
                print("Image is empty - reduce Minimal bead brightness")
                minmass = minmass / 10
                settings["Find"]["Minimal bead brightness"] = minmass
            else:
                output_empty = False
                
        
        output['abstime'] = output['frame'] / settings["MSD"]["effective_fps"]
 
    
        nd.handle_data.WriteJson(ParameterJsonFile, settings) 
    
    
        if SaveFig == True:
            from NanoObjectDetection.PlotProperties import axis_font, title_font
            
#            params, title_font, axis_font = nd.visualize.GetPlotParameters(settings)
            fig = plt.figure()

            plt.imshow(nd.handle_data.DispWithGamma(frames_np[0,:,:] ,gamma = gamma))
            plt.scatter(output["x"],output["y"], s = 20, facecolors='none', edgecolors='r', linewidths=0.3)
 
        
            plt.title("Identified Particles in first frame", **title_font)
            plt.xlabel("long. Position [Px]")
            plt.ylabel("trans. Position [Px]", **axis_font)
            save_folder_name = settings["Plot"]["SaveFolder"]
            save_image_name = 'Optimize_First_Frame'
    
            settings = nd.visualize.export(save_folder_name, save_image_name, settings, use_dpi = 300)
        
    
    return output




def link_df(obj, ParameterJsonFile, SearchFixedParticles = False, max_displacement = None, dark_time = None):
    """
    Defines the parameters for the trackpy routine tp.link, which forms trajectories
    out of particle positions, out of the json file.
    
    important parameters:
    SearchFixedParticles = Defines whether fixed or moving particles are under current investigation
    dark_time            = settings["Link"]["Dark time"] ... maximum number of frames a particle can disappear
    max_displacement     = ["Link"]["Max displacement"]  ... maximum displacement between two frames

    """
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    dark_time = settings["Link"]["Dark time"]
    
    
    if SearchFixedParticles == False:
        max_displacement = settings["Link"]["Max displacement"]
    else:
        max_displacement = settings["Link"]["Max displacement fix"]
    
    t1_orig = tp.link_df(obj, max_displacement, memory=dark_time)
    
    nd.handle_data.WriteJson(ParameterJsonFile, settings) 
    
    
    return t1_orig



def filter_stubs(traj_all, ParameterJsonFile, FixedParticles = False, BeforeDriftCorrection = False, min_tracking_frames = None):
    """
    Defines the parameters for the trackpy routine tp.filter_stubs, which cuts to short trajectories,
    out of the json file.
    
    important parameters:
    FixedParticles        = Defines weather fixed or moving particles are under current investigation
    Fixed Particles must have long trajectories to ensure that they are really stationary
    
    BeforeDriftCorrection = Defines if the particles have been already drift corrected
    Before drift correction short trajectories are ok
    """

    settings = nd.handle_data.ReadJson(ParameterJsonFile)

    if (FixedParticles == True) and (BeforeDriftCorrection == True):
        # stationry particles
        min_tracking_frames = settings["Link"]["Dwell time stationary objects"]
        
    elif (FixedParticles == False) and (BeforeDriftCorrection == True):
        # moving particle before drift correction
        min_tracking_frames = settings["Link"]["Min tracking frames before drift"]
        
    else:
        # moving particle after drift correction
        min_tracking_frames = settings["Link"]["Min_tracking_frames"]
        

    traj_min_length = tp.filter_stubs(traj_all, min_tracking_frames)
    
    # RF 190408 remove Frames because the are doupled and panda does not like it
#    traj_min_length = traj_min_length.drop(columns="frame")
#    traj_min_length = traj_min_length.reset_index()

    particle_number = traj_all['particle'].unique(); #particlue numbers that fulfill all previous requirements
    amount_particles = len(particle_number); #total number of valid particles
    
    valid_particle_number = traj_min_length['particle'].unique(); #particlue numbers that fulfill all previous requirements
    amount_valid_particles = len(valid_particle_number); #total number of valid particles
    
    
    if (FixedParticles == True) and (BeforeDriftCorrection == True):
        print('Number of stationary objects (might detected multiple times after beeing dark):', amount_valid_particles)
    
    elif (FixedParticles == False) and (BeforeDriftCorrection == True):
        print("Too short trajectories removed! Before: %d, After = %d" %(amount_particles,amount_valid_particles))
            
    else:
        print("Too short trajectories removed! Before: %d, After = %d" %(amount_particles,amount_valid_particles))

    nd.handle_data.WriteJson(ParameterJsonFile, settings) 
    

    return traj_min_length



def RemoveSpotsInNoGoAreas(obj, t2_long_fix, ParameterJsonFile, min_distance = None):
    """
    In case of stationary/ fixed objects a moving particle should not come to close.
    This is because stationary objects might be very bright clusters which overshine the image of the dim
    moving particle. Thus a 'not-go-area' is defined
    """
    
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    if settings["StationaryObjects"]["Analyze fixed spots"] == 1:
        #required minimum distance in pixels between moving and stationary particles
        min_distance = settings["StationaryObjects"]["Min distance to stationary object"]

        stationary_particles = t2_long_fix.groupby("particle").mean()

        # loop through all stationary objects (contains position (x,y) and time of existent (frame))
        num_loop_elements = len(stationary_particles)
        for loop_t2_long_fix in range(0,num_loop_elements):
            nd.visualize.update_progress("Remove Spots In No Go Areas", (loop_t2_long_fix+1)/num_loop_elements)

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
        print("!!! PARTICLES CAN ENTER NO GO AREA WITHOUT GETTING CUT !!!")
        
    nd.handle_data.WriteJson(ParameterJsonFile, settings) 
    
        
    return obj



def close_gaps(t1):
    """
    # FILL GAPS IN THE TRAJECTORY BY NEAREST NEIGHBOUR
    # NECESSARY FOR FOLLOWING FILTERING WHERE AGAIN A NEAREST NEIGHBOUR IS APPLIED
    # OTHERWISE THE MEDIAN FILL WILL JUST IGNORE MISSING TIME POINTS
    """

    valid_particle_number = t1['particle'].unique(); #particlue numbers that fulfill all previous requirements
    amount_valid_particles = len(valid_particle_number);

    # use a copy of t1
    t1_before = t1.copy();
    
    # SORT BY PARTICLE AND THEN BY FRAME
    t1_search_gap = t1_before.sort_values(by = ['particle', 'frame'])
    t1_search_gap["RealData"] = True
    #t1_search_gap.head()
    
    #fill the gaps now - get the header
    t1_gapless = pd.DataFrame(columns = t1_search_gap.columns)           
     
    #for loop_particle in range(0,int(num_last_particles)):
    for i, loop_particle in enumerate(valid_particle_number):
        nd.visualize.update_progress("Close gaps in trajectories", (i+1) / amount_valid_particles)
    
        #grab values for specific particle
        t1_loop = t1_search_gap[t1_search_gap.particle == loop_particle]
        # read number of detected frames
        num_catched_frames = t1_loop.shape[0]
        # check if particle is not empty
        if num_catched_frames > 0:
            # calculate possible number of frames
            start_frame = t1_loop.frame.min()
            end_frame = t1_loop.frame.max()
    #        start_frame = t1_loop.frame_as_column.min()
    #        end_frame = t1_loop.frame_as_column.max()
            num_frames = end_frame - start_frame + 1
            # check if frames are missing
            if num_catched_frames < num_frames:
                # insert missing indicies by nearest neighbur
                index_frames = np.linspace(start_frame, end_frame, num_frames, dtype='int')
                t1_loop = t1_loop.reindex(index_frames)

                t1_loop.loc[t1_loop["RealData"] != True, "RealData"] = False
#                t1_loop["measured"] = True
#                t1_loop.loc[np.isnan(t1_loop["particle"]), "measured"] = False
                
                t1_loop = t1_loop.interpolate('nearest')
                
            # cat not data frame together
            t1_gapless = pd.concat([t1_gapless, t1_loop])
            
            
    return t1_gapless




def calc_intensity_fluctuations(t1_gapless, ParameterJsonFile, dark_time = None, PlotIntMedianFit = False, PlotIntFlucPerBead = False):
    """
    Calculates the intensity fluctuation of a particle along its trajectory
    """
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    dark_time = settings["Link"]["Dark time"]
    
    # MEDIAN FILTER OF MASS
    # CALCULATE RELATIVE STEP HEIGHTS OVER TIME
    
    # rolling median filter
    # NOT VERY ACCURATE BUT DOES IT FOR THE MOMENT.
    rolling_median_filter = t1_gapless.groupby('particle')['mass'].rolling(2*dark_time, center=True).median()
    
    # get it back to old format
    rolling_median_filter = rolling_median_filter.to_frame()
    rolling_median_filter = rolling_median_filter.reset_index(level='particle')
    
    # insert median filtered mass in original data frame
    t1_gapless['mass_smooth'] = rolling_median_filter['mass'].values
    
    # CALC DIFFERENCE
    my_diff = t1_gapless[['particle','mass_smooth']].diff()
    
       
    # remove gap if NaN
    my_diff.loc[pd.isna(my_diff['particle']),'mass_smooth']=0 # RF 180906
    
    # remove gap when new particle is occurs 
    my_diff.loc[my_diff['particle'] > 0 ,'mass_smooth']=0 # RF 180906    
    
    # remove NaN when median filter is to close on the edge defined by dark time in the median filter
    my_diff.loc[pd.isna(my_diff['mass_smooth']),'mass_smooth']=0 # RF 180906
    
    
    # relative step is median smoothed difference over its value
    #t1_search_gap_filled['rel_step'] = abs(my_diff['mass_smooth']) / t1_search_gap_filled['mass_smooth']
    # step height
    my_step_height = abs(my_diff['mass_smooth'])
    # average step offset (top + bottom )/2
    my_step_offest = t1_gapless.groupby('particle')['mass_smooth'].rolling(2).mean()
    my_step_offest = my_step_offest.to_frame().reset_index(level='particle')
    # relative step
    #t1_search_gap_filled['rel_step'] = my_step_height / my_step_offest.mass_smooth
    t1_gapless['rel_step'] = np.array(my_step_height) / np.array(my_step_offest.mass_smooth)

    if PlotIntMedianFit == True:
        nd.visualize.IntMedianFit(t1_gapless)
    
    if PlotIntFlucPerBead == True:
        nd.visualize.MaxIntFluctuationPerBead(t1_gapless)

    nd.handle_data.WriteJson(ParameterJsonFile, settings) 

    return t1_gapless




def split_traj(t2_long, t3_gapless, ParameterJsonFile):
    """
    Define settings for split trajectory at high intensity jumps
    """
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    t4_cutted, settings = split_traj_at_high_steps(t2_long, t3_gapless, settings)
    
    nd.handle_data.WriteJson(ParameterJsonFile, settings) 

    return t4_cutted



def split_traj_at_long_trajectorie(t4_cutted, settings, Min_traj_length = None, Max_traj_length = None):
    """
    Splits trajectory if they are too long
    This might be usefull, to have sufficently long trajectorys all at the same length.
    Otherwise they have very diffent conficence intervalls
    E.g: 2 particles: 1: 500 frames, 2: 2000 frames
    particle 2 is splitted into 4 500frames
    
    Splitting of a particle is fine, because a particle can go out of focus and return later
    and is assigned as new particle too.
    
    Important is too look at the temporal component, thus particle 2 never exists twice
    """

    keep_tail = settings["Split"]["Max_traj_length_keep_tail"]
    
    if Max_traj_length is None:
        Max_traj_length = int(settings["Split"]["Max_traj_length"])
     
    if Min_traj_length is None:
        Min_traj_length = int(settings["Link"]["Min_tracking_frames"])
        
    free_particle_id = np.max(t4_cutted["particle"]) + 1
    
#    Max_traj_length = 1000

    t4_cutted["true_particle"] = t4_cutted["particle"]
    
    traj_length = t4_cutted.groupby(["particle"]).frame.max() - t4_cutted.groupby(["particle"]).frame.min()
    
    # split when two times longer required
    split_particles = traj_length > Max_traj_length
    
    
    particle_list = split_particles.index[split_particles]
    
    particle_list = np.asarray(particle_list.values,dtype = 'int')
    
    num_particle_list = len(particle_list)

    for count, test_particle in enumerate(particle_list):
        nd.visualize.update_progress("Split too long trajectories", (count+1) / num_particle_list)


#        start_frame = t4_cutted[t4_cutted["particle"] == test_particle]["frame"].iloc[0]
#        end_frame   = t4_cutted[t4_cutted["particle"] == test_particle]["frame"].iloc[-1]
       
        start_frame = t4_cutted[t4_cutted["particle"] == test_particle]["frame"].iloc[0]
#        end_frame   = t4_cutted[t4_cutted["particle"] == test_particle]["frame"].iloc[-1]
        
        
        traj_length = len(t4_cutted[t4_cutted["particle"] == test_particle])
        
        
        print("traj_length", traj_length)
        while traj_length > Max_traj_length:
            if (traj_length > 2*Max_traj_length) or (keep_tail == 0):
                start_frame = t4_cutted[t4_cutted["particle"] == test_particle].iloc[Max_traj_length]["frame"]
                t4_cutted.loc[(t4_cutted["particle"] == test_particle) & (t4_cutted["frame"] >= start_frame), "particle"] = free_particle_id
    
                test_particle = free_particle_id
                free_particle_id = free_particle_id + 1
                
                traj_length = len(t4_cutted[t4_cutted["particle"] == test_particle])
            else:
                break
            
    if keep_tail == 0:
        t4_cutted = tp.filter_stubs(t4_cutted, Min_traj_length)

    return t4_cutted


def split_traj_at_high_steps(t2_long, t3_gapless, settings, max_rel_median_intensity_step = None,
                             min_tracking_frames_before_drift = None, PlotTrajWhichNeedACut = False, NumParticles2Plot = 3,
                             PlotAnimationFiltering = False, rawframes_ROI = -1):
    """
    Splits the trajectory at high intensity jumps.
    This is motivated by the idea, that an intensity jump is more likely to happen because of a wrong assignment in
    the trajectory building routine, than a real intensity jump due to radius change (I_scatterung ~ R^6) or heavy
    substructures/intensity holes in the laser moder
    """
    t4_cutted = t3_gapless.copy()
        
    max_rel_median_intensity_step = settings["Split"]["Max rel median intensity step"]
    
    min_tracking_frames_before_drift = settings["Link"]["Min tracking frames before drift"]
    
    
    # check if threshold is exceeded
    split_particle_at = t3_gapless[t3_gapless['rel_step'] > max_rel_median_intensity_step]
    
    # get frame and particle number where threshold is broken
    split_particle_at = split_particle_at[['frame','particle']]
    #split_particle_at = split_particle_at[['frame_as_column','particle']]
    
    # how many trajectories need splitting
    num_split_particles = split_particle_at.shape[0]  
    
    # currently last bead. so the number of the new bead is defined and not unique
    num_last_particle = np.max(t3_gapless['particle']) 
    #num_last_particle = np.max(t1_before['particle']) 
    
    # loop variable in case of plotting
    if PlotTrajWhichNeedACut == True:
        counter_display = 1
    
    # now split the beads at the gap into two beads

    for x in range(0,num_split_particles):
        nd.visualize.update_progress("Close gaps in trajectories", (x+1) / num_split_particles)
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
    
        t4_cutted.at[((t4_cutted.particle == particle_to_split) & (t4_cutted.frame < first_new_frame)),'particle'] = \
        num_last_particle + 1 

        # to remove the step which is now not here anymore
        t4_cutted.loc[(t4_cutted.particle == particle_to_split) & (t4_cutted.frame == first_new_frame),"rel_step"] = 0
        
        # num_last_particle ++
        num_last_particle = num_last_particle + 1;
        
        # just for visualization
        if PlotTrajWhichNeedACut == True:            
            if counter_display <= NumParticles2Plot:
                counter_display = counter_display + 1
                nd.visualize.CutTrajectorieAtStep(t3_gapless, particle_to_split, max_rel_median_intensity_step)

            

    # get rid of too short tracks - again because some have been splitted    
    #t1=t1.rename(columns={'frame_as_column':'frame'})
    t4_cutted = tp.filter_stubs(t4_cutted, min_tracking_frames_before_drift) # filtering out of artifacts that are seen for a short time only
    # the second argument is the maximum amount of frames that a particle is supposed not to be seen in order
    # not to be filtered out.
    #t1=t1.rename(columns={'frame':'frame_as_column'})
    print('Trajectories with risk of wrong assignments :',t3_gapless['particle'].nunique())
    print('Trajectories with reduced risk of wrong assignments :', t2_long['particle'].nunique())
    # Compare the number of particles in the unfiltered and filtered data.
    
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



