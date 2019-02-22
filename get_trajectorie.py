# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 12:23:39 2019

@author: foersterronny
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

import nanoobject_detection as nd
import matplotlib.pyplot as plt # Libraries for plotting
import matplotlib as mpl # I import this entirely here, as it's needed to change colormaps
from tqdm import tqdm# progress bar

from pdb import set_trace as bp #debugger

# In[]
def OptimizeParamFindSpots(rawframes_ROI, settings, SaveFig, gamma = 0.8, diameter=None , minmass=None, separation=None):
    diameter, settings = nd.handle_data.SpecificValueOrSettings(diameter, settings, "Processing", "Estimated particle size [px]")
    minmass, settings =  nd.handle_data.SpecificValueOrSettings(minmass, settings, "Processing", "Minimal bead brightness")
    separation, settings = nd.handle_data.SpecificValueOrSettings(separation, settings, "Processing", "Separation data")
    
    obj, settings = nd.get_trajectorie.batch_np(rawframes_ROI[0:1,:,:], settings, diameter = diameter,
                                      minmass = minmass, separation = separation)

    params, title_font, axis_font = nd.visualize.GetPlotParameters(settings)
    
#    mpl.rcParams.update(params)
    


    fig = plt.figure()

    plt.imshow(nd.handle_data.DispWithGamma(rawframes_ROI[0,:,:] ,gamma = gamma))

    plt.scatter(obj["x"],obj["y"], s = 20, facecolors='none', edgecolors='r', linewidths=0.3)
    
    plt.title("Identified Particles in first frame", **title_font)
    plt.xlabel("long. Position [Px]")
    plt.ylabel("trans. Position [Px]", **axis_font)
    
       
    if SaveFig == True:
        save_folder_name = settings["Gui"]["SaveFolder"]
        save_image_name = 'Optimize_First_Frame'
    
        settings = nd.visualize.export(save_folder_name, save_image_name, settings)
        
        # close it because its reopened anyway
        plt.close(fig)
    

    return obj, settings



def batch_np(frames_np, settings, UseLog = False, diameter = None, minmass=None, maxsize=None, separation=None,
          noise_size=1, smoothing_size=None, threshold=None, invert=False,
          percentile=64, topn=None, preprocess=True, max_iterations=10,
          filter_before=None, filter_after=None, characterize=True,
          engine='auto', output=None, meta=None):
    """Locate Gaussian-like blobs of some approximate size in a set of images.

    Preprocess the image by performing a band pass and a threshold.
    Locate all peaks of brightness, characterize the neighborhoods of the peaks
    and take only those with given total brightness ("mass"). Finally,
    refine the positions of each peak.

    Parameters
    ----------
    frames_np : HERE COMES THE DIFFERENCE
            NP ARRAY OF IMAGESTACK
            THAT SHOULD BE MUCH FASTER THE PIMS
    diameter : odd integer or tuple of odd integers
        This may be a single number or a tuple giving the feature's
        extent in each dimension, useful when the dimensions do not have
        equal resolution (e.g. confocal microscopy). The tuple order is the
        same as the image shape, conventionally (z, y, x) or (y, x). The
        number(s) must be odd integers. When in doubt, round up.
    minmass : float
        The minimum integrated brightness.
        Default is 100 for integer images and 1 for float images, but a good
        value is often much higher. This is a crucial parameter for eliminating
        spurious features.
        .. warning:: The mass value was changed since v0.3.3
    maxsize : float
        maximum radius-of-gyration of brightness, default None
    separation : float or tuple
        Minimum separtion between features.
        Default is diameter + 1. May be a tuple, see diameter for details.
    noise_size : float or tuple
        Width of Gaussian blurring kernel, in pixels
        Default is 1. May be a tuple, see diameter for details.
    smoothing_size : float or tuple
        The size of the sides of the square kernel used in boxcar (rolling
        average) smoothing, in pixels
        Default is diameter. May be a tuple, making the kernel rectangular.
    threshold : float
        Clip bandpass result below this value.
        Default, None, defers to default settings of the bandpass function.
    invert : boolean
        Set to True if features are darker than background. False by default.
    percentile : float
        Features must have a peak brighter than pixels in this
        percentile. This helps eliminate spurious peaks.
    topn : integer
        Return only the N brightest features above minmass.
        If None (default), return all features above minmass.
    preprocess : boolean
        Set to False to turn off bandpass preprocessing.
    max_iterations : integer
        max number of loops to refine the center of mass, default 10
    filter_before : boolean
        filter_before is no longer supported as it does not improve performance.
    filter_after : boolean
        This parameter has been deprecated: use minmass and maxsize.
    characterize : boolean
        Compute "extras": eccentricity, signal, ep. True by default.
    engine : {'auto', 'python', 'numba'}
    output : {None, trackpy.PandasHDFStore, SomeCustomClass}
        If None, return all results as one big DataFrame. Otherwise, pass
        results from each frame, one at a time, to the put() method
        of whatever class is specified here.
    meta : filepath or file object, optional
        If specified, information relevant to reproducing this batch is saved
        as a YAML file, a plain-text machine- and human-readable format.
        By default, this is None, and no file is saved.

    Returns
    -------
    DataFrame([x, y, mass, size, ecc, signal])
        where mass means total integrated brightness of the blob,
        size means the radius of gyration of its Gaussian-like profile,
        and ecc is its eccentricity (0 is circular).

    See Also
    --------
    locate : performs location on a single image
    minmass_v03_change : to convert minmass from v0.2.4 to v0.3.0
    minmass_v04_change : to convert minmass from v0.3.3 to v0.4.0

    Notes
    -----
    This is an implementation of the Crocker-Grier centroid-finding algorithm.
    [1]_

    Locate works with a coordinate system that has its origin at the center of
    pixel (0, 0). In almost all cases this will be the topleft pixel: the
    y-axis is pointing downwards.

    References
    ----------
    .. [1] Crocker, J.C., Grier, D.G. http://dx.doi.org/10.1006/jcis.1996.0217

    """
    
    
    UseLog = settings["Processing"]["Analyze in log"]
    
    if UseLog == True:
        frames_np = nd.handle_data.LogData(frames_np)
    
    
    separation, settings = nd.handle_data.SpecificValueOrSettings(separation, settings, "Processing", "Separation data")
    minmass, settings =  nd.handle_data.SpecificValueOrSettings(minmass, settings, "Processing", "Minimal bead brightness")
         
    if UseLog == False:
        diameter, settings = nd.handle_data.SpecificValueOrSettings(diameter, settings, "Processing", "Estimated particle size [px]")
    else:
        diameter, settings = nd.handle_data.SpecificValueOrSettings(diameter, settings, "Processing", "Estimated particle size (log-scale) [px]")
    
#    settings = None, diameter = None, minmass=100, maxsize=None, separation=None,
#          noise_size=1, smoothing_size=None, threshold=None, invert=False,
#          percentile=64, topn=None, preprocess=True, max_iterations=10,
#          filter_before=None, filter_after=None, characterize=True,
#          engine='auto', output=None, meta=None

    all_features = []
    
    num_enumerate_elements = frames_np.shape[0]

    for i, image in enumerate(frames_np):
        nd.visualize.update_progress("Find Particles", (i+1)/num_enumerate_elements)
        features = tp.locate(image, diameter, minmass, maxsize, separation,
                          noise_size, smoothing_size, threshold, invert,
                          percentile, topn, preprocess, max_iterations,
                          filter_before, filter_after, characterize,
                          engine)
        if hasattr(image, 'frame_no') and image.frame_no is not None:
            frame_no = image.frame_no
            # If this works, locate created a 'frame' column.
        else:
            frame_no = i
            features['frame'] = i  # just counting iterations
            
       

        if len(features) == 0:
            continue

        if output is None:
            all_features.append(features)
        else:
            output.put(features)


    if output is None:
        if len(all_features) > 0:
            
            output = pd.concat(all_features).reset_index(drop=True)
        else:  # return empty DataFrame
            warnings.warn("No maxima found in any frame.")
            output = pd.DataFrame(columns=list(features.columns) + ['frame'])
    
#    tp.subpx_bias(output)
    
    output['abstime'] = output['frame'] / settings["Exp"]["fps"]
 

    return output, settings
 



# Test
#for i,j in enumerate(mylist):
#    print(i)
#    time.sleep(0.1)
#    update_progress("Some job", i/100.0)
#update_progress("Some job", 1)


def link_df(obj, settings, SearchFixedParticles = False, max_displacement = None, dark_time = None):
    
    dark_time, settings = nd.handle_data.SpecificValueOrSettings(dark_time, settings, "Processing", "Dark time [frames]") 
    
    
    if SearchFixedParticles == False:
        max_displacement, settings = nd.handle_data.SpecificValueOrSettings(max_displacement, settings, "Processing", "Max displacement [px]")
    else:
        max_displacement, settings = nd.handle_data.SpecificValueOrSettings(max_displacement, settings, "Processing", "Max displacement fix [px]")
    
    
    t1_orig = tp.link_df(obj, max_displacement, memory=dark_time)
    
    return t1_orig, settings



def filter_stubs(traj_all, settings, FixedParticles = False, BeforeDriftCorrection = False, min_tracking_frames = None):
    if (FixedParticles == True) and (BeforeDriftCorrection == True):
        # stationry particles
        min_tracking_frames, settings = nd.handle_data.SpecificValueOrSettings(min_tracking_frames, settings, "Processing", "Dwell time stationary objects") 
    elif (FixedParticles == False) and (BeforeDriftCorrection == True):
        # moving particle before drift correction
        min_tracking_frames, settings = nd.handle_data.SpecificValueOrSettings(min_tracking_frames, settings, "Processing", "Min tracking frames before drift")
    else:
        # moving particle after drift correction
        min_tracking_frames, settings = nd.handle_data.SpecificValueOrSettings(min_tracking_frames, settings, "Processing", "min_tracking_frames")

    traj_min_length = tp.filter_stubs(traj_all, min_tracking_frames)

    particle_number = traj_all['particle'].unique(); #particlue numbers that fulfill all previous requirements
    amount_particles = len(particle_number); #total number of valid particles
    
    valid_particle_number = traj_min_length['particle'].unique(); #particlue numbers that fulfill all previous requirements
    amount_valid_particles = len(valid_particle_number); #total number of valid particles
    
    
    if (FixedParticles == True) and (BeforeDriftCorrection == True):
        print('Number of stationary objects (might detected multiple times after beeing dark):', amount_valid_particles)
    elif (FixedParticles == False) and (BeforeDriftCorrection == True):
        print("To short particles removed! Before: %d, After = %d" %(amount_particles,amount_valid_particles))
        min_tracking_frames, settings = nd.handle_data.SpecificValueOrSettings(min_tracking_frames, settings, "Processing", "Min tracking frames before drift")
    else:
        min_tracking_frames, settings = nd.handle_data.SpecificValueOrSettings(min_tracking_frames, settings, "Processing", "min_tracking_frames")
        print("To short particles removed! Before: %d, After = %d" %(amount_particles,amount_valid_particles))

    return traj_min_length, settings



def RemoveSpotsInNoGoAreas(obj, t2_long_fix, settings, min_distance = None):
    
    if settings["Processing"]["Analyze fixed spots"] == 1:
        #required minimum distance in pixels between moving and stationary particles
        min_distance, settings = nd.handle_data.SpecificValueOrSettings(min_distance, settings, "Processing", "Min distance to stationary object [px]")
        
    
        # loop through all stationary objects (cotains position (x,y) and time of existent (frame))
        num_loop_elements = len(t2_long_fix)
        for loop_t1_fix in range(0,len(t2_long_fix)):
            nd.visualize.update_progress("Remove Spots In No Go Areas", (loop_t1_fix+1)/num_loop_elements)
            #print(loop_t1_fix)
            
            # stationary object to check if it disturbs other particles
            my_check = obj.iloc[obj]
        
            # SEMI EXPENSIVE STEP: calculate the position and time mismatch between all objects 
            # and stationary object under investigation    
            mydiff = obj[['x','y','frame']] - my_check[['x','y','frame']]
            
            # get the norm
            # THIS ALSO ACCOUNT FOR THE TIME DIMENSION!
            # --> IF THE STATIONARY OBJECT VANISHED ITS "SHADOW" CAN STILL DELETE A MOVING PARTICLE
            mynorm = np.linalg.norm(mydiff.values,axis=1)
            
            # check for which particles the criteria of minimum distance is fulfilled
            check_distance = mynorm > min_distance 
            
            # keep only the good ones
            obj = obj[check_distance]
            
    else:
        print("!!! PARTICLES CAN ENTER NO GO AREA WITHOUT GETTING CUT !!!")
        
    return obj, settings



def close_gaps(t1):
    # FILL GAPS IN THE TRAJECTORY BY NEAREST NEIGHBOUR
    # NECESSARY FOR FOLLOWING FILTERING WHERE AGAIN A NEAREST NEIGHBOUR IS APPLIED
    # OTHERWISE THE MEDIAN FILL WILL JUST IGNORE MISSING TIME POINTS

    valid_particle_number = t1['particle'].unique(); #particlue numbers that fulfill all previous requirements
    amount_valid_particles = len(valid_particle_number);

    # use a copy of t1
    t1_before = t1.copy();
    
    # SORT BY PARTICLE AND THEN BY FRAME
    t1_search_gap = t1_before.sort_values(by = ['particle', 'frame'])

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
#               
#                t1_loop["measured"] = True
#                t1_loop.loc[np.isnan(t1_loop["particle"]), "measured"] = False
                
                t1_loop = t1_loop.interpolate('nearest')
                
            # cat not data frame together
            t1_gapless = pd.concat([t1_gapless, t1_loop])
                           
    
    return t1_gapless



def calc_intensity_fluctuations(t1_gapless, settings, dark_time = None, PlotIntMedianFit = False, PlotIntFlucPerBead = False):
    dark_time, settings = nd.handle_data.SpecificValueOrSettings(dark_time, settings, "Processing", "Dark time [frames]") 
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


    return t1_gapless, settings




def split_traj_at_high_steps(t2_long, t3_gapless, settings, max_rel_median_intensity_step = None,
                             min_tracking_frames_before_drift = None, PlotTrajWhichNeedACut = False, NumParticles2Plot = 3,
                             PlotAnimationFiltering = False, rawframes_ROI = -1):
    
    max_rel_median_intensity_step, settings = nd.handle_data.SpecificValueOrSettings(max_rel_median_intensity_step, settings, 
                                                                           "Processing", "Max rel median intensity step") 
    
    min_tracking_frames_before_drift, settings = nd.handle_data.SpecificValueOrSettings(min_tracking_frames_before_drift, settings, 
                                                                           "Processing", "Min tracking frames before drift") 
    
    
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
    
       
        #t1.at[((t1_before.particle == particle_to_split) & (t1_before.frame < first_new_frame)),'particle'] = num_last_particle + 1        
        t2_long.at[((t2_long.particle == particle_to_split) & (t2_long.frame < first_new_frame)),'particle'] = num_last_particle + 1 
    #    t1.at[((t1.particle == particle_to_split) & (t1.frame_as_column < first_new_frame)),'particle'] = num_last_particle + 1 
         
        # num_last_particle ++
        num_last_particle = num_last_particle + 1;
        
        # just for visualization
        if PlotTrajWhichNeedACut == True:            
            if counter_display <= NumParticles2Plot:
                counter_display = counter_display + 1
                nd.visualize.CutTrajectorieAtStep(t3_gapless, particle_to_split, max_rel_median_intensity_step)

            
     
    # get rid of too short tracks - again because some have been splitted    
    #t1=t1.rename(columns={'frame_as_column':'frame'})
    t2_long = tp.filter_stubs(t2_long, min_tracking_frames_before_drift) # filtering out of artifacts that are seen for a short time only
    # the second argument is the maximum amount of frames that a particle is supposed not to be seen in order
    # not to be filtered out.
    #t1=t1.rename(columns={'frame':'frame_as_column'})
    print('Trajectories with risk of wrong assignments :',t3_gapless['particle'].nunique())
    print('Trajectories with reduced risk of wrong assignments ::', t2_long['particle'].nunique())
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

    t4_cutted = t2_long.copy()
    
    return t4_cutted, settings






def DriftCorrection(t_drift, settings, Do_transversal_drift_correction = None, drift_smoothing_frames = None, rolling_window_size = None,
                    min_particle_per_block = None, min_tracking_frames = None, PlotGlobalDrift = False, PlotDriftAvgSpeed = False, PlotDriftTimeDevelopment = False, 
                    PlotDriftFalseColorMapFlow = False, PlotDriftVectors = False, PlotDriftFalseColorMapSpeed = False,
                    PlotDriftCorrectedTraj = False):
    
    """
    ******************************************************************************
    Calculate and remove overall drift from trajectories
    Attention: Only makes sense if more than 1 particle is regarded
    Otherwise the average drift is exactly the particles movement and thus the trajectory vanishes!
    """

    
    Do_transversal_drift_correction, settings = nd.handle_data.SpecificValueOrSettings(Do_transversal_drift_correction, settings, 
                                                                           "Processing", "Do transversal drift correction")     
    
    drift_smoothing_frames, settings = nd.handle_data.SpecificValueOrSettings(drift_smoothing_frames, settings, 
                                                                           "Processing", "Drift smoothing frames") 
    
    rolling_window_size, settings = nd.handle_data.SpecificValueOrSettings(rolling_window_size, settings, 
                                                                           "Processing", "Drift rolling window size") 
    
    min_particle_per_block, settings = nd.handle_data.SpecificValueOrSettings(min_particle_per_block, settings, 
                                                                           "Processing", "Min particle per block") 
    
    min_tracking_frames, settings = nd.handle_data.SpecificValueOrSettings(min_tracking_frames, settings, 
                                                                           "Processing", "min_tracking_frames") 
    
       

    if Do_transversal_drift_correction == False:
        print('Mode: global drift correction')
        # That's not to be used if y-depending correction (next block) is performed!
        
        # Attention: Strictly this might be wrong:
        # Drift might be different along y-positions of channel.
        # It might be more appropriate to divide into subareas and correct for drift individually there
        # That's done if Do_transversal_drift_correction==1
        d = tp.compute_drift(t_drift, drift_smoothing_frames) # calculate the overall drift (e.g. drift of setup or flow of particles)
        t_no_drift = tp.subtract_drift(t_drift.copy(), d) # subtract overall drift from trajectories (creates new dataset)
        
        if PlotGlobalDrift == True:
            nd.visualize.PlotGlobalDrift() # plot the calculated drift
    
    
    else:
        print('Mode: transversal correction')
        # Y-Depending drift-correction
        # RF: Creation of y-sub-zones and calculation of drift
        # SW 180717: Subtraction of drift from trajectories
    
        
        # how many particles are needed to perform a drift correction
        #min_particle_per_block = 40
        
    #    # use blocks above and below for averaging (more particles make drift correction better)
    #    # e.g. 2 means y subarea itself and the two above AND below
    #    rolling_window_size = 5
    
        average_over_total_block = 2 * rolling_window_size + 1
          
        # sort y values to have in each sub area the same amount of particles
        all_y_sorted = t_drift.y.values.copy()
        all_y_sorted.sort()
        
        y_min = all_y_sorted[0];   # min y of tracked particle
        y_max = all_y_sorted[-1];   # max y of tracked particle
        num_data_points = len(all_y_sorted)
        
        # total number of captured frames
        num_frames = t_drift.index.max() - t_drift.index.min() + 1
        
        # this is difficult to explain
        # we have >num_data_points< data points and want to split them such, that each sub y has >min_particle_per_block< in it
        # in each frame >num_frames<
        # Because of the averaging with neighbouring areas the effective number is lifted >average_over_total_block<
        #start: distribute num_data_points over all number of frames and blocks
        number_blocks = int(num_data_points / min_particle_per_block / num_frames * average_over_total_block)
        
        
        #sub_y = np.linspace(y_min,y_max,number_blocks+1)
        sub_y = np.zeros(number_blocks+1)
        
        for x in range(0,number_blocks):
            use_index = int(num_data_points * (x / number_blocks))
            sub_y[x] = all_y_sorted[use_index]
        sub_y[-1] = y_max
        
        #average y-range for later
        y_range = (sub_y[:-1] + sub_y[1:]) / 2
        
        # delete variable to start again
        if 'calc_drift' in locals():
            del calc_drift 
            del total_drift
            total_drift = pd.DataFrame(columns = ['y','x','frame'])
            calc_drift_diff = pd.DataFrame()

        # Creating a copy of t1 which will contain a new column with each values ysub-position
        t_drift_ysub = t_drift.copy()
        t_drift_ysub['ysub']=np.nan # Defining values to nan
            
        for x in range(0,number_blocks):
            print(x)
               
            # calc which subareas of y are in the rolling window.
            sub_y_min = x - rolling_window_size
            if sub_y_min  < 0:
                sub_y_min = 0
            
            sub_y_max = x + 1 + rolling_window_size
            if sub_y_max  > number_blocks:
                sub_y_max = number_blocks;
                
            # select which particles are in the current subarea
            use_part = (t_drift['y'] >= sub_y[sub_y_min]) & (t_drift['y'] < sub_y[sub_y_max])
            use_part_subtract = (t_drift['y'] >= sub_y[x]) & (t_drift['y'] < sub_y[x+1]) 
            
            # get their indices
            use_part_index = np.where(np.array(use_part)==True)   # find index of elements true
            
            # WHAT IS THIS VARIABLE ACTUALLY GOOD FOR?
            use_part_subtract_index = np.where(np.array(use_part_subtract)==True)
            
            # Writing x as ysub into copy of t1. That's needed to treat data differently depending on y-sub-position
            # Python 3.5 t1_ysub['ysub'].iloc[use_part_subtract_index]=x # I believe that's not an elegant way of doing it. Maybe find a better approach       
            t_drift_ysub.loc[use_part,'ysub'] = x # RF 180906
            
    
            
            # how many particles are in each frames
            use_part_index = list(use_part_index)[0]
            use_part_subtract_index = list(use_part_subtract_index)[0]
            num_particles_block = len(use_part_index)
                
            # check if drift_smoothing_frames is not longer than the video is long
            num_frames = settings["ROI"]["frame_max"] - settings["ROI"]["frame_min"] + 1
            if num_frames < drift_smoothing_frames:
                sys.exit("Number of frames is smaller than drift_smoothing_frames")

#            raise NameError('HiThere')
            # make the drift correction with the subframe
            calc_drift_y = tp.compute_drift(t_drift.iloc[use_part_index], drift_smoothing_frames) # calculate the drift of this y block
            calc_drift_y_diff=calc_drift_y.diff(periods=1).fillna(value=0)
            calc_drift_y_diff['ysub']=x
                
            calc_drift_y['frame'] = calc_drift_y.index.values
            calc_drift_y['y_range'] = y_range[x]
        
            # calculate entire drift with starting and end position
            start_pos = calc_drift_y.set_index('y_range')[['y', 'x', 'frame']].iloc[[0],:]
            end_pos = calc_drift_y.set_index('y_range')[['y', 'x', 'frame']].iloc[[-1],:]
            my_total = end_pos - start_pos
            my_total ['num_particles'] = num_particles_block
            
            if x == 0:
                calc_drift = calc_drift_y
                total_drift = my_total
                calc_drift_diff = calc_drift_y_diff # that's going to be the look-up for the drift, depending on y-sub
                
            else:
                calc_drift = pd.concat([calc_drift, calc_drift_y])  #prepare additional index
                total_drift = pd.concat([total_drift, my_total])
                calc_drift_diff = pd.concat([calc_drift_diff,calc_drift_y_diff])
        
        # to distinguish that we're not looking at positions but the difference of positions
        calc_drift_diff=calc_drift_diff.rename(columns={'x':'x_diff1', 'y':'y_diff1'}) 
        
        # Adding frame as a column
        calc_drift_diff['frame']=calc_drift_diff.index 
        
        # Indexing by y-sub-area and frame
        calc_drift_diff=calc_drift_diff.set_index(['ysub','frame']) 
        
        # Adding frame as a calumn to particle data
        t_drift_ysub['frame']=t_drift_ysub.index 
        
        # Indexing particle-data analogously to drift-lookup
        t_drift_ysub=t_drift_ysub.set_index(['ysub','frame']) 
        
        # Adding drift-lookup into particle data, using frame and ysub
        t_drift_ysub_diff=pd.merge(t_drift_ysub,calc_drift_diff, left_index=True, right_index=True, how='inner') 
        
        # Releasing frame from index -> allows to sort easier by frame
        t_drift_ysub_diff=t_drift_ysub_diff.reset_index('frame') 
        
        cumsums_x=t_drift_ysub_diff.sort_values(['particle','frame']).groupby(by='particle')['x_diff1'].cumsum() 
        
        # Calculating particle history in x direction:
        # sorting by particle first, then frame, grouping then by particle and calculating the cumulative sum of displacements
        cumsums_y=t_drift_ysub_diff.sort_values(['particle','frame']).groupby(by='particle')['y_diff1'].cumsum()
        # same in y-direction
        
         # Sorting particle data in the same way
        t_no_drift_ysub_diff_sort=t_drift_ysub_diff.sort_values(['particle','frame'])
        
         # UNSICHER: + oder - ?????
        t_no_drift_ysub_diff_sort['x_corr']=t_drift_ysub_diff.sort_values(['particle','frame'])['x']-cumsums_x
        
        # subtracting drift-history for each particle
         # UNSICHER: + oder - ?????
        t_no_drift_ysub_diff_sort['y_corr']=t_drift_ysub_diff.sort_values(['particle','frame'])['y']-cumsums_y
        # same in y-direction
        
        #tm_sub=t1_ysub_diff_sort.copy()
        # just giving a more descriptive name to particle data
        t_no_drift_sub = t_no_drift_ysub_diff_sort 
        
        # dropping axes that wouldn't be needed any longer
        t_no_drift_sub = t_no_drift_sub.drop(['x', 'y', 'x_diff1', 'y_diff1'], axis=1) 
        
        #tm_sub=tm_sub.rename(columns={'x':'x', 'y':'y'}) 
        # renaming the corrected position into original names to keep the remaining code working with it
        t_no_drift_sub = t_no_drift_sub.rename(columns={'x_corr':'x', 'y_corr':'y'}) 
        
        # Bringing tm_sub back into a format that later parts of the code need to work with it
        t_no_drift_sub_store=t_no_drift_sub.copy()
        
        # Forgetting about ysub - which isn't needed anymore - and making frame the only index again
        t_no_drift_sub.set_index('frame', drop=True, inplace=True) 
        
        # Sorting by frame
        t_no_drift_sub=t_no_drift_sub.sort_index() 
        
        # Adding frame as a column
        t_no_drift_sub['frame'] = t_no_drift_sub.index 
        t_no_drift_sub = t_no_drift_sub[['x','y','mass','size','ecc','signal','raw_mass','ep','frame','abstime','particle']] # Ordering as needed later
        
        # Set this, if y-depending-drift-correction is to be used
        t_no_drift = t_no_drift_sub 
        
#        t_no_drift = tp.filter_stubs(t_no_drift, min_tracking_frames) 
        t_no_drift = t_no_drift.sort_values('frame')
        
        # insert y_range
        # total_drift.index = y_range RF180906 is that needed?
            
        # set two new indices - first frame than y_range        
        calc_drift = calc_drift.set_index(['frame'])
        
        # calc velocity as deviation of drift
        # average speed for display
        avg_frames = 30
        calc_drift[['velocity_y', 'velocity_x','new_y_range']] = calc_drift[['y','x','y_range']].diff(avg_frames)/avg_frames
        
        
        # Delete lines where new y range begins
        # ronny does not like python yet
        calc_drift_copy = calc_drift[abs(calc_drift['new_y_range']) == 0].copy()
        
        # still not...
        del calc_drift
        calc_drift = calc_drift_copy.copy()
        del calc_drift_copy
        
        # Do some plotting of the drift stuff
        
        
    if PlotDriftAvgSpeed == True:
        nd.visualize.DriftAvgSpeed()
       
    if PlotDriftTimeDevelopment == True:
        nd.visualize.DriftTimeDevelopment()  

    if PlotDriftFalseColorMapFlow == True:
        nd.visualize.DriftFalseColorMapFlow()
    
    if PlotDriftVectors == True:
        nd.visualize.DriftVectors()

    if PlotDriftFalseColorMapSpeed == True:
        nd.visualize.DriftFalseColorMapSpeed()

    if PlotDriftCorrectedTraj == True:
        nd.visualize.DriftCorrectedTraj()
    
    print('drift correction --> finished')
    
    return t_no_drift, settings
    
