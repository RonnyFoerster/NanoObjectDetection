# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:48:34 2019

@author: foersterronny
"""

import numpy as np # Library for array-manipulation
import pandas as pd # Library for DataFrame Handling
import trackpy as tp # trackpy offers all tools needed for the analysis of diffusing particles
import sys
import NanoObjectDetection as nd



def Main(t_drift, ParameterJsonFile, PlotGlobalDrift = True, SaveDriftPlots = True, PlotDriftAvgSpeed = False, PlotDriftTimeDevelopment = False, PlotDriftFalseColorMapFlow = False, PlotDriftVectors = False, PlotDriftFalseColorMapSpeed = False, PlotDriftCorrectedTraj = False):
    
    """
    Calculate and remove overall drift from trajectories
    
    The entire movement consists of Brownian motion and drift.
    To measure pure Brownian motion, the drift needs to be calculated and subtracted.
    
    There are currently three options to choose from
    1) No drift correction - this is dangerous. Be sure that this is what you want.
    
    2) Global Drift
    Calculated the drift of all particles between neighbouring frames
    
    3) Transversal drift corretion
    Splits the fiber into several transversal pieces. Each of them is treated independently. 
    This is motivated by the idea of laminar flow, where particles close to 
    the channel wall are in a lower current than the ones in the center.
    However this method requires a lot of particles and makes sense for smaller 
    fiber diameters where laminar flow is significant.
    
    
    Parameters
    ----------
    t_drift : pandas
        trajectories with drift
    ParameterJsonFile : 
        

    Returns
    -------
    t_no_drift : pandas
        trajectories without drift
    
    """
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    ApplyDriftCorrection = settings["Drift"]["Apply"]    
    
    if ApplyDriftCorrection == 0:
        nd.logger.warning("No drift correction applied.")
        t_no_drift = t_drift
        
    else:
        nd.logger.info("Drift correction: starting...")
        
        # leave saturated data point out, because they are not precisely localized
        t_drift = t_drift[t_drift.saturated == False]
        
        if settings["Help"]["Drift"] == "auto":
            #estimate how many frames it needs to have enough particle to make the drift estimation
            num_particles_per_frame = t_drift.groupby("frame")["particle"].count().mean()

            # calculated how many frames are merged to make a good drift estimation
            avg_frames = nd.ParameterEstimation.Drift(ParameterJsonFile, num_particles_per_frame)

        
        # get relevant parameters
        LaminarFlow            = settings["Drift"]["LaminarFlow"]    
        avg_frames             = settings["Drift"]["Avg frames"]    
        rolling_window_size    = settings["Drift"]["Drift rolling window size"]    
        min_particle_per_block = settings["Drift"]["Min particle per block"]    


        # two different types of drift correction
        
        if LaminarFlow == False:
            # Attention: This ignores laminar flow, but needs fewer frames (and thus time) to get a good estimation

            nd.logger.warning("211008: Test a drift meets Kolmogorov Test method with data!")
            
            PureBrownianMotion = False
            PlotErrorIfTestFails = False
            
            CheckBrownianMotion = settings["MSD"]["CheckBrownianMotion"]

            if CheckBrownianMotion == True:
                nd.logger.info("Combination of Drift and Kolmogorov Test is switched of, because it is not validated")

            # here comes the drift correction
            t_no_drift, my_drift = GlobalEstimation(t_drift, avg_frames)

            
            # if CheckBrownianMotion == False:
            #     nd.logger.info("Kolmogorow-Smirnow test: off")
            #     t_no_drift, my_drift = GlobalEstimation(t_drift, avg_frames)
                
            # else:
            #     nd.logger.info("Kolmogorow-Smirnow test: on")
            #     while PureBrownianMotion == False:
            #         t_no_drift, my_drift = GlobalEstimation(t_drift, avg_frames)
               
            #         valid_particle_number = t_no_drift.particle.unique()
               
            #         # check if the trajectories follow brownian motion after the drift correction.
    
            #         #check if the histogram of each particle displacement is Gaussian shaped
            #         t_no_drift_true_brown_x = nd.get_trajectorie.CheckForPureBrownianMotion(t_no_drift, PlotErrorIfTestFails)
                    
            #         valid_particle_number_x = t_no_drift_true_brown_x.particle.unique()
                    
            #         # check it in y too
            #         t_no_drift_true_brown_xy = nd.get_trajectorie.CheckForPureBrownianMotion(t_no_drift_true_brown_x, PlotErrorIfTestFails, yEval = True)
    
            #         valid_particle_number_xy = t_no_drift_true_brown_xy.particle.unique()
            
                    
            #         # only when all trajectories return pure brownian motion is fullfilled
            #         if len(t_no_drift) == len(t_no_drift_true_brown_xy):
            #             PureBrownianMotion = True
            #         else:
            #             nd.logger.info("Rerun drift correrction with pure brownian motion particles")
            #             # vaild_id = t_no_drift_true_brown_xy.particle.unique()
            #             t_drift = t_drift[t_drift.particle.isin(valid_particle_number_xy)]
            
            # remove to short trajectories for further processing
            t_no_drift = nd.get_trajectorie.filter_stubs(t_no_drift, ParameterJsonFile, Mode = "Moving After Drift")
            

            # plot the calculated drift
            if PlotGlobalDrift == True:
                nd.visualize.PlotGlobalDrift(my_drift, settings, save=SaveDriftPlots) 
        
            
        
        else:
            t_no_drift, total_drift, calc_drift, number_blocks, y_range  = TransversalEstimation(settings, t_drift, avg_frames, rolling_window_size, min_particle_per_block)
            
            # do some plotting if wished
            if PlotDriftAvgSpeed == True:
                nd.visualize.DriftAvgSpeed(total_drift[['y','x']])
               
            if PlotDriftTimeDevelopment == True:
                nd.visualize.DriftTimeDevelopment(calc_drift, number_blocks)  
        
            if PlotDriftFalseColorMapFlow == True:
                nd.visualize.DriftFalseColorMapFlow(calc_drift, number_blocks, y_range)
            
            if PlotDriftVectors == True:
                nd.visualize.DriftVectors(calc_drift, number_blocks, y_range)
        
            if PlotDriftFalseColorMapSpeed == True:
                nd.visualize.DriftFalseColorMapSpeed(calc_drift, number_blocks, y_range)
        
            # if PlotDriftCorrectedTraj == True:
            #     nd.visualize.DriftCorrectedTraj(tm_sub)
        
        nd.logger.info("Drift correction: ...finished")

    #save the pandas
    if settings["Plot"]["save_data2csv"] == 1:
        nd.handle_data.pandas2csv(t_no_drift, settings["Plot"]["SaveFolder"], "traj_no_drift")


    return t_no_drift



def GlobalEstimation(t_drift, drift_smoothing_frames):
    """
    Estimates the drift for all particles in a frame together
    Attention: This ignores laminar flow, but needs fewer frames (and thus time) to get a good estimation

    Parameters
    ----------
    t_drift : pandas
        trajectory with drift
    drift_smoothing_frames : int or numpy
        number of frames whose average motion of all particles is defined as the drift.

    Returns
    -------
    t_no_drift : pandas
        trajectory without drift
    my_drift : pandas
        drift

    """
    
    
    nd.logger.info("Mode: global drift correction (frame per frame)")
            
    # calculate the overall drift (e.g. drift of setup or flow of particles)
    my_drift = tp.compute_drift(t_drift, drift_smoothing_frames) 
    
    # There is is a bug in tracky. If there is no particle in a frame, it will sometimes not calculate the drift in the next frame with a particle. So a small workaround here
    
    # get a list of all frames
    full_index = t_drift.frame.sort_values().unique()

    # interpolate the missing frames (the ones without a particle in it)
    my_drift = my_drift.reindex(full_index)
    my_drift = my_drift.interpolate(method = 'linear')          

    # subtract overall drift from trajectories
    t_no_drift = tp.subtract_drift(t_drift.copy(), my_drift) 

    t_no_drift = t_no_drift.drop(columns = "frame").reset_index()

    return t_no_drift, my_drift


def TransversalEstimation(settings, t_drift, drift_smoothing_frames, rolling_window_size, min_particle_per_block):   
    """
    Y-Depending drift-correction
    RF: Creation of y-sub-zones and calculation of drift
    SW 180717: Subtraction of drift from trajectories
    That code should work but not ideal sure. Have a look if you rely on it!

    Parameters
    ----------
    settings : TYPE
        DESCRIPTION.
    t_drift : pandas
        trajectory with drift.
    drift_smoothing_frames : int or numpy
        number of frames the drift is averaged to reduce noise.
    rolling_window_size : int or numpy
        number of neighbouring transversal blocks which are used for fitting. (If this is not existing, the temporal or transversal resolution is very poor)
    min_particle_per_block : int or numpy
        numer of particles which should be in one block

    Returns
    -------
    t_no_drift: trajectory without drift.
    total_drift: number of particles in each block
    calc_drift: drift as function of frame and transverval (y) components
    number_blocks: number of blocks considering the number of particles in each frame and how many particles one block should have
    y_range: y-coordinates of the blocks
    

    """
    nd.logger.info("Mode: transversal drift correction (laminar flow)")

    
    nd.logger.warning('That code should work but not ideal sure. Have a look if you rely on it!')
            
    
    # how many particles are needed to perform a drift correction
    #min_particle_per_block = 40
    
#    # use blocks above and below for averaging (more particles make drift correction better)
#    # e.g. 2 means y subarea itself and the two above AND below
#    rolling_window_size = 5

    #total number of averaged blocks
    average_over_total_block = 2 * rolling_window_size + 1
      
    # sort y values to have in each sub area the same amount of particles
    all_y_sorted = t_drift.y.values.copy()
    all_y_sorted.sort()
    
    y_min = all_y_sorted[0];   # min y of tracked particle
    y_max = all_y_sorted[-1];   # max y of tracked particle
    num_data_points = len(all_y_sorted)
    
    # total number of captured frames
    num_frames = t_drift.index.max() - t_drift.index.min() + 1
    
    """
    This is difficult to explain =/
    We have >num_data_points< data points and want to split them such, that each sub y has >min_particle_per_block< in it in each frame >num_frames<.
    Because of the averaging with neighbouring areas the effective number is lifted >average_over_total_block<
    start: distribute num_data_points over all number of frames and blocks
    """
    
    # calculate how many transversal blocks we can form
    number_blocks = int(num_data_points / min_particle_per_block / num_frames * average_over_total_block)
    
    # get the y-coordinates of all particles in a block
    sub_y = np.zeros(number_blocks+1)
    
    for x in range(0,number_blocks):
        use_index = int(num_data_points * (x / number_blocks))
        sub_y[x] = all_y_sorted[use_index]
    sub_y[-1] = y_max
    
    #average y-range in each block
    y_range = (sub_y[:-1] + sub_y[1:]) / 2
    
    # delete variable to start again
    if 'calc_drift' in locals():
        del calc_drift 
        del total_drift
        total_drift = pd.DataFrame(columns = ['y','x','frame'])
        calc_drift_diff = pd.DataFrame()

    # Creating a copy of the trajectory which will contain a new column with each values ysub-position
    t_drift_ysub = t_drift.copy()
    t_drift_ysub['ysub']=np.nan # Defining values to nan
    
    # loop the drift correction over the transversal blocks
    for x in range(0,number_blocks):
        print(x)
           
        # calc which subareas of y are in the rolling window.
        sub_y_min = x - rolling_window_size
        if sub_y_min  < 0:
            sub_y_min = 0
        
        sub_y_max = x + 1 + rolling_window_size
        if sub_y_max  > number_blocks:
            sub_y_max = number_blocks;
            
        # select which particles are in the surrounding blocks for the averaging
        use_part = (t_drift['y'] >= sub_y[sub_y_min]) & (t_drift['y'] < sub_y[sub_y_max])
        
        # select which particles are in the current block and are corrected
        use_part_subtract = (t_drift['y'] >= sub_y[x]) & (t_drift['y'] < sub_y[x+1]) 
        
        # get their indices
        use_part_index = np.where(np.array(use_part)==True)   # find index of elements true
        
        # WHAT IS THIS VARIABLE ACTUALLY GOOD FOR?
        use_part_subtract_index = np.where(np.array(use_part_subtract)==True)
        
        # Writing x as ysub into copy of t1. That's needed to treat data differently depending on y-sub-position
        # Python 3.5 t1_ysub['ysub'].iloc[use_part_subtract_index]=x 
        # I believe that's not an elegant way of doing it. Maybe find a better approach       
        t_drift_ysub.loc[use_part,'ysub'] = x # RF 180906
        

        
        # how many particles are in each frames
        use_part_index = list(use_part_index)[0]
        use_part_subtract_index = list(use_part_subtract_index)[0]
        num_particles_block = len(use_part_index)
            
        # check if drift_smoothing_frames is not longer than the video is long
        num_frames = settings["ROI"]["frame_max"] - settings["ROI"]["frame_min"] + 1
        if num_frames < drift_smoothing_frames:
            sys.exit("Number of frames is smaller than drift_smoothing_frames")


        # make the drift correction with the subframe
        calc_drift_y = tp.compute_drift(t_drift.iloc[use_part_index], drift_smoothing_frames)
        
        # calculate the drift of this y block
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
    
    #reindex https://stackoverflow.com/questions/25122099/move-column-by-name-to-front-of-table-in-pandas
    cols = t_no_drift_sub.columns.tolist()
    cols.insert(0, cols.pop(cols.index("y")))
    cols.insert(0, cols.pop(cols.index("x")))
    
    t_no_drift_sub = t_no_drift_sub.reindex(columns= cols)# Ordering as needed later
    
    # Set this, if y-depending-drift-correction is to be used
    t_no_drift = t_no_drift_sub 
    
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
    
    return t_no_drift, total_drift, calc_drift, number_blocks, y_range



def DriftCorrection(t_drift, ParameterJsonFile, Do_transversal_drift_correction = None, drift_smoothing_frames = None, rolling_window_size = None, min_particle_per_block = None, min_tracking_frames = None, PlotGlobalDrift = True, SaveDriftPlots = True, PlotDriftAvgSpeed = False, PlotDriftTimeDevelopment = False, PlotDriftFalseColorMapFlow = False, PlotDriftVectors = False, PlotDriftFalseColorMapSpeed = False, PlotDriftCorrectedTraj = False):
    
    nd.logger.warning("This is an old function. Use nd.Drift.Main from now on. It is still executed.")
    
    t5_no_drift = Main(t_drift, ParameterJsonFile, Do_transversal_drift_correction = None, drift_smoothing_frames = None, rolling_window_size = None, min_particle_per_block = None, PlotGlobalDrift = True, SaveDriftPlots = True, PlotDriftAvgSpeed = False, PlotDriftTimeDevelopment = False, PlotDriftFalseColorMapFlow = False, PlotDriftVectors = False, PlotDriftFalseColorMapSpeed = False, PlotDriftCorrectedTraj = False)
    
    return t5_no_drift