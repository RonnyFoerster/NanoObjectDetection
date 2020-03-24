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


from pdb import set_trace as bp #debugger

#%%
def DriftCorrection(t_drift, ParameterJsonFile, Do_transversal_drift_correction = None, drift_smoothing_frames = None, rolling_window_size = None, min_particle_per_block = None, min_tracking_frames = None, PlotGlobalDrift = False, PlotDriftAvgSpeed = False, PlotDriftTimeDevelopment = False, PlotDriftFalseColorMapFlow = False, PlotDriftVectors = False, PlotDriftFalseColorMapSpeed = False, PlotDriftCorrectedTraj = False):
    
    """
    Calculate and remove overall drift from trajectories
    
    The drift needs to be removed, because the entire movement consists of brownian motion and drift
    In order to measure the brownian motion, the drift needs to be calculated and subtracted
    
    There are currently three options to choose from
    1) No drift correction - this is dangerous. However, if just a few particles are tracked the 
    average drift is most the particles movement and thus the trajectory vanishes!
    
    2) Global Drift
    Calculated the drift of all particles between neighbouring frames
    
    3) Transversal drift corretion
    Splits the fiber in several "subfibers". Each of them is treated independent. This is motivated by the idea of laminar
    flow, where particles on the side have a lower current than the ones in the middle
    However this method requires a lot of particles and makes sense for small fiber diameters where laminar flow is
    significant.
    
    """
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    ApplyDriftCorrection = settings["Drift"]["Apply"]    
    
    if ApplyDriftCorrection == 0:
        t_no_drift = t_drift
        
    else:
        
        if settings["Help"]["Drift"] == "auto":
            num_particles_per_frame = t_drift.groupby("frame")["particle"].count().mean()

            nd.ParameterEstimation.Drift(ParameterJsonFile, num_particles_per_frame)

        
        
        Do_transversal_drift_correction = settings["Drift"]["Do transversal drift correction"]    
        drift_smoothing_frames          = settings["Drift"]["Drift smoothing frames"]    
        rolling_window_size             = settings["Drift"]["Drift rolling window size"]    
        min_particle_per_block          = settings["Drift"]["Min particle per block"]    
        min_tracking_frames             = settings["Link"]["Min_tracking_frames"]
        
           
    
        if Do_transversal_drift_correction == False:
            print('Mode: global drift correction')
            # That's not to be used if y-depending correction (next block) is performed!
            
            # Attention: Strictly this might be wrong:
            # Drift might be different along y-positions of channel.
            # It might be more appropriate to divide into subareas and correct for drift individually there
            # That's done if Do_transversal_drift_correction==1
            my_drift = tp.compute_drift(t_drift, drift_smoothing_frames) # calculate the overall drift (e.g. drift of setup or flow of particles)
            
            """
            this is a bug in tracky. If there is no particle in a frame, it will sometimes not calculate the drift 
            in the next frame with a particle. So a small workaround here
            """

#            full_index = t_drift.frame.unique()
            full_index = t_drift.sort_values("frame").frame.unique()
            my_drift = my_drift.reindex(full_index)
            my_drift = my_drift.interpolate(method = 'linear')          

            t_no_drift = tp.subtract_drift(t_drift.copy(), my_drift) # subtract overall drift from trajectories (creates new dataset)

                
#            my_a = t_drift[t_drift.particle==103].x
#            my_b = t_no_drift[t_no_drift.particle==103].x
#            
#            plt.plot(my_drift,'o')
#            plt.plot(my_a,'x')
#            plt.plot(my_b,'x')
#            plt.plot(my_a - my_b,'x')

            
            if PlotGlobalDrift == True:
                nd.visualize.PlotGlobalDrift(my_drift) # plot the calculated drift
        
        
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
            
            #reindex https://stackoverflow.com/questions/25122099/move-column-by-name-to-front-of-table-in-pandas
            cols = t_no_drift_sub.columns.tolist()
            cols.insert(0, cols.pop(cols.index("y")))
            cols.insert(0, cols.pop(cols.index("x")))
            
            t_no_drift_sub = t_no_drift_sub.reindex(columns= cols)# Ordering as needed later
            
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
                nd.visualize.DriftFalseColorMapFlow(calc_drift, number_blocks, y_range)
            
            if PlotDriftVectors == True:
                nd.visualize.DriftVectors()
        
            if PlotDriftFalseColorMapSpeed == True:
                nd.visualize.DriftFalseColorMapSpeed()
        
            if PlotDriftCorrectedTraj == True:
                nd.visualize.DriftCorrectedTraj()
        
        print('drift correction --> finished')
        
        nd.handle_data.WriteJson(ParameterJsonFile, settings) 

    
    return t_no_drift