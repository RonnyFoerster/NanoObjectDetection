# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 09:52:27 2019

@author: Ronny Förster und Stefan Weidlich

"""

import numpy as np # Library for array-manipulation
import pandas as pd # Library for DataFrame Handling
import math # offering some maths functions
import warnings
import NanoObjectDetection as nd
from pdb import set_trace as bp #debugger
import matplotlib.pyplot as plt # Libraries for plotting
import sys
import scipy
import scipy.constants

def Main(t6_final, ParameterJsonFile, obj_all, microns_per_pixel = None, frames_per_second = None, amount_summands = None, amount_lagtimes = None,
         amount_lagtimes_auto = None, Histogramm_Show = True, MSD_fit_Show = False, EvalOnlyLongestTraj = 0, Max_traj_length = None):
    """ calculate diameters of individual particles via mean squared displacement analysis
    
    Procedure:
    1.) filter out too short tracks
    2.) construct matrix with defined lag-times for each particle. Amount of lag-times is set.
    3.) calculate mean and variance of lag-times for each particle
    4.) regress each particle individually linearly: msd per lag-time 
    5.) slope of regression used to calculate size of particle
    6.) standard error of lagtime=1 used as error on slope
    
    Parameters to be adjusted:
    # amount_summands ... If msd for a given lag-time is calculated by less than this amount of summands, the corresponding lag-time is disregarded in the fit
    # amount_lagtimes ... If the particle has less than this amount of valid lag-times that were fitted, 
    the particle wouldn't be taken into account. From 181120, SW: Not more than this datapoints are taken into account either
    # cutoff_size [nm] ... particles with size exceeding this will not be plotted
    #
    # binning = 25 # Amount of bins used in histogram
    """

    #%% read the parameters
    settings = nd.handle_data.ReadJson(ParameterJsonFile)

    microns_per_pixel = settings["MSD"]["effective_Microns_per_pixel"]
    frames_per_second = settings["MSD"]["effective_fps"]
    temp_water = settings["Exp"]["Temperature"]
    
    if temp_water < 250:
        raise Exception("Temperature is below 250K! Check if the temperature is inserted in K not in C!")
    
    solvent = settings["Exp"]["solvent"]
    
    #required since Viscosity_auto has old typo
    try:
        visc_auto = settings["Exp"]["Viscosity_auto"]
    except:
        visc_auto = settings["Exp"]["Viscocity_auto"]
        
    
    if visc_auto == 1:
        settings["Exp"]["Viscosity"] = nd.handle_data.GetViscosity(temperature = temp_water, solvent = solvent)
    
    
    #required since Viscosity has old typo
    try:
        visc_water = settings["Exp"]["Viscosity"]
    except:
        visc_water = settings["Exp"]["Viscocity"]

        
    
    min_rel_error = settings["MSD"]["Min rel Error"]
    
    # check if only the longest trajectory shall be evaluated
    EvalOnlyLongestTraj = settings["MSD"]["EvalOnlyLongestTraj"]
    
    if EvalOnlyLongestTraj == 1:
        longest_particle, longest_traj = nd.handle_data.GetTrajLengthAndParticleNumber(t6_final)
    
        settings["Split"]["ParticleWithLongestTrajBeforeSplit"] = longest_particle
    
        t6_final_use = t6_final[t6_final["particle"] == longest_particle].copy()
    else:
        t6_final_use = t6_final.copy()
    

    Max_traj_length = int(settings["Split"]["Max_traj_length"])
    Min_traj_length = int(settings["Link"]["Min_tracking_frames"])
    
    if Max_traj_length is None:
        t6_final_use = nd.get_trajectorie.split_traj_at_long_trajectorie(t6_final_use, settings)
    else:
        t6_final_use = nd.get_trajectorie.split_traj_at_long_trajectorie(t6_final_use, settings, Min_traj_length, Max_traj_length)

    
    amount_summands = settings["MSD"]["Amount summands"]
    amount_lagtimes_auto = settings["MSD"]["Amount lagtimes auto"]
 
    if amount_lagtimes_auto == 1:
        if settings["Exp"]["gain"] == "unknown":
            raise ValueError("Number of considered lagtimes cant be estimated automatically, if gain is unknown. Measure gain, or change 'Amount lagtime _auto' values to 0.")
        
        
    
    do_rolling = settings["Time"]["DoRolling"]
    my_rolling = settings["Time"]["Frames"]
                
    
    # insert hindrance factor parameters
    UseHindranceFac =  settings["MSD"]["EstimateHindranceFactor"]
    
    if UseHindranceFac == True:
        # fiber diameter is needed
        if settings["Fiber"]["Shape"] == "round":
            fibre_diameter_nm = settings["Fiber"]["TubeDiameter_nm"]
            
        if settings["Fiber"]["Shape"] == "hex":
            #calc out of hex diameters if wanted
            if settings["Fiber"]["CalcTubeDiameterOutOfSidelength"] == 0:
                fibre_diameter_nm = settings["Fiber"]["TubeDiameter_nm"]
            else:
                side_length_um = settings["Fiber"]["ARHCF_hex_sidelength_um"]
                diameter_inner_um, diameter_outer_um = nd.handle_data.ARHCF_HexToDiameter(side_length_um)
                diameter_inner_nm = np.round(1000 * diameter_inner_um,0)
                fibre_diameter_nm, settings = nd.handle_data.SpecificValueOrSettings(diameter_inner_nm, settings, "Fiber", "TubeDiameter_nm")
    else:
        fibre_diameter_nm = None
    
    MSD_fit_Show = settings["Plot"]['MSD_fit_Show']
    MSD_fit_Save = settings["Plot"]['MSD_fit_Save']
        
    if MSD_fit_Save == True:
        MSD_fit_Show = True
    
    particle_list_value=list(t6_final_use.particle.drop_duplicates())

    sizes_df_lin=pd.DataFrame(columns={'diameter','particle'})       
    sizes_df_lin_rolling = pd.DataFrame()   
    
    any_successful_check = False

    # go through the particles
    num_loop_elements = len(particle_list_value)
    for i,particleid in enumerate(particle_list_value): # iteratig over all particles
        print("\n Particle number: ",  round(particleid))

#        nd.visualize.update_progress("Analyze Particles", (i+1)/num_loop_elements)
#        print("Particle Id: ", particleid)
        # select track to analyze
        eval_tm = t6_final_use[t6_final_use.particle==particleid]
        start_frame = int(eval_tm.iloc[0].frame)

        mean_mass = eval_tm["mass"].mean()
        mean_size = eval_tm["size"].mean()
        mean_ecc = eval_tm["ecc"].mean()
        mean_signal = eval_tm["signal"].mean()
        mean_raw_mass = eval_tm["raw_mass"].mean()
        mean_ep = eval_tm["ep"].mean()
        max_step = eval_tm["rel_step"].max()
        true_particle = eval_tm["true_particle"].max()
        
        # defines how many lagtimes are considered for the fit
        if amount_lagtimes_auto == 1:
            max_counter = 10
            #maximum value of the min lagtimes, sotaht the minimum number of summands can be achieved for the first lagtime, given the trajectory length of the particle
            length = len(eval_tm)
            
            lagtimes_min_max = np.int(np.floor(length/settings["MSD"]["Amount summands"]))
            lagtimes_min = CalculateLagtimes_min(eval_tm, lagtimes_min_max)
            p_min_old = 2
            lagtimes_max = lagtimes_min + p_min_old - 1
        else:
            max_counter = 1
            lagtimes_min = settings["MSD"]["lagtimes_min"]
            lagtimes_max = settings["MSD"]["lagtimes_max"]
        
        if lagtimes_min > 0:
            stable_num_lagtimes = False
            counter = 0
    
                # iterate msd fit till it converges
            while ((stable_num_lagtimes == False) and (counter < max_counter)):
                counter = counter + 1 
                # Calc MSD
    
                nan_tm_sq, amount_frames_lagt1, enough_values, traj_length = \
                CalcMSD(eval_tm, microns_per_pixel, amount_summands, lagtimes_min = lagtimes_min, lagtimes_max = lagtimes_max)
    
    #            print("enough values:  ",enough_values)
    
                if enough_values == True:  
                    if any_successful_check == False:
                        any_successful_check = True
                        if MSD_fit_Show == True:
                            plt.figure()
                            
                    #iterative to find optimal number of lagtimes in the fit    
                    # Average MSD (several (independent) values for each lagtime)
    
                    lagt_direct, mean_displ_direct, mean_displ_sigma_direct = \
                    AvgMsdRolling(nan_tm_sq, frames_per_second, DoRolling = False, lagtimes_min = lagtimes_min, lagtimes_max = lagtimes_max)
    
                    diff_direct_lin, diff_std = \
                    FitMSDRolling(lagt_direct, amount_frames_lagt1, mean_displ_direct, mean_displ_sigma_direct, \
                               PlotMsdOverLagtime = MSD_fit_Show, DoRolling = False)
                        
                    if do_rolling == True:
                        lagt_direct, mean_displ_direct, mean_displ_sigma_direct = \
                        AvgMsdRolling(nan_tm_sq, frames_per_second, my_rolling = my_rolling, DoRolling = True, lagtimes_min = lagtimes_min)
                        bp()
                        diff_direct_lin_rolling = \
                        FitMSDRolling(lagt_direct, amount_frames_lagt1, mean_displ_direct, mean_displ_sigma_direct, \
                                   PlotMsdOverLagtime = MSD_fit_Show, my_rolling = my_rolling, DoRolling = True)
    
    
                    if amount_lagtimes_auto == 1:
                        # calculate theoretical best number of considered lagtimes
                        p_min = OptimalMSDPoints(settings, mean_ep, mean_raw_mass, diff_direct_lin, amount_frames_lagt1)
    
                        if p_min == p_min_old:
                            stable_num_lagtimes = True 
                        else:
                            lagtimes_max = lagtimes_min + p_min - 1
                            print("p_min_before = ",p_min_old)
                            print("p_min_after  = ",p_min)
                            
                            p_min_old = p_min
                            
                            # drop last line, because it is done again
    #                        sizes_df_lin = sizes_df_lin[:-1]                   
    
    
            if enough_values == True: 
                red_ep = ReducedLocalPrecision(settings, mean_raw_mass, diff_direct_lin)
                
                # get the fit error if switches on (and working)
    #            rel_error_diff, diff_std = DiffusionError(traj_length, red_ep, diff_direct_lin, min_rel_error, lagtimes_max)
                
                diameter = DiffusionToDiameter(diff_direct_lin, UseHindranceFac, fibre_diameter_nm, temp_water, visc_water)

                sizes_df_lin = ConcludeResults(sizes_df_lin, diff_direct_lin, diff_std, diameter, \
                                       particleid, traj_length, amount_frames_lagt1, start_frame, \
                                       mean_mass, mean_size, mean_ecc, mean_signal, mean_raw_mass, mean_ep, \
                                       red_ep, max_step, true_particle)
                
                if do_rolling == True:
                    red_ep_rolling = ReducedLocalPrecision(settings, mean_raw_mass, diff_direct_lin_rolling)
               
                    # get the fit error if switches on (and working)
                    rel_error_diff_rolling, diff_std_rolling = \
                    DiffusionError(traj_length, red_ep_rolling, diff_direct_lin_rolling, min_rel_error)
                    
                    diameter_rolling = DiffusionToDiameter(diff_direct_lin_rolling, UseHindranceFac, fibre_diameter_nm, temp_water, visc_water, DoRolling = True)
                    
                    sizes_df_lin_rolling = ConcludeResultsRolling(sizes_df_lin_rolling, diff_direct_lin_rolling, \
                                           diff_std_rolling, diameter_rolling, \
                                           particleid, traj_length, amount_frames_lagt1, start_frame, \
                                           mean_mass, mean_size, mean_ecc, mean_signal, mean_raw_mass, mean_ep, \
                                           red_ep, max_step)
                    
                else:
                    sizes_df_lin_rolling = -1

        else:
            print("Localization precision to low for MSD fit")
    
    if MSD_fit_Show == True:    
        # get the axis from the msd plot right
        ax_msd = plt.gca()
    
        #get maximum value of lagtime and msd
        y_min, y_max, x_min, x_max = GetLimitOfPlottedLineData(ax_msd)
        
        ax_msd.set_xlim([0, 1.1*x_max])
        ax_msd.set_ylim([0, 1.1*y_max])
    
    sizes_df_lin = sizes_df_lin.set_index('particle')

    if do_rolling == True:
        sizes_df_lin_rolling = sizes_df_lin_rolling.set_index('frame')
    else:
        sizes_df_lin_rolling = "Undone"

    if MSD_fit_Save == True:
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "MSD Fit", settings)
    
    nd.handle_data.WriteJson(ParameterJsonFile, settings) 

    
    return sizes_df_lin, sizes_df_lin_rolling, any_successful_check

 
def GetLimitOfPlottedLineData(ax):
    
    x_min = np.inf
    x_max = -np.inf

    y_min = np.inf
    y_max = -np.inf
    
    
    for counter, ax_line in enumerate(ax.lines):
         x_min_test = np.min(ax_line.get_xdata())
         if x_min_test < x_min:
             x_min = x_min_test
             
         x_max_test = np.max(ax_line.get_xdata())
         if x_max_test > x_max:
             x_max = x_max_test

         y_min_test = np.min(ax_line.get_ydata())
         if y_min_test < y_min:
             y_min = y_min_test
             
         y_max_test = np.max(ax_line.get_ydata())
         if y_max_test > y_max:
             y_max = y_max_test
             
    return y_min, y_max, x_min, x_max


def CalcMSD(eval_tm, microns_per_pixel = 1, amount_summands = 5, lagtimes_min = 1, lagtimes_max = 2):

    nan_tm_sq = 0
    amount_frames_lagt1 = 0
#    nd.visualize  
    #    fig=plt.figure()
    #    ax = fig.add_subplot(1, 1, 1)
    #    ax.set_yscale('log')
    #    ax.set_xscale('log')
    #    plt.title('Mean-Square-Diffusion: Min. #summands = %r and track-length = %r' %(amount_summands, amount_lagtimes))
    #    plt.ylabel(r'$\langle \Delta x^2 \rangle$ [$\mu$m$^2$]')
    #    plt.xlabel('lag time $t$ [s]')
    #     
    #    sizes_df_lin=pd.DataFrame(columns={'diameter','particle'})         
      
    
    min_frame=np.min(eval_tm.frame) # frame that particle's trajectory begins
    max_frame=np.max(eval_tm.frame) # frame that particle's trajectory ends
    
    # Checking already here, if particle-track is long enough:
    length_indexer = max_frame - min_frame + 1
    traj_length = length_indexer

    if length_indexer < (lagtimes_max + amount_summands * lagtimes_max):
        enough_values = False
        print("Trajectory is to short to have enough data points. \n Trajectorie length must be larger than (amount_summands * lagtimes_max). \n Consider optimizing parameters Min_tracking_frames, amount_summands, lagtimes_max.")
    
    else:
        # columns has two parts. The lagtimes:
        my_columns = range(lagtimes_min, lagtimes_max + 1)
        #now add the zero where the position is int
        my_columns = [0, *my_columns]
        
        nan_tm = pd.DataFrame(index=np.arange(min_frame,max_frame+1),columns = my_columns) 
        # setting up a matrix (dataframe) that has the correct size to put in all needed combinations of displacement and lag-times:
        # rows: frames: all frames from first appearance to diappearance are created. even if particle isn't recorded for some time,
        # in this case the respective positions will be filled with nan (that's the reason for the name of the variable)
        # columns: 0: initial position in respective frame
        # columns: all others: displacement from original position
        
        
        nan_tm[0]=eval_tm.x*microns_per_pixel # filling column 0 with position of respective frame
#        nan_tm[0]=eval_tm.y*microns_per_pixel # filling column 0 with position of respective frame
    
        for row in range(lagtimes_min, lagtimes_max + 1):
            nan_tm[row]=nan_tm[0].diff(row) # putting the displacement of position into the columns, each representing a lag-time
        
        # Checking already here if enough non NAN values exist to calculate
        # if not, particle is disregarded. another check of observation-number for each lagt is then obsolet

        amount_frames_lagt1 = nan_tm[0].count() # SW, 181125: Might be better to use the highest lag-t for filtering instead of lag-t=1?!

        tm_lagt_max = nan_tm[lagtimes_max]
        nan_tm_lagt_max = tm_lagt_max[np.isnan(tm_lagt_max) == False] 
        
        amount_frames_lagt_max = len(nan_tm_lagt_max)
        
#        amount_frames_lagt_max = 

        if amount_frames_lagt_max < (lagtimes_max + 1 + amount_summands * lagtimes_max):
            enough_values = False
            print("To many holes in trajectory to have enough data points. \n Number of data points must be larger than (amount_summands * lagtimes_max). \n Consider optimizing parameters Dark tim, Min_tracking_frames, amount_summands, lagtimes_max")

        else:
            enough_values = True    
            nan_tm_sq=nan_tm**2 # Squaring displacement    
     
    return nan_tm_sq, amount_frames_lagt1, enough_values, traj_length
    


def AvgMsd(nan_tm_sq, frames_per_second):
    mean_displ_direct = pd.Series() # To hold the mean sq displacement of a particle
    mean_displ_variance_direct = pd.Series() # To hold the variance of the msd of a particle
#    my_columns = nan_tm_sq.columns
    
    for column in nan_tm_sq.columns[1:]:
        # That iterates over lag-times for the respective particle and
        # calculates the msd in the following manner:
        # 1. Build sub-blocks for statistically independent lag-time measurements
        nan_indi_means = rolling_with_step(nan_tm_sq[column], column, column, mean_func)
        
        # 2. Calculate mean msd for the sublocks
        mean_displ_direct_loop = nan_indi_means.mean(axis=0)
        mean_displ_direct = mean_displ_direct.append(pd.Series(index=[column], data=[mean_displ_direct_loop]), sort=False)
        
        # 3. Check how many independent measurements are present (this number is used for filtering later. 
        # Also the iteration is limited to anaylzing
        # those lag-times only that can possibly yield enough entries according to the chosen filter). 
        len_nan_tm_sq_loop = nan_indi_means.count()
        
        # 4. Calculate the mean of these sub-means --> that's the msd for each lag-time
        # 5. Calculate the variance of these sub-means --> that's used for variance-based fitting 
        # when determining the slope of msd over time
        
        mean_displ_variance_direct_loop = nan_indi_means.var(axis=0)*(2/(len_nan_tm_sq_loop-1))
        mean_displ_variance_direct = mean_displ_variance_direct.append(pd.Series(index=[column], data=[mean_displ_variance_direct_loop]), sort=False)
        
        mean_displ_sigma_direct = np.sqrt(mean_displ_variance_direct)
        
    lagt_direct = mean_displ_direct.index/frames_per_second # converting frames into physical time: 
    # first entry always starts with 1/frames_per_second
    
    return lagt_direct, mean_displ_direct, mean_displ_sigma_direct



def AvgMsdRolling(nan_tm_sq, frames_per_second, my_rolling = 100, DoRolling = False, lagtimes_min = 1, lagtimes_max = 2):
    
    num_cols = len(nan_tm_sq.columns) - 1  
    
    lagt_direct = np.linspace(lagtimes_min,lagtimes_max ,num_cols) /frames_per_second # converting frames into physical time: 
    # first entry always starts with 1/frames_per_second
    
    if DoRolling == False:
#        mean_displ_direct = pd.DataFrame(index = [0], columns = nan_tm_sq.columns.tolist()[1:])
        mean_displ_direct = np.zeros(num_cols)
        mean_displ_variance_direct = mean_displ_direct.copy()    

        for column in range(num_cols):
            eval_column = column + lagtimes_min
            # That iterates over lag-times for the respective particle and
            # calculates the msd in the following manner:
            # 1. Build sub-blocks for statistically independent lag-time measurements

            nan_indi_means = rolling_with_step(nan_tm_sq[eval_column], eval_column, eval_column, mean_func)
            
            # 2. Calculate mean msd for the sublocks
            mean_displ_direct[column] = nan_indi_means.mean(axis=0)
#            mean_displ_direct[column] = nan_indi_means.median(axis=0)
            
            
            # 3. Check how many independent measurements are present (this number is used for filtering later. 
            # Also the iteration is limited to anaylzing
            # those lag-times only that can possibly yield enough entries according to the chosen filter). 
            
            len_nan_tm_sq_loop = nan_indi_means.count()
            
            # 4. Calculate the mean of these sub-means --> that's the msd for each lag-time
            # 5. Calculate the variance of these sub-means --> that's used for variance-based fitting 
            # when determining the slope of msd over time
            
            # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
            
            #OLD RONNY DOES NOT KNOW WHY THERE IS THIS FACTOR OF 2
#            mean_displ_variance_direct_loop = nan_indi_means.var(axis=0)*(2/(len_nan_tm_sq_loop-1))
            mean_displ_variance_direct_loop = nan_indi_means.var(axis=0) / (len_nan_tm_sq_loop-1)
            
            mean_displ_variance_direct[column] = mean_displ_variance_direct_loop

    else:   
        mean_displ_direct = pd.DataFrame(index = nan_tm_sq.index, columns = nan_tm_sq.columns.tolist()[1:])
        mean_displ_variance_direct = mean_displ_direct.copy()    

        for column in mean_displ_direct.columns:
            my_rol = np.int(np.floor(my_rolling/column))
            nan_indi_means = rolling_with_step(nan_tm_sq[column], column, column, mean_func)
            mean_displ_direct[column] = nan_indi_means.rolling(my_rol, center = True, min_periods = 1).mean()
            
            len_nan_tm_sq_loop = nan_indi_means.rolling(my_rol, center = True).count()            
           
            mean_displ_variance_direct[column] = nan_indi_means.rolling(my_rol, center = True, min_periods = 1).var(ddof = False)*(2/(len_nan_tm_sq_loop-1))
            
            # mean_displ_direct[column] = nan_indi_means.rolling(my_rolling, center = True).mean()
            # mean_displ_direct = mean_displ_direct.append(pd.Series(index=[column], data=[mean_displ_direct_loop]))

            #remove NaN at the side with nearest neighbour
            mean_displ_direct = FillNanInPanda(mean_displ_direct)
            mean_displ_variance_direct = FillNanInPanda(mean_displ_variance_direct)
    
    mean_displ_sigma_direct = np.sqrt(mean_displ_variance_direct)
    

    
    return lagt_direct, mean_displ_direct, mean_displ_sigma_direct



def FillNanInPanda(panda_in):
    # fills gaps in between linearly
    panda_out = panda_in.interpolate()
    
    # fill NaN at the beginning and the end with nearest neighbour
    panda_out = panda_out.fillna(method = "bfill")
    panda_out = panda_out.fillna(method = "ffill")
    
    return panda_out




def FitMSDRolling(lagt_direct, amount_frames_lagt1, mean_displ_direct, mean_displ_sigma_direct, PlotMsdOverLagtime = False,  my_rolling = 100, DoRolling = False):

    if DoRolling == True:
        mean_displ_sigma_direct = mean_displ_sigma_direct.median()

    # new method
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    def lin_func(x,m,n):
        y = m * x + n
        return y


#    CovCalculateable = (len(lagt_direct) >= 3)

    [fit_values, fit_cov]= scipy.optimize.curve_fit(lin_func, lagt_direct, mean_displ_direct, sigma = mean_displ_sigma_direct, absolute_sigma=True , bounds = ([0,0],[np.inf, np.inf]))
    
    diff_direct_lin = np.squeeze(fit_values[0]/2)
    var_diff_direct_lin = np.squeeze(fit_cov[0,0]/4)
    std_diff_direct_lin = np.sqrt(var_diff_direct_lin)
    
#    if CovCalculateable == True:

        
#    else:
#        bp()
#        std_diff_direct_lin = "to few data points"   

    
    if PlotMsdOverLagtime == True:
        
        if DoRolling == True:
            mean_displ_direct = mean_displ_direct.median()
            fit_values = np.median(fit_values,axis=1)    
            
#        mean_displ_fit_direct_lin=lagt_direct.map(lambda x: x*fit_values[0]+ fit_values[1])
        mean_displ_fit_direct_lin = lagt_direct *fit_values[0]+ fit_values[1]
    
        nd.visualize.MsdOverLagtime(lagt_direct, mean_displ_direct, mean_displ_fit_direct_lin)
    
    
    
    return diff_direct_lin, std_diff_direct_lin


#    def lin_func_flow(x,m,n,V):
#        y = m * x + n + V * (x**2)
#        return y
# 
#    [fit_values_flow, fit_cov_flow]= scipy.optimize.curve_fit(lin_func_flow, lagt_direct, mean_displ_direct, sigma = mean_displ_sigma_direct)
    

#    [fit_values, fit_cov]= scipy.optimize.curve_fit(lin_func, lagt_direct, mean_displ_direct, sigma = mean_displ_sigma_direct)
      
#    fit_values = fit_values_flow
#    fit_cov = fit_cov_flow
    
#    print("flow= ", np.abs(fit_values_flow[2]))
#    
#    if (fit_values_flow[0] < 0) | (np.abs(fit_values_flow[2]) > 20):
#        diff_direct_lin = 1000
#        std_diff_direct_lin = 0
#    
##    [fit_values, fit_cov]= np.polyfit(lagt_direct, mean_displ_direct, 1, w = 1/mean_displ_sigma_direct, cov=True) 
#    else:
#        [fit_values, fit_cov]= scipy.optimize.curve_fit(lin_func, lagt_direct, mean_displ_direct, sigma = mean_displ_sigma_direct)
#        
#        diff_direct_lin = np.squeeze(fit_values[0]/2)
#        var_diff_direct_lin = np.squeeze(fit_cov[0,0]/4)
#        std_diff_direct_lin = np.sqrt(var_diff_direct_lin)

#    if DoRolling == False:
#        bp()
#        weigths = np.squeeze(np.asarray(mean_displ_sigma_direct))
#    else:
#        bp()

#    if 1 == 0:
#        #old method
#        if (len(lagt_direct) >= 5):
#            [fit_values, fit_cov]= np.polyfit(lagt_direct, mean_displ_direct.T, 1, w = 1/mean_displ_sigma_direct, cov=True) 
#    
#        else:
#            try:
#                fit_values= np.polyfit(lagt_direct, mean_displ_direct.T, 1, w = 1/mean_displ_sigma_direct) 
#            except:
#                print("not good!")

#    diff_direct_lin = fit_values[0]/2
#    diff_direct_lin = np.squeeze(diff_direct_lin)   






def DiffusionToDiameter(diffusion, UseHindranceFac = 0, fibre_diameter_nm = None, temp_water = 295, visc_water = 9.5e-16, DoRolling = False):
    const_Boltz = scipy.constants.Boltzmann # Boltzmann-constant
    
    diameter = (2*const_Boltz*temp_water/(6*math.pi *visc_water)*1e9) /diffusion # diameter of each particle


    if UseHindranceFac == True:
        if diameter < fibre_diameter_nm:
            if DoRolling == False:
                diamter_corr = EstimateHindranceFactor(diameter, fibre_diameter_nm, DoPrint = True)
            else:
                diamter_corr = diameter.copy()
                for counter, diameter_rolling in enumerate(diameter):
                    diamter_corr[counter] = EstimateHindranceFactor(diameter_rolling, fibre_diameter_nm, DoPrint = False)
                
        else:
            diamter_corr = diameter
            print("WARNING: STATIONARY PARTICLE STILL INSIDE")
#            sys.exit("Here is something wrong: The diameter is calculated to be larger than the core diameter. \
#                     Possible Reasons: \
#                     \n 1 - drift correctes motion instead of drift because to few particles inside. \
#                     \n 2 - stationary particle remains, which seems very big because it diffuses to little. ")
#            
    else:
        print("WARNING: hindrance factor ignored. You just need to have the fiber diameter!")   
        diamter_corr = diameter
    
    return diamter_corr



def EstimateHindranceFactor(diam_direct_lin, fibre_diameter_nm, DoPrint = True):
    
    diam_direct_lin_corr = diam_direct_lin
    diam_direct_lin_corr_old = 0
    
    my_iter = 0        
    while np.abs(diam_direct_lin_corr - diam_direct_lin_corr_old) > 0.1:
        my_iter = my_iter + 1
        diam_direct_lin_corr_old = diam_direct_lin_corr
        
#        corr_visc = 1 + 2.105 * (diam_direct_lin_corr / fibre_diameter_nm)
#        hindrance = hindrance_fac(fibre_diameter_nm,diam_direct_lin_corr)
        hindrance = hindrance_fac(fibre_diameter_nm,diam_direct_lin_corr)
        corr_visc = 1 / hindrance
        diam_direct_lin_corr = diam_direct_lin / corr_visc # diameter of each particle
        
        # this steps helps converging. Otherwise it might jump from 1/10 to 10 to 1/10 ...
        diam_direct_lin_corr = np.sqrt(diam_direct_lin_corr * diam_direct_lin_corr_old)
#        print(my_iter, 'diamter:',diam_direct_lin,'corr_visc:',corr_visc,'corr diameter:',diam_direct_lin_corr)
        
        if my_iter > 100:
            print("Iteration does not converge. Abort !!!")
            bp()
            input("PRESS ENTER TO CONTINUE.")
            diam_direct_lin_corr = diam_direct_lin_corr_old

    if DoPrint == True:
#        print("After iteration %d: Starting Diameter: %.1f nm; corr. viscosity: %.3f; corr. diameter: %.2nmf" % (my_iter, round(diam_direct_lin,1), round(corr_visc,3), round(diam_direct_lin_corr,2)))
        print("Starting Diameter: %.1fnm; hindrance factor: %.3f; Corrected Diameter: %.2fnm" % (round(diam_direct_lin,1), round(corr_visc,3), round(diam_direct_lin_corr,2)))


    return diam_direct_lin_corr



def ConcludeResults(sizes_df_lin, diff_direct_lin, diff_std, diameter,
                    particleid, traj_length, amount_frames_lagt1, start_frame,
                    mean_mass, mean_size, mean_ecc, mean_signal, mean_raw_mass, mean_ep,
                    red_ep, max_step, true_particle):

    # Storing results in df:
#    sizes_df_lin = sizes_df_lin.append(pd.DataFrame(data={'particle': [particleid],
#                                                          'diffusion': [diff_direct_lin],
#                                                          'diffusion std': [diff_std],
#                                                          'diameter': [diameter],
#                                                          'ep': [mean_ep],
#                                                          'redep' : [red_ep], 
#                                                          'signal': [mean_signal],
#                                                          'mass': [mean_mass],
#                                                          'rawmass': [mean_raw_mass],
#                                                          'max step': [max_step],
#                                                          'first frame': [start_frame],
#                                                          'traj length': [traj_length],
#                                                          'valid frames':[amount_frames_lagt1],
#                                                          'size': [mean_size],
#                                                          'ecc': [mean_ecc],
#                                                          'true_particle': [true_particle]}),sort=False)
    
    sizes_df_lin = sizes_df_lin.append(pd.DataFrame(data={'particle': [particleid],
                                                          'diffusion': [diff_direct_lin],
                                                          'diffusion std': [diff_std],
                                                          'diameter': [diameter],
                                                          'ep': [mean_ep],
                                                          'redep' : [red_ep], 
                                                          'signal': [mean_signal],
                                                          'mass': [mean_mass],
                                                          'rawmass': [mean_raw_mass],
                                                          'max step': [max_step],
                                                          'first frame': [start_frame],
                                                          'traj length': [traj_length],
                                                          'valid frames':[amount_frames_lagt1],
                                                          'size': [mean_size],
                                                          'ecc': [mean_ecc],
                                                          'true_particle': [true_particle]}), sort=False)
    
    
    
    return sizes_df_lin



def ConcludeResultsRolling(sizes_df_lin_rolling, diff_direct_lin_rolling, diff_std_rolling, diameter_rolling, 
                    particleid, traj_length, amount_frames_lagt1, start_frame,
                    mean_mass, mean_size, mean_ecc, mean_signal, mean_raw_mass, mean_ep,
                    red_ep, max_step):
    

    # Storing results in df:
    new_panda = pd.DataFrame(data={'particle': particleid,
                                  'frame': np.arange(start_frame,start_frame + len(diff_direct_lin_rolling), dtype = int),
                                  'diffusion': diff_direct_lin_rolling,
                                  'diffusion std': diff_std_rolling,
                                  'diameter': diameter_rolling,
                                  'ep': mean_ep,
                                  'redep' : red_ep, 
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



def OptimalMSDPoints(settings, ep, raw_mass, diffusivity, amount_frames_lagt1):
    """
    
    XAVIER MICHALET AND ANDREW J. BERGLUND, PHYSICAL REVIEW E 85, 2012
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4917385/
    """
    red_x = ReducedLocalPrecision(settings, raw_mass, diffusivity)
    
    f_b = 2
    bp()
    if type(red_x) != type('abc'): # exclude that red_x is a string ("unknown")
        if red_x >= 0:
            f_b = 2 + 1.35 * np.power(red_x,0.6)
    
    L_b = 0.8 + 0.564 * amount_frames_lagt1
    
    value_1 = L_b
    value_2 = (f_b * L_b) / np.power(np.power(f_b,3) + np.power(L_b,3),1/3)
       
    p_min = np.int(np.min(np.round([value_1, value_2])))

    return p_min
    
    
    
def ReducedLocalPrecision(settings, mass, diffusion, DoRolling = False):
    """ calculate reduced square localization error from experimental parameters
    
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4917385/#FD13
    Eq. 13
    """
    
#    local_precision_um = ep * settings["Exp"]["Microns_per_pixel"]
    lagtime_s = 1/settings["Exp"]["fps"]
    exposure_time_s = settings["Exp"]["ExposureTime"]
    
    NA = settings["Exp"]["NA"]
    lambda_nm = settings["Exp"]["lambda"]
    gain = settings["Exp"]["gain"]
    
    rayleigh_nm = 2 * 0.61 * lambda_nm / NA
    rayleigh_um = rayleigh_nm / 1000
    
    # we use rayleigh as resolution of the system
    # 2* because it is coherent
    # not sure here
    if gain == "unknown":
        red_x = "unknown"
        
    else:
        num_photons = mass / gain
        static_local_precision_um = rayleigh_um / np.power(num_photons ,1/2)
        
        # Eq. 13:
        red_x = np.power(static_local_precision_um,2) / (diffusion * lagtime_s) \
        * (1 + (diffusion * exposure_time_s / np.power(rayleigh_um,2))) \
        - 1/3 * (exposure_time_s / lagtime_s)
    
    #    red_x = np.power(local_precision_um,2) / (diffusion * lagtime_s) \
    #    * (1 + (diffusion * exposure_time_s / np.power(rayleigh_um,2))) \
    #    - 1/3 * (exposure_time_s / lagtime_s)
        
        if red_x < 0:
            red_x = 0
    
    return red_x
    


def DiffusionError(traj_length, red_x, diffusion, min_rel_error, lagtimes_max, DoRolling = False):
    
    if red_x != "unknown":
        """
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4917385/#FD7
        Eq 7
        """
        if DoRolling == False:
            delta = DeltaErrorEstimation(red_x,traj_length)
        else:
            bp()
            
        if np.isnan(delta) == True:
            rel_error = min_rel_error
        else:
            rel_error = np.power(2/(traj_length-1) * (1+delta**2),1/2)
    
    else:
        #Foerster2019 ARHCF-paper
        rel_error = np.sqrt((2*lagtimes_max) / (3*(traj_length-lagtimes_max)))


    
    diffusion_std = diffusion * rel_error
    
    "Min rel Error"
    
    return rel_error, diffusion_std





def DeltaErrorEstimation(red_x,traj_length):
    """
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4917385/#APP1
    Eq A5 - A8
    """

    # old red_x**2 is wrong
#    y1 = 1 / np.power(1 + 2*red_x**2 , 1/2)
#    y2 = (1+red_x) / np.power(1 + 2*red_x**2 , 3/2)

    y1 = 1 / np.sqrt(1 + 2*red_x)
    y2 = (1+red_x) / np.power(1 + 2*red_x , 3/2)
    
    
    delta = (1-y1) / np.sqrt(y2 - y1**2)

    
    return delta


def OptimizeTrajLenght(t6_final, ParameterJsonFile, obj_all, microns_per_pixel = None, frames_per_second = None, amount_summands = None, amount_lagtimes = None,
         amount_lagtimes_auto = None, Histogramm_Show = True, MSD_fit_Show = False, EvalOnlyLongestTraj = 0):
    
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
        sizes_df_lin_new, any_successful_check = nd.CalcDiameter.Main(t6_final, ParameterJsonFile, obj_all, 
                                                                  Max_traj_length = current_max_traj_length, MSD_fit_Show = True)


        if cur_ind_particles == 1:
            sizes_df_lin = sizes_df_lin_new
        else:
            
            sizes_df_lin = sizes_df_lin.append(sizes_df_lin_new, sort=False)
            
        cur_ind_particles = cur_ind_particles * 2


    return sizes_df_lin, any_successful_check



def rolling_with_step(s, window, step, func): 
    # Defining a function that allows to calculate another function over a window of a series,
    # while the window moves with steps 
    # See https://ga7g08.github.io/2015/01/30/Applying-python-functions-in-moving-windows/
    vert_idx_list = np.arange(0, s.size - window, step)
    hori_idx_list = np.arange(window)
    A, B = np.meshgrid(hori_idx_list, vert_idx_list)
    idx_array = A + B
    x_array = s.values[idx_array]
    idx = s.index[vert_idx_list + int(window/2.)]
    d = func(x_array)
    return pd.Series(d, index=idx)



def mean_func(d):
    # Function that calculates the mean of a series, while ignoring NaN and not yielding a warning when all are NaN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(d, axis=1)


def hindrance_fac(diam_channel,diam_particle):
    """ calculate the hindrance factor for diffusion from particle and channel dimensions
    according to eq. 16 from "Hindrance Factors for Diffusion and Convection in Pores", 
    Dechadilok and Deen (2006)
    """
    
    l = diam_particle / diam_channel # relative particle size
    
    H = 1 + 9/8*l*np.log(l) - 1.56034 * l \
    + 0.528155 * np.power(l,2) \
    + 1.915210 * np.power(l,3) \
    - 2.819030 * np.power(l,4) \
    + 0.270788 * np.power(l,5) \
    + 1.101150 * np.power(l,6) \
    - 0.435933 * np.power(l,7)
    
    return H



def ContinousIndexingTrajectory(t):
    min_frame = t.index.values.min() # frame that particle's trajectory begins
    max_frame = t.index.values.max() # frame that particle's trajectory ends
    
    new_index = np.linspace(min_frame,max_frame, (max_frame - min_frame + 1))
    
    t = t.reindex(index=new_index)
    
    return t



def CalculateLagtimes_min(eval_tm, lagtimes_min_max = 10, min_snr = 10):
    """ calculate the minimum lagtime for the MSD fit
    
    If the framerate is to high or the localization precision to bad, small lagtimes contain only noise
    The function searches the first lagtime which has a msd which is at least 10 times beyond the noise floor
    """
        
    eval_tm = ContinousIndexingTrajectory(eval_tm)
    msd_offset = np.square(eval_tm.ep).mean()
    
    valid_lagtimes_min = False
    lagtimes_min = 1
    
    # this function needs and abort-criteria too
    # for 
    
    while valid_lagtimes_min == False:
        msd = np.square(eval_tm.x.diff(lagtimes_min)).mean()
        
        # check if SNR of MSD is better the minimum SNR
        current_snr = msd/msd_offset
        if current_snr > min_snr:
            valid_lagtimes_min = True
        else:
            print("msd offset is: ", msd_offset)
            print("msd (at first lagtime = {:d}) is: {:f}".format(lagtimes_min,msd))

            lagtimes_min = lagtimes_min + 1
            if lagtimes_min > lagtimes_min_max:
                lagtimes_min = -1
                
                valid_lagtimes_min = True
    
    return lagtimes_min



def InvDiameter(sizes_df_lin, settings):
    
    inv_diam = 1/sizes_df_lin["diameter"]
    
    t_max = settings["MSD"]["lagtimes_max"]
#    print("t_max = ", t_max)
    
    f = sizes_df_lin["traj length"]
#    print("f = ", f)
    
#    rel_error = sizes_df_lin["diffusion std"] / sizes_df_lin["diffusion"]
    rel_error = np.sqrt((2*t_max)/(3*(f-t_max)))
#    print("rel_error = ", rel_error)
    inv_diam_std = inv_diam * rel_error
    
    return inv_diam,inv_diam_std



def StatisticOneParticle(sizes_df_lin):
    
    # mean diameter is 1/(mean diffusion) not mean of all diameter values
    diam_inv_mean = (1/sizes_df_lin.diameter).mean()
    diam_inv_std  = (1/sizes_df_lin.diameter).std()
    
    return diam_inv_mean, diam_inv_std
    
    

def FitMeanDiameter(sizes_df_lin, settings):    
    inv_diam,inv_diam_std = nd.CalcDiameter.InvDiameter(sizes_df_lin, settings)
    
    diff = inv_diam
    diff_std = inv_diam_std
    diff_weight = 1/(diff_std**2)
    
    mean_diff = np.average(diff, weights = diff_weight)
    
#    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean    
    mean_diff_std = np.sqrt(1/np.sum(diff_weight))

    snr = mean_diff / mean_diff_std
    
    diam = 1/mean_diff
    
    diam_std = diam / snr
    
    return diam, diam_std
    
    
    
    
    
#def FitMSD(lagt_direct, amount_frames_lagt1, mean_displ_direct, mean_displ_sigma_direct, PlotMsdOverLagtime = False):
#
#    if (len(lagt_direct) >= 5):
#        [fit_values, fit_cov]= np.polyfit(lagt_direct, mean_displ_direct, 1, w = 1/mean_displ_sigma_direct, cov=True) 
#    else:
#        fit_values= np.polyfit(lagt_direct, mean_displ_direct, 1, w = 1/mean_displ_sigma_direct) 
##        fit_cov = []
#    
#     
#    if PlotMsdOverLagtime == True:
#
#        mean_displ_fit_direct_lin=lagt_direct.map(lambda x: x*fit_values[0])+ fit_values[1]
#        # order of regression_coefficients depends on method for regression (0 and 1 are switched in the two methods)
#        # fit results into non-log-space again
#        nd.visualize.MsdOverLagtime(lagt_direct, mean_displ_direct, mean_displ_fit_direct_lin)
#    
#    return diff_direct_lin, std_diff_direct_lin
    
    
    