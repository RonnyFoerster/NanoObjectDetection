# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 09:52:27 2019

@author: foersterronny
"""

import numpy as np # Library for array-manipulation
import pandas as pd # Library for DataFrame Handling
import math # offering some maths functions
import warnings
import nanoobject_detection as nd



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
    
   
def Main(t5_no_drift, settings, obj_all, microns_per_pixel = None, frames_per_second = None, amount_summands = None, amount_lagtimes = None,
         amount_lagtimes_auto = None, cutoff_size = None, binning = None, PlotDiameterHistogramm = True, PlotMsdOverLagtime = False):
    '''
    Calculating msd of individual particles:
    
    1.) filtering out too short tracks
    2.) constructing matrix with defined lag-times for each particle. Amount of lag-times is set.
    3.) calculating mean and variance of lag-times for each particle
    4.) regressing each particle individually linearly: msd per lag-time 
    5.) slope of regression used to calculate size of particle
    6.) standard error of lag-time=1 used as error on slope
    
    Parameters to be adjusted:
    # amount_summands ... If msd for a given lag-time is calculated by less than this amount of summands, the corresponding lag-time is disregarded in the fit
    # amount_lagtimes ... If the particle has less than this amount of valid lag-times that were fitted, 
    the particle wouldn't be taken into account. From 181120, SW: Not more than this datapoints are taken into account either
    # cutoff_size [nm] ... particles with size exceeding this will not be plotted
    # binning = 25 # Amount of bins used in histogram
    '''
    
    microns_per_pixel, settings    = nd.handle_data.SpecificValueOrSettings(microns_per_pixel, settings, "Exp", "Microns_per_pixel")
    frames_per_second, settings    = nd.handle_data.SpecificValueOrSettings(frames_per_second, settings, "Exp", "fps")
    binning, settings              = nd.handle_data.SpecificValueOrSettings(binning, settings, "Gui", "Bins")
    cutoff_size, settings          = nd.handle_data.SpecificValueOrSettings(cutoff_size, settings, "Gui", "Cutoff Size [nm]")
    amount_summands, settings      = nd.handle_data.SpecificValueOrSettings(amount_summands, settings, "Processing", "Amount summands")
    amount_lagtimes, settings      = nd.handle_data.SpecificValueOrSettings(amount_lagtimes, settings, "Processing", "Amount lagtimes")
    amount_lagtimes_auto, settings = nd.handle_data.SpecificValueOrSettings(amount_lagtimes_auto, settings, "Processing", "Amount lagtimes auto")
    
    
    particle_list_value=list(t5_no_drift.particle.drop_duplicates())
    
    sizes_df_lin=pd.DataFrame(columns={'diameter','particle'})       
    
    any_successful_check = False
    
    
    num_loop_elements = len(particle_list_value)
    for i,particleid in enumerate(particle_list_value): # iteratig over all particles
        nd.visualize.update_progress("Analyze Particles", (i+1)/num_loop_elements)

        # select track to analyze
        eval_tm = t5_no_drift[t5_no_drift.particle==particleid]
        
        
        stable_num_lagtimes = False
        counter = 0
        
        if amount_lagtimes_auto == 1:
            max_counter = 10
            current_amount_lagtimes = 2
        else:
            max_counter = 1
            current_amount_lagtimes = amount_lagtimes
         
            
        while ((stable_num_lagtimes == False) and (counter < max_counter)):
            counter = counter + 1 
            # Calc MSD
            nan_tm_sq, amount_frames_lagt1, enough_values = CalcMSD(eval_tm, microns_per_pixel, amount_summands, 
                                                                current_amount_lagtimes, cutoff_size, binning)

            if enough_values == True:  
                any_successful_check = True

                #iterative to find optimal number of lagtimes in the fit    
                # Average MSD (several (independent) values for each lagtime)
                lagt_direct, mean_displ_direct, mean_displ_sigma_direct = AvgMsd(nan_tm_sq, frames_per_second)
                
                # Fit MSD (slope is proportional to diffusion coefficent)
                sizes_df_lin, diffusivity = FitMSD(settings, obj_all, lagt_direct, amount_frames_lagt1, mean_displ_direct, mean_displ_sigma_direct,
                                      sizes_df_lin, particleid, binning, cutoff_size, PlotMsdOverLagtime = PlotMsdOverLagtime)
                
                # calculate theoretical best number of considered lagtimes
                p_min = OptimalMSDPoints(settings, obj_all, diffusivity, amount_frames_lagt1)

                
                if p_min == current_amount_lagtimes:
                    stable_num_lagtimes = True 
                else:
                    current_amount_lagtimes = p_min
                    print("p_min_before = ",p_min)
                    print("p_min_after  = ", current_amount_lagtimes)
                    
                    # drop last line, because it is done again
                    sizes_df_lin = sizes_df_lin[:-1]                   
                    
                
            else:
                any_successful_check = True
        
        
    if any_successful_check == True:   
        if PlotDiameterHistogramm == True:
#            print(sizes_df_lin)
            nd.visualize.DiameterHistogramm(sizes_df_lin, binning, cutoff_size)
    else:
        print("NO PARTICLE WAS MEASURED LONG ENOUGH FOR A GOOD STATISTIC !")
    
    
    return sizes_df_lin, settings
    
def CalcMSD(eval_tm, microns_per_pixel, amount_summands = 5, amount_lagtimes = 5, cutoff_size = 100, binning = 25):

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
    if length_indexer < (amount_lagtimes + amount_summands * amount_lagtimes):
        enough_values = False
    
    else:
        nan_tm = pd.DataFrame(index=min_frame+range(max_frame+1-min_frame),columns=range(amount_lagtimes + 1)) 
        # setting up a matrix (dataframe) that has the correct size to put in all needed combinations of displacement and lag-times:
        # rows: frames: all frames from first appearance to diappearance are created. even if particle isn't recorded for some time,
        # in this case the respective positions will be filled with nan (that's the reason for the name of the variable)
        # columns: 0: initial position in respective frame
        # columns: all others: displacement from original position
        nan_tm[0]=eval_tm.x*microns_per_pixel # filling column 0 with position of respective frame
    
        for row in range(1,len(nan_tm.columns)):
            nan_tm[row]=nan_tm[0].diff(row) # putting the displacement of position into the columns, each representing a lag-time
        
        # Checking already here if enough non NAN values exist to calculate
        # if not, particle is disregarded. another check of observation-number for each lagt is then obsolet
        amount_frames_lagt1 = nan_tm[1].count() # SW, 181125: Might be better to use the highest lag-t for filtering instead of lag-t=1?!
        if amount_frames_lagt1 < (amount_lagtimes + 1 + amount_summands * amount_lagtimes):
            enough_values = False

        else:
            enough_values = True    
            nan_tm_sq=nan_tm**2 # Squaring displacement    
     
    return nan_tm_sq, amount_frames_lagt1, enough_values
    


def AvgMsd(nan_tm_sq, frames_per_second):
    mean_displ_direct=pd.Series() # To hold the mean sq displacement of a particle
    mean_displ_variance_direct=pd.Series() # To hold the variance of the msd of a particle

    for column in range(1,len(nan_tm_sq.columns)):
        # That iterates over lag-times for the respective particle and
        # calculates the msd in the following manner:
        # 1. Build sub-blocks for statistically independent lag-time measurements
        # 2. Calculate mean msd for the sublocks
        # 3. Check how many independent measurements are present (this number is used for filtering later. 
        # Also the iteration is limited to anaylzing
        # those lag-times only that can possibly yield enough entries according to the chosen filter). 
        # 4. Calculate the mean of these sub-means --> that's the msd for each lag-time
        # 5. Calculate the variance of these sub-means --> that's used for variance-based fitting 
        # when determining the slope of msd over time
        nan_indi_means=rolling_with_step(nan_tm_sq[column], column, column, mean_func)
        len_nan_tm_sq_loop=nan_indi_means.count()
        mean_displ_direct_loop=nan_indi_means.mean(axis=0)
        mean_displ_direct=mean_displ_direct.append(pd.Series(index=[column], data=[mean_displ_direct_loop]))
        mean_displ_variance_direct_loop=nan_indi_means.var(axis=0)*(2/(len_nan_tm_sq_loop-1))
        mean_displ_variance_direct=mean_displ_variance_direct.append(pd.Series(index=[column], 
                                                                               data=[mean_displ_variance_direct_loop]))
        
        mean_displ_sigma_direct=np.sqrt(mean_displ_variance_direct)
        
        lagt_direct=mean_displ_direct.index/frames_per_second # converting frames into physical time: 
        # first entry always starts with 1/frames_per_second
    
    return lagt_direct, mean_displ_direct, mean_displ_sigma_direct



def FitMSD(settings, obj_all, lagt_direct, amount_frames_lagt1, mean_displ_direct, mean_displ_sigma_direct, sizes_df_lin, particleid,
           binning, cutoff_size, temp_water = 295, visc_water = 9.5e-16, PlotMsdOverLagtime = False):
    
    const_Boltz = 1.38e-23 # Boltzmann-constant
    
    if (len(lagt_direct) >= 4):
        [fit_values, fit_cov]= np.polyfit(lagt_direct, mean_displ_direct, 1, w = 1/mean_displ_sigma_direct, cov=True) 
    else:
        fit_values= np.polyfit(lagt_direct, mean_displ_direct, 1, w = 1/mean_displ_sigma_direct) 
#        fit_cov = []

    
    diff_direct_lin = fit_values[0]/2 # diffusivity of each particle
#    diff_direct_var = fit_cov[0,0] / 4 #always scare the scalars in var propagation
    
    diam_direct_lin = (2*const_Boltz*temp_water/(6*math.pi *visc_water)*1e9) /diff_direct_lin # diameter of each particle
    
    #https://en.wikipedia.org/wiki/Propagation_of_uncertainty
#    diam_direct_var = (diam_direct_lin/diff_direct_lin)**2 * diff_direct_var # diameter of each particle
#    diam_direct_std = np.sqrt(diam_direct_var)
    diam_direct_std = "still to do"
    
        
    # Calculating error of diffusivity:
    # "Single-Particle Tracking: The Distribution of Diffusion Coefficients"
    # Biophysical Journal Volume 72 1997 1744-1753
    rel_error_slope = 0.8165 * diff_direct_lin / np.sqrt(amount_frames_lagt1)
    rel_error_slope = "still to do"
    
    # Storing results in df:
    sizes_df_lin = sizes_df_lin.append(pd.DataFrame(data={'diameter': [diam_direct_lin],
                                                        'particle': [particleid],
                                                        'slope': [diff_direct_lin],
                                                        'rel_error_slope': [rel_error_slope],
                                                        'diameter_std': [diam_direct_std],
                                                        'frames':[amount_frames_lagt1]}),sort=False)
    
    
    if PlotMsdOverLagtime == True:
        mean_displ_fit_direct_lin=lagt_direct.map(lambda x: x*fit_values[0])+ fit_values[1]
        # order of regression_coefficients depends on method for regression (0 and 1 are switched in the two methods)
        # fit results into non-log-space again
        nd.visualize.MsdOverLagtime(lagt_direct, mean_displ_direct, mean_displ_fit_direct_lin)
    
    return sizes_df_lin, diff_direct_lin
    

def OptimalMSDPoints(settings, obj_all, diffusivity, amount_frames_lagt1):
    # XAVIER MICHALET AND ANDREW J. BERGLUND PHYSICAL REVIEW E 85, 2012
    red_x = ReducedLocalPrecision(settings, obj_all, diffusivity)
    
    
    if red_x >= 0:
        f_b = 2 + 1.35 * np.power(red_x,0.6)
    else:
        f_b = 2
    L_b = 0.8 + 0.564 * amount_frames_lagt1
    
    
    value_1 = L_b
    value_2 = (f_b * L_b) / np.power(np.power(f_b,3) + np.power(L_b,3),1/3)
       
    p_min = np.int(np.min(np.round([value_1, value_2])))

    
    return p_min
    
    
    
def ReducedLocalPrecision(settings, obj_all, diffusivity):
    local_precision_px = np.mean(obj_all["ep"])
    local_precision_um = local_precision_px * settings["Exp"]["Microns_per_pixel"]
    lagtime_s = 1/settings["Exp"]["fps"]
    exposure_time_s = settings["Exp"]["ExposureTime_s"]
    
    NA = settings["Exp"]["NA"]
    lambda_nm = settings["Exp"]["lambda_nm"]
    
    rayleigh_nm = 2 * 0.61 * lambda_nm / NA
    rayleigh_um = rayleigh_nm / 1000
    
    # we use rayleigh as resolution of the system
    # 2* because it is coherent
    # not sure here

    
    
    red_x = np.power(local_precision_um,2) / (diffusivity * lagtime_s) \
    * (1 + (diffusivity * exposure_time_s / np.power(rayleigh_um,2))) \
    - 1/3 * (exposure_time_s / lagtime_s)

    
    return red_x
    
    