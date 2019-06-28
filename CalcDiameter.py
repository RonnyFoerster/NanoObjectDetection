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


def Main(t6_final, ParameterJsonFile, obj_all, microns_per_pixel = None, frames_per_second = None, amount_summands = None, amount_lagtimes = None,
         amount_lagtimes_auto = None, Histogramm_Show = True, MSD_fit_Show = False, EvalOnlyLongestTraj = 0, Max_traj_length = None):
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
    #
    # binning = 25 # Amount of bins used in histogram
    '''

    #%% read the parameters
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
  
    microns_per_pixel = settings["MSD"]["effective_Microns_per_pixel"]
    frames_per_second = settings["MSD"]["effective_fps"]
    
    temp_water = settings["Exp"]["Temperature"]
    solvent = settings["Exp"]["solvent"]
    
    if settings["Exp"]["Viscocity_auto"] == 1:
        settings["Exp"]["Viscocity"] = nd.handle_data.GetViscocity(temperature = temp_water, solvent = solvent)
    
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
        print("Particle number: ",  round(particleid))
#        if particleid == 7:
#            bp()
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
            lagtimes_min = CalculateLagtimes_min(eval_tm)
            p_min_old = 2
            lagtimes_max = lagtimes_min + p_min_old - 1
            
        else:
            max_counter = 1
            lagtimes_min = settings["MSD"]["lagtimes_min"]
            lagtimes_max = settings["MSD"]["lagtimes_max"]
        
        stable_num_lagtimes = False
        counter = 0
         
            # iterate msd fit till it converges
        while ((stable_num_lagtimes == False) and (counter < max_counter)):
            counter = counter + 1 
            # Calc MSD

            nan_tm_sq, amount_frames_lagt1, enough_values, traj_length = \
            CalcMSD(eval_tm, microns_per_pixel, amount_summands, lagtimes_min = lagtimes_min, lagtimes_max = lagtimes_max)

            if enough_values == True:  
                
                if any_successful_check == False:
                    any_successful_check = True
                    if MSD_fit_Show == True:
                        plt.plot()

                #iterative to find optimal number of lagtimes in the fit    
                # Average MSD (several (independent) values for each lagtime)

                lagt_direct, mean_displ_direct, mean_displ_sigma_direct = \
                AvgMsdRolling(nan_tm_sq, frames_per_second, DoRolling = False)
                
                diff_direct_lin = \
                FitMSDRolling(lagt_direct, amount_frames_lagt1, mean_displ_direct, mean_displ_sigma_direct, \
                           PlotMsdOverLagtime = MSD_fit_Show, DoRolling = False)
                    
                do_rolling = settings["Time"]["DoRolling"]
                my_rolling = settings["Time"]["Frames"]
                if do_rolling == True:
                    lagt_direct, mean_displ_direct, mean_displ_sigma_direct = \
                    AvgMsdRolling(nan_tm_sq, frames_per_second, my_rolling = my_rolling, DoRolling = True)
                    
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
                        sizes_df_lin = sizes_df_lin[:-1]                   


        if enough_values == True: 
            red_ep = ReducedLocalPrecision(settings, mean_raw_mass, diff_direct_lin)
            
            # get the fit error if switches on (and working)
            
            rel_error_diff, diff_std = DiffusionError(traj_length, red_ep, diff_direct_lin, min_rel_error)
            
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
        
    sizes_df_lin = sizes_df_lin.set_index('particle')
    
    if do_rolling == True:
        sizes_df_lin_rolling = sizes_df_lin_rolling.set_index('frame')
    else:
        sizes_df_lin_rolling = "Undone"

    if MSD_fit_Save == True:
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "MSD Fit", settings)
    
    nd.handle_data.WriteJson(ParameterJsonFile, settings) 

    
    return sizes_df_lin, sizes_df_lin_rolling, any_successful_check

 



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


        if amount_frames_lagt1 < (lagtimes_max + 1 + amount_summands * lagtimes_max):
            enough_values = False

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
        mean_displ_direct = mean_displ_direct.append(pd.Series(index=[column], data=[mean_displ_direct_loop]))
        
        # 3. Check how many independent measurements are present (this number is used for filtering later. 
        # Also the iteration is limited to anaylzing
        # those lag-times only that can possibly yield enough entries according to the chosen filter). 
        len_nan_tm_sq_loop = nan_indi_means.count()
        
        # 4. Calculate the mean of these sub-means --> that's the msd for each lag-time
        # 5. Calculate the variance of these sub-means --> that's used for variance-based fitting 
        # when determining the slope of msd over time
        
        mean_displ_variance_direct_loop = nan_indi_means.var(axis=0)*(2/(len_nan_tm_sq_loop-1))
        mean_displ_variance_direct = mean_displ_variance_direct.append(pd.Series(index=[column], 
                                                                               data=[mean_displ_variance_direct_loop]))
        
        mean_displ_sigma_direct = np.sqrt(mean_displ_variance_direct)
        
    lagt_direct = mean_displ_direct.index/frames_per_second # converting frames into physical time: 
    # first entry always starts with 1/frames_per_second
    
    return lagt_direct, mean_displ_direct, mean_displ_sigma_direct



def AvgMsdRolling(nan_tm_sq, frames_per_second, my_rolling = 100, DoRolling = False):
    
    num_cols = len(nan_tm_sq.columns) - 1
    
    lagt_direct = np.linspace(1,num_cols,num_cols) /frames_per_second # converting frames into physical time: 
    # first entry always starts with 1/frames_per_second
    
    if DoRolling == False:
#        mean_displ_direct = pd.DataFrame(index = [0], columns = nan_tm_sq.columns.tolist()[1:])
        mean_displ_direct = np.zeros(num_cols)
        mean_displ_variance_direct = mean_displ_direct.copy()    
                
        for column in range(num_cols):           
            eval_column = column + 1
            # That iterates over lag-times for the respective particle and
            # calculates the msd in the following manner:
            # 1. Build sub-blocks for statistically independent lag-time measurements
            nan_indi_means = rolling_with_step(nan_tm_sq[eval_column], eval_column, eval_column, mean_func)
            
            # 2. Calculate mean msd for the sublocks
            mean_displ_direct[column] = nan_indi_means.mean(axis=0)
            
            # 3. Check how many independent measurements are present (this number is used for filtering later. 
            # Also the iteration is limited to anaylzing
            # those lag-times only that can possibly yield enough entries according to the chosen filter). 
            
            len_nan_tm_sq_loop = nan_indi_means.count()
            
            # 4. Calculate the mean of these sub-means --> that's the msd for each lag-time
            # 5. Calculate the variance of these sub-means --> that's used for variance-based fitting 
            # when determining the slope of msd over time
            
            mean_displ_variance_direct_loop = nan_indi_means.var(axis=0)*(2/(len_nan_tm_sq_loop-1))
            
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



def FitMSD(lagt_direct, amount_frames_lagt1, mean_displ_direct, mean_displ_sigma_direct, PlotMsdOverLagtime = False):
    
    const_Boltz = 1.38e-23 # Boltzmann-constant

    if (len(lagt_direct) >= 5):
        [fit_values, fit_cov]= np.polyfit(lagt_direct, mean_displ_direct, 1, w = 1/mean_displ_sigma_direct, cov=True) 
    else:
        fit_values= np.polyfit(lagt_direct, mean_displ_direct, 1, w = 1/mean_displ_sigma_direct) 
#        fit_cov = []
    
    
    diff_direct_lin = fit_values[0]/2
        
    if PlotMsdOverLagtime == True:

        mean_displ_fit_direct_lin=lagt_direct.map(lambda x: x*fit_values[0])+ fit_values[1]
        # order of regression_coefficients depends on method for regression (0 and 1 are switched in the two methods)
        # fit results into non-log-space again
        nd.visualize.MsdOverLagtime(lagt_direct, mean_displ_direct, mean_displ_fit_direct_lin)
    
    return diff_direct_lin




def FitMSDRolling(lagt_direct, amount_frames_lagt1, mean_displ_direct, mean_displ_sigma_direct, PlotMsdOverLagtime = False,  my_rolling = 100, DoRolling = False):

    if DoRolling == True:
        mean_displ_sigma_direct = mean_displ_sigma_direct.median()
    
#    if DoRolling == False:
#        bp()
#        weigths = np.squeeze(np.asarray(mean_displ_sigma_direct))
#    else:
#        bp()

    if (len(lagt_direct) >= 5):
        [fit_values, fit_cov]= np.polyfit(lagt_direct, mean_displ_direct.T, 1, w = 1/mean_displ_sigma_direct, cov=True) 
    else:
        try:
            fit_values= np.polyfit(lagt_direct, mean_displ_direct.T, 1, w = 1/mean_displ_sigma_direct) 
        except:
            bp()


    diff_direct_lin = fit_values[0]/2
    diff_direct_lin = np.squeeze(diff_direct_lin)

        
    if PlotMsdOverLagtime == True:
        
        if DoRolling == True:
            mean_displ_direct = mean_displ_direct.median()
            fit_values = np.median(fit_values,axis=1)    
            
#        mean_displ_fit_direct_lin=lagt_direct.map(lambda x: x*fit_values[0]+ fit_values[1])
        mean_displ_fit_direct_lin = lagt_direct *fit_values[0]+ fit_values[1]
    
            
        nd.visualize.MsdOverLagtime(lagt_direct, mean_displ_direct, mean_displ_fit_direct_lin)
    
    
    
    return diff_direct_lin



def DiffusionToDiameter(diffusion, UseHindranceFac = 0, fibre_diameter_nm = None, temp_water = 295, visc_water = 9.5e-16, DoRolling = False):
    const_Boltz = 1.38e-23 # Boltzmann-constant
    
    diameter = (2*const_Boltz*temp_water/(6*math.pi *visc_water)*1e9) /diffusion # diameter of each particle

    if UseHindranceFac == True:
        if np.max(diameter) < fibre_diameter_nm:
            if DoRolling == False:
                diamter_corr = EstimateHindranceFactor(diameter, fibre_diameter_nm, DoPrint = True)
            else:
                diamter_corr = diameter.copy()
                for counter, diameter_rolling in enumerate(diameter):
                    diamter_corr[counter] = EstimateHindranceFactor(diameter_rolling, fibre_diameter_nm, DoPrint = False)
                
        else:
            sys.exit("Here is something wrong. Since the diameter is calculated to be large than the core diameter. \
                     Possible Reasons: \
                     \n 1 - drift correctes motion instead of drift because to few particles inside. \
                     \n 2 - stationary particle remains, which seems very big because it diffuses to little. ")
            
    else:
#        print("WARNING: hindrance factor ignored. You just need to have the fiber diameter!")   
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
        if DoPrint == True:
            print("Iter: %d: Starting Diameter: %.1f; corr. viscocity: %.3f; corr. diameter: %.2f" % (my_iter, round(diam_direct_lin,1), round(corr_visc,3), round(diam_direct_lin_corr,2)))

        if my_iter > 100:
            print("Iteration does not converge. Abort !!!")
            bp()
            input("PRESS ENTER TO CONTINUE.")
            diam_direct_lin_corr = diam_direct_lin_corr_old

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
                                                          'true_particle': [true_particle]}))
    
    
    
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
    # XAVIER MICHALET AND ANDREW J. BERGLUND PHYSICAL REVIEW E 85, 2012
    red_x = ReducedLocalPrecision(settings, raw_mass, diffusivity)
    
    
    if red_x >= 0:
        f_b = 2 + 1.35 * np.power(red_x,0.6)
    else:
        f_b = 2
    L_b = 0.8 + 0.564 * amount_frames_lagt1
    
    
    value_1 = L_b
    value_2 = (f_b * L_b) / np.power(np.power(f_b,3) + np.power(L_b,3),1/3)
       
    p_min = np.int(np.min(np.round([value_1, value_2])))

    
    return p_min
    
    
    
def ReducedLocalPrecision(settings, mass, diffusion, DoRolling = False):
    """
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4917385/#FD7
    Eq 13
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
        red_x = -1
    else:
            
        num_photons = mass / gain
        static_local_precision_um = rayleigh_um / np.power(num_photons ,1/2)
    
        red_x = np.power(static_local_precision_um,2) / (diffusion * lagtime_s) \
        * (1 + (diffusion * exposure_time_s / np.power(rayleigh_um,2))) \
        - 1/3 * (exposure_time_s / lagtime_s)
    
    #    red_x = np.power(local_precision_um,2) / (diffusion * lagtime_s) \
    #    * (1 + (diffusion * exposure_time_s / np.power(rayleigh_um,2))) \
    #    - 1/3 * (exposure_time_s / lagtime_s)
    
        
        if red_x < 0:
            red_x = 0
    
    return red_x
    

def DiffusionError(traj_length, red_x, diffusion, min_rel_error, DoRolling = False):
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
            
            sizes_df_lin = sizes_df_lin.append(sizes_df_lin_new)
            
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


def hindrance_fac(diam_fibre,diam_particle):
    l = diam_particle / diam_fibre
    # Eq 16 from "Hindrance Factors for Diffusion and Convection in Pores"  Dechadilok and Deen
    h = 1 + 9/8*l*np.log(l) - 1.56034 * l \
    + 0.528155 * np.power(l,2) \
    + 1.915210 * np.power(l,3) \
    - 2.819030 * np.power(l,4) \
    + 0.270788 * np.power(l,5) \
    + 1.101150 * np.power(l,6) \
    - 0.435933 * np.power(l,7)
    return h


def CalculateLagtimes_min(eval_tm, min_snr = 10):
    """
    Calculates the minimum lagtime for the MSD fit.
    If the framerate is to high or the localization precision to bad, small lagtimes contain only noise
    The function searches the first lagtime which has a msd which is at least 10 times beyond the noise floor
    """
        
    msd_offset = np.power(eval_tm.ep,2).mean()
    
    valid_lagtimes_min = False
    lagtimes_min = 1
    while valid_lagtimes_min == False:
        msd = np.power(eval_tm.x.diff(1),2).mean()
    
        # check if SNR of MSD is better the minimum SNR
        current_snr = msd/msd_offset
        if current_snr > min_snr:
            valid_lagtimes_min = True
        else:
            lagtimes_min = lagtimes_min + 1
    
    return lagtimes_min


def InvDiameter(sizes_df_lin):
    
    inv_diam = 1/sizes_df_lin["diameter"]
    
    rel_error = sizes_df_lin["diffusion std"] / sizes_df_lin["diffusion"]
    
    inv_diam_std = inv_diam * rel_error
    
    return inv_diam,inv_diam_std



def StatisticOneParticle(sizes_df_lin):
    
    # mean diameter is 1/(mean diffusion) not mean of all diameter values
    diam_inv_mean = (1/sizes_df_lin.diameter).mean()
    diam_inv_std  = (1/sizes_df_lin.diameter).std()
    
    return diam_inv_mean, diam_inv_std
    
    
    
    
    
    #    
#def MSD_Var_f(diff, lagt_direct, ep, traj_length):
#    """
#    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3055791/
#    Appendix D 15
#    """
#    N = traj_length
#    K = N - n
#    x = (ep**2) / (diff * lagt_direct)
#    if n <= K:
#        f = n / (6*(K**2)) * (4*(n**2)*K  + 2*K - n**3 + n) \
#            + 1/K * (2*n*x + (1 + 1/2*(1-n/K)) * (x**2))
#    else:
#        f = 1 / (6*K) * (6*(n**2)*K - 4*n*(K**2) + 4*n + K**3 - K) \
#            + 1/K * (2*n*x + (x**2))
#            
#    
#    return
    
    
    
    
    
    
    
    
    