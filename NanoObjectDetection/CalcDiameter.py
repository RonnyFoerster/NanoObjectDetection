# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 09:52:27 2019

@author: Ronny Förster, Stefan Weidlich, Mona Nissen

This module calculates the diameter of a diffusing particle out of its trajectory.

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
import multiprocessing
from joblib import Parallel, delayed


def MSD_par(eval_tm, settings, yEval, any_successful_check, t_beforeDrift):
    """
    Function that calculates MSD, fits it, optimizes the parameters and estimates the diameter out of the trajectory. This function is executed in parallel later

    Parameters
    ----------
    eval_tm : pandas
        trajectory of a single particle.
    settings : TYPE
        DESCRIPTION.
    yEval : boolean
        defines which direction is evaluated.
    any_successful_check : boolean
        help variable that defines if any particle was evaluated successfully.
    t_beforeDrift : pandas
        trajectory BEFORE the drift correction

    Returns
    -------
    sizes_df_lin : pandas
        summary of a estiamted parameters.
    any_successful_check : TYPE
        see above.

    """
    
    # run the trajectory to results routine
    sizes_df_particle, OptimizingStatus = OptimizeMSD(eval_tm, settings, yEval, any_successful_check, t_beforeDrift = t_beforeDrift)

    # OptimizingStatus return
    if OptimizingStatus == "Successful":
        any_successful_check = True
        # after the optimization is done -save the result in a large pandas.DataFrame
        sizes_df_lin = sizes_df_particle
    else:
        sizes_df_lin = "invalid"
        
    return sizes_df_lin, any_successful_check 


def Main2(t6_final, ParameterJsonFile, MSD_fit_Show = False, yEval = False, processOutput = True, t_beforeDrift = None):
    """
    Main function to retrieve the diameter ouf of an trajectory
    """
    
    # read the parameters
    settings = nd.handle_data.ReadJson(ParameterJsonFile)

    # preprocess trajectories if defined in json
    # (e.g. only use longest or split too long trajectories)
    t6_final_use, settings = SelectTrajectories(t6_final, settings)

    # list of particles id
    particle_list_value = list(t6_final_use.particle.drop_duplicates())

    #setup return variable
    sizes_df_lin=pd.DataFrame(columns={'diameter','particle'})       
    
    nd.logger.warning("Variable sizes_df_lin_rolling is not longer calculated. That might be inserted later again. If you have this return variable in your CalcDiameter.Main2 function that will cause an error.")
        
    #boolean for plotting stuff later (without this python gets stucked)
    any_successful_check = False
   
    # gets the number of avaialbe CPU on the executing machine to make it parallel
    num_cores = multiprocessing.cpu_count()

    nd.logger.info("MSD analysis in parallel (Number of cores: %s)", num_cores)
    
    # the parallel programming cannot execute the previous plotting routines. However, trackpy can be used instead (look online)
    if settings["Plot"]["MSD_fit"] == 1:
        nd.logger.error("MSD Plot sacrificed for parallel processing. Try TrackPy instead if needed. Set MSD_fit to 0.")
    
    # return the number of prints you see during execution, depending on your logger mode
    num_verbose = nd.handle_data.GetNumberVerbose()
                                                                  
    # for jj in particle_list_value:
    #     print(jj)
    #     MSD_par(t6_final_use[t6_final_use.particle == jj].copy(), settings, yEval, any_successful_check, t_beforeDrift)
        

    # Estimate diameter out of trajectory in parallel (see joblib for detailed information)
    output_list = Parallel(n_jobs=num_cores, verbose=num_verbose)(delayed(MSD_par)(t6_final_use[t6_final_use.particle == jj].copy(), settings, yEval, any_successful_check, t_beforeDrift) for jj in particle_list_value)

    
    # separate valid and invalid entries from output_list and story all valid into size_df_lin_valid
    size_df_lin_valid = []
    
    for ii,jj in enumerate(output_list):
        # copy only the valid
        if type(jj[0]) == str:
            nd.logger.debug("remove to short trajectory of particle id: %.0f", particle_list_value[ii])
        else:
            size_df_lin_valid.append(jj[0])
            
            if jj[1] == True:
                any_successful_check = True
      
    # convert data structure here
    sizes_df_lin = pd.concat(size_df_lin_valid)

    # plot the predefined figures
    sizes_df_lin = Main2Plots(sizes_df_lin, settings, any_successful_check, ParameterJsonFile, yEval)
          
    return sizes_df_lin, any_successful_check                            
    

def Main2Plots(sizes_df_lin, settings, any_successful_check, ParameterJsonFile, yEval):
    """
    Plots the defined figures in settings

    Parameters
    ----------
    sizes_df_lin : pandas
        summary of results.
    settings : TYPE
        DESCRIPTION.
    any_successful_check : TYPE
        DESCRIPTION.
    ParameterJsonFile : TYPE
        DESCRIPTION.
    yEval : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    min_brightness = settings["PostProcessing"]["MinimalBrightness"]
    if min_brightness > 0:
        if settings["Plot"]["UseRawMass"] == "mean":
            sizes_df_lin = sizes_df_lin[sizes_df_lin["rawmass_mean"] > min_brightness]
    
    
    MaxDiameter = settings["PostProcessing"]["MaxDiameter"]
    if MaxDiameter > 0:
        sizes_df_lin = sizes_df_lin[sizes_df_lin.diameter < MaxDiameter]
    
    
    if settings["PostProcessing"]["ForceUltraUniform"] == 1:
        sizes_df_lin = nd.PostProcessing.ForceUltraUniformParticle(settings, sizes_df_lin, ShowPlot = True)
    
    if any_successful_check == False:
        nd.logger.warning("No particle made it to the end!")
      
    if settings["Plot"]["MSD_fit"] == 1:
        nd.logger.error("RF 211004: implement this!")
      
    else:
        ExportResultsMain(ParameterJsonFile, settings, sizes_df_lin, yEval = yEval)       
    
    nd.logger.info("WARNING sizes_df_lin_rolling is removed!!!")
    
    return sizes_df_lin


def SelectTrajectories(t6_final, settings):
    """
    select a subset of particle trajectories, e.g. only the longest

    Parameters
    ----------
    t6_final : pandas
        all particles trajectory.
    settings : TYPE
        DESCRIPTION.

    Returns
    -------
    t6_final_use : TYPE
        selected trajectories.
    settings : TYPE
        DESCRIPTION.

    """

    # choose only the longest trajectory
    EvalOnlyLongestTraj = settings["MSD"]["EvalOnlyLongestTraj"]
    
    # minimum trajectory length
    Min_traj_length = settings["Link"]["Min_tracking_frames"]
    
    # maximum trajectory length
    Max_traj_length = settings["Split"]["Max_traj_length"]
    
    # eval longest trajectory only
    if EvalOnlyLongestTraj == 1:
        # return particle-id and length of longest trajectory
        longest_particle, longest_traj = nd.handle_data.GetTrajLengthAndParticleNumber(t6_final)
    
        # saves it to settings
        settings["Split"]["ParticleWithLongestTrajBeforeSplit"] = longest_particle
    
        # select the longest trajectory only
        t6_final_use = t6_final[t6_final["particle"] == longest_particle].copy()
    else:
        t6_final_use = t6_final.copy()
        
        
    # split too long trajectories
    if Max_traj_length is None:
        t6_final_use = nd.get_trajectorie.split_traj_at_long_trajectory(t6_final_use, settings)
    else:
        t6_final_use = nd.get_trajectorie.split_traj_at_long_trajectory(t6_final_use, settings, Min_traj_length, Max_traj_length)
        

    return t6_final_use, settings



def MSDFitLagtimes(settings, eval_tm, amount_lagtimes_auto = None):
    """
    Define how many lagtimes are considered for the MSD fit 
    If this is set to auto, the function will only return the starting values and the optimization will be done later  \n
    See part 6 in X. Michalet 2010, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3055791/ )

    Parameters
    ----------
    settings : TYPE
        DESCRIPTION.
    eval_tm : TYPE
        trajectory data of a single particle.
    amount_lagtimes_auto : TYPE, optional
        boolean that defines whether to set the number of lagtimes automatically. The default is None.

    Returns
    -------
    lagtimes_min : TYPE
        DESCRIPTION.
    lagtimes_max : TYPE
        DESCRIPTION.
    max_counter : TYPE
        DESCRIPTION.           
    """ 
    
    if amount_lagtimes_auto == None:
        amount_lagtimes_auto = settings["MSD"]["Amount lagtimes auto"]
    
    if amount_lagtimes_auto == 1:            
        # maximum iteration
        max_counter = 10
       
        # the lower lagtimes are always considered no matter how noise they are 
        # (X. Michalet 2010, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3055791/ )
        
        traj_length = len(eval_tm)
        
        # start value accoring to Michaelt 2010
        lagtimes_min = 1
        lagtimes_max = int(np.ceil(traj_length/10))
        
        # lagtimes must be 2 at least
        if lagtimes_max == 1:
            lagtimes_max = 2

        # avoid to large lagtimes as starting values, otherwise it takes to long (and the value will be much lower than 100 anyway)
        if lagtimes_max > 100:
            lagtimes_max = 100
        
        
        nd.logger.debug("Currently considered lagtimes (offset, slope): %s", lagtimes_max)              
        
    else:
        # no iteration but select lagtimes min and max values
        max_counter = 1
        lagtimes_min = settings["MSD"]["lagtimes_min"]
        lagtimes_max = settings["MSD"]["lagtimes_max"]

    # lagtimes max can be different for offset and slope
    lagtimes_max = [lagtimes_max, lagtimes_max]

    return lagtimes_min, lagtimes_max, max_counter



def ExportResultsMain(ParameterJsonFile, settings, sizes_df_lin, yEval = False):
    """
    Exports/Save the results

    Parameters
    ----------
    ParameterJsonFile : TYPE
        DESCRIPTION.
    settings : TYPE
        DESCRIPTION.
    sizes_df_lin : TYPE
        DESCRIPTION.
    yEval : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """    
    # set particle-id to index
    sizes_df_lin = sizes_df_lin.set_index('particle')
    
    if settings["Plot"]["save_data2csv"] == True:
        # save the summary pandas sizes_df_lin
        sizes_df_lin2csv(settings, sizes_df_lin, yEval = yEval)        
    
    # write settings to parameter file ParameterJsonFile
    nd.handle_data.WriteJson(ParameterJsonFile, settings)



def sizes_df_lin2csv(settings, sizes_df_lin, yEval = False):
    """
    Saves the pandas type sizes_df_lin to a csv

    Parameters
    ----------
    settings : TYPE
        DESCRIPTION.
    sizes_df_lin : TYPE
        DESCRIPTION.
    yEval : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """  
    
    save_folder_name = settings["Plot"]["SaveFolder"]
    
    # choose transversal or longitudinal file name
    if yEval == True:
        save_file_name = "sizes_df_lin_trans"
    else:
        save_file_name = "sizes_df_lin_long"
        
    # get full path where data shall be stored
    my_dir_name, entire_path_file, time_string = nd.visualize.CreateFileAndFolderName(save_folder_name, save_file_name, d_type = 'csv')
    
    # save it
    sizes_df_lin.to_csv(entire_path_file)
    
    nd.logger.info('Data stored in: %s', format(my_dir_name))



def OptimizeMSD(eval_tm, settings, yEval, any_successful_check, max_counter = None, lagtimes_min = None, lagtimes_max = None, t_beforeDrift = None):
    """
    Calculates the MSD, finds the optimal number of fitting points (iterativly), fits the MSD-curve and saves the retrieved values to a large pandas sizes_df_particle

    Parameters
    ----------
    eval_tm : pandas
        trajectory of a single particle.
    settings : TYPE
        DESCRIPTION.
    yEval : TYPE
        DESCRIPTION.
    any_successful_check : TYPE
        DESCRIPTION.
    MSD_fit_Show : TYPE, optional
        DESCRIPTION. The default is None.
    max_counter : integer
        sets the number of iterations for the choice of the lagtime. The default is None.
    lagtimes_min : TYPE, optional
        minimal considered lagtime. The default is None.
    lagtimes_max : TYPE, optional
        maximal considered lagtime. The default is None.
    t_beforeDrift : TYPE, optional
        trajectory of a single particle, before drift correction. The default is None.

    Returns
    -------
    sizes_df_particle : pandas
        summary of retrieved values.
    OptimizingStatus : TYPE
        DESCRIPTION.

    """
        
    # if t_beforeDrift not given, just use the first particles trajectory instead (better than nothing)
    if t_beforeDrift is not None:
        particleid = eval_tm["particle"].iloc[0]
        t_beforeDrift = t_beforeDrift[t_beforeDrift.particle==particleid]

    # use only unsaturated particle positions. Saturation leads to low accuracy localization
    eval_tm_valid = eval_tm[eval_tm.saturated == False]
    
    # get the start Parameters of the MSD fitting if not all given
    if (lagtimes_min == None) or (lagtimes_max == None) or (max_counter == None):
        lagtimes_min, lagtimes_max, max_counter = MSDFitLagtimes(settings, eval_tm_valid)
    
    # get if the lagtimes shall be optimized automatically
    amount_lagtimes_auto = settings["MSD"]["Amount lagtimes auto"]
    
    # boolean to leave optimization counter of the loop
    OptimizingStatus = "Continue"
    counter = 0 # loop counter 
        
    # CALCULATE MSD, FIT IT AND OPTIMIZE PARAMETERS
    while OptimizingStatus == "Continue":
        # calculate MSD
        nan_tm_sq, amount_frames_lagt1, enough_values, traj_length, nan_tm = \
        CalcMSD(eval_tm_valid, settings, lagtimes_min = lagtimes_min, 
                lagtimes_max = lagtimes_max, yEval = yEval)
        
        # just continue if there are enough data points
        if enough_values in ["TooShortTraj", "TooManyHoles"]:
            OptimizingStatus = "Abort"
            
        else:
            # ... if Kolmogorow Smirnow Test was done before already
            if 'stat_sign' in (eval_tm_valid.keys()):
                stat_sign = eval_tm_valid['stat_sign'].mean() # same value for all
            else:
            # check if datapoints in first lagtime are normal distributed
                _, stat_sign, _ = KolmogorowSmirnowTest(nan_tm, traj_length)
                
            # only continue of trajectory shows brownian (gaussian) motion
            nd.logger.debug("Statistically threshold of 0.01 is a bit empirically.")
            if stat_sign < 0.01:
                OptimizingStatus = "Abort"
            else:    
                # Avg each lagtime and fit the MSD Plot
                # calc MSD and fit linear function through it
                msd_fit_para, diff_direct_lin, diff_std = \
                AvgAndFitMSD(nan_tm_sq, settings, lagtimes_min, lagtimes_max, amount_frames_lagt1)
                        
                # recalculate fitting range p_min if auto is selected
                if amount_lagtimes_auto == 1:
                    lagtimes_max, OptimizingStatus = UpdateP_Min(settings, eval_tm_valid, msd_fit_para, diff_direct_lin, amount_frames_lagt1, lagtimes_max)
               
        
                # Check if a new iteration shall be done
                if max_counter == 1: # dont do optimization
                    OptimizingStatus = "Successful"        
                elif counter < max_counter: #iterate again
                    counter = counter + 1           
                else: # abort since it does not converge
                    nd.logger.debug("Does not converge")
                    OptimizingStatus = "Abort"


    # save all the results of the particle into a pandas
    sizes_df_particle=pd.DataFrame(columns={'diameter','particle'})  
    
    if OptimizingStatus == "Successful":               
        # put all retrieved values into one large pandas
        sizes_df_particle = ConcludeResultsMain(settings, eval_tm, sizes_df_particle, diff_direct_lin, traj_length, lagtimes_max, amount_frames_lagt1, stat_sign, msd_fit_para, t_beforeDrift = t_beforeDrift)
    
    return sizes_df_particle, OptimizingStatus


def GetVisc(settings):
    """
    Returns the viscosity of the liquid

    """
    
    temp_water = settings["Exp"]["Temperature"]
    solvent = settings["Exp"]["solvent"]
    
    #required since Viscosity has old typo
    try:
        visc_auto = settings["Exp"]["Viscosity_auto"]
    except:
        visc_auto = settings["Exp"]["Viscocity_auto"]          
    
    if visc_auto == 1:
        settings["Exp"]["Viscosity"] = nd.Experiment.GetViscosity(temperature = temp_water, solvent = solvent)
    
    #required since Viscosity has old typo
    try:
        visc_water = settings["Exp"]["Viscosity"]
    except:
        visc_water = settings["Exp"]["Viscocity"]
        
    return settings, visc_water



def GetParameterOfTraj(eval_tm, t_beforeDrift=None):
    """
    extract information from trajectory DataFrame of a single particle. Check trackpy for list and description of all the parameters

    Parameters
    ----------
    eval_tm : pandas.DataFrame
        ideally containing all information about a single trajectory..
    t_beforeDrift : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    start_frame : TYPE
        DESCRIPTION.
    mean_mass : TYPE
        DESCRIPTION.
    mean_size : TYPE
        DESCRIPTION.
    mean_ecc : TYPE
        DESCRIPTION.
    mean_signal : TYPE
        DESCRIPTION.
    raw_mass_mean : TYPE
        DESCRIPTION.
    raw_mass_median : TYPE
        DESCRIPTION.
    raw_mass_max : TYPE
        DESCRIPTION.
    mean_ep : TYPE
        DESCRIPTION.
    max_step : TYPE
        DESCRIPTION.
    true_particle : TYPE
        DESCRIPTION.
    start_x : TYPE
        DESCRIPTION.
    start_y : TYPE
        DESCRIPTION.

    """    

    start_frame = int(eval_tm.iloc[0].frame)
	
    if t_beforeDrift is None:
        # check if x and y columns exist (sometimes just one is given (RF))
        # print("Ronny thinks that is missleading.")
        if "x" in eval_tm:
            start_x = eval_tm.iloc[0].x
        else:
            start_x = -1            
            
        if "y" in eval_tm:
            start_y = eval_tm.iloc[0].y
        else:
            start_y = -1
            
    else:
        if "x" in t_beforeDrift:
            start_x = t_beforeDrift.iloc[0].x
        else:
            start_x = -1            
            
        if "y" in t_beforeDrift:
            start_y = t_beforeDrift.iloc[0].y       
        else:
            start_y = -1
            
    # total (sometimes saturaed) trajectory
    eval_tm_valid = eval_tm[eval_tm.saturated == False]
    
    # get the mean value over the entire traj
    mean_mass     = eval_tm_valid["mass"].mean()
    mean_signal   = eval_tm_valid["signal"].mean()    
    
    raw_mass_mean = eval_tm_valid["raw_mass"].mean()
    raw_mass_median = eval_tm["raw_mass"].median()
    raw_mass_max = eval_tm_valid["raw_mass"].max()
    
    
    # localizable (unsaturated) frames
    mean_size = eval_tm["size"].mean()
    mean_ecc = eval_tm["ecc"].mean()
    mean_ep = eval_tm["ep"].mean()
    
    if ["rel_step"] in list(eval_tm.keys()):
        max_step = eval_tm["rel_step"].max()
    else:
        max_step = 0
    true_particle = eval_tm["true_particle"].max()
    
    return start_frame, mean_mass, mean_size, mean_ecc, mean_signal, raw_mass_mean, raw_mass_median, raw_mass_max, mean_ep, max_step, true_particle, start_x, start_y



def  KolmogorowSmirnowTest(nan_tm, traj_length, MinSignificance = 0.1, PlotErrorIfTestFails = False, PlotAlways = False, ID='unknown', processOutput = False):
    """
    Perform a Kolmogorow-Smirnow test on the displacement values of a trajectory. The histogram of the misplacement should be normal distributed. KS-Test checks how well this is achieved. If the mismatch is to large, an error is given, because the trajectory must have an error

    Parameters
    ----------
    nan_tm : pandas
        misplacement of trajectory.
    traj_length : TYPE
        DESCRIPTION.
    MinSignificance : TYPE, optional
        This is the threshold value for the obtained significance. The default is 0.1.
    PlotErrorIfTestFails : Boolean, optional
        DESCRIPTION. The default is False.
    PlotAlways : Boolean, optional
        DESCRIPTION. The default is False.
    ID : TYPE, optional
        DESCRIPTION. The default is 'unknown'.
    processOutput : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    traj_has_error : TYPE
        DESCRIPTION.
    stat_sign : TYPE
        DESCRIPTION.
    dx : TYPE
        DESCRIPTION.

    """

    
    # get movement between two CONSECUTIVE frames
    dx = nan_tm[1][np.isnan(nan_tm[1]) == False]

    # fit the normal distribution
    mu, std = scipy.stats.norm.fit(dx)
    
    #HG was here
    if len(dx) == 0:
        traj_has_error = False
        stat_sign = np.nan
        nd.logger.warning("Trajectory has only holes. No Kolmogorow-Smirnow test.")
        
    else:
        # perform Kolmogorow-Smirnow test if data can be represented by best normal fit
        [D_Kolmogorow, _] = scipy.stats.kstest(dx, 'norm', args=(mu, std))

        # calculate statistical significance out of it
        # https://de.wikipedia.org/wiki/Kolmogorow-Smirnow-Test
        stat_sign = 2*np.power(np.e,-2*traj_length*(D_Kolmogorow**2))

        # stat_sign says how likely it is, that this trajectory has not normal distributed displacement values
        traj_has_error = ( stat_sign < MinSignificance )
    
    
    if ((traj_has_error == True) and (PlotErrorIfTestFails == True)) or PlotAlways == True:
        # plot the misplacement histogramm including the fit
        
        # form cumulative density function (CDF)
        # dx over cdf
        dx_exp = np.sort(dx)
        
        # here comes the cdf value (from 0 to 1 in equidistant steps)
        N = len(dx_exp)
        cdf_exp = np.array(range(N))/float(N-1)
        
        plt.figure()
        plt.plot(dx_exp, cdf_exp, '-g', label = 'CDF - Data')
        plt.xlabel("dx")
        plt.ylabel("CDF")
        if type(ID) != str :
            plt.title('Particle ID = {}'.format(int(ID)) )
        
        #compare with theory
        #dx_theory = np.linspace(cdf_exp[0],dx_exp[-1],N)             
        cdf_theory = scipy.stats.norm.cdf(dx_exp, loc = mu, scale = std)     
        plt.plot(dx_exp, cdf_theory, '--r', label = 'CDF - Fit')
        plt.legend()
        plt.show()

    
    # print result if wanted
    if processOutput == True:
        nd.logger.debug("Kolmogorow-Smirnow test significance: ", stat_sign)   
    
    return traj_has_error, stat_sign, dx



def UpdateP_Min(settings, eval_tm, msd_fit_para, diff_direct_lin, amount_frames_lagt1, lagtimes_max_old):
    """
    calculate theoretical best number of considered lagtimes. The ideal number of lagtimes can be different for the slope and the offset. \n
    
    Strictly after the method from Michalet 2012 - Appendix B    
    
    a is the offset/intersept
    b is the slope

    Parameters
    ----------
    settings : TYPE
        DESCRIPTION.
    eval_tm : TYPE
        DESCRIPTION.
    msd_fit_para : TYPE
        DESCRIPTION.
    diff_direct_lin : TYPE
        DESCRIPTION.
    amount_frames_lagt1 : TYPE
        DESCRIPTION.
    lagtimes_max_old : TYPE
        DESCRIPTION.

    Returns
    -------
    lagtimes_max : TYPE
        max lagtime for b(slope) and a(offset) 
    OptimizingStatus : TYPE
        DESCRIPTION.

    """
    
    # input after step 1: new fitting parameters msd_fit_para
    
    # now step 2: compute new red_x ()
    
    # Eq B3 and 4
    L_a = 3 + np.power(4.5*np.power(amount_frames_lagt1, 0.4) - 8.5, 1.2)
    
    L_b = 0.8 + 0.564 * amount_frames_lagt1

    # change maximal lagtime if slope or offset is negativ (Michalet2012)
    if msd_fit_para[0] < 0:
        # negative slope
        lagtimes_max = np.array([L_a, L_b], dtype = 'int')
     
    else:
        #calculate best considered maximal lagtime
    
        #Michalet 2012; Eq B1 in 13
        t_frame = 1 / settings["Exp"]["fps"]
        
        # get reduced localization accuracy depending on the mode
        red_x = Estimate_X(settings, msd_fit_para[0], msd_fit_para[1], t_frame)

        # set it to zero if negative
        if red_x < 0:
            red_x = 0

        # step 3 - compute new lagtimes

        # for the offset - Eq. B3
        f_a = 2 + 1.6 * np.power(red_x, 0.51)        
        
        p_min_a = (f_a*L_a) / (np.power((f_a**3) + (L_a**3),1/3))
        p_min_a = np.floor(p_min_a)    

        # 2 points are required at least
        if p_min_a < 2:
            p_min_a = 2
        
        # for the slope - Eq. B4
        f_b = 2 + 1.35 * np.power(red_x, 0.6)
        
        value_1 = L_b
        value_2 = (f_b * L_b) / np.power(np.power(f_b,3) + np.power(L_b,3),1/3)
           
        p_min_b = np.floor(np.min([value_1, value_2]))
        
        # 2 points are required at least
        if p_min_b < 2:
            p_min_b = 2
        
        #max lagtime for a(offset) and b(slope)
        lagtimes_max = np.array([p_min_b, p_min_a], dtype = 'int')

    # check if input and output number of lagtimes (FOR THE SLOPE) is identical - continue or finish optimization then
    if np.min(lagtimes_max_old[1] == lagtimes_max[1]):
        OptimizingStatus = "Successful"
    else:
        OptimizingStatus = "Continue"
        nd.logger.debug("Current considered lagtimes (offset, slope): %s", lagtimes_max)  
       
    return lagtimes_max, OptimizingStatus



def Estimate_X(settings, slope, offset, t_frame):
    """
    calculated reduced localication accuracy out of the fitting parameters.
    different modes how to do this
    """
    
    if settings["MSD"]["Estimate_X"] == "Exp":
        red_x = RedXOutOfMsdFit(slope, offset, t_frame)
        
    elif settings["MSD"]["Estimate_X"] == "Theory":
        diffusion = 1
        expTime = settings["Exp"]["ExposureTime"]
        NA = settings["Exp"]["NA"]
        wavelength = settings["Exp"]["lambda"]
        photons = 1
        red_x = nd.Theory.RedXOutOfTheory(diffusion, expTime, t_frame, NA, wavelength, photons)
        
    else:
        nd.logger.warning("ERROR in json settings[MSD][Estimate_X]. This should be either Exp or Theory!")
        
        
    return red_x


def RedXOutOfMsdFit(slope, offset, t_frame):
    """
    calculated reduced localication accuracy out of the fitting parameters

    Parameters
    ----------
    slope : TYPE
        DESCRIPTION.
    offset : TYPE
        DESCRIPTION.
    t_frame : TYPE
        DESCRIPTION.

    Returns
    -------
    red_x : TYPE
        DESCRIPTION.

    """   
    
    # Michalet 2012 using Eq 4 in offset of Eq 10
    
    red_x = offset / (t_frame*slope)
    
    # do not allow theoretical unallowed x
    # look at defintion of x. minimum of x is achieved with sigma^2 = 0 and R maximum = 1/4
    if red_x < (-1/2):
        red_x = -1/2
        
    return red_x




def AvgAndFitMSD(nan_tm_sq, settings, lagtimes_min, lagtimes_max, amount_frames_lagt1):
    """
    wrapper for AvgMsd and FitMSD

    Parameters
    ----------
    nan_tm_sq : TYPE
        DESCRIPTION.
    settings : TYPE
        DESCRIPTION.
    lagtimes_min : TYPE
        DESCRIPTION.
    lagtimes_max : TYPE
        max lagtime for slope and offset.
    amount_frames_lagt1 : TYPE
        DESCRIPTION.

    Returns
    -------
    msd_fit_para : TYPE
        DESCRIPTION.
    diff_direct_lin : TYPE
        DESCRIPTION.
    diff_std : TYPE
        DESCRIPTION.
    
    
    
    Returns
    -------
    msd_fit_para, diff_direct_lin, diff_std
    """
    
    frames_per_second = settings["Exp"]["fps"]
    
    # calc the MSD per lagtime
    lagt_direct, msd_direct, msd_direct_std = \
    AvgMsd(nan_tm_sq, frames_per_second, lagtimes_min = lagtimes_min, lagtimes_max = lagtimes_max)

    # fit a linear function through it
    msd_fit_para, diff_direct_lin, diff_std = \
    FitMSD(lagt_direct, amount_frames_lagt1, msd_direct, msd_direct_std)                    

    return msd_fit_para, diff_direct_lin, diff_std



def CalcMSD(eval_tm, settings = None, microns_per_pixel = 1, amount_summands = 5, 
            lagtimes_min = 1, lagtimes_max = 2, yEval = False):
    """ calculate all squared displacement values for a given trajectory

    Parameters
    ----------
    eval_tm : pandas.DataFrame
        Trajectory data of a single particle.
    settings : dict, optional
        Parameter from json file. The default is None.
    microns_per_pixel : float, optional
        DESCRIPTION. The default is 1.
    amount_summands : int, optional
        minimum number of independent misplacement a trajectory has. The default is 5.
    lagtimes_min : int, optional
        DESCRIPTION. The default is 1.
    lagtimes_max : int, optional
        max lagtime for slope and offset. The default is 2.

    Returns
    -------
    nan_tm_sq, amount_frames_lagt1, enough_values, traj_length, nan_tm
    """
    # define return variables
    nan_tm, nan_tm_sq, amount_frames_lagt1 = [0, 0, 0]    
    
    if settings != None:
        microns_per_pixel = settings["Exp"]["Microns_per_pixel"]
        
        # describe how many independent misplacements are required 
        if settings["MSD"]["Amount lagtimes auto"] == 1:
            # auto function handles this differently so the value can be ignored
            amount_summands = 0
        else:
            amount_summands = settings["MSD"]["Amount summands"]
   
    # first and last frame of a trajectory
    min_frame = np.min(eval_tm.frame) 
    max_frame = np.max(eval_tm.frame)
    
    # check if particle track is long enough
    length_of_traj = max_frame - min_frame + 1
    traj_length = int(length_of_traj)
    
    # largest considered lagtime
    max_lagtimes_max = np.max(lagtimes_max)
    
    # fulfill the condition of enough statistically independent data points
    # at least amount_summands independent misplacements at max lagtimes
    if length_of_traj < (max_lagtimes_max *(1 + amount_summands)):
        enough_values = "TooShortTraj"
        nd.logger.debug("Trajectory is too short to have enough data points.")
        nd.logger.debug("It's length must be larger than (amount_summands + 1) * lagtimes_max.")
        nd.logger.debug("Consider optimizing parameters Min_tracking_frames, amount_summands, lagtimes_max.")

    else:
        # set up a matrix (DataFrame) of correct size to put in all needed 
        # combinations of displacements and lag-times:
        # 
        # index         frame number
        # column 0      initial position in respective frame
        # column 1..    displacement from original position for corresponding lagtime 1..
        #
        # Note:
        # If a particle isn't recorded for some time, missing positions have "nan" values.
        
        my_columns = range(lagtimes_min, max_lagtimes_max + 1)
        my_columns = [0, *my_columns]
        nan_tm = pd.DataFrame(index=np.arange(min_frame,max_frame+1),columns = my_columns) 
        
        # sort trajectory by frame 
        eval_tm = eval_tm.set_index("frame")
        
        # fill column 0 with position of respective frame. choose y or x direction. transition from pixels to real distance by calibrated magnification
        if yEval == False:
            nan_tm[0] = eval_tm.x * microns_per_pixel 
        else:
            nan_tm[0] = eval_tm.y * microns_per_pixel
    
        # calculate the displacement for all individual lagtimes
        # and store it in the corresponding column
        for lagtime in range(lagtimes_min, max_lagtimes_max + 1):
            nan_tm[lagtime] = nan_tm[0].diff(lagtime) 
        
        # count the number of valid frames (i.e. != nan)
        amount_frames_lagt1 = nan_tm[1].count()
        amount_frames_lagt_max = nan_tm[max_lagtimes_max].count() 

        # check if too many holes are in the trajectory
        # i.e. not enough statistically independent data present
        if amount_frames_lagt_max < (max_lagtimes_max *(1+amount_summands) + 1):
            enough_values = "TooManyHoles"
            nd.logger.debug("Too many holes in trajectory to have enough data points. Number of data points must be larger than (amount_summands * lagtimes_max). Consider optimizing parameters Dark time, Min_tracking_frames, amount_summands, lagtimes_max")
            nd.logger.debug("amount_frames_lagt_max: %s", amount_frames_lagt_max)
            nd.logger.debug("amount_summands: %s", amount_summands)
            nd.logger.debug("lagtimes_max: %s", max_lagtimes_max)

        else:
            enough_values = True    
            nan_tm_sq = nan_tm**2 # square all displacement values
     
    return nan_tm_sq, amount_frames_lagt1, enough_values, traj_length, nan_tm
    


def AvgMsd(nan_tm_sq, frames_per_second, lagtimes_min = 1, lagtimes_max = 2):
    """ calculate the mean of the squared displacement values for all lagtime values
    along with its standard deviation
    
    Parameters
    ----------
    nan_tm_sq : TYPE
        squared displacement value of each lagtime
    frames_per_second : TYPE
        frames_per_second 
    lagtimes_min : TYPE, optional
        MSD fit starting point. The default is 1.
    lagtimes_max : TYPE, optional
        MSD fit end point. The default is 2.

    Returns
    -------
    lagt_direct: list with two elements. First is the physical lagtime for fitting the offset, the second element is for the slope.
    , msd_direct, msd_direct_std

    """
    
    num_cols = len(nan_tm_sq.columns) - 1  
    
    # converting frames into physical time: 
    lagt_direct_slope = np.linspace(lagtimes_min,lagtimes_max[0],lagtimes_max[0]) /frames_per_second     
        
    lagt_direct_offset = np.linspace(lagtimes_min,lagtimes_max[1],lagtimes_max[1]) /frames_per_second 
    
    lagt_direct = (lagt_direct_offset, lagt_direct_slope)
    

    msd_direct = np.zeros(num_cols)
    msd_direct_variance = msd_direct.copy()

    for column in range(num_cols):
        eval_column = column + lagtimes_min
        # That iterates over lag-times for the respective particle and
        # calculates the MSD in the following manner:
        # 1. Build sub-blocks for statistically independent lag-time measurements

        nan_indi_means = rolling_with_step(nan_tm_sq[eval_column], eval_column, eval_column, mean_func)
        
        # 2. Calculate mean of squared displacement values for the sublocks
        msd_direct[column] = nan_indi_means.mean(axis=0)
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
        msd_direct_variance_loop = nan_indi_means.var(axis=0) / (len_nan_tm_sq_loop-1)
        
        msd_direct_variance[column] = msd_direct_variance_loop
    
    msd_direct_std = np.sqrt(msd_direct_variance)
    

    return lagt_direct, msd_direct, msd_direct_std



def FitMSD(lagt_direct, amount_frames_lagt1, msd_direct, msd_direct_std):
    """
    Fits a linear function through the MSD data

    Parameters
    ----------
    lagt_direct : TYPE
        lagt_direct: list with two elements. First is the physical lagtime for fitting the slope, the second element is for the offset.
    amount_frames_lagt1 : TYPE
        DESCRIPTION.
    msd_direct : TYPE
        DESCRIPTION.
    msd_direct_std : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    # new method
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    def lin_func(x,m,n):
        y = m * x + n
        return y

    #switch warning off, because scipy fitting warns that covariance (which is not needed here) cant be calculated
    np.warnings.filterwarnings('ignore')

    #fit slope
    t = lagt_direct[0]
    msd = msd_direct[:len(t)]
    fit_values_slope, _ = scipy.optimize.curve_fit(lin_func, t, msd)

    #fit offset
    t = lagt_direct[1]
    msd = msd_direct[:len(t)]
    fit_values_offset, _ = scipy.optimize.curve_fit(lin_func, t, msd)

    np.warnings.filterwarnings('default')

    # slope = 2 * Diffusion_coefficent
    diff_direct_lin = np.squeeze(fit_values_slope[0]/2)
        
    msd_fit_para = [fit_values_slope[0], fit_values_offset[1]]    
    
    # calculate reduced localization accuracy
    red_x = RedXOutOfMsdFit(msd_fit_para[0], msd_fit_para[1], t[0])
    
    # get diffusion error via relative error from eq.(12) in Michalet 2012
    std_diff_direct_lin = diff_direct_lin * nd.Theory.CRLB(amount_frames_lagt1,red_x)
    # * np.sqrt((2+4*np.sqrt(1+2*red_x))/(amount_frames_lagt1-1))
       
    
    return msd_fit_para, diff_direct_lin, std_diff_direct_lin



def DiffusionToDiameter(diffusion, UseHindranceFac = 0, fibre_diameter_nm = None, temp_water = 295, visc_water = 9.5e-16):
    """
    convert diffusion value to hydrodynamic diameter 
    (optionally) with taking the hindrance factor into account    

    Parameters
    ----------
    diffusion : TYPE
        DESCRIPTION.
    UseHindranceFac : Boolean, optional
        Defines if the hindrance factor shall be calcualted and corrected. The default is 0.
    fibre_diameter_nm : TYPE, optional
        DESCRIPTION. The default is None.
    temp_water : TYPE, optional
        DESCRIPTION. The default is 295.
    visc_water : TYPE, optional
        DESCRIPTION. The default is 9.5e-16.

    Returns
    -------
    diamter_corr : TYPE
        DESCRIPTION.


    """
    const_Boltz = scipy.constants.Boltzmann 
    
    # diameter of each particle in nm
    diameter = (2*const_Boltz*temp_water/(6*math.pi *visc_water)*1e9) /diffusion 

    # revert the effect of the hindrance factor
    if UseHindranceFac == True:
        if diameter < fibre_diameter_nm:
            diamter_corr = EstimateHindranceFactor(diameter, fibre_diameter_nm, DoPrint = True)
            
                
        else:
            diamter_corr = diameter
            nd.logger.warning("Stationary particle still inside")
       
    else:
        diamter_corr = diameter
    
    return diamter_corr



def EstimateHindranceFactor(diam_direct_lin, fibre_diameter_nm, DoPrint = True):
    """
    Revert the effect of the hindrance factor

    Parameters
    ----------
    diam_direct_lin : TYPE
        diameter out of the MSD analysis in nm.
    fibre_diameter_nm : TYPE
        DESCRIPTION.
    DoPrint : TYPE, optional
        if true, use the logger. The default is True.

    Returns
    -------
    None.

    """
    
    diam_direct_lin_corr = diam_direct_lin
    diam_direct_lin_corr_old = 0
    
    my_iter = 0        
    
    # run iteration till correction of diameter is below 0.1 nm
    while np.abs(diam_direct_lin_corr - diam_direct_lin_corr_old) > 0.1:
        my_iter = my_iter + 1
        
        # update current (old) diameter
        diam_direct_lin_corr_old = diam_direct_lin_corr
        
        #  calculate hindrance factor
        H,Kd = hindrance_fac(fibre_diameter_nm,diam_direct_lin_corr)

        # Dechadilok 2006, Eq 1
        corr_visc = 1 / Kd
        diam_direct_lin_corr = diam_direct_lin / corr_visc # diameter of each particle
        
        # this steps helps converging. Otherwise it might jump from 1/10 to 10 to 1/10 ...
        diam_direct_lin_corr = np.sqrt(diam_direct_lin_corr * diam_direct_lin_corr_old)

        
        if my_iter > 100:
            nd.logger.debug("Iteration does not converge. Abort !!!")
            diam_direct_lin_corr = diam_direct_lin_corr_old

    if DoPrint == True:
        nd.logger.debug("Starting Diameter: %.1fnm; hindrance factor: %.3f; Corrected Diameter: %.2fnm", round(diam_direct_lin,1), round(corr_visc,3), round(diam_direct_lin_corr,2))


    return diam_direct_lin_corr



def ConcludeResultsMain(settings, eval_tm, sizes_df_lin, diff_direct_lin, traj_length, lagtimes_max, amount_frames_lagt1, stat_sign, msd_fit_para, t_beforeDrift=None):
    """
    organize the results and write them in one large pandas.DataFrame
    """
    
    UseHindranceFac = settings["MSD"]["EstimateHindranceFactor"]
    temp_water = settings["Exp"]["Temperature"]
    
    settings, visc_water = GetVisc(settings)
    
    fibre_diameter_nm = settings["Fiber"]["TubeDiameter_nm"]
    
    # get parameters of the trajectory to analyze it
    start_frame, mean_mass, mean_size, mean_ecc, mean_signal, raw_mass_mean, raw_mass_median, raw_mass_max, mean_ep, max_step, true_particle, start_x, start_y = GetParameterOfTraj(eval_tm, t_beforeDrift = t_beforeDrift)
    
    particleid = int(eval_tm.iloc[0].particle)
    
    lagtime = 1/settings["Exp"]["fps"]
    
    red_x = RedXOutOfMsdFit(msd_fit_para[0], msd_fit_para[1], lagtime)
                    
#     get the fit error if switches on (and working)
    rel_error_diff, diff_std = DiffusionError(traj_length, red_x, diff_direct_lin)

    diameter = DiffusionToDiameter(diff_direct_lin, UseHindranceFac, fibre_diameter_nm, temp_water, visc_water)


    sizes_df_lin = ConcludeResults(sizes_df_lin, diff_direct_lin, diff_std, diameter,                            particleid, traj_length, amount_frames_lagt1, start_frame, mean_mass, mean_size, mean_ecc, mean_signal, raw_mass_mean, raw_mass_median, raw_mass_max, mean_ep, red_x, max_step, true_particle, stat_sign = stat_sign, start_x=start_x, start_y=start_y)


    return sizes_df_lin



def ConcludeResults(sizes_df_lin, diff_direct_lin, diff_std, diameter, particleid, traj_length, amount_frames_lagt1, start_frame, mean_mass, mean_size, mean_ecc, mean_signal, raw_mass_mean, raw_mass_median, raw_mass_max, mean_ep, red_x, max_step, true_particle, stat_sign = None, start_x=0,start_y=0):
    """ write all the valuable parameters in one large pandas.DataFrame """

    # store results in DataFrame:   
    sizes_df_lin = sizes_df_lin.append(pd.DataFrame(data={'particle': [particleid],
                                                          'diffusion': [diff_direct_lin],
                                                          'diffusion std': [diff_std],
                                                          'diameter': [diameter],
                                                          'ep': [mean_ep],
                                                          'red_x' : [red_x], 
                                                          'signal': [mean_signal],
                                                          'mass': [mean_mass],
                                                          'rawmass_mean': [raw_mass_mean],
                                                          'rawmass_median': [raw_mass_median],
                                                          'rawmass_max': [raw_mass_max],
                                                          'max step': [max_step],
                                                          'first frame': [start_frame],
                                                          'traj length': [traj_length],
                                                          'valid frames':[amount_frames_lagt1],
                                                          'size': [mean_size],
                                                          'ecc': [mean_ecc],
                                                          'stat_sign': [stat_sign],
                                                          'true_particle': [true_particle],
                                                          'x in first frame': [start_x],
                                                          'y in first frame': [start_y]
                                                          }), sort=False)
    
    return sizes_df_lin

    


def DiffusionError(traj_length, red_x, diffusion):
    """
    Calculates the minimum diffusion error by the MSD analysis

    Parameters
    ----------
    traj_length : TYPE
        trajectory length.
    red_x : TYPE
        reduced localization accuracy.
    diffusion : TYPE
        retrieved diffusion coefficent.

    Returns
    -------
    rel_error : TYPE
        DESCRIPTION.
    diffusion_std : TYPE
        DESCRIPTION.

    """    
    
    if red_x == "gain missing":
        red_x = 0 # assume the ideal case
        # MN: I think this is okay since it's meant to estimate the MIN. error
        
    # rel. error is well approximated by CRLB for optimized least square fit 
    rel_error = nd.Theory.CRLB(traj_length, red_x)
  
    diffusion_std = diffusion * rel_error
    
    return rel_error, diffusion_std



def rolling_with_step(s, window, step, func): 
    """ apply a given function "func" over a window of a series "s",
    while the window moves with stepsize "step" 
    cf. https://github.com/pandas-dev/pandas/issues/15354
    
    Idea: Reorder the 1d input (s) into a 2d array, in which each coloum contains the values inside the window the func shall be applied on. the rows are the different windows.
    """
    
    # vert_idx_list = np.arange(0, s.size - window, step) # old version #30
    
    # start index of window    
    vert_idx_list = np.arange(0, s.size - window + 1, step)
    
    # increment in a window (this is the same for windows)
    hori_idx_list = np.arange(window)
    
    # get the indices in each window to apply the function on
    A, B = np.meshgrid(hori_idx_list, vert_idx_list)
    idx_array = A + B
    
    # get the values for the windows
    x_array = s.values[idx_array]
    
    # apply function
    d = func(x_array)
    
    # get the index of the MIDDLE of the window. thus the window is centered
    idx = s.index[vert_idx_list + int(window/2.)]
    
    # return it as a pandas
    return pd.Series(d, index=idx)



def mean_func(d):
    """ function that calculates the mean of a series, while ignoring NaN and 
    not yielding a warning when all are NaN 
    
    MN: obsolete? pd.Series.mean() function handles this automatically
    """
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
    
    # calculate the required local hindrance factor
    Kd = local_hindrance_fac(H, l)
    
    return H, Kd



def local_hindrance_fac(H, my_lambda):
    """ calculate the hindrance factor for diffusion from particle and channel dimensions
    according to eq. 16 from "Hindrance Factors for Diffusion and Convection in Pores", 
    Dechadilok and Deen (2006)
    """
    #Eq. 9
    Phi = (1-my_lambda)**2
    Kd = H / Phi
    
    return Kd
  
  

def SummaryEval(settings, rawframes_pre, obj_moving, traj_moving, traj_no_drift, traj_final, sizes_df_lin):
    """
    Plots a lot of info about the entire evaluation
    """
    
    # num frames
    num_frames = rawframes_pre.shape[0]
    nd.logger.info("number of frames: %i", num_frames)
    
    fov_length = rawframes_pre.shape[2] * settings["Exp"]["Microns_per_pixel"]
    fov_crosssection = np.pi * (settings["Fiber"]["TubeDiameter_nm"]/2/1000)**2
    
    # volume in um^3
    fov_volume_um = fov_length * fov_crosssection
    
    fov_volume_nl = fov_volume_um * 1E-6
    nd.logger.info("Observed volume: %.2f nl", fov_volume_nl)
    
    # located particles per frame
    loc_part_frame = len(obj_moving) / num_frames
    nd.logger.info("Located particles per frames: %.1f", loc_part_frame)
    
    # formed trajectory - min trajectory before drift correction
    traj_frame_before = len(traj_moving) / num_frames
    nd.logger.info("Trajectory per frames - before testing: %.1f", traj_frame_before)
    
    # formed trajectory - min trajectory after drift correction
    traj_frame_after = len(traj_no_drift) / num_frames
    nd.logger.info("Trajectory per frames - after drift correction: %.1f", traj_frame_before)
    
    # valid trajectory - used in msd
    traj_frame_msd = len(traj_final) / num_frames 
    nd.logger.info("Trajectory per frames - evaluted by MSD: %.1f", traj_frame_msd)
    
    # analyze components when existing
    if "comp" in sizes_df_lin.keys():
        #evalue the components
        comp = np.sort(sizes_df_lin["comp"].unique())
        num_comp = len(comp)
        
        # number trajectory points each component has
        comp_traj = np.zeros(num_comp)
    
        for ii in range(num_comp):
            eval_comp = sizes_df_lin[sizes_df_lin["comp"] == ii]
            comp_traj[ii] = int(np.sum(eval_comp["traj length"]))
            
        # components per frame
        comp_frame = comp_traj / num_frames
        
        comp_concentration = comp_frame / fov_volume_nl
    
        for ii in range(num_comp):
            nd.logger.info("Component %i: particles per frames: %.2f", ii, comp_frame[ii])
    
        for ii in range(num_comp):
            nd.logger.info("Component %i: %.2f (particles/nL)", ii, comp_concentration[ii])    
    
    
    
    
def InvDiameter(sizes_df_lin, settings, useCRLB=True):
    nd.logger.error("InvDiameter has moved to statistics.py! Please change if you see this")


def StatisticOneParticle(sizes):
    nd.logger.errort("StatisticOneParticle has moved to statistics.py! Please change if you see this")
    
    
def StatisticDistribution(sizes_df_lin, num_dist_max=10, 
                          weighting=True, showICplot=False, useAIC=True):
    nd.logger.error("StatisticDistribution has moved to statistics.py! Please change if you see this")    
    

def Main(t6_final, ParameterJsonFile, obj_all, microns_per_pixel = None, frames_per_second = None, amount_summands = None, amount_lagtimes = None, amount_lagtimes_auto = None, Histogramm_Show = True, MSD_fit_Show = False, EvalOnlyLongestTraj = 0, Max_traj_length = None, yEval = False, 
         processOutput=True, t_beforeDrift=None):
    
    nd.logger.error("This is an old function. Call Main2 instead...")
