# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 09:52:27 2019

@author: Ronny Förster, Stefan Weidlich, Mona Nissen

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


def Main2(t6_final, ParameterJsonFile, MSD_fit_Show = False, yEval = False, 
          processOutput = True, t_beforeDrift = None):
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
    
    nd.logger.warning("Variable sizes_df_lin_rolling is not longer calculated. That might be inserted later again. If you have this return variable in your CalcDiameter.Main(2) function that might cause an error.")
        
    #boolean for plotting stuff
    any_successful_check = False


    # LOOP THROUGH ALL THE PARTICLES
    for i,particleid in enumerate(particle_list_value):
        nd.logger.debug("Particle number: %s",  int(particleid))

        # select trajectory to analyze
        eval_tm = t6_final_use[t6_final_use.particle==particleid]
        

        # CALCULATE MSD, FIT IT AND OPTIMIZE PARAMETERS
        sizes_df_particle, OptimizingStatus = OptimizeMSD(eval_tm, settings, yEval, any_successful_check, t_beforeDrift = t_beforeDrift)

        if OptimizingStatus == "Successful":
            any_successful_check = True
            # after the optimization is done -save the result in a large pandas.DataFrame
            sizes_df_lin = sizes_df_lin.append(sizes_df_particle)
                          
    
    if any_successful_check == False:
        nd.logger.warning("No particle made it to the end!")
        
    else:
        ExportResultsMain(ParameterJsonFile, settings, sizes_df_lin)       
    
    return sizes_df_lin, any_successful_check



def GetSettingsParameters(settings):
    """ get required parameters out of the settings
    """
    temp_water = settings["Exp"]["Temperature"]
    amount_lagtimes_auto = settings["MSD"]["Amount lagtimes auto"]
    MSD_fit_Show = settings["Plot"]['MSD_fit_Show']
    MSD_fit_Save = settings["Plot"]['MSD_fit_Save']
    do_rolling = settings["Time"]["DoRolling"]
 
    # sanity checks    
    if MSD_fit_Save == True:
        MSD_fit_Show = True
        
    if amount_lagtimes_auto == 1:
        if settings["Exp"]["gain"] == "unknown":
            raise ValueError("Number of considered lagtimes can't be estimated "
                             "automatically, if gain is unknown. Measure gain, or change 'Amount lagtime auto' values to 0.")
    if temp_water < 250:
        raise Exception("Temperature is below 250K! Check if the temperature is inserted in K not in C!")
   
    return temp_water, amount_lagtimes_auto, MSD_fit_Show, MSD_fit_Save, do_rolling



def SelectTrajectories(t6_final, settings):
    """ select a subset of particle trajectories, e.g. only the longest
    """
    
    EvalOnlyLongestTraj = settings["MSD"]["EvalOnlyLongestTraj"]
    Min_traj_length = settings["Link"]["Min_tracking_frames"]
    Max_traj_length = settings["Split"]["Max_traj_length"]
    
    if EvalOnlyLongestTraj == 1:
        longest_particle, longest_traj = nd.handle_data.GetTrajLengthAndParticleNumber(t6_final)
    
        settings["Split"]["ParticleWithLongestTrajBeforeSplit"] = longest_particle
    
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
    """ define how many lagtimes are considered for the fit 
    (initial values for optimization only!)
    
    amount_lagtimes_auto:   boolean that defines whether to set the number of lagtimes automatically
    eval_tm:                trajectory data of a single particle
    """ 
    if amount_lagtimes_auto == None:
        amount_lagtimes_auto = settings["MSD"]["Amount lagtimes auto"]
    
    if amount_lagtimes_auto == 1:            
        # maximum iteration
        max_counter = 10
       
        # the lower lagtimes are always considered no matter how noise they are 
        # (X. Michalet 2010, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3055791/ )
        
        traj_length = len(eval_tm)
        
        lagtimes_min = 1
        lagtimes_max = np.int(traj_length/10)
        
        # lagtimes must be 2 at least
        if lagtimes_max == 1:
            lagtimes_max = 2
        
        # special rule for simulated data of very long track lengths
        if settings["Simulation"]["SimulateData"] == 1:
            if traj_length > 10:
                nd.logger.warning("Ronny is not sure if this satisfies every user!")
                nd.logger.debug("Use 100 Lagtimes as a starting value, instead of TrajLength/100")
                lagtimes_max = 10
        
        nd.logger.debug("Currently considered lagtimes (offset, slope): %s", lagtimes_max)              
    else:
        max_counter = 1
        lagtimes_min = settings["MSD"]["lagtimes_min"]
        lagtimes_max = settings["MSD"]["lagtimes_max"]

    # lagtimes max is different for offset and slope
    lagtimes_max = [lagtimes_max, lagtimes_max]

    return lagtimes_min, lagtimes_max, max_counter



def ExportResultsMain(ParameterJsonFile, settings, sizes_df_lin):
    MSD_fit_Show = settings["Plot"]['MSD_fit_Show']
    MSD_fit_Save = settings["Plot"]['MSD_fit_Save']
    
    AdjustMSDPlot(MSD_fit_Show)
    
    sizes_df_lin = sizes_df_lin.set_index('particle')

    if MSD_fit_Save == True:
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "MSD_Fit", settings, ShowPlot = settings["Plot"]["MSD_fit_Show"])
    
    if settings["Plot"]["save_data2csv"] == True:
        sizes_df_lin2csv(settings, sizes_df_lin)        
    
    nd.handle_data.WriteJson(ParameterJsonFile, settings)



def sizes_df_lin2csv(settings, sizes_df_lin):
    save_folder_name = settings["Plot"]["SaveFolder"]
    save_file_name = "sizes_df_lin"
    my_dir_name, entire_path_file, time_string = nd.visualize.CreateFileAndFolderName(save_folder_name, save_file_name, d_type = 'csv')
    sizes_df_lin.to_csv(entire_path_file)
    nd.logger.info('Data stored in: %s', format(my_dir_name))



def OptimizeMSD(eval_tm, settings, yEval, any_successful_check, MSD_fit_Show = None, max_counter = None, lagtimes_min = None, lagtimes_max = None, t_beforeDrift = None):
    """
    Calculates the MSD, finds the optimal number of fitting points (iterativly) and fits the MSD-curve
    """
    
    if t_beforeDrift is not None:
        particleid = eval_tm["particle"].iloc[0]
        t_beforeDrift = t_beforeDrift[t_beforeDrift.particle==particleid]

    
    # get the start Parameters of the MSD fitting if not all given
    if (lagtimes_min == None) or (lagtimes_max == None) or (max_counter == None):
        lagtimes_min, lagtimes_max, max_counter = MSDFitLagtimes(settings, eval_tm)
    
    amount_lagtimes_auto = settings["MSD"]["Amount lagtimes auto"]
    
    # boolean to leave optimization counter of the loop
    OptimizingStatus = "Continue"
    counter = 0 # loop counter 
        
    # CALCULATE MSD, FIT IT AND OPTIMIZE PARAMETERS
    while OptimizingStatus == "Continue":
        """  1 - calculate MSD """
        nan_tm_sq, amount_frames_lagt1, enough_values, traj_length, nan_tm = \
        CalcMSD(eval_tm, settings, lagtimes_min = lagtimes_min, 
                lagtimes_max = lagtimes_max, yEval = yEval)
        
        # just continue if there are enough data points
        if enough_values in ["TooShortTraj", "TooManyHoles"]:
            OptimizingStatus = "Abort"
            
        else:
            # ... if Kolmogorow Smirnow Test was done before already
            if 'stat_sign' in (eval_tm.keys()):
                stat_sign = eval_tm['stat_sign'].mean() # same value for all
            else:
            # check if datapoints in first lagtime are normal distributed
                _, stat_sign, _ = KolmogorowSmirnowTest(nan_tm, traj_length)
                
            # only continue of trajectory shows brownian (gaussian) motion
            if stat_sign < 0.01:
                OptimizingStatus = "Abort"
            else:    
                """ Avg each lagtime and fit the MSD Plot """
                # calc MSD and fit linear function through it
                msd_fit_para, diff_direct_lin, diff_std = \
                AvgAndFitMSD(nan_tm_sq, settings, lagtimes_min, lagtimes_max, amount_frames_lagt1)
                        
                # recalculate fitting range p_min
                if amount_lagtimes_auto == 1:
                    lagtimes_max, OptimizingStatus = UpdateP_Min(settings, eval_tm, msd_fit_para, diff_direct_lin, amount_frames_lagt1, lagtimes_max)
               
        
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
        # open a window to plot for the MSD if not open yet
        any_successful_check = CreateNewMSDPlot(any_successful_check, MSD_fit_Show)
                
        sizes_df_particle = ConcludeResultsMain(settings, eval_tm, sizes_df_particle, diff_direct_lin, traj_length, lagtimes_max, amount_frames_lagt1, stat_sign, msd_fit_para, DoRolling = False, t_beforeDrift = t_beforeDrift)
    
    
        # plot MSD if wanted
        if MSD_fit_Show == None:
            MSD_fit_Show = settings["Plot"]['MSD_fit_Show']
    
        # plot MSD if wanted
        if MSD_fit_Show == True:
            AvgAndFitMSD(nan_tm_sq, settings, lagtimes_min, lagtimes_max, amount_frames_lagt1, MSD_fit_Show = True)


    return sizes_df_particle, OptimizingStatus




def GetVisc(settings):
    
    temp_water = settings["Exp"]["Temperature"]
    solvent = settings["Exp"]["solvent"]
    
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
        
    return settings, visc_water



def GetParameterOfTraj(eval_tm, t_beforeDrift=None):
    """ extract information from trajectory DataFrame of a single particle
    
    Parameters
    ----------
    eval_tm : pandas.DataFrame
        ... ideally containing all information about a single trajectory.


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
            
    mean_mass = eval_tm["mass"].mean()
    mean_size = eval_tm["size"].mean()
    mean_ecc = eval_tm["ecc"].mean()
    mean_signal = eval_tm["signal"].mean()
    mean_raw_mass = eval_tm["raw_mass"].mean()
    mean_ep = eval_tm["ep"].mean()
    max_step = eval_tm["rel_step"].max()
    true_particle = eval_tm["true_particle"].max()
    
    return start_frame, mean_mass, mean_size, mean_ecc, mean_signal, mean_raw_mass, mean_ep, max_step, true_particle, start_x, start_y



def CreateNewMSDPlot(any_successful_check, MSD_fit_Show):
    """ for first time call, open a new plot window for the MSD ensemble fit """
    if any_successful_check == False:
        any_successful_check = True
        if MSD_fit_Show == True:
            plt.figure()

    return any_successful_check


def CheckIfTrajectoryHasError(nan_tm, traj_length, MinSignificance = 0.1, PlotErrorIfTestFails = False, PlotAlways = False, ID='unknown', processOutput = False):
    nd.logger.error("Function name is old. Use KolmogorowSmirnowTest instead")
    
    return KolmogorowSmirnowTest(nan_tm, traj_length, MinSignificance = 0.1, PlotErrorIfTestFails = False, PlotAlways = False, ID='unknown', processOutput = False)


def  KolmogorowSmirnowTest(nan_tm, traj_length, MinSignificance = 0.1, PlotErrorIfTestFails = False, PlotAlways = False, ID='unknown', processOutput = False):
    """ perform a Kolmogorow-Smirnow test on the displacement values of a trajectory
    """
    
    # get movement between two CONSECUTIVE frames
    dx = nan_tm[1][np.isnan(nan_tm[1]) == False]

    # fit the normal distribution
    mu, std = scipy.stats.norm.fit(dx)
    
    # perform Kolmogorow-Smirnow test if data can be represented by best normal fit
    [D_Kolmogorow, _] = scipy.stats.kstest(dx, 'norm', args=(mu, std))

    # calculate statistical significance out of it
    # https://de.wikipedia.org/wiki/Kolmogorow-Smirnow-Test
    stat_sign = 2*np.power(np.e,-2*traj_length*(D_Kolmogorow**2))

    # stat_sign says how likely it is, that this trajectory has not normal distributed displacement values
    traj_has_error = ( stat_sign < MinSignificance )
    
    
    if ((traj_has_error == True) and (PlotErrorIfTestFails == True)) or PlotAlways == True:
        # if PlotErrorIfTestFails == True:
        #print("Error in Traj. This can be plotted, if code here is switched on.")
        dx_exp = np.sort(dx)
        N = len(dx_exp)
        cdf_exp = np.array(range(N))/float(N)
        
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
    
    #        bp()
    
    if processOutput == True:
        # print("Trajectory has error. Particle ID: ", particleid)
        nd.logger.debug("Kolmogorow-Smirnow test significance: ", stat_sign)
    
    
    return traj_has_error, stat_sign, dx



def UpdateP_Min(settings, eval_tm, msd_fit_para, diff_direct_lin, amount_frames_lagt1, lagtimes_max_old):
    """ calculate theoretical best number of considered lagtimes
    
    method from Michalet 2012
    """
    # Eq B3 and 4
    # L_a = 3 + np.sqrt(4.5*np.power(amount_frames_lagt1, 0.4) - 8.5) #error in exponent 201124 RF
    L_a = 3 + np.power(4.5*np.power(amount_frames_lagt1, 0.4) - 8.5, 1.2)
    
    L_b = 0.8 + 0.564 * amount_frames_lagt1

    # change maximal lagtime if slope or offset is negativ (Michalet2012)
    if msd_fit_para[0] < 0:
        # negative slope
        lagtimes_max = np.array([L_a, L_b], dtype = 'int')
 
    # negative offset is not a problem. it arises from the motion blur
    # elif msd_fit_para[1] < 0:
    #     # negative offset
    #     lagtimes_max = [2,2]
        
    else:
        #calculate best considered maximal lagtime
    
        #Michalet 2012; Eq B1 in 13
        #Fengjis idea
        t_frame = 1 / settings["Exp"]["fps"]
        
        # red_x = RedXOutOfMsdFit(msd_fit_para[0], msd_fit_para[1], t_frame)
        
        # get reduced localization accuracy depending on the mode
        if settings["MSD"]["Estimate_X"] == "Exp":
            red_x = RedXOutOfMsdFit(msd_fit_para[0], msd_fit_para[1], t_frame)
        
        elif settings["MSD"]["Estimate_X"] == "Theory":
            expTime = settings["Exp"]["ExposureTime"]
            NA = settings["Exp"]["NA"]
            wavelength = settings["Exp"]["lambda"]
            
            #correct gain by bit depth which was corrected previous
            gain = settings["Exp"]["gain_corr"]
            mass = eval_tm.mass.mean()
            # convert ADU into photons by camera gain
            photons = mass * gain
            red_x = nd.Theory.RedXOutOfTheory(diff_direct_lin, expTime, t_frame, NA, wavelength, photons)
        else:
            nd.logger.warning("ERROR in json settings[MSD][Estimate_X]. This should be either Exp or Theory!")


        # set it to zero if negative
        if red_x < 0:
            red_x = 0

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
        
        lagtimes_max = np.array([p_min_a, p_min_b], dtype = 'int')


    if np.min(lagtimes_max_old == lagtimes_max):
        OptimizingStatus = "Successful"
    else:
        OptimizingStatus = "Continue"
        nd.logger.debug("Current considered lagtimes (offset, slope): %s", lagtimes_max)  
       
    return lagtimes_max, OptimizingStatus


def Estimate_X(settings, slope, offset, t_frame):
    # calculated reduced localication accuracy out of the fitting parameters
    # different modes how to do this
    
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
    # calculated reduced localication accuracy out of the fitting parameters
    
    # Michalet 2012 using Eq 4 in offset of Eq 10
    
    red_x = offset / (t_frame*slope)
    
    # do not allow theoretical unallowed x
    # look at defintion of x. minimum of x is achieved with sigma^2 = 0 and R maximum = 1/4
    if red_x < (-1/2):
        red_x = -1/2
        
    return red_x



 
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



def AvgAndFitMSD(nan_tm_sq, settings, lagtimes_min, lagtimes_max, amount_frames_lagt1, 
                 DoRolling = False, MSD_fit_Show = False):
    """ wrapper for AvgMsdRolling and FitMSDRolling
    
    Returns
    -------
    msd_fit_para, diff_direct_lin, diff_std
    """
    
    frames_per_second = settings["Exp"]["fps"]
    
    # calc the MSD per lagtime
    lagt_direct, mean_displ_direct, mean_displ_sigma_direct = \
    AvgMsdRolling(nan_tm_sq, frames_per_second, DoRolling = False, 
                  lagtimes_min = lagtimes_min, lagtimes_max = lagtimes_max)

    # fit a linear function through it
    msd_fit_para, diff_direct_lin, diff_std = \
    FitMSDRolling(lagt_direct, amount_frames_lagt1, mean_displ_direct, 
                  mean_displ_sigma_direct, PlotMsdOverLagtime = MSD_fit_Show, DoRolling = False)                    

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
        DESCRIPTION. The default is 5.
    lagtimes_min : int, optional
        DESCRIPTION. The default is 1.
    lagtimes_max : int, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    nan_tm_sq, amount_frames_lagt1, enough_values, traj_length, nan_tm
    """
    # define return variables
    nan_tm, nan_tm_sq, amount_frames_lagt1 = [0, 0, 0]    
    
    if settings != None:
        microns_per_pixel = settings["Exp"]["Microns_per_pixel"]
        
        if settings["MSD"]["Amount lagtimes auto"] == 1:
            amount_summands = 0                                  # MN: does this make sense??
        else:
            amount_summands = settings["MSD"]["Amount summands"]
   
    min_frame = np.min(eval_tm.frame) # frame where particle trajectory begins
    max_frame = np.max(eval_tm.frame) # ... and where it ends
    
    # check if particle track is long enough
    length_of_traj = max_frame - min_frame + 1
    traj_length = int(length_of_traj)
    max_lagtimes_max = np.max(lagtimes_max)
    
    # fulfill the condition of enough statistically independent data points
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
        
        # fill column 0 with position of respective frame
        if yEval == False:
            nan_tm[0] = eval_tm.x * microns_per_pixel 
        else:
            nan_tm[0] = eval_tm.y * microns_per_pixel
    
        # calculate the displacement for all individual lagtimes
        # and store it in the corresponding column
        for lagtime in range(lagtimes_min, max_lagtimes_max + 1):
            nan_tm[lagtime] = nan_tm[0].diff(lagtime) 
        
        """"""
        # count the number of valid frames (i.e. != nan)
        amount_frames_lagt1 = nan_tm[0].count()                     # MN: shouldn't this be nan_tm[1] ?
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
    


def AvgMsdRolling(nan_tm_sq, frames_per_second, my_rolling = 100, DoRolling = False, 
                  lagtimes_min = 1, lagtimes_max = 2):
    """ calculate the mean of the squared displacement values for all lagtime values
    along with its standard deviation
    
    Parameters
    ----------
    nan_tm_sq : TYPE
        squared displacement value of each lagtime
    frames_per_second : TYPE
        frames_per_second 
    my_rolling : TYPE, optional
        DESCRIPTION. The default is 100.
    DoRolling : bool, optional
        True if the msd shall be evaluated time dependent (rolling). The default is False.
    lagtimes_min : TYPE, optional
        MSD fit starting point. The default is 1.
    lagtimes_max : TYPE, optional
        MSD fit end point. The default is 2.

    Returns
    -------
    lagt_direct, mean_displ_direct, mean_displ_sigma_direct

    """
    
    num_cols = len(nan_tm_sq.columns) - 1  
    
    # converting frames into physical time: 
    lagt_direct_slope = np.linspace(lagtimes_min,lagtimes_max[0],lagtimes_max[0]) /frames_per_second 
    
    lagt_direct_offset = np.linspace(lagtimes_min,lagtimes_max[1],lagtimes_max[1]) /frames_per_second 
    
    lagt_direct = (lagt_direct_slope, lagt_direct_offset)
    
    # lagt_direct = np.linspace(lagtimes_min,lagtimes_max ,num_cols) /frames_per_second # converting frames into physical time: 
    # first entry always starts with 1/frames_per_second
    
    #lag times for the offset
    
    
    if DoRolling == False:
#        mean_displ_direct = pd.DataFrame(index = [0], columns = nan_tm_sq.columns.tolist()[1:])
        mean_displ_direct = np.zeros(num_cols)
        mean_displ_variance_direct = mean_displ_direct.copy()    

        for column in range(num_cols):
            eval_column = column + lagtimes_min
            # That iterates over lag-times for the respective particle and
            # calculates the MSD in the following manner:
            # 1. Build sub-blocks for statistically independent lag-time measurements

            nan_indi_means = rolling_with_step(nan_tm_sq[eval_column], eval_column, eval_column, mean_func)
            
            # 2. Calculate mean of squared displacement values for the sublocks
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
        bp()
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

    #switch warning off, because scipy fitting warns that covariance (which is not needed here) cant be calculated
    np.warnings.filterwarnings('ignore')

    #fit slope
    t = lagt_direct[0]
    msd = mean_displ_direct[:len(t)]
    # [fit_values_slope, fit_cov_slope]= scipy.optimize.curve_fit(lin_func, t, msd)
    fit_values_slope, _ = scipy.optimize.curve_fit(lin_func, t, msd)

    #fit offset
    t = lagt_direct[1]
    msd = mean_displ_direct[:len(t)]
    # [fit_values_offset, fit_cov_offset]= scipy.optimize.curve_fit(lin_func, t, msd)
    fit_values_offset, _ = scipy.optimize.curve_fit(lin_func, t, msd)

    np.warnings.filterwarnings('default')

    # slope = 2 * Diffusion_coefficent
    diff_direct_lin = np.squeeze(fit_values_slope[0]/2)
    
    # var_diff_direct_lin = np.squeeze(fit_cov_slope[0,0]/4)
    # std_diff_direct_lin = np.sqrt(var_diff_direct_lin)
    
    msd_fit_para = [fit_values_slope[0], fit_values_offset[1]]    
    
    # calculate reduced localization accuracy
    red_x = RedXOutOfMsdFit(msd_fit_para[0], msd_fit_para[1], t[0])
    
    # get diffusion error via relative error from eq.(12) in Michalet 2012
    std_diff_direct_lin = diff_direct_lin * nd.Theory.CRLB(amount_frames_lagt1,red_x)
    # * np.sqrt((2+4*np.sqrt(1+2*red_x))/(amount_frames_lagt1-1))


    # plot it if wanted    
    if PlotMsdOverLagtime == True:
        
        if DoRolling == True:
            mean_displ_direct = mean_displ_direct.median()
            
            # fit_values = np.median(fit_values,axis=1)
            
            raise ValueError("Update this!")
            
#        mean_displ_fit_direct_lin=lagt_direct.map(lambda x: x*fit_values[0]+ fit_values[1])

        # choose lagtime which are longer for good plot
        if len(lagt_direct[0]) > len(lagt_direct[1]):
            lagt_direct = np.append(lagt_direct[0], 0)
        else:
            lagt_direct = np.append(lagt_direct[1], 0)

        mean_displ_fit_direct_lin = lagt_direct *msd_fit_para[0]+ msd_fit_para[1]
        nd.visualize.MsdOverLagtime(lagt_direct, mean_displ_direct, mean_displ_fit_direct_lin)
        
    
    return msd_fit_para, diff_direct_lin, std_diff_direct_lin



def DiffusionToDiameter(diffusion, UseHindranceFac = 0, fibre_diameter_nm = None, temp_water = 295, visc_water = 9.5e-16, DoRolling = False):
    """ convert diffusion value to hydrodynamic diameter 
    (optionally) with taking the hindrance factor into account
    """
    const_Boltz = scipy.constants.Boltzmann 
    
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
            nd.logger.warning("Stationary particle still inside")
#            sys.exit("Here is something wrong: The diameter is calculated to be larger than the core diameter. \
#                     Possible Reasons: \
#                     \n 1 - drift correctes motion instead of drift because to few particles inside. \
#                     \n 2 - stationary particle remains, which seems very big because it diffuses to little. ")
#            
    else:
        # MN 202012: I prefer to switch this off - particularly for simulations
        # print("WARNING: Hindrance factor ignored. You just need to have the fiber diameter!")   
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
        H,Kd = hindrance_fac(fibre_diameter_nm,diam_direct_lin_corr)

        corr_visc = 1 / Kd
        diam_direct_lin_corr = diam_direct_lin / corr_visc # diameter of each particle
        
        # this steps helps converging. Otherwise it might jump from 1/10 to 10 to 1/10 ...
        diam_direct_lin_corr = np.sqrt(diam_direct_lin_corr * diam_direct_lin_corr_old)
#        print(my_iter, 'diamter:',diam_direct_lin,'corr_visc:',corr_visc,'corr diameter:',diam_direct_lin_corr)
        
        if my_iter > 100:
            nd.logger.debug("Iteration does not converge. Abort !!!")
            bp()
            input("PRESS ENTER TO CONTINUE.")
            diam_direct_lin_corr = diam_direct_lin_corr_old

    if DoPrint == True:
#        print("After iteration %d: Starting Diameter: %.1f nm; corr. viscosity: %.3f; corr. diameter: %.2nmf" % (my_iter, round(diam_direct_lin,1), round(corr_visc,3), round(diam_direct_lin_corr,2)))
        nd.logger.debug("Starting Diameter: %.1fnm; hindrance factor: %.3f; Corrected Diameter: %.2fnm", round(diam_direct_lin,1), round(corr_visc,3), round(diam_direct_lin_corr,2))


    return diam_direct_lin_corr



def ConcludeResultsMain(settings, eval_tm, sizes_df_lin, diff_direct_lin, traj_length, lagtimes_max, amount_frames_lagt1, stat_sign, msd_fit_para, DoRolling = False, t_beforeDrift=None):
    """ organize the results and write them in one large pandas.DataFrame
    """
    
    mean_raw_mass = eval_tm["raw_mass"].mean()
    UseHindranceFac = settings["MSD"]["EstimateHindranceFactor"]
    temp_water = settings["Exp"]["Temperature"]
    
    settings, visc_water = GetVisc(settings)
    
    #fibre_diameter_nm = GetFiberDiameter(settings)
    fibre_diameter_nm = settings["Fiber"]["TubeDiameter_nm"]
    
    # get parameters of the trajectory to analyze it
    start_frame, mean_mass, mean_size, mean_ecc, mean_signal, mean_raw_mass, mean_ep, max_step, true_particle, start_x, start_y = GetParameterOfTraj(eval_tm, t_beforeDrift = t_beforeDrift)
    
    particleid = eval_tm.particle.unique()
    
    # red_x = ReducedLocalPrecision(settings, mean_raw_mass, diff_direct_lin)
    lagtime = 1/settings["Exp"]["fps"]
    
    red_x = RedXOutOfMsdFit(msd_fit_para[0], msd_fit_para[1], lagtime)
                    
#     get the fit error if switches on (and working)
    rel_error_diff, diff_std = DiffusionError(traj_length, red_x, diff_direct_lin, lagtimes_max)

    diameter = DiffusionToDiameter(diff_direct_lin, UseHindranceFac, fibre_diameter_nm, 
                                   temp_water, visc_water, DoRolling = DoRolling)

    if DoRolling == False:
        sizes_df_lin = ConcludeResults(sizes_df_lin, diff_direct_lin, diff_std, diameter, \
                           particleid, traj_length, amount_frames_lagt1, start_frame, \
                           mean_mass, mean_size, mean_ecc, mean_signal, mean_raw_mass, mean_ep, \
                           red_x, max_step, true_particle, stat_sign = stat_sign,
                           start_x=start_x, start_y=start_y)
    else:
        nd.logger.error("This is not implemented yet.")

    return sizes_df_lin



def ConcludeResults(sizes_df_lin, diff_direct_lin, diff_std, diameter,
                    particleid, traj_length, amount_frames_lagt1, start_frame,
                    mean_mass, mean_size, mean_ecc, mean_signal, mean_raw_mass, mean_ep,
                    red_x, max_step, true_particle, stat_sign = None, start_x=0,start_y=0):
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
                                                          'rawmass': [mean_raw_mass],
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



def ConcludeResultsRolling(sizes_df_lin_rolling, diff_direct_lin_rolling, diff_std_rolling, diameter_rolling, 
                    particleid, traj_length, amount_frames_lagt1, start_frame,
                    mean_mass, mean_size, mean_ecc, mean_signal, mean_raw_mass, mean_ep,
                    red_x, max_step):
    
    # Storing results in df:
    new_panda = pd.DataFrame(data={'particle': particleid,
                                  'frame': np.arange(start_frame,start_frame + len(diff_direct_lin_rolling), dtype = int),
                                  'diffusion': diff_direct_lin_rolling,
                                  'diffusion std': diff_std_rolling,
                                  'diameter': diameter_rolling,
                                  'ep': mean_ep,
                                  'red_x' : red_x, 
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



def AdjustMSDPlot(MSD_fit_Show):
    if MSD_fit_Show == True:    
        # get the axis from the msd plot right
        ax_msd = plt.gca()
    
        #get maximum value of lagtime and msd
        y_min, y_max, x_min, x_max = GetLimitOfPlottedLineData(ax_msd)
        
        ax_msd.set_xlim([0, 1.1*x_max])
        ax_msd.set_ylim([0, 1.1*y_max])


        #set limits according to smallest value
        ax = plt.gca()
        x_max = 0
        y_max = 0
        y_min = 0
        
        for ii,ax_ii in enumerate(ax.lines):
            # print(ii)
            x_max_loop = np.max(ax_ii.get_xdata())
            y_max_loop = np.max(ax_ii.get_ydata())
            y_min_loop = np.min(ax_ii.get_ydata())
                            
            if x_max < x_max_loop: x_max = x_max_loop
            if y_max < y_max_loop: y_max = y_max_loop
            if y_min > y_min_loop: y_min = y_min_loop
    
        plt.xlim(0,x_max * 1.1)
        plt.ylim(y_min,y_max * 1.1)

    
    
def ReducedLocalPrecision(settings, raw_mass, diffusion, DoRolling = False):
    """ calculate reduced square localization error from experimental parameters
    
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4917385/#FD13
    Eq. 13
    """
    
#    local_precision_um = ep * settings["Exp"]["Microns_per_pixel"]
    lagtime_s = 1/settings["Exp"]["fps"]
    exposure_time_s = settings["Exp"]["ExposureTime"]
    
    NA = settings["Exp"]["NA"]
    lambda_nm = settings["Exp"]["lambda"]
    gain = settings["Exp"]["gain_corr"]
    
    # rayleigh_nm = 2 * 0.61 * lambda_nm / NA
    # rayleigh_um = rayleigh_nm / 1000
    
    sigma_um = nd.Theory.SigmaPSF(NA, lambda_nm / 1000)
    
    # we use rayleigh as resolution of the system
    # 2* because it is coherent
    # not sure here
    if gain == "unknown":
        red_x = "gain missing"
        
    else:
#        num_photons = raw_mass / gain
        num_photons = raw_mass * gain #gain in photoelectron/ADU
        
        # old with rayleigh limit
        # static_local_precision_um = rayleigh_um / np.power(num_photons ,1/2)

    #    red_x = np.power(local_precision_um,2) / (diffusion * lagtime_s) \
    #    * (1 + (diffusion * exposure_time_s / np.power(rayleigh_um,2))) \
    #    - 1/3 * (exposure_time_s / lagtime_s)

    
        static_local_precision_um = sigma_um / np.power(num_photons ,1/2)

        # Eq. 13:
        red_x = np.power(static_local_precision_um,2) / (diffusion * lagtime_s) \
        * (1 + (diffusion * exposure_time_s / np.power(sigma_um,2))) \
        - 1/3 * (exposure_time_s / lagtime_s)
    
        
        # if red_x < 0:
        #     red_x = 0
    
    
    return red_x
    


def DiffusionError(traj_length, red_x, diffusion, lagtimes_max, DoRolling = False):
    """ ... """    
    if red_x == "gain missing":
        red_x = 0 # assume the ideal case
        # MN: I think this is okay since it's meant to estimate the MIN. error
        
    # rel. error is well approximated by CRLB for optimized least square fit 
    rel_error = nd.Theory.CRLB(traj_length, red_x)
    
    # old version:
    # # Qian 1991 / Foerster2019 ARHCF-paper
    # rel_error = np.sqrt((2*lagtimes_max) / (3*(traj_length-lagtimes_max))) 
  
    diffusion_std = diffusion * rel_error
    
    return rel_error, diffusion_std



def OptimizeTrajLenght(t6_final, ParameterJsonFile, obj_all, microns_per_pixel = None, 
                       frames_per_second = None, amount_summands = None, amount_lagtimes = None, 
                       amount_lagtimes_auto = None, Histogramm_Show = True, MSD_fit_Show = False, 
                       EvalOnlyLongestTraj = 0):
    """ ... """
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
    


def ContinousIndexingTrajectory(t):
    min_frame = t.index.values.min() # frame that particle's trajectory begins
    max_frame = t.index.values.max() # frame that particle's trajectory ends
    
    new_index = np.linspace(min_frame,max_frame, (max_frame - min_frame + 1))
    
    t = t.reindex(index=new_index)
    
    return t



def InvDiameter(sizes_df_lin, settings, useCRLB=True):
    sys.exit("InvDiameter has moved to statistics.py! Please change if you see this")


def StatisticOneParticle(sizes):
    sys.exit("StatisticOneParticle has moved to statistics.py! Please change if you see this")
    
    
def StatisticDistribution(sizes_df_lin, num_dist_max=10, 
                          weighting=True, showICplot=False, useAIC=True):
    sys.exit("StatisticDistribution has moved to statistics.py! Please change if you see this")
    
    

def Main(t6_final, ParameterJsonFile, obj_all, microns_per_pixel = None, 
         frames_per_second = None, amount_summands = None, amount_lagtimes = None, 
         amount_lagtimes_auto = None, Histogramm_Show = True, MSD_fit_Show = False, 
         EvalOnlyLongestTraj = 0, Max_traj_length = None, yEval = False, 
         processOutput=True, t_beforeDrift=None):
    
    
    nd.logger.warning("This is an old function. Call Main2 instead...")
    sizes_df_lin, sizes_df_lin_rolling, any_successful_check = Main2(t6_final, ParameterJsonFile, MSD_fit_Show = False, yEval = False, processOutput = True, t_beforeDrift = None)
    
    
    
    # """ calculate diameters of individual particles via mean squared displacement analysis
    
    # Procedure:
    # 0.) filter trajectories (optional)
    # 1.) set the amount of lag-times (globally or individually)
    # 2.) construct matrix with squared displacement values for all defined lag-times 
    #     (for each trajectory)
    # 3.) check if displacement values are normal distributed (Kolmogorow-Smirnow test)
    # 4.) calculate mean of the squared displacement values ...
    
    # # [documentation still under construction...]
    # # 3.) calculate mean and variance of lag-times for each particle
    # # 4.) regress each particle individually linearly: MSD per lag-time 
    # # 5.) slope of regression used to calculate size of particle
    # # 6.) standard error of lagtime=1 used as error on slope
    

    # Parameters
    # ----------
    # t6_final : pandas.DataFrame
    #     Trajectory data.
    # ParameterJsonFile : path
    #     Location of the json parameter file.
    # obj_all : pandas.DataFrame
    #     Object location data.
    # microns_per_pixel : float, optional
    #     Pixel to micrometer conversion number (usually contained in json parameter file). 
    #     The default is None.
    # frames_per_second : float, optional
    #     Framerate at which the trajectories were taken. (usually contained in json parameter file)
    #     The default is None.
    # amount_summands : int, optional
    #     If the MSD for a given lag-time is calculated by less than this amount of summands, 
    #     the corresponding lag-time is disregarded in the fit. The default is None.
    # amount_lagtimes : int, optional
    #     Number of lag-times to be considered.
    #     If the particle has less than this amount of valid lag-times to fit, 
    #     the particle is not taken into account. The default is None.
    # amount_lagtimes_auto : 0, 1 or None, optional
    #     DESCRIPTION. The default is None.
    # Histogramm_Show : bool, optional
    #     DESCRIPTION. The default is True.
    # MSD_fit_Show : bool, optional
    #     DESCRIPTION. The default is False.
    # EvalOnlyLongestTraj : 0 or 1, optional
    #     If set to 1, only the longest trajectory in t6_final is evaluated. The default is 0.
    # Max_traj_length : int, optional
    #     If longer trajectories are present, they get cut at this value. The default is None.
    # yEval : bool, optional
    #     If set to True, the MSD analysis is done along the transverse (y) axis
    #     instead of along the fiber (x). The default is False.
    # processOutput : bool, optional
    #     Get some more info printout during processing. The default is True.
    # t_beforeDrift : pandas.DataFrame or None, optional
    #     Trajectory data needed to calculate the initial positions "x/y in first frame"
    #     before drift correction. If not given, the values from t6_final are taken and
    #     are usually not identical to the values in the rawdata. The default is None.

    # Returns
    # -------
    # sizes_df_lin : pandas.DataFrame
    #     DESCRIPTION.
    # sizes_df_lin_rolling : pandas.DataFrame
    #     DESCRIPTION.
    # any_successful_check : bool
    #     If Ture, at least one trajectory could be evaluated.

    # """
    # # read the parameters
    # settings = nd.handle_data.ReadJson(ParameterJsonFile)

    # # get parameters
    # temp_water, amount_lagtimes_auto, MSD_fit_Show, MSD_fit_Save, do_rolling = GetSettingsParameters(settings)

    # """ 0.) """
    # # preprocess trajectories if defined in json
    # # (e.g. only use longest or split too long trajectories)
    # t6_final_use, settings = SelectTrajectories(t6_final, settings)

    # # list of particles id
    # particle_list_value = list(t6_final_use.particle.drop_duplicates())

    # #setup return variable
    # sizes_df_lin=pd.DataFrame(columns={'diameter','particle'})       
    # sizes_df_lin_rolling = pd.DataFrame()   
    
    # #boolean for plotting stuff
    # any_successful_check = False
    
    # # #boolean output if msd evaluation contributes to final result
    # # successfull_fit = []


    # # LOOP THROUGH ALL THE PARTICLES
    # for i,particleid in enumerate(particle_list_value):
    #     if processOutput == True:
    #         nd.logger.debug("Particle number: %s",  int(particleid))

    #     # select trajectory to analyze
    #     eval_tm = t6_final_use[t6_final_use.particle==particleid]
    #     if t_beforeDrift is None:
    #         t_bDrift = None
    #     else:
    #         t_bDrift = t_beforeDrift[t_beforeDrift.particle==particleid]
        
    #     """ 1.) """
    #     # define which lagtimes are used to fit the MSD data
    #     # if "auto" than max_counter defines the number of iteration steps
    #     lagtimes_min, lagtimes_max, max_counter = MSDFitLagtimes(settings, amount_lagtimes_auto, eval_tm)
        
    #     # boolean to leave optimization counter of the loop    
    #     OptimizingStatus = "Continue"
    #     counter = 0
    #     error_counter = 0

        
    #     # CALCULATE MSD, FIT IT AND OPTIMIZE PARAMETERS
    #     while OptimizingStatus == "Continue":
            
    #         """ 2.) """
    #         # calculate MSD
    #         nan_tm_sq, amount_frames_lagt1, enough_values, traj_length, nan_tm = \
    #         CalcMSD(eval_tm, settings, lagtimes_min = lagtimes_min, 
    #                 lagtimes_max = lagtimes_max, yEval = yEval)

    #         # just continue if there are enough data points
    #         if enough_values in ["TooShortTraj", "TooManyHoles"]:
    #             OptimizingStatus = "Abort"
                
    #         else:
    #             # open a window to plot MSD into
    #             any_successful_check = CreateNewMSDPlot(any_successful_check, MSD_fit_Show)
                
    #             """ 3.) """
    #             # check if datapoints in first lagtime are normal distributed
    #             traj_has_error, stat_sign, dx = CheckIfTrajectoryHasError(nan_tm, traj_length, MinSignificance = 0.01)
                
    #             # only continue if trajectory is good, otherwise plot the error
    #             if traj_has_error == True:
    #                 OptimizingStatus = "Abort"
    #                 if processOutput == True:
    #                     nd.logger.debug("Trajectory has error. Particle ID: %s", particleid)
    #                     nd.logger.debug("Kolmogorow-Smirnow test significance: ", stat_sign)
    #                 error_counter += 1
                    
    #             else:    
    #                 """ 4.) """
    #                 # calc MSD and fit linear function through it
    #                 msd_fit_para, diff_direct_lin, diff_std = \
    #                 AvgAndFitMSD(nan_tm_sq, settings, lagtimes_min, lagtimes_max, amount_frames_lagt1, DoRolling = False)
                            
    #                 # recalculate fitting range p_min
    #                 if amount_lagtimes_auto == 1:
    #                     lagtimes_max, OptimizingStatus = UpdateP_Min(settings, eval_tm, msd_fit_para, diff_direct_lin, amount_frames_lagt1, lagtimes_max)

    #                 # do it time resolved                    
    #                 if do_rolling == True:
    #                     bp()
    #                     diff_direct_lin_rolling = \
    #                     AvgAndFitMSD(nan_tm_sq, settings, lagtimes_min, lagtimes_max, amount_frames_lagt1, DoRolling = True)                        


    #         if max_counter == 1:
    #             # dont do optimization
    #             OptimizingStatus = "Successful"
            
    #         else:
    #             # decide if the previous loop is executed again
    #             if counter >= max_counter:
    #                 # abort since it does not converge
    #                 nd.logger.debug("Does not converge")
    #                 OptimizingStatus = "Abort"
    #             else:
    #                 # run again till it converges or max number of iterations is exceeded
    #                 counter = counter + 1


    #     if OptimizingStatus == "Successful":
    #         # after the optimization is done -save the result in a large pandas.DataFrame
    #         # summarize
    #         sizes_df_lin = ConcludeResultsMain(settings, eval_tm, sizes_df_lin, diff_direct_lin, traj_length, lagtimes_max, amount_frames_lagt1, stat_sign, msd_fit_para, DoRolling = False, t_beforeDrift=t_bDrift)
    
    #         # plot MSD if wanted
    #         if MSD_fit_Show == True:
    #             AvgAndFitMSD(nan_tm_sq, settings, lagtimes_min, lagtimes_max, 
    #                          amount_frames_lagt1, DoRolling = False, MSD_fit_Show = True)
                       
    #         # do it time resolved                    
    #         if do_rolling == True:
    #             bp()                        
    #             sizes_df_lin_rolling = ConcludeResultsMain(settings, eval_tm, diff_direct_lin_rolling,
    #                                                        diff_direct_lin, traj_length, lagtimes_max,
    #                                                        amount_frames_lagt1, stat_sign, msd_fit_para,
    #                                                        DoRolling = True)
    # # ============= here ends the long loop over all trajectories =============
        
    # if len(sizes_df_lin) == 0:
    #     nd.logger.warning("No particle made it to the end!!!")
        
    # else:
    #     AdjustMSDPlot(MSD_fit_Show)
        
    #     sizes_df_lin = sizes_df_lin.set_index('particle')
    
    #     if do_rolling == True:
    #         sizes_df_lin_rolling = sizes_df_lin_rolling.set_index('frame')
    #     else:
    #         sizes_df_lin_rolling = "Undone"
    
    #     if MSD_fit_Save == True:
    #         settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "MSD_Fit", settings, ShowPlot = settings["Plot"]["MSD_fit_Show"])
        
    #     if settings["Plot"]["save_data2csv"] == True:
    #         save_folder_name = settings["Plot"]["SaveFolder"]
    #         save_file_name = "sizes_df_lin"
    #         my_dir_name, entire_path_file, time_string = nd.visualize.CreateFileAndFolderName(save_folder_name, save_file_name, d_type = 'csv')
    #         sizes_df_lin.to_csv(entire_path_file)
    #         nd.logger.info('Data stored in: %s', format(my_dir_name))
        
    #     nd.handle_data.WriteJson(ParameterJsonFile, settings) 

    
    # return sizes_df_lin, sizes_df_lin_rolling, any_successful_check    


# def GetFiberDiameter(settings):
#     """ take or calculate the fiber channel diameter

#     Parameters
#     ----------
#     settings : dict
#         Parameter dictionary defined via parameter json file.

#     Returns
#     -------
#     fibre_diameter_nm : float
#         Diameter of the fiber channel in [nm].
#     """
    
#     if settings["Fiber"]["Shape"] == "round":
#         fibre_diameter_nm = settings["Fiber"]["TubeDiameter_nm"]
    
#     elif settings["Fiber"]["Shape"] == "square":
#         fibre_diameter_nm = settings["Fiber"]["TubeDiameter_nm"] 
#         # in a first approximation at least...
        
#     elif settings["Fiber"]["Shape"] == "hex":
#         #calc out of hex diameters if wanted
#         if settings["Fiber"]["CalcTubeDiameterOutOfSidelength"] == 0:
#             fibre_diameter_nm = settings["Fiber"]["TubeDiameter_nm"]
#         else:
#             side_length_um = settings["Fiber"]["ARHCF_hex_sidelength_um"]
#             diameter_inner_um, diameter_outer_um = nd.handle_data.ARHCF_HexToDiameter(side_length_um)
#             diameter_inner_nm = np.round(1000 * diameter_inner_um,0)
#             fibre_diameter_nm, settings = nd.handle_data.SpecificValueOrSettings(diameter_inner_nm, settings, "Fiber", "TubeDiameter_nm")
#     else:
#         fibre_diameter_nm = None
        
#     return fibre_diameter_nm