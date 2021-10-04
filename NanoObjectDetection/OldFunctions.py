# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:37:39 2021

Store old function that are not used anymore, but maybe are of use later, for debugging, version rollback, etc

@author: foersterronny
"""

def GetSettingsParameters(settings):
    """
    Get required parameters out of the settings

    Parameters
    ----------
    settings : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.
    Exception
        DESCRIPTION.

    Returns
    -------
    temp_water : TYPE
        DESCRIPTION.
    amount_lagtimes_auto : Boolean
        Defines if the number of lagtimes is set automatically.
    MSD_fit_Show : TYPE
        Show MSD curve.
    MSD_fit_Save : TYPE
        Save MSD curve.
    do_rolling : TYPE
        Rolling means time dependent evaluations.

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


def CheckIfTrajectoryHasError(nan_tm, traj_length, MinSignificance = 0.1, PlotErrorIfTestFails = False, PlotAlways = False, ID='unknown', processOutput = False):
    nd.logger.error("Function name is old. Use KolmogorowSmirnowTest instead")
    
    return KolmogorowSmirnowTest(nan_tm, traj_length, MinSignificance = 0.1, PlotErrorIfTestFails = False, PlotAlways = False, ID='unknown', processOutput = False)



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


def CreateNewMSDPlot(any_successful_check, settings):
    """ for first time call, open a new plot window for the MSD ensemble fit """
    if any_successful_check == False:
        any_successful_check = True
        
        MSD_fit_Show = settings["Plot"]["MSD_fit_Show"]
        MSD_fit_Save = settings["Plot"]["MSD_fit_Save"]
        
        if (MSD_fit_Show == True) or (MSD_fit_Save == True):
            plt.figure("MSD-Plot", clear = True)

    return any_successful_check




def ContinousIndexingTrajectory(t):
    min_frame = t.index.values.min() # frame that particle's trajectory begins
    max_frame = t.index.values.max() # frame that particle's trajectory ends
    
    new_index = np.linspace(min_frame,max_frame, (max_frame - min_frame + 1))
    
    t = t.reindex(index=new_index)
    
    return t


def OptimizeTrajLenght(t6_final, ParameterJsonFile, obj_all, microns_per_pixel = None, frames_per_second = None, amount_summands = None, amount_lagtimes = None,  Histogramm_Show = True, EvalOnlyLongestTraj = 0):
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
        sizes_df_lin_new, any_successful_check = nd.CalcDiameter.Main(t6_final, ParameterJsonFile, obj_all, Max_traj_length = current_max_traj_length)


        if cur_ind_particles == 1:
            sizes_df_lin = sizes_df_lin_new
        else:
            
            sizes_df_lin = sizes_df_lin.append(sizes_df_lin_new, sort=False)
            
        cur_ind_particles = cur_ind_particles * 2


    return sizes_df_lin, any_successful_check