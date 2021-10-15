# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:37:39 2021

Store old function that are not used anymore, but maybe are of use later, for debugging, version rollback, etc

@author: foersterronny
"""


def GetVarOfSettings(settings, key, entry):
    """ read the variable inside a dictonary
    
    settings: dict
    key: type of setting
    entry: variable name
    old function - should no be in use anymore
    """
    
    print("nd.handle_data.GetVarOfSettings is an old function, which should not be used anymore. Consider replacing it by settings[key][entry].")
    
    if entry in list(settings[key].keys()):
        # get value
        value = settings[key][entry]
        
    else:
        print("!!! Parameter settings['%s']['%s'] not found !!!" %(key, entry))
        # see if defaul file is available
        if "DefaultParameterJsonFile" in list(settings["Exp"].keys()):
            print('!!! Default File found !!!')
            path_default = settings["Exp"]["DefaultParameterJsonFile"]
            
            settings_default = ReadJson(path_default)
            default_value = settings_default[key][entry]
            
            ReadDefault = 'invalid'
            while (ReadDefault in ["y", "n"]) == False:
                # Do until input is useful
                ReadDefault = nd.handle_data.GetInput("Shall default value %s been used?" %default_value, ["y", "n"])
            
            if ReadDefault == "y":
                value = default_value

            else:
                SetYourself = 'invalid'
                while (SetYourself in ["y", "n"]) == False:
                    # Do until input is useful
                    ReadDefault = nd.handle_data.GetInput("Do you wanna set the value yourself?", ["y", "n"])

                 
                if ReadDefault == "y":
                    value = input("Ok. Set the value of settings['%s']['%s']: " %(key, entry))
                else:
                    sys.exit("Well... you had your chances!")
                                               
    return value


def RotImages(rawframes_np, ParameterJsonFile, Do_rotation = None, rot_angle = None):
    """ rotate the rawimage by rot_angle """
    import scipy # it's not necessary to import the full library here => to be shortened
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    Do_rotation = settings["PreProcessing"]["Do_or_apply_data_rotation"]
    
    if Do_rotation == True:
        nd.logger.info('Rotation of rawdata: start removing')
        rot_angle = settings["PreProcessing"]["rot_angle"]

    
        if rawframes_np.ndim == 2:
            im_out = scipy.ndimage.interpolation.rotate(rawframes_np, angle = rot_angle, axes=(1, 0), reshape=True, output=None, order=1, mode='constant', cval=0.0, prefilter=True)
        else:
            im_out = scipy.ndimage.interpolation.rotate(rawframes_np, angle = rot_angle, axes=(1, 2), reshape=True, output=None, order=1, mode='constant', cval=0.0, prefilter=True)

        nd.logger.info("Rotation of rawdata: Applied with an angle of %d" %rot_angle)
        
    else:
        im_out = rawframes_np
        nd.logger.info("Rotation of rawdata: Not Applied")

    nd.handle_data.WriteJson(ParameterJsonFile, settings)

    return im_out


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



def RollingPercentilFilter(rawframes_np, settings, PlotIt = True):
    """
    Old function that removes a percentile/median generates background image from the raw data.
    The background is calculated time-dependent.
    """
    nd.logger.warning('THIS IS AN OLD FUNCTION! SURE YOU WANNA USE IT?')

    nd.logger.info('Remove background by rolling percentile filter: starting...')
    
    rolling_length = ["PreProcessing"]["RollingPercentilFilter_rolling_length"]
    rolling_step = ["PreProcessing"]["RollingPercentilFilter_rolling_step"]
    percentile_filter = ["PreProcessing"]["RollingPercentilFilter_percentile_filter"]   
    
    for i in range(0,len(rawframes_np)-rolling_length,rolling_step):
        my_percentil_value = np.percentile(rawframes_np[i:i+rolling_length], percentile_filter, axis=0)
        rawframes_np[i:i+rolling_step] = rawframes_np[i:i+rolling_step] - my_percentil_value


    if PlotIt == True:
        nd.visualize.Plot2DImage(rawframes_np[0,:,0:500], title = "Raw Image (rolling percentilce subtracted) (x=[0:500])", xlabel = "x [Px]", ylabel = "y [Px]", ShowColorBar = False)

    nd.logger.info('Remove background by rolling percentile filter: ...finished')

    return rawframes_np



def UseSuperSampling(image_in, ParameterJsonFile, fac_xy = None, fac_frame = None):
    """ supersamples the data in x, y and frames by integer numbers
    
    e.g.: fac_frame = 5 means that only every fifth frame is kept
    """
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    DoSimulation = settings["Simulation"]["SimulateData"]
    
    if DoSimulation == 1:
        nd.logger.info("No data. Do a simulation later on")
        image_super = 0
                
    else:
        # supersampling  
        if settings["Subsampling"]["Apply"] == 0:
            nd.logger.info("Supersampling NOT applied")
            fac_xy = 1
            fac_frame = 1
            
        else:
            fac_xy = settings["Subsampling"]['fac_xy']
            fac_frame = settings["Subsampling"]['fac_frame']           
            nd.logger.info("Supersampling IS applied. With factors %s in xy and %s in frame ", fac_xy, fac_frame)
            
        image_super = image_in[::fac_frame, ::fac_xy, ::fac_xy]
            
            
        settings["MSD"]["effective_fps"] = round(settings["Exp"]["fps"] / fac_frame,2)
        settings["MSD"]["effective_Microns_per_pixel"] = round(settings["Exp"]["Microns_per_pixel"] / fac_xy,5)
        
        
        if (settings["Subsampling"]["Save"] == 1) and (settings["Subsampling"]["Apply"] == 1):
            data_folder_name = settings["File"]["data_folder_name"]
            SaveROIToTiffStack(image_super, data_folder_name)
        
        
        WriteJson(ParameterJsonFile, settings)
    
    return image_super



def Correlation(ParameterJsonFile, sizes_df_lin, show_plot = None, save_plot = None):
    """ produces grid plot for the investigation of particle data correlation
    
    for more information, see
    https://stackoverflow.com/questions/30942577/seaborn-correlation-coefficient-on-pairgrid
    """
    import NanoObjectDetection as nd
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    sns.set(style="white")
#    mygraph = sns.pairplot(sizes_df_lin[["diameter", "mass","signal"]])
    
    mycol = list(sizes_df_lin.columns)
    #remove diamter std because it is note done yet
#    mycol = [x for x in mycol if x!= "diameter std"]

    mygraph = sns.pairplot(sizes_df_lin[mycol])
    
    mygraph.map_lower(corrfunc)
    mygraph.map_upper(corrfunc)

    if save_plot == True:
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "Correlation",
                                       settings, data = sizes_df_lin)
        
    plt.show()



def Pearson(ParameterJsonFile, sizes_df_lin, show_plot = None, save_plot = None):
    import NanoObjectDetection as nd
    from NanoObjectDetection.PlotProperties import axis_font
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    sns.set_context("paper", rc={"font.size":int(axis_font["size"]),
                                 "axes.titlesize":8,
                                 "axes.labelsize":100})
                            
    mycorr = sizes_df_lin.corr()
    
    plt.figure()
    sns.set(font_scale=0.6)
#    mygraph = sns.heatmap(np.abs(mycorr), annot=True, vmin=0, vmax=1)
    mygraph = sns.heatmap(mycorr, annot=True, vmin=-1, vmax=1, cmap="bwr", cbar_kws={'label': 'Pearson Coefficent'})
    
#    sns.set(font_scale=1.2)
#    mygraph = sns.clustermap(np.abs(mycorr), annot=True)
    

    if save_plot == True:
        print("Saving the Pearson heat map take a while. No idea why =/")
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "PearsonCoefficent",
                                       settings, data = sizes_df_lin)

    plt.show()
    
    
    
def split_traj(t2_long, t3_gapless, ParameterJsonFile):
    """ wrapper function for 'split_traj_at_high_steps' , returns both the original
    output and one without missing time points
    """

    nd.logger.warning("split_traj is an old function which is not used anymore. Use split_traj_at_high_steps instead.")


    t4_cutted, t4_cutted_no_gaps = split_traj_at_high_steps(t2_long, t3_gapless, ParameterJsonFile)


    # close gaps to have a continous trajectory
    # t4_cutted_no_gaps = nd.get_trajectorie.close_gaps(t4_cutted)

    return t4_cutted, t4_cutted_no_gaps