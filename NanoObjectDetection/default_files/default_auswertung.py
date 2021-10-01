# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:39:56 2019

@author: foersterronny
"""

# Standard Libraries
from __future__ import division, unicode_literals, print_function # For compatibility of Python 2 and 3
# from importlib import reload # only used for debugging --> reload(package_name)

# for easy debugging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# own Library
import NanoObjectDetection as nd


#%% path of parameter file
# this must be replaced by any json file
ParameterJsonFile = r'Insert Json Path here, like: Z:\Datenauswertung\Ronny_Foerster\sdl_80nm\parameter.json'


#%% check if the python version and the library are good
nd.CheckSystem.CheckAll(ParameterJsonFile)
settings = nd.handle_data.ReadJson(ParameterJsonFile)

#%% read in the raw data to numpy
rawframes_np = nd.handle_data.ReadData2Numpy(ParameterJsonFile)


#%% choose ROI if wanted
# ROI (includes a help how to find it)
rawframes_super = nd.handle_data.RoiAndSuperSampling(settings, ParameterJsonFile, rawframes_np)


#%% standard image preprocessing
rawframes_pre, static_background = nd.PreProcessing.Main(rawframes_super, ParameterJsonFile)


#%% help with the parameters for finding objects 
settings = nd.handle_data.ReadJson(ParameterJsonFile)

nd.AdjustSettings.Main(rawframes_super, rawframes_pre, ParameterJsonFile)
    

#%% find the objects
obj_all = nd.get_trajectorie.FindSpots(rawframes_np, rawframes_pre, ParameterJsonFile)
objects_per_frame = nd.particleStats.ParticleCount(obj_all, rawframes_pre.shape[0])
print(objects_per_frame.describe())


#%% identify static objects
# find trajectories of very slow diffusing (maybe stationary) objects
t1_orig_slow_diff = nd.get_trajectorie.link_df(obj_all, ParameterJsonFile, SearchFixedParticles = True)

# delete trajectories which are not long enough. Stationary objects have long trajcetories and survive the test   
t2_stationary = nd.get_trajectorie.filter_stubs(t1_orig_slow_diff, ParameterJsonFile, FixedParticles = True, BeforeDriftCorrection = True)


#%% cut trajectories if a moving particle comes too close to a stationary object
obj_moving = nd.get_trajectorie.RemoveSpotsInNoGoAreas(obj_all, t2_stationary, ParameterJsonFile)

 

#%% form trajectories of valid particle positions
t1_orig = nd.get_trajectorie.link_df(obj_moving, ParameterJsonFile, SearchFixedParticles = False) 


#%% remove too short trajectories
t2_long = nd.get_trajectorie.filter_stubs(t1_orig, ParameterJsonFile, FixedParticles = False, BeforeDriftCorrection = True)
   

#%% identify and close gaps in the trajectories
t4_cutted, t4_cutted_no_gaps = nd.get_trajectorie.CutTrajAtIntensityJump_Main(ParameterJsonFile, t2_long)


#%% drift correction
t5_no_drift = nd.Drift.Main(t4_cutted, ParameterJsonFile, PlotGlobalDrift = True)


#%% only long trajectories are used in the MSD plot in order to get a good fit
t6_final = nd.get_trajectorie.filter_stubs(t5_no_drift, ParameterJsonFile, FixedParticles = False, BeforeDriftCorrection = False, PlotErrorIfTestFails = False)


#%% calculate the MSD and process to diffusion and diameter - LONGITUDINAL
sizes_df_lin_x, any_successful_check = nd.CalcDiameter.Main2(t6_final, ParameterJsonFile, MSD_fit_Show = True)

nd.CalcDiameter.SummaryEval(settings, rawframes_pre, obj_moving, t1_orig, t2_long, t5_no_drift, t6_final, sizes_df_lin_x)

#%% visualize results - LONGITUDINAL
nd.visualize.PlotDiameters(ParameterJsonFile, sizes_df_lin_x, any_successful_check)


#%% repeat for TRANSVERSAL direction
sizes_df_lin_y, any_successful_check = nd.CalcDiameter.Main2(t6_final, ParameterJsonFile, MSD_fit_Show = True, yEval = True)

nd.visualize.PlotDiameters(ParameterJsonFile, sizes_df_lin_y, any_successful_check, yEval = True)


