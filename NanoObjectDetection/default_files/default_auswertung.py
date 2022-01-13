# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:39:56 2019

@author: foersterronny
"""

# Standard Libraries
from __future__ import division, unicode_literals, print_function # For compatibility of Python 2 and 3

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


#%% choose ROI (Region of Interest) if wanted
rawframes_super = nd.handle_data.RoiAndSuperSampling(settings, ParameterJsonFile, rawframes_np)


#%% standard image preprocessing
rawframes_pre, rawframes_int, static_background = nd.PreProcessing.Main(rawframes_super, ParameterJsonFile)


#%% help with the parameters for finding objects 
settings = nd.handle_data.ReadJson(ParameterJsonFile)
nd.AdjustSettings.Main(rawframes_super, rawframes_pre, rawframes_int, ParameterJsonFile)
    

#%% find the objects
obj_all = nd.get_trajectorie.FindSpots(rawframes_np, rawframes_int, ParameterJsonFile)


#%% identify static objects
# find trajectories of very slow diffusing (maybe stationary) objects
traj_fixed = nd.get_trajectorie.Link(obj_all, ParameterJsonFile, SearchFixedParticles = True)


#%% cut trajectories if a moving particle comes too close to a stationary object
obj_moving = nd.get_trajectorie.RemoveSpotsInNoGoAreas(obj_all, traj_fixed, ParameterJsonFile)
 

#%% form trajectories of valid particle positions
traj_moving = nd.get_trajectorie.Link(obj_moving, ParameterJsonFile, SearchFixedParticles = False)


#%% drift correction
traj_no_drift = nd.Drift.Main(traj_moving, ParameterJsonFile, PlotGlobalDrift = True)


#%% calculate the MSD and process to diffusion and diameter - LONGITUDINAL
eval_dim = "x"

sizes_df_lin_x, any_successful_check, traj_final_x  = nd.CalcDiameter.Main2(traj_no_drift, ParameterJsonFile, MSD_fit_Show = True, eval_dim = eval_dim)

nd.visualize.PlotDiameters(ParameterJsonFile, sizes_df_lin_x, any_successful_check, eval_dim = eval_dim)


#%% repeat for TRANSVERSAL direction
eval_dim = "y"

sizes_df_lin_y, any_successful_check, traj_final_y = nd.CalcDiameter.Main2(traj_no_drift, ParameterJsonFile, MSD_fit_Show = True, eval_dim = eval_dim)

nd.visualize.PlotDiameters(ParameterJsonFile, sizes_df_lin_y, any_successful_check, eval_dim = eval_dim)


#%% repeat for 2d evaluation
eval_dim = "2d"

sizes_df_lin2, any_successful_check, traj_final2 = nd.CalcDiameter.Main2(traj_no_drift, ParameterJsonFile, MSD_fit_Show = True, eval_dim = eval_dim)

nd.visualize.PlotDiameters(ParameterJsonFile, sizes_df_lin2, any_successful_check, eval_dim = eval_dim)


#%% visualize results - LONGITUDINAL
nd.CalcDiameter.SummaryEval(settings, rawframes_pre, obj_moving, traj_moving, traj_no_drift, traj_final_x, sizes_df_lin_x)


#%% Animation
nd.Animate.AnimateDiameterAndRawData_Big(rawframes_np, sizes_df_lin_x.copy(), traj_final_x, ParameterJsonFile, DoSave = False)