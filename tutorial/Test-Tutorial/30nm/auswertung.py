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
ParameterJsonFile = r'C:\Users\foersterronny\Desktop\Test-Tutorial\30nm\parameter.json'


#%% check if the python version and the library are good
nd.CheckSystem.CheckAll(ParameterJsonFile)
settings = nd.handle_data.ReadJson(ParameterJsonFile)

#%% read in the raw data to numpy
rawframes_np = nd.handle_data.ReadData2Numpy(ParameterJsonFile)


#%% choose ROI (Region of Interest) if wanted
rawframes_super = nd.handle_data.RoiAndSuperSampling(settings, ParameterJsonFile, rawframes_np)


#%% standard image preprocessing
rawframes_pre, static_background = nd.PreProcessing.Main(rawframes_super, ParameterJsonFile)


#%% help with the parameters for finding objects 
settings = nd.handle_data.ReadJson(ParameterJsonFile)
nd.AdjustSettings.Main(rawframes_super, rawframes_pre, ParameterJsonFile)
    

#%% find the objects
obj_all, objects_per_frame = nd.get_trajectorie.FindSpots(rawframes_np, rawframes_pre, ParameterJsonFile)


#%% identify static objects
# find trajectories of very slow diffusing (maybe stationary) objects
traj_fixed = nd.get_trajectorie.Link(obj_all, ParameterJsonFile, SearchFixedParticles = True)


#%% cut trajectories if a moving particle comes too close to a stationary object
obj_moving = nd.get_trajectorie.RemoveSpotsInNoGoAreas(obj_all, traj_fixed, ParameterJsonFile)
 

#%% form trajectories of valid particle positions
traj_moving = nd.get_trajectorie.Link(obj_moving, ParameterJsonFile, SearchFixedParticles = False)


#%% drift correction
traj_no_drift = nd.Drift.Main(traj_moving, ParameterJsonFile, PlotGlobalDrift = True)


#%% only long trajectories are used in the MSD plot in order to get a good fit
traj_final = nd.get_trajectorie.filter_stubs(traj_no_drift, ParameterJsonFile, Mode = "Moving After Drift", PlotErrorIfTestFails = False)


#%% calculate the MSD and process to diffusion and diameter - LONGITUDINAL
sizes_df_lin_x, any_successful_check = nd.CalcDiameter.Main2(traj_final, ParameterJsonFile, MSD_fit_Show = True)

nd.CalcDiameter.SummaryEval(settings, rawframes_pre, obj_moving, traj_moving, traj_no_drift, traj_final, sizes_df_lin_x)

#%% visualize results - LONGITUDINAL
nd.visualize.PlotDiameters(ParameterJsonFile, sizes_df_lin_x, any_successful_check)


#%% repeat for TRANSVERSAL direction
sizes_df_lin_y, any_successful_check = nd.CalcDiameter.Main2(traj_final, ParameterJsonFile, MSD_fit_Show = True, yEval = True)

nd.visualize.PlotDiameters(ParameterJsonFile, sizes_df_lin_y, any_successful_check, yEval = True)


