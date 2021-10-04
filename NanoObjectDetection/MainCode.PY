# In[0]:
# coding: utf-8
"""
Analyzis of Gold-Particle data for ARHCF paper

Created on 20181001 by Stefan Weidlich (SW)
Based on previous skript of Ronny Förster (RF) and SW


******************************************************************************
Importing neccessary libraries
"""
# Standard Libraries
from __future__ import division, unicode_literals, print_function # For compatibility with Python 2 and 3
from importlib import reload # only used for debugging --> reload(package_name)

# Own Library
import NanoObjectDetection as nd

# path of parameter file
ParameterJsonFile = "Z:\\Datenauswertung\\Torsten_Wieduwilt\\190322\\100nist_2000_40fps_0\\100nist_2000_40fps_0.json"


#%% check if the python version and the library are good
nd.CheckSystem.CheckAll()


#%% read in the raw data into numpy
rawframes_np = nd.handle_data.ReadData2Numpy(ParameterJsonFile)


#%% choose ROI if wantend
# ROI (includes a help how to find it)
settings = nd.handle_data.ReadJson(ParameterJsonFile)
if settings["Help"]["ROI"] == 1:
            nd.AdjustSettings.FindROI(rawframes_np)

rawframes_ROI = nd.handle_data.UseROI(rawframes_np, ParameterJsonFile)    

# supersampling  
rawframes_super = nd.handle_data.UseSuperSampling(rawframes_ROI, ParameterJsonFile)    


#%% standard image preprocessing
rawframes_pre = nd.PreProcessing.Main(rawframes_super, ParameterJsonFile)


#%% rotate the images if necessary 
# Check if rotated data shall be used or not
rawframes_rot = nd.handle_data.RotImages(rawframes_pre, ParameterJsonFile)
   
del rawframes_ROI, rawframes_super, rawframes_pre


#%% help with the parameters for finding objects 
settings = nd.handle_data.ReadJson(ParameterJsonFile)

if settings["Help"]["Bead brightness"] == 1:
    obj_first = nd.AdjustSettings.FindSpot(rawframes_rot, ParameterJsonFile)

if settings["Help"]["Bead size"] == 1:
    nd.AdjustSettings.SpotSize(rawframes_rot, ParameterJsonFile)   
    

#%% find the objects
obj_all = nd.get_trajectorie.FindSpots(rawframes_rot, ParameterJsonFile)


#%% identify static objects
# find trajectories of very slow diffusing (maybe stationary) objects
t1_orig_slow_diff = nd.get_trajectorie.link_df(obj_all, ParameterJsonFile, SearchFixedParticles = True)

# delete trajectories which are not long enough. stationary objects have long trajcetories and survive the test   
t2_stationary = nd.get_trajectorie.filter_stubs(t1_orig_slow_diff, ParameterJsonFile, FixedParticles = True, BeforeDriftCorrection = True)



#%% cut trajectories when a moving particles comes to close to a stationary object
obj_moving = nd.get_trajectorie.RemoveSpotsInNoGoAreas(obj_all, t2_stationary, ParameterJsonFile)
   


#%% form trajectories of valid particle positions
t1_orig = nd.get_trajectorie.link_df(obj_moving, ParameterJsonFile, SearchFixedParticles = False) 



#%% remove to short trajectories
t2_long = nd.get_trajectorie.filter_stubs(t1_orig, ParameterJsonFile, FixedParticles = False, BeforeDriftCorrection = True)
   


#%% identify and close gaps in the trajectory
t3_gapless = nd.get_trajectorie.close_gaps(t2_long)



#%% calculate intensity fluctuations as a sign of wrong assignment
t3_gapless = nd.get_trajectorie.calc_intensity_fluctuations(t3_gapless, ParameterJsonFile)



#%% split trajectories if necessary (e.g. to large intensity jumps)
t4_cutted = nd.get_trajectorie.split_traj(t2_long, t3_gapless, ParameterJsonFile)


#%% drift correction
t5_no_drift = nd.Drift.DriftCorrection(t4_cutted, ParameterJsonFile, PlotGlobalDrift = False)


#%% only long trajectories are used in the msd plort in order to get a good fit
t6_final = nd.get_trajectorie.filter_stubs(t5_no_drift, ParameterJsonFile, FixedParticles = False, BeforeDriftCorrection = False)


#%% calculate the msd and process to diffusion and diameter
sizes_df_lin, any_successful_check = nd.CalcDiameter.Main(t6_final, ParameterJsonFile, obj_all, MSD_fit_Show = True)

#sizes_df_lin, any_successful_check = nd.CalcDiameter.OptimizeTrajLenght(t6_final, ParameterJsonFile, obj_all, MSD_fit_Show = True)

#%% visualiz results
nd.visualize.PlotDiameters(ParameterJsonFile, sizes_df_lin, any_successful_check)




