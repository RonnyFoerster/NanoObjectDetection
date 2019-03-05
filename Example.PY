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
ParameterJsonFile = r'Z:\Datenauswertung\19_ARHCF\190802\ARHCF_Au_Mona_Harvard.json'



# In[0] - check if the python version and the library are good
nd.CheckSystem

# In[1] - read the json parameter file
settings = nd.handle_data.ReadJson(ParameterJsonFile)



# In[2] - read in the raw data into numpy
rawframes_np = nd.handle_data.ReadData2Numpy(settings)

# choose ROI if wantend
# that is good for optimizing the parameters
if settings["ROI"]["UseROI"] == 1:
    rawframes_ROI, settings = nd.handle_data.UseROI(rawframes_np, settings,
                                     x_min = 25, x_max = 80, y_min = 500, y_max = 600,
                                     frame_min = 0, frame_max = 200)
else:
    rawframes_ROI = rawframes_np
  


# In[2] - standard image preprocessing
rawframes_pre, settings = nd.PreProcessing.Main(rawframes_ROI, settings)



# In[3] - rotate the images if necessary 
# Check if rotated data shall be used or not
rawframes_rot, settings = nd.handle_data.RotImages(rawframes_pre, settings)
   
    

# In[4] - help with the parameters for finding objects 
if settings["Help"]["Help"] == 1:
    settings["Find"]["Minimal bead brightness"] =  42
    obj_first, settings = nd.AdjustSettings.FindSpot(rawframes_rot, settings)

    settings["Find"]["Estimated particle size [px]"] =  (3,3)
    settings = nd.AdjustSettings.SpotSize(rawframes_rot, settings)   
    

# In[5] - find the objects
obj_all, settings = nd.get_trajectorie.batch_np(rawframes_rot, settings, UseLog = False)



# In[6] - identify static objects
# find trajectories of very slow diffusing (maybe stationary) objects
t1_orig_slow_diff, settings = nd.get_trajectorie.link_df(obj_all, settings, SearchFixedParticles = True)

# delete trajectories which are not long enough. stationary objects have long trajcetories and survive the test   
t2_stationary, settings = nd.get_trajectorie.filter_stubs(t1_orig_slow_diff, settings, FixedParticles = True, BeforeDriftCorrection = True)



# In[7] - cut trajectories when a moving particles comes to close to a stationary object
obj_moving, settings = nd.get_trajectorie.RemoveSpotsInNoGoAreas(obj_all, t2_stationary, settings)
   


# In[8] - form trajectories of valid particle positions
t1_orig, settings = nd.get_trajectorie.link_df(obj_moving, settings, SearchFixedParticles = False) 



# In[9] - remove to short trajectories
t2_long, settings = nd.get_trajectorie.filter_stubs(t1_orig, settings, FixedParticles = False, BeforeDriftCorrection = True)
   


# In[10] - identify and close gaps in the trajectory
t3_gapless = nd.get_trajectorie.close_gaps(t2_long)



# In[11] - calculate intensity fluctuations as a sign of wrong assignment
t3_gapless, settings = nd.get_trajectorie.calc_intensity_fluctuations(t3_gapless, settings)



# In[12] - split trajectories if necessary (e.g. to large intensity jumps)
t4_cutted, settings = nd.get_trajectorie.split_traj(t2_long, t3_gapless, settings)



# In[13] - drift correction
t5_no_drift, settings = nd.get_trajectorie.DriftCorrection(t4_cutted, settings, drift_smoothing_frames = 20)


# In[14] - only long trajectories are used in the msd plort in order to get a good fit
t6_final, settings = nd.get_trajectorie.filter_stubs(t5_no_drift, settings, FixedParticles = False, BeforeDriftCorrection = False)



# In[15] - calculate the msd and process to diffusion and diameter
sizes_df_lin, settings = nd.CalcDiameter.Main(t6_final, settings, obj_all, MSD_fit_Show = True)





