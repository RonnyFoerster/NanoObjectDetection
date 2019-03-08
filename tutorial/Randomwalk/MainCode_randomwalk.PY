# In[0]:
# coding: utf-8
"""
THIS IS A TUTORIAL FOR A BROWNIAN MOTION RANDOM WALK
THE DATA IS SIMULATED AND ANALYSED

from Stefan Weidlich and Ronny Förster

******************************************************************************
Importing neccessary libraries

Problem fixing: If you get get the "ModuleNotFoundError" you have to install
the missing package. This can be for example in the Anaconda Promt (run as Admin)
To find out what you have write in there... google is your friend
Running this the first time might take a while till all the packages are installed
"""
# Standard Libraries
from __future__ import division, unicode_literals, print_function # For compatibility with Python 2 and 3
from importlib import reload # only used for debugging --> reload(package_name)

# Own Library
import NanoObjectDetection as nd

# path of parameter json file
ParameterJsonFile = r'C:\ProgramData\Anaconda3\Lib\site-packages\NanoObjectDetection\tutorial\Randomwalk\tutorial_60nm_randomwalk.json'

#%% check if the python version and the library are good
"""
Python and Trackpy have a lot of version. The script does not work for the 'old' versions
Thus they are checked here, if they meet the minimum requirements.
"""
nd.CheckSystem.CheckAll()


#%% find the objects
"""
Simulate Data points

Things to learn/ try:
    
"""

rawframes_rot = None
obj_all = nd.get_trajectorie.FindSpots(rawframes_rot, ParameterJsonFile)



#%% form trajectories of valid particle positions
"""
Link the particles to a trajectory

Things to learn/ try:
    
"""

t1_orig = nd.get_trajectorie.link_df(obj_all, ParameterJsonFile, SearchFixedParticles = False) 


#%% remove to short trajectories
"""
Throw away to short trajectoies

Things to learn/ try:
Change in json and run everything from simulation again
    a1) change: Exp / fps
    --> reduce and more time passes between two measurements linking fails more often
    a2) change: Simulation / Diameter
    --> reduce and particles diffuse faster --> linking fails more often
    a3) change: Simulation / EstimationPrecision
    --> reduce and particles seems to diffuse faster due to unprecise position measurement --> linking fails more often
    b) change: Link / Max displacement --> play with a and b
    --> reduce a --> enhance b to enable linking again
    c) change: Split / Max_traj_length    
 
"""
t2_long = nd.get_trajectorie.filter_stubs(t1_orig, ParameterJsonFile, FixedParticles = False, BeforeDriftCorrection = True)
   


#%% identify and close gaps in the trajectory
"""
Identify and close gaps

Things to learn/ try:
The simulation does not have gaps - ToDo (?)
"""

t3_gapless = nd.get_trajectorie.close_gaps(t2_long)



#%% calculate intensity fluctuations as a sign of wrong assignment
"""
Calculate intensity fluctuations

Things to learn/ try:
The simulation does not have fluctuations - ToDo (?)
"""

t3_gapless = nd.get_trajectorie.calc_intensity_fluctuations(t3_gapless, ParameterJsonFile)



#%% split trajectories if necessary (e.g. to large intensity jumps)
"""
Split doubted trajectories

Things to learn/ try:
The simulation does not have doubts - ToDo - simulate several particles that cross (?)
"""

t4_cutted = nd.get_trajectorie.split_traj(t2_long, t3_gapless, ParameterJsonFile)


#%% drift correction
"""
Drift correction

Things to learn/ try:
The simulation does not have drift - ToDo (?)
"""

t5_no_drift = nd.Drift.DriftCorrection(t4_cutted, ParameterJsonFile, PlotGlobalDrift = False)


#%% only long trajectories are used in the msd plort in order to get a good fit
"""
Remove to short trajectories

Things to learn/ try:
Change in json and run everything from simulation again
    a) change: Simulation / NumberOfFrames
    b) change: Link / Min_tracking_frames
    c) change: Split / Max_traj_length    
"""

t6_final = nd.get_trajectorie.filter_stubs(t5_no_drift, ParameterJsonFile, FixedParticles = False, BeforeDriftCorrection = False)


#%% calculate the msd and process to diffusion and diameter
"""
Remove to short trajectories

Things to learn/ try:
    a) change: Simulation / EstimationPrecision (run from simulation again)
    --> higher error in MSD --> diameter has larger error
    b) change: Split / Max_traj_length
    --> longer trajectorie length leads to better results
"""
sizes_df_lin, any_successful_check = nd.CalcDiameter.Main(t6_final, ParameterJsonFile, obj_all, MSD_fit_Show = True)

#sizes_df_lin, any_successful_check = nd.CalcDiameter.OptimizeTrajLenght(t6_final, ParameterJsonFile, obj_all, MSD_fit_Show = True)

#%% visualiz results
"""
Play with all the paramters in the plot part
"""

nd.visualize.PlotDiameters(ParameterJsonFile, sizes_df_lin, any_successful_check)




