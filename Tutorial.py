# coding: utf-8
"""
THIS IS A TUTORIAL FOR THE NANOOBEJCTDETECTION PACKAGE

You can run it all at once. But it is of more use to go through it block by block and always have a look at your json file

It contains a random walk, an ARHCF image and a nanobore (under construction)

For some useful information it is sometime good to lock at the SAME step in ANOTHER example.

INFORMATION:
    
1. Copy the tutorial json file to your system (not in the package folder)

2. Open the json file with python/ spyder, etc

    a.	Change the entry under “File” > “json” to path + name of the json file
    
        i.	E.g:  "C:\\RonnyFoerster\\tutorial_60nm_randomwalk.json"
        
    b.	Change the entry under “Plot” > “SaveFolder” to the path where the images shall be safed
    
        i.	E.g: “C:\\Users\\foersterronny\\Desktop\\TryNanoObjectDetection”
        
    c.	Change the entry under “Plot” > “SaveFolder” to the path where the properties shall be safed
    
        i.	E.g: “C:\\Users\\foersterronny\\Desktop\\TryNanoObjectDetection”
        
    d.	Safe and close
        
    e.	Change “ParameterJsonFile” to path + name of the json file
    
        i.	E.g:  "C:\\RonnyFoerster\\190304_60nm_randomwalk.json"
        
    f.	Run the code

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



def RonnyRandomWalk(): 
    #%% Information
    """
    Here comes the first tutorial - the RANDOM walk 
    The data is first simulated and then analyzed
     """
     
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
    
    
    
    







def MonaARHCF(): 
    #%% Information
    """
    Here comes the Tutorial of the ARHCF fiber
    
    Data acquired by Mona Nissen (Au50_922fps_mainChanOnly_16s_0-5msET.tif). You see just a subimage (rawframes_np[:,:,985:1130]; Au50_922fps_mainChanOnly_16s_0-5msET_roix_985_1130.tif) of the entire data set, due to copyright and limited space on github.

    """
    # path of parameter json file
    ParameterJsonFile = r'C:\ProgramData\Anaconda3\Lib\site-packages\NanoObjectDetection\tutorial\ARHCF_50nm\tutorial_50nm_beads.json'

    #%% check if the python version and the library are good
    """
    Python and Trackpy have a lot of version. The script does not work for the 'old' versions
    Thus they are checked here, if they meet the minimum requirements.
    """
    nd.CheckSystem.CheckAll()
    
    
    #%% read in the raw data into numpy
    """
    Reads in the image
    
    Things to learn/ try:
        The images transfered into the RAM
        Other programm load every image when needed - this can be very slow
    """
    rawframes_np = nd.handle_data.ReadData2Numpy(ParameterJsonFile)
    
    
    #%% choose ROI if wantend
    """
    Choose a ROI in case not the entire image shall be used. This can be very usefull for setting up a new system,
    where you wanna have to processing time short
    
    Things to learn/ try:
        Set the Help --> ROI value to one and see the maximum image of your datastack to localize particles
        Try the supersampling by getting a factor in Subsampling --> fac_frame' or fac_xy
        """
    # ROI (includes a help how to find it)
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    if settings["Help"]["ROI"] == 1:
                nd.AdjustSettings.FindROI(rawframes_np)
    
    rawframes_ROI = nd.handle_data.UseROI(rawframes_np, ParameterJsonFile)    
    
    # supersampling  
    rawframes_super = nd.handle_data.UseSuperSampling(rawframes_ROI, ParameterJsonFile)    
    
    
    #%% standard image preprocessing
    """
    Standard image processing here
    
    Things to learn/ try:
        Play with all parameters in the preprocessing section
        Let it be plotted by changing the values in Plot--> Background_Show or Laserfluctuation_Show to 1
        See that by change from 0 to 1 they are switched on off
        Try the Background_Save value (1) and save your plot in Plot --> SaveFolder
        Save also the json file and the data by changing Plot --> save_json and "save_data2csv" to 1
        Look for it in the explorer
        """

    rawframes_pre = nd.PreProcessing.Main(rawframes_super, ParameterJsonFile)
    
    
    #%% rotate the images if necessary 
    """
    Try the rotation of the image
    
    Here is still a bit of work to do
    
    Things to learn/ try:
        
    """

    # Check if rotated data shall be used or not
    rawframes_rot = nd.handle_data.RotImages(rawframes_pre, ParameterJsonFile)
       
    del rawframes_ROI, rawframes_super, rawframes_pre
    
    
    #%% help with the parameters for finding objects 
    """
    Parameter optimization
    
    Things to learn/ try:
        Try the help for getting the right bead brightness and size 
        Change Help --> Bead brightness or Bead size to 1 and let you help    
    """

    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    if settings["Help"]["Bead brightness"] == 1:
        obj_first = nd.AdjustSettings.FindSpot(rawframes_rot, ParameterJsonFile)
    
    if settings["Help"]["Bead size"] == 1:
        nd.AdjustSettings.SpotSize(rawframes_rot, ParameterJsonFile)   
        

        
    
    #%% find the objects
    """
    Simulate Data points
    
    Things to learn/ try:
        
    """
    
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





def StefanNanoBore():
    print("Stefans stuff")




