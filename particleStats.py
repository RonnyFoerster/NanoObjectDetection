# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:39:16 2019

@author: nissenmona
"""

# In[0]:
"""
particle statistics and concentration calculation


******************************************************************************
Importing neccessary libraries
"""

import numpy as np # library for array-manipulation
from scipy.spatial import cKDTree
import pandas as pd
import NanoObjectDetection as nd

#import trackpy as tp
#import matplotlib.pyplot as plt

#%% statistics on the found objects --> concentration estimation

#totalframenb = len(rawframes_rot[:,0,0]) # total number of frames in current video
# length of current FoV in um:
#fovlength = settings["Exp"]["Microns_per_pixel"] * len(rawframes_rot[0,0,:]) 


def ParticleCount(obj_matrix,totalframe):
    """ calculate the number of visible particles (= found objects) in all frames 
    
    obj_matrix: DataFrame as created by the nd.get_trajectorie.FindSpots function
    totalframe: overall number of frames in use
    """
    
    obj_per_frame = pd.Series(np.zeros(totalframe)) # initialize DataFrame for results
    
    for fnum in range(totalframe): # look through all frames and count particles present
        obj_per_frame[fnum] = len(obj_matrix[obj_matrix.frame==fnum])
    
    return obj_per_frame
    


def EstimateConcentration(obj_matrix,totalframe,fovlen,ParamJsonFilepath):
    """ calculate particle concentration [1/nL] from particle position data 
    
    obj_matrix: DataFrame as created by the nd.get_trajectorie.FindSpots function
    totalframe: overall number of frames in use
    fovlen:     length of the current field of view (along the fiber; in um)
    ParamJsonFilepath: location of the json parameter file
    """
    
    obj_per_frame = ParticleCount(obj_matrix,totalframe)
    opf_mean =  obj_per_frame.mean()
    
    settings = nd.handle_data.ReadJson(ParamJsonFilepath)
    channelshape = settings["Fiber"]["Shape"]
    channeldiameter_um = 0.001 * settings["Fiber"]["TubeDiameter_nm"]
    
    if channelshape == "round":
        volume = 0.25 * np.pi * fovlen * channeldiameter_um**2
        conc = opf_mean/volume # particles/um³
        
    elif channelshape == "hex":
        # A_hex = 2*sqrt(3)*r²
        volume = 0.5*3**0.5 * channeldiameter_um**2 * fovlen
        conc = opf_mean/volume # particles/um³
        
    elif channelshape == "square":
        volume = fovlen * channeldiameter_um**2
        conc = opf_mean/volume # particles/um³
        
    else:
        print("Invalid/unknown channel shape.")
        conc = "unknown"
    
    # NB: 1000 um³ = 1 nL = 0.001 uL
    return 1000*conc # particles/nL



def NNDistances(xy_positions_singleFrame):
    """ find the nearest neighbor of each object and calculate the distance in between 
    
    (cf. https://stackoverflow.com/questions/12923586/nearest-neighbor-search-python,
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html)
    
    xy_positions_singleFrame: np.array with x and y position values
    """
    
    itemNb = len(xy_positions_singleFrame)
    distances = np.zeros(itemNb)
    IDs = np.zeros(itemNb)
#
#    step2: Create an instance of a cKDTree as such:
    myKdTree = cKDTree(xy_positions_singleFrame)#, leafsize=100)
    #Play with the leafsize to get the fastest result for your dataset
#
#    step 3: Query the cKDTree for the SECOND Nearest Neighbor (within 6 units max.):
    i = 0
    for item in xy_positions_singleFrame:
        distances[i], IDs[i] = myKdTree.query(item, k=[2])#, distance_upper_bound=6)
        i += 1
#    for each item in your array, distances_IDs will be a tuple of the distance between 
#    the two points, and the index of the location of the point in your array
    
    return np.array([distances.ravel(),IDs.ravel()]).transpose() # np.c_[distances,IDs]


## UNDER CONSTRUCTION...
#    
#def NNDistances_allFrames(obj_matrix, trajLinked=False):
#    """ calculate nearest neighbors for a complete data set (output of nd.get_trajectorie.FindSpots)
#    
#    """
#    if trajLinked:
#        xyframe_matrix = obj_matrix[['x','y','frame','particle']].copy()
#    else:
#        xyframe_matrix = obj_matrix[['x','y','frame']].copy()
#    
#    # to be completed...
#    
#    return distances_IDs_allFrames

