# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 11:09:16 2019

@author: foersterronny
"""
import nanoobject_detection as nd
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace as bp #debugger

def Main(rawframes_np, settings):
    # check if constant background shall be removed
    if settings["PreProcessing"]["Remove_CameraOffset"] == 1:
        print('Constant camera background: removed')
        rawframes_np = nd.PreProcessing.SubtractCameraOffset(rawframes_np, settings)
    else:
        print('Constant camera background: not removed')
    
    if settings["PreProcessing"]["Remove_Laserfluctuation"] == 1:
        print('Laser fluctuations: removed')
        rawframes_np = nd.PreProcessing.RemoveLaserfluctuation(rawframes_np)    
        # WARNING - this needs a roughly constant amount of particles in the object!
    else:
        print('Laser fluctuations: not removed')
    
    if settings["PreProcessing"]["Remove_StaticBackground"] == 1:
        print('Static background: removed')
        rawframes_np, static_background = nd.PreProcessing.Remove_StaticBackground(rawframes_np)
    else:
        print('Static background: not removed')
        
    if settings["PreProcessing"]["RollingPercentilFilter"] == 1:
        print('Rolling percentil filter: applied')
        rawframes_np = nd.PreProcessing.RollingPercentilFilter(rawframes_np, settings)
    else:
        print('Rolling percentil filter: not applied')
    
    if settings["PreProcessing"]["ClipNegativeValue"] == 1:
        print('Negative values: removed')
        print("Ronny does not love clipping.")
        rawframes_np[rawframes_np < 0] = 0
    else:
        print('Negative values: kept')
        
        
    return rawframes_np, settings



def SubtractCameraOffset(rawframes_np, settings, ShowBackground = False):
    
    #That generates one image that holds the minimum-vaues for each pixel of all times
    rawframes_pixelCountOffsetArray = nd.handle_data.min_rawframes(rawframes_np)
        
    # calculates the minimum of all pixel counts. Assumption:
    # this is the total offset
    offsetCount=np.min(rawframes_pixelCountOffsetArray) 
    
    # I'm now subtracting the offset (only the offset, not the full background) from the complete data. Assumption:
    # Whenever there is a change in intensity, e.g. by fluctuations in incoupling,the lightsource etc., this affects mututally background and signal
    rawframes_np=rawframes_np-offsetCount
    
    ShowBackground, settings = nd.handle_data.SpecificValueOrSettings(None,settings,"Plot",'ShowBackground')
    
    if ShowBackground == True:
        # Plot it

        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "CameraBackground", settings)
        
        # close it because its reopened anyway
        plt.close(fig)
    
    return rawframes_np


def RemoveLaserfluctuation(rawframes_np):
    # Mean-counts of a given frame
    tot_intensity, rel_intensity = nd.handle_data.total_intensity(rawframes_np)
    
    rawframes_np = rawframes_np / rel_intensity[:, None, None]
    
    return rawframes_np


def Remove_StaticBackground(rawframes_np):
    '''
    Subtracting back-ground and take out points that are constantly bright
    '''

    static_background = nd.handle_data.min_rawframes(rawframes_np)
    
    rawframes_np = rawframes_np - static_background # Now, I'm subtracting a background, in case there shall be anything left
    
    return rawframes_np, static_background
    

def RollingPercentilFilter(rawframes_np, settings):
    rolling_length = ["PreProcessing"]["RollingPercentilFilter_rolling_length"]
    rolling_step = ["PreProcessing"]["RollingPercentilFilter_rolling_step"]
    percentile_filter = ["PreProcessing"]["RollingPercentilFilter_percentile_filter"]   
    
    for i in range(0,len(rawframes_np)-rolling_length,rolling_step):
        my_percentil_value = np.percentile(rawframes_np[i:i+rolling_length], percentile_filter, axis=0)
        rawframes_np[i:i+rolling_step] = rawframes_np[i:i+rolling_step] - my_percentil_value

    return rawframes_np


