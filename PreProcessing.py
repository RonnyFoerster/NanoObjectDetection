# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 11:09:16 2019

@author: Ronny Förster und Stefan Weidlich
"""

# In[]
import NanoObjectDetection as nd
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace as bp #debugger
from scipy import ndimage

# In[]
def Main(rawframes_np, ParameterJsonFile):
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    DoSimulation = settings["Simulation"]["SimulateData"]
    
    if DoSimulation == 1:
        print("No data. Do a simulation later on")
        rawframes_np = 0
                
    else:
        # check if constant background shall be removed
        if settings["PreProcessing"]["Remove_CameraOffset"] == 1:
            print('Constant camera background: removed')
            rawframes_np = nd.PreProcessing.SubtractCameraOffset(rawframes_np, settings)
        else:
            print('Constant camera background: not removed')
        
        if settings["PreProcessing"]["Remove_Laserfluctuation"] == 1:
            print('Laser fluctuations: removed')
            rawframes_np = nd.PreProcessing.RemoveLaserfluctuation(rawframes_np, settings)    
            # WARNING - this needs a roughly constant amount of particles in the object!
        else:
            print('Laser fluctuations: not removed')
        
        if settings["PreProcessing"]["Remove_StaticBackground"] == 1:
            print('Static background: removed')
            rawframes_np, static_background = nd.PreProcessing.Remove_StaticBackground(rawframes_np, settings)
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
            
        if settings["PreProcessing"]["EnhanceSNR"] == 1:
            print('Convolve rawframe by PSF to enhance SNR')
            rawframes_np = nd.PreProcessing.ConvolveWithPSF(rawframes_np, settings)
        else:
            print('Image SNR not enhanced by a gaussian average')
            
         
        nd.handle_data.WriteJson(ParameterJsonFile, settings)
        
    return rawframes_np



def SubtractCameraOffset(rawframes_np, settings):
    #That generates one image that holds the minimum-vaues for each pixel of all times
    rawframes_pixelCountOffsetArray = nd.handle_data.min_rawframes(rawframes_np)
        
    # calculates the minimum of all pixel counts. Assumption:
    # this is the total offset
    offsetCount=np.min(rawframes_pixelCountOffsetArray) 
    
    # I'm now subtracting the offset (only the offset, not the full background) from the complete data. Assumption:
    # Whenever there is a change in intensity, e.g. by fluctuations in incoupling,the lightsource etc., this affects mututally background and signal
    rawframes_np=rawframes_np-offsetCount
    
    
    return rawframes_np


def RemoveLaserfluctuation(rawframes_np, settings):
    Laserfluctuation_Show = settings["Plot"]['Laserfluctuation_Show']
    Laserfluctuation_Save = settings["Plot"]['Laserfluctuation_Save']
    
    if Laserfluctuation_Save == True:
        Laserfluctuation_Show = True
    
    
    # Mean-counts of a given frame
    tot_intensity, rel_intensity = nd.handle_data.total_intensity(rawframes_np, Laserfluctuation_Show)
    
    rawframes_np = rawframes_np / rel_intensity[:, None, None]


    if Laserfluctuation_Save == True:
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "Intensity Fluctuations", \
                                       settings, data = rel_intensity, data_header = "Intensity Fluctuations")
        
    return rawframes_np


def Remove_StaticBackground(rawframes_np, settings, Background_Show = False, Background_Save = False):
    Background_Show = settings["Plot"]['Background_Show']
    Background_Save = settings["Plot"]['Background_Save']
    
    if Background_Save == True:
        Background_Show = True
    '''
    Subtracting back-ground and take out points that are constantly bright
    '''

#    static_background = nd.handle_data.min_rawframes(rawframes_np,  display = Background_Show)
    print("remove median")
#    static_background = nd.handle_data.percentile_rawframes(rawframes_np, 50, display = Background_Show)
    
    static_background_max = np.percentile(rawframes_np,50, axis = 0)
    #assumes that each pixel is at least in 50% of the time specimen free and shows bg only
    
    num_frames = rawframes_np.shape[0]
    
    #repmat 
    static_background_max = np.dstack([static_background_max]*num_frames)
    
    #transpose so that 0dim is time again
    static_background_max = np.transpose(static_background_max, (2, 0, 1))    
    
    # select the values that are bg only
    mask_background = rawframes_np > static_background_max
    
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ma.mean.html
    a = np.ma.array(rawframes_np, mask=mask_background)
    
    static_background = a.mean(axis = 0)
    
    # average them
#    static_background = np.mean(static_background, axis = 0)
    
    
    rawframes_np_no_bg = rawframes_np - static_background # Now, I'm subtracting a background, in case there shall be anything left
    
    if Background_Save == True:
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "CameraBackground", settings)
    
    
    return rawframes_np_no_bg, static_background
    

def RollingPercentilFilter(rawframes_np, settings):
    rolling_length = ["PreProcessing"]["RollingPercentilFilter_rolling_length"]
    rolling_step = ["PreProcessing"]["RollingPercentilFilter_rolling_step"]
    percentile_filter = ["PreProcessing"]["RollingPercentilFilter_percentile_filter"]   
    
    for i in range(0,len(rawframes_np)-rolling_length,rolling_step):
        my_percentil_value = np.percentile(rawframes_np[i:i+rolling_length], percentile_filter, axis=0)
        rawframes_np[i:i+rolling_step] = rawframes_np[i:i+rolling_step] - my_percentil_value

    return rawframes_np


def ConvolveWithPSF(rawframes_np, settings):  
    
    if settings["PreProcessing"]["KernelSize"] == 'auto':
        diameter_partice = settings["Find"]['Estimated particle size'][0]
        gauss_kernel_rad = 0.68 * diameter_partice / 2
    else:
        gauss_kernel_rad = settings["PreProcessing"]["KernelSize"]
        
    print("Gauss Kernel in px:", gauss_kernel_rad)
    
    rawframes_filtered = np.real(np.fft.ifftn(ndimage.fourier_gaussian(np.fft.fftn(rawframes_np, axes = (1,2)), sigma=[0,gauss_kernel_rad  ,gauss_kernel_rad]), axes = (1,2)))
    

    return rawframes_filtered




