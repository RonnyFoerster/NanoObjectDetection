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
from joblib import Parallel, delayed
import multiprocessing

# In[]
def Main(rawframes_np, ParameterJsonFile):
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    DoSimulation = settings["Simulation"]["SimulateData"]
    
    rawframes_np = np.float32(rawframes_np)
    
    if DoSimulation == 1:
        print("No data. Do a simulation later on")
        rawframes_np = 0
                
    else:
        if settings["PreProcessing"]["Remove_Laserfluctuation"] == 1:
            rawframes_np = nd.PreProcessing.RemoveLaserfluctuation(rawframes_np, settings)    
            # WARNING - this needs a roughly constant amount of particles in the object!
        else:
            print('Laser fluctuations: not removed')

        # check if constant background shall be removed
        if settings["PreProcessing"]["Remove_CameraOffset"] == 1:
            rawframes_np = nd.PreProcessing.SubtractCameraOffset(rawframes_np, settings)
        else:
            print('Constant camera background: not removed')
        
        if settings["PreProcessing"]["Remove_StaticBackground"] == 1:            
            rawframes_np, static_background = nd.PreProcessing.Remove_StaticBackground(rawframes_np, settings)            
        else:
            print('Static background: not removed')
      
        
        if settings["PreProcessing"]["RollingPercentilFilter"] == 1:            
            rawframes_np = nd.PreProcessing.RollingPercentilFilter(rawframes_np, settings)            
        else:
            print('Rolling percentil filter: not applied')
        
        
        if settings["PreProcessing"]["ClipNegativeValue"] == 1:
            print('Negative values: start removing')
            print("Ronny does not love clipping.")
            rawframes_np[rawframes_np < 0] = 0
            print('Negative values: removed')
        else:
            print('Negative values: kept')
            
            
        if settings["PreProcessing"]["EnhanceSNR"] == 1:            
            rawframes_np = nd.PreProcessing.ConvolveWithPSF(rawframes_np, settings)            
        else:
            print('Image SNR not enhanced by a gaussian average')
            
         
        nd.handle_data.WriteJson(ParameterJsonFile, settings)
        
    return rawframes_np



def SubtractCameraOffset(rawframes_np, settings):
    print('Constant camera background: start removing')
    
    #That generates one image that holds the minimum-vaues for each pixel of all times
    rawframes_pixelCountOffsetArray = nd.handle_data.min_rawframes(rawframes_np)
        
    # calculates the minimum of all pixel counts. Assumption:
    # this is the total offset
    offsetCount=np.min(rawframes_pixelCountOffsetArray) 
    
    # I'm now subtracting the offset (only the offset, not the full background) from the complete data. Assumption:
    # Whenever there is a change in intensity, e.g. by fluctuations in incoupling,the lightsource etc., this affects mututally background and signal
    rawframes_np=rawframes_np-offsetCount
    
    print('Constant camera background: removed')  
    
    return rawframes_np


def RemoveLaserfluctuation(rawframes_np, settings):
    print('Laser fluctuations: start removing')
    
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
        
    print('Laser fluctuations: removed')
        
    return rawframes_np


def Remove_StaticBackground(rawframes_np, settings, Background_Show = False, Background_Save = False):
    print('Static background: start removing')
    Background_Show = settings["Plot"]['Background_Show']
    Background_Save = settings["Plot"]['Background_Save']
    
    if Background_Save == True:
        Background_Show = True
    '''
    Subtracting back-ground and take out points that are constantly bright
    '''

#    static_background = nd.handle_data.min_rawframes(rawframes_np,  display = Background_Show)
#    print("remove median")
#    static_background = nd.handle_data.percentile_rawframes(rawframes_np, 50, display = Background_Show)


    
    def CalcBackgroundParallel(rawframes_np_loop, num_frames):
        if 1 == 0:
            #old
            #assumes that each pixel is at least in 50% of the time specimen free and shows bg only
            static_background_max = np.median(rawframes_np_loop, axis = 0)
            
            #repmat 
            static_background_max = np.squeeze(np.dstack([static_background_max]*num_frames))
            
            #transpose so that 0dim is time again
            static_background_max = np.transpose(static_background_max, (1, 0))   
            
            mask_background = rawframes_np_loop > static_background_max
            
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ma.mean.html
            a = np.ma.array(rawframes_np_loop, mask=mask_background)
            
            static_background = a.mean(axis = 0)
            
        else:
            min_percentile = int(0.2*rawframes_np_loop.shape[0])
            max_percentile = int(0.8*rawframes_np_loop.shape[0])
            
            static_background = np.mean(np.sort(rawframes_np_loop,axis = 0)[min_percentile:max_percentile,:],axis = 0)

        
        return static_background    

    
    num_cores = multiprocessing.cpu_count()
    
    num_frames = rawframes_np.shape[0]
    num_lines = rawframes_np.shape[1]
    
    inputs = range(num_lines)

    static_background_list = Parallel(n_jobs=num_cores)(delayed(CalcBackgroundParallel)(rawframes_np[:,loop_line,:].copy(), num_frames) for loop_line in inputs)

    static_background = np.asarray(static_background_list)

    rawframes_np_no_bg = rawframes_np - static_background # Now, I'm subtracting a background, in case there shall be        


    if 1 == 0:
        #do it seriel
        print("remove median")
    #    static_background_max = np.percentile(rawframes_np,50, axis = 0)
        static_background_max = np.median(rawframes_np, axis = 0)
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
    
    
    print('Static background: removed')
    
    return rawframes_np_no_bg, static_background
    

def RollingPercentilFilter(rawframes_np, settings):
    print('Rolling percentil filter: start applying')
    
    rolling_length = ["PreProcessing"]["RollingPercentilFilter_rolling_length"]
    rolling_step = ["PreProcessing"]["RollingPercentilFilter_rolling_step"]
    percentile_filter = ["PreProcessing"]["RollingPercentilFilter_percentile_filter"]   
    
    for i in range(0,len(rawframes_np)-rolling_length,rolling_step):
        my_percentil_value = np.percentile(rawframes_np[i:i+rolling_length], percentile_filter, axis=0)
        rawframes_np[i:i+rolling_step] = rawframes_np[i:i+rolling_step] - my_percentil_value

    print('Rolling percentil filter: applied')

    return rawframes_np



def EstimageSigmaPSF(settings):
    #estimate best sigma
    #https://en.wikipedia.org/wiki/Numerical_aperture
    NA = settings["Exp"]["NA"]
    n  = 1
    
    # fnumber
    N = 1/(2*np.tan(np.arcsin(NA / n)))
    
    # approx PSF by gaussian
    # https://en.wikipedia.org/wiki/Airy_disk
    lambda_nm = settings["Exp"]["lambda"]
    sigma_nm = 0.45 * lambda_nm * N
    sigma_um = sigma_nm / 1000
    sigma_px = sigma_um / settings["Exp"]["Microns_per_pixel"]
    
    
    # old
#    #        diameter_partice = settings["Find"]['Estimated particle size'][0]
##        gauss_kernel_rad = 0.68 * diameter_partice / 2
#        rayleigh_nm = 0.61 * settings["Exp"]["lambda"] / settings["Exp"]["NA"]
#        rayleigh_um = rayleigh_nm / 1000
#        rayleigh_px = rayleigh_um / settings["Exp"]["Microns_per_pixel"]
    
    return sigma_px


def ConvolveWithPSF(rawframes_np, settings):  
    print('Convolve rawframe by PSF to enhance SNR: start removing')
    
    if settings["PreProcessing"]["KernelSize"] == 'auto':        
        gauss_kernel_rad = EstimageSigmaPSF(settings)
    else:
        gauss_kernel_rad = settings["PreProcessing"]["KernelSize"]
        
    print("Gauss Kernel in px:", gauss_kernel_rad)

    if 1==0:
        print("Do it seriell")
        #get number of frames for iteration    
        num_frames = rawframes_np.shape[0]   
        rawframes_filtered = rawframes_np.copy()
        
        print("Do FT of:", num_frames, "frames. That might take a while")
        for loop_frames in range(num_frames):
    #        print("number of current frame", loop_frames, "of:", num_frames, "\r")
            rawframes_filtered[loop_frames,:,:] = np.real(np.fft.ifft2(ndimage.fourier_gaussian(np.fft.fft2(rawframes_np[loop_frames,:,:]), sigma=[gauss_kernel_rad  ,gauss_kernel_rad])))
            
    #        rawframes_filtered[loop_frames,:,:] =  np.real(np.fft.ifft(ndimage.fourier_gaussian(np.fft.fft(rawframes_np[loop_frames,:,:]), sigma=[gauss_kernel_rad  ,gauss_kernel_rad])))
    #        
    #        rawframes_filtered[loop_frames,:,:] = np.real(np.fft.ifftn(ndimage.fourier_gaussian(np.fft.fftn(np.squeeze(rawframes_np[loop_frames,:,:]), axes = (0,1)), sigma=gauss_kernel_rad), axes = (0,1)))
            
            if np.mod(loop_frames,100) == 0:
                print("Number of frames done: ", loop_frames)
        
    
    else:
       
        def ApplyPSFKernelParallel(image_frame, gauss_kernel_rad):
            image_frame_filtered = np.real(np.fft.ifft2(ndimage.fourier_gaussian(np.fft.fft2(image_frame), sigma=[gauss_kernel_rad  ,gauss_kernel_rad])))
            
            return image_frame_filtered
        
        if rawframes_np.ndim == 3:
            print("Do it parallel")
            num_cores = multiprocessing.cpu_count()
            
            num_frames = rawframes_np.shape[0]
            
            inputs = range(num_frames)
        
        #    rawframes_filtered_list = Parallel(n_jobs=num_cores)(delayed(ApplyPSFKernelParallel)(np.squeeze(rawframes_np[loop_frame,:,:]).copy(), gauss_kernel_rad) for loop_frame in inputs)

            rawframes_filtered_list = Parallel(n_jobs=num_cores)(delayed(ApplyPSFKernelParallel)(rawframes_np[loop_frame,:,:].copy(), gauss_kernel_rad) for loop_frame in inputs)
        
        
            rawframes_filtered = np.asarray(rawframes_filtered_list)
            
            print("Parallel finished")
            
        else:
            rawframes_filtered = ApplyPSFKernelParallel(rawframes_np, gauss_kernel_rad)
        
    

    
#    rawframes_filtered = np.real(np.fft.ifftn(ndimage.fourier_gaussian(np.fft.fftn(rawframes_np, axes = (1,2)), sigma=[0,gauss_kernel_rad  ,gauss_kernel_rad]), axes = (1,2)))
    
#    import cv2
#    
#    rawframes_filtered = cv2.GaussianBlur(rawframes_np, (0,0), gauss_kernel_rad)

    print('Convolve rawframe by PSF to enhance SNR: removed')

    return rawframes_filtered




