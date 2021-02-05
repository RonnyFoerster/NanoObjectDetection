# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 11:09:16 2019

@author: Ronny Förster and Stefan Weidlich
"""


import NanoObjectDetection as nd
import numpy as np
import matplotlib.pyplot as plt
#from pdb import set_trace as bp #debugger
from scipy import ndimage
from joblib import Parallel, delayed
import multiprocessing


def Main(rawframes_np, ParameterJsonFile):
    """
    Main Function of rawimage preprocessing
    1 - LASER FLUCTUATION: reduced by normalizing every image to have the same total intensity
    2 - CAMERA OFFSET: subtracted by defined value
    3 - BACKGROUND (BG): Estimated bg for each pixel by median filter over time
    4 - TIME DEPENDENT BACKGROUND: Similar to 3 but background can change in time
    this should be avoided. If the bg changes in the experiment, that setup should be optimized
    5 - CLIP NEGATIVE VALUE: RF does not like this at all
    6 - ENHANCE SNR: Convolve image with PSF- maintains signal, while reducing noise
    7 - ROTATE RAW IMAGE: Should be avoided experimentally, but if it happened with rare specimen...
    """
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    DoSimulation = settings["Simulation"]["SimulateData"]
    
    rawframes_np = np.float32(rawframes_np)
    
    if DoSimulation == 1:
        print("No data. Do a simulation later on")
        rawframes_np = 0
                
    else:
        # 1 - LASER FLUCTUATION
        if settings["PreProcessing"]["Remove_Laserfluctuation"] == 1:
            rawframes_np = RemoveLaserfluctuation(rawframes_np, settings)
        else:
            print('Laser fluctuations: not removed')


        # 2 - CAMERA OFFSET
        if settings["PreProcessing"]["Remove_CameraOffset"] == 1:
            rawframes_np = SubtractCameraOffset(rawframes_np, settings)
        else:
            print('Constant camera background: not removed')
    
        
        # 3 - BACKGROUND (BG)
        if settings["PreProcessing"]["Remove_StaticBackground"] == 1:            
            rawframes_np, static_background = Remove_StaticBackground(rawframes_np, settings)  
        else:
            static_background = "NotDone"
            print('Static background: not removed')
    
    
        # 4 - TIME DEPENDENT BACKGROUND
        if settings["PreProcessing"]["RollingPercentilFilter"] == 1:            
            rawframes_np = nd.PreProcessing.RollingPercentilFilter(rawframes_np, settings)    
        else:
            print('Rolling percentil filter: not applied')
      

        # 6 - ENHANCE SNR
        if settings["PreProcessing"]["EnhanceSNR"] == 1:            
            rawframes_np, settings = ConvolveWithPSF_Main(rawframes_np, settings)   
        else:
            print('Image SNR not enhanced by a gaussian average')
 

        # 7 - ROTATE RAW IMAGE
        if settings["PreProcessing"]["Do_or_apply_data_rotation"] == 1:
            rawframes_np = nd.handle_data.RotImages(rawframes_np, ParameterJsonFile)
        else:
            print('Image Rotation: not applied')
            
          
        # DTYPE CANT BE FLOAT FOR TRACKPY! Decive int datatype below
        rawframes_np = np.round(rawframes_np)
            
        # 5 - CLIP NEGATIVE VALUE
        if settings["PreProcessing"]["ClipNegativeValue"] == 1:
            print('Negative values: start removing')
            print("Ronny does not love clipping.")
            rawframes_np[rawframes_np < 0] = 0
            rawframes_np = rawframes_np.astype("uint16")
            print('Negative values: removed')
            print('DType : UINT16 - good')
            
        else:
            print('Negative values: kept')
            # check if int16 is enough
            if np.max(np.abs(rawframes_np)) < 32767:
                rawframes_np = rawframes_np.astype("int16")
                print('DType : INT16 - good')
            else:
                rawframes_np = rawframes_np.astype("int32")
                print('DType : INT32 - This is only usefull if you have a true 16 bit image depth.')
            
            
        nd.handle_data.WriteJson(ParameterJsonFile, settings)
        
        # make it a 16bit image again (DTYPE MUST BE INT FOR TRACKPY!)
        # since it can have negative values from the bg subrations make it at int32
        rawframes_np = np.round(rawframes_np)
        rawframes_np = rawframes_np.astype("int32")

        
    return rawframes_np, static_background



def SubtractCameraOffset(rawframes_np, settings, PlotIt = True):
    print('\nConstant camera background: start removing')
    
    #That generates one image that holds the minimum-vaues for each pixel of all times
    rawframes_pixelCountOffsetArray = nd.handle_data.min_rawframes(rawframes_np)
        
    # calculates the minimum of all pixel counts. Assumption:
    # this is the total offset
    offsetCount=np.min(rawframes_pixelCountOffsetArray) 
    
    # I'm now subtracting the offset (only the offset, not the full background) from the complete data. Assumption:
    # Whenever there is a change in intensity, e.g. by fluctuations in incoupling,the lightsource etc., this affects mututally background and signal
    rawframes_np=rawframes_np-offsetCount
    
    print("Camera offset is: ", offsetCount)
    
    if PlotIt == True:
        # show rawimage
        nd.visualize.Plot2DImage(rawframes_np[0,:,0:500], title = "Raw Image (x=[0:500])", xlabel = "x [Px]", ylabel = "y [Px]", ShowColorBar = False)
    
    
    print('Constant camera background: removed')  
    
    return rawframes_np



def RemoveLaserfluctuation(rawframes_np, settings, PlotIt = True):
    print('\nLaser fluctuations: start removing')
    print("WARNING - this needs a roughly constant amount of particles in the object!")
    
    Laserfluctuation_Show = settings["Plot"]['Laserfluctuation_Show']
    Laserfluctuation_Save = settings["Plot"]['Laserfluctuation_Save']
    
    if Laserfluctuation_Save == True:
        Laserfluctuation_Show = True
    
    
    # Mean-counts of a given frame
    tot_intensity, rel_intensity = nd.handle_data.total_intensity(rawframes_np, Laserfluctuation_Show)
    
    rawframes_np = rawframes_np / rel_intensity[:, None, None]


    if Laserfluctuation_Save == True:
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "Intensity_Fluctuations", \
                                       settings, data = rel_intensity, data_header = "Intensity Fluctuations")
    
    if PlotIt == True:
        plt.figure()
        plt.imshow(rawframes_np[0,:,0:350])
    
    print('Laser fluctuations: removed')
    
    return rawframes_np


def StaticBackground_Median(rawframes_np_loop, num_frames):
    """
    assumes that each pixel is at least in 50% of the time specimen free and shows bg only
    However this can introduce artefacts on very black regions, because median give a discrete background. Mean gives a better estimation if preselected
    E.g. [0,0,0,0,0,1,1,1,1] --> median: 0, mean: 0.44
    but  [0,0,0,0,1,1,1,1,1] --> median: 1, mean: 0.55
    """
    
    static_background_max = np.median(rawframes_np_loop, axis = 0)
    
    #repmat in order to subtract 2d background from 3d rawimage
    static_background_max = np.squeeze(np.dstack([static_background_max]*num_frames))
    
    #transpose so that 0dim is time again
    static_background_max = np.transpose(static_background_max, (1, 0))   
    
    mask_background = rawframes_np_loop > static_background_max
    
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ma.mean.html
    a = np.ma.array(rawframes_np_loop, mask=mask_background)
    
    static_background = a.mean(axis = 0)
            
    return static_background


def StaticBackground_Mean(rawframes_np_loop):
    """
    Calculate the background by a combination of mean and median
    E.g. [0,0,0,0,0,1,1,1,1] --> median: 0, mean: 0.44
    but  [0,0,0,0,1,1,1,1,1] --> median: 1, mean: 0.55
    
    1 - sort the values in each pixel
    2 - grab the middle percentile from 30% to 70%. Do a mean here.
    
    """
    
    min_percentile = int(0.3*rawframes_np_loop.shape[0])
    max_percentile = int(0.7*rawframes_np_loop.shape[0])
    
    static_background = np.mean(np.sort(rawframes_np_loop,axis = 0)[min_percentile:max_percentile,:],axis = 0)         
    
    return static_background
 
           

def Remove_StaticBackground(rawframes_np, settings, Background_Show = False, Background_Save = False, ShowColorBar = True, ExternalSlider = False):
    print('\nStatic background: start removing')
    Background_Show = settings["Plot"]['Background_Show']
    Background_Save = settings["Plot"]['Background_Save']
    
    if Background_Save == True:
        Background_Show = True
    '''
    Subtracting back-ground and take out points that are constantly bright
    '''

    # prepare multiprocessing (parallel computing)
    num_cores = multiprocessing.cpu_count()
    print("Do that parallel. Number of cores: ", num_cores)    
    
    num_frames = rawframes_np.shape[0]
    num_lines = rawframes_np.shape[1]
    
    inputs = range(num_lines)

    # execute background estimation in parallel for each line
    static_background_list = Parallel(n_jobs=num_cores, verbose = 5)(delayed(StaticBackground_Mean)(rawframes_np[:,loop_line,:].copy()) for loop_line in inputs)

    # get output back to proper array
    static_background = np.asarray(static_background_list)

    # Remove the background
    rawframes_np_no_bg = rawframes_np - static_background 


    if ExternalSlider == False:
        if Background_Show == True:
            #plt.imshow(static_background)
            nd.visualize.Plot2DImage(static_background,title = "Background image", \
                                     xlabel = "[Px]", ylabel = "[Px]", ShowColorBar = ShowColorBar)
            
        
        if Background_Save == True:
            settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "CameraBackground", settings)
    
    
    print('Static background: removed')
    
    return rawframes_np_no_bg, static_background
    


def RollingPercentilFilter(rawframes_np, settings, PlotIt = True):
    """
    Old function that removes a percentile/median generates background image from the raw data.
    The background is calculated time-dependent.
    """
    print('\n THIS IS AN OLD FUNCTION! SURE YOU WANNA USE IT?')

    print('Rolling percentil filter: start applying')
    
    rolling_length = ["PreProcessing"]["RollingPercentilFilter_rolling_length"]
    rolling_step = ["PreProcessing"]["RollingPercentilFilter_rolling_step"]
    percentile_filter = ["PreProcessing"]["RollingPercentilFilter_percentile_filter"]   
    
    for i in range(0,len(rawframes_np)-rolling_length,rolling_step):
        my_percentil_value = np.percentile(rawframes_np[i:i+rolling_length], percentile_filter, axis=0)
        rawframes_np[i:i+rolling_step] = rawframes_np[i:i+rolling_step] - my_percentil_value


    if PlotIt == True:
        nd.visualize.Plot2DImage(rawframes_np[0,:,0:500], title = "Raw Image (rolling percentilce subtracted) (x=[0:500])", xlabel = "x [Px]", ylabel = "y [Px]", ShowColorBar = False)

    print('Rolling percentil filter: applied')

    return rawframes_np



def ConvolveWithPSF(image_frame, gauss_kernel_rad):
    """
    convolves a 2d image with a gaussian kernel by multipliying in fourierspace
    """
    

    
    PSF_Type = "Gauss"
    if PSF_Type == "Gauss":
        image_frame_filtered = np.real(np.fft.ifft2(ndimage.fourier_gaussian(np.fft.fft2(image_frame), sigma=[gauss_kernel_rad  ,gauss_kernel_rad])))
    
    return image_frame_filtered



def ConvolveWithPSF_Main(rawframes_np, settings, ShowFirstFrame = False, ShowColorBar = True, ExternalSlider = False, PlotIt = True):  
    print('\nConvolve rawframe by PSF to enhance SNR: start removing')
     
    CalcAiryDisc(settings, rawframes_np[0,:,:])
     
    PSF_Type = "Gauss"
    
    # set parameters depending on PSF type.
    if PSF_Type == "Gauss":
        # Standard Gaussian PSF
        # estimate PSF by experimental settings
        if settings["PreProcessing"]["KernelSize"] == 'auto':        
            settings["PreProcessing"]["KernelSize"] = nd.ParameterEstimation.EstimageSigmaPSF(settings)
        
        # set PSF Kernel
        gauss_kernel_rad = settings["PreProcessing"]["KernelSize"]
            
        print("Gauss Kernel in px:", gauss_kernel_rad)
    
    elif PSF_Type == "Airy":
        print("RF: Implements the AIRY DISC")
        # Calculate the Airy disc for the given experimental parameters
        # this is for data with little or now aberrations

    if rawframes_np.ndim == 2:
        rawframes_filtered = ConvolveWithPSF(rawframes_np, gauss_kernel_rad)
        
    else:
        #3D case requires efficent looping over the frames
        
        #get number of frames for iteration    
        num_frames = rawframes_np.shape[0] 
    
        if 1==0:
            print("OLD METHOD - Do it seriell")
    
            # save space to save it in
            rawframes_filtered = np.zeros_like(rawframes_np)
            
            print("Do FT of:", num_frames, "frames. That might take a while")
            for loop_frames in range(num_frames):
                rawframes_filtered[loop_frames,:,:] = ConvolveWithPSF(rawframes_np[loop_frames,:,:], gauss_kernel_rad)
                
                # show the progres every 100 frames            
                if np.mod(loop_frames,100) == 0:
                    print("Number of frames done: ", loop_frames)
                  
        else:
            print("Do it parallel")
            
            # setup and execute parallel processing of the filterung
            num_cores = multiprocessing.cpu_count()
            print("Number of parallel cores: ", num_cores)
            
            inputs = range(num_frames)
        
            rawframes_filtered_list = Parallel(n_jobs=num_cores, verbose = 5)(delayed(ConvolveWithPSF)(rawframes_np[loop_frame,:,:].copy(), gauss_kernel_rad) for loop_frame in inputs)
        
            # make it into a proper array
            rawframes_filtered = np.asarray(rawframes_filtered_list)
            
            print("Parallel finished")
                

    # Do some plotting if requries  
    if ExternalSlider == False:
        if ShowFirstFrame == True:
            if rawframes_filtered.ndim == 2:
                disp_data = rawframes_filtered
            else:
                disp_data = rawframes_filtered[0,:,:]
                
            nd.visualize.Plot2DImage(disp_data,title = "Filtered image", xlabel = "[Px]", ylabel = "[Px]", ShowColorBar = ShowColorBar)

    if PlotIt == True:
        if rawframes_filtered.ndim == 2:
            nd.visualize.Plot2DImage(rawframes_np[:,0:500], title = "Raw Image (convolved by PSF) (x=[0:500])", xlabel = "x [Px]", ylabel = "y [Px]", ShowColorBar = False)
        else:
            nd.visualize.Plot2DImage(rawframes_np[0,:,0:500], title = "Raw Image (convolved by PSF) (x=[0:500])", xlabel = "x [Px]", ylabel = "y [Px]", ShowColorBar = False)


    print('Convolve rawframe by PSF to enhance SNR: removed')

    return rawframes_filtered, settings


def CalcAiryDisc(settings, img):
    import pyotf.otf
    
    wl = settings["Exp"]["lambda"]
    na = settings["Exp"]["NA"]
    ni = settings["Exp"]["n_immersion"]
    res = settings["Exp"]["Microns_per_pixel"] * 1000
    size_x = int(img.shape[0])
    size_y = int(img.shape[1])
    size = int(np.max([size_x, size_y]))
    zres = res
    zsize = 99
    
    args = (wl, na, ni, res, size, zres, zsize)
    # wl, na, ni, res, size, zres=None, zsize=None, vec_corr="none", condition="sine"
    kwargs = dict(vec_corr="none", condition="none")
    psf3d = pyotf.otf.SheppardPSF(*args, **kwargs)
    
    psfi3d = psf3d.PSFi
    
    #get focus slice
    psfi2d = psfi3d[50,:,:]
    
    psfi2d = psfi2d /np.max(psfi2d)

    # cut out psf same size as image
    center = int(np.ceil(size/2))
    
    left_border_x = int(np.ceil(size/2 - size_x/2))
    right_border_x = left_border_x + size_x

    left_border_y = int(np.ceil(size/2 - size_y/2))
    right_border_y = left_border_y + size_y
    
    
    print(center, left_border_x, right_border_x)

    psfi2d_roi = psfi2d[left_border_x : right_border_x, left_border_y : right_border_y]

    return psfi2d_roi