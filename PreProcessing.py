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
import scipy 


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

    
    # required float instead of int and make negative value (bg subtraction) possible
    rawframes_np = np.float32(rawframes_np)
    
    if DoSimulation == 1:
        nd.logger.info("No data. Do a simulation later on")
        rawframes_np = 0
                
    else:
        # 1 - LASER FLUCTUATION
        if settings["PreProcessing"]["Remove_Laserfluctuation"] == 1:
            rawframes_np = RemoveLaserfluctuation(rawframes_np, settings)
        else:
            nd.logger.info('Laser fluctuations: not removed')


        # 2 - CAMERA OFFSET
        if settings["PreProcessing"]["Remove_CameraOffset"] == 1:
            rawframes_np = SubtractCameraOffset(rawframes_np, settings)
        else:
            nd.logger.info('Constant camera background: not removed')
    
        
        # 3 - BACKGROUND (BG)
        if settings["PreProcessing"]["Remove_StaticBackground"] == 1:            
            rawframes_np, static_background = Remove_StaticBackground(rawframes_np, settings)  
        else:
            static_background = "NotDone"
            nd.logger.info('Static background: not removed')
    
    
        # 4 - TIME DEPENDENT BACKGROUND
        if settings["PreProcessing"]["RollingPercentilFilter"] == 1:            
            rawframes_np = nd.PreProcessing.RollingPercentilFilter(rawframes_np, settings)    
        else:
            nd.logger.info('Rolling percentil filter: not applied')
      

        # 5 - ENHANCE SNR
        if settings["PreProcessing"]["EnhanceSNR"] == 1:            
            rawframes_np, settings = ConvolveWithPSF_Main(rawframes_np, settings)   
        else:
            nd.logger.info('Image SNR not enhanced by a gaussian average')
 

        # 6 - ROTATE RAW IMAGE
        if settings["PreProcessing"]["Do_or_apply_data_rotation"] == 1:
            rawframes_np = nd.handle_data.RotImages(rawframes_np, ParameterJsonFile)
        else:
            nd.logger.info('Image Rotation: not applied')
            
            
            
        # 7 - CLIP NEGATIVE VALUE
        if settings["PreProcessing"]["ClipNegativeValue"] == 1:
            nd.logger.info('Set negative pixel values to 0: staring...')
            nd.logger.warning("Ronny does not love clipping.")
            rawframes_np[rawframes_np < 0] = 0
            nd.logger.info('Set negative pixel values to 0: ...finished')
        else:
            nd.logger.info("Negative values in image kept")

        
        # Transform to correct (ideal) dtype
        # rawframes_np = IdealDType(rawframes_np, settings)
        
        # Transform to int dtype, because trackpy requires this
        # uint16 or int 16 is good compromise from precision and memory
        rawframes_np, settings = MakeInt16(rawframes_np, settings)


        # save the settings
        nd.handle_data.WriteJson(ParameterJsonFile, settings)

    return rawframes_np, static_background



def MakeInt16(rawframes_np, settings):
    #make a uint16 or int16 dtype out of float
    nd.logger.info("Convert image to integer DType: starting...")
    
    min_value = np.min(rawframes_np) #check later if negative
    max_value = np.max(np.abs(rawframes_np))

    max_uint16 = 2**16-1
    max_int16 = np.floor(max_uint16/2)
    
    if min_value < 0:
        # SIGNED dtype needed
        img_scale_fac = max_int16/max_value
        rawframes_np = np.multiply(rawframes_np, img_scale_fac, out = rawframes_np)
        rawframes_np = np.round(rawframes_np, out = rawframes_np)
        rawframes_np = rawframes_np.astype("int16")
        
        nd.logger.info("DType: int16")
    else :
        # UNSIGNED dtype possible
        img_scale_fac = max_uint16/max_value
        rawframes_np = np.multiply(rawframes_np, img_scale_fac, out = rawframes_np)
        rawframes_np = np.round(rawframes_np, out = rawframes_np)
        rawframes_np = rawframes_np.astype("uint16")
        
        nd.logger.info("DType: uint16")
        
    settings["Exp"]["img-scale-fac"] = img_scale_fac
    
    if settings["Exp"]["gain"] != "unknown":
        settings["Exp"]["gain_corr"] = settings["Exp"]["gain"] / img_scale_fac

    nd.logger.info("Convert image to integer DType ...finished")    

    return rawframes_np, settings



def TryInt16(rawframes_np):
    rawframes_np = np.round(rawframes_np)
    rawframes_np = rawframes_np.astype("uint16")
    
    return rawframes_np
                

def SubtractCameraOffset(rawframes_np, settings, PlotIt = True):
    nd.logger.info('Remove constant camera background: starting...')
    
    #That generates one image that holds the minimum-vaues for each pixel of all times
    rawframes_pixelCountOffsetArray = nd.handle_data.min_rawframes(rawframes_np)
        
    # calculates the minimum of all pixel counts. Assumption:
    # this is the total offset
    offsetCount=np.min(rawframes_pixelCountOffsetArray) 
    
    # I'm now subtracting the offset (only the offset, not the full background) from the complete data. Assumption:
    # Whenever there is a change in intensity, e.g. by fluctuations in incoupling,the lightsource etc., this affects mututally background and signal
    rawframes_np=rawframes_np-offsetCount
    
    nd.logger.info("Camera offset is: %s", offsetCount)
    
    if PlotIt == True:
        # show rawimage
        nd.visualize.Plot2DImage(rawframes_np[0,:,0:500], title = "Raw Image (x=[0:500])", xlabel = "x [Px]", ylabel = "y [Px]", ShowColorBar = False)
    
    
    print('Remove constant camera background: ...finished')  
    
    return rawframes_np



def RemoveLaserfluctuation(rawframes_np, settings, PlotIt = True):
    nd.logger.info('Removing laser fluctuations: starting...')
    nd.logger.warning("WARNING - this needs a roughly constant amount of particles in the object!")
    
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
    
    nd.logger.info('Removing laser fluctuations: ...finished')
    
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
    nd.logger.info('Remove static background: starting...')
    Background_Show = settings["Plot"]['Background_Show']
    Background_Save = settings["Plot"]['Background_Save']
    
    if Background_Save == True:
        Background_Show = True
    '''
    Subtracting back-ground and take out points that are constantly bright
    '''

    # prepare multiprocessing (parallel computing)
    num_cores = multiprocessing.cpu_count()
    nd.logger.info("Do that parallel. Number of cores: %s", num_cores)    
    
    num_frames = rawframes_np.shape[0]
    num_lines = rawframes_np.shape[1]
    
    inputs = range(num_lines)

    num_verbose = nd.handle_data.GetNumberVerbose()

    # execute background estimation in parallel for each line
    static_background_list = Parallel(n_jobs=num_cores, verbose = num_verbose)(delayed(StaticBackground_Mean)(rawframes_np[:,loop_line,:].copy()) for loop_line in inputs)

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
    
    
    nd.logger.info('Remove static background: ...finished')
    
    return rawframes_np_no_bg, static_background
    


def RollingPercentilFilter(rawframes_np, settings, PlotIt = True):
    """
    Old function that removes a percentile/median generates background image from the raw data.
    The background is calculated time-dependent.
    """
    nd.logger.warning('THIS IS AN OLD FUNCTION! SURE YOU WANNA USE IT?')

    nd.logger.info('Remove background by rolling percentile filter: starting...')
    
    rolling_length = ["PreProcessing"]["RollingPercentilFilter_rolling_length"]
    rolling_step = ["PreProcessing"]["RollingPercentilFilter_rolling_step"]
    percentile_filter = ["PreProcessing"]["RollingPercentilFilter_percentile_filter"]   
    
    for i in range(0,len(rawframes_np)-rolling_length,rolling_step):
        my_percentil_value = np.percentile(rawframes_np[i:i+rolling_length], percentile_filter, axis=0)
        rawframes_np[i:i+rolling_step] = rawframes_np[i:i+rolling_step] - my_percentil_value


    if PlotIt == True:
        nd.visualize.Plot2DImage(rawframes_np[0,:,0:500], title = "Raw Image (rolling percentilce subtracted) (x=[0:500])", xlabel = "x [Px]", ylabel = "y [Px]", ShowColorBar = False)

    nd.logger.info('Remove background by rolling percentile filter: ...finished')

    return rawframes_np



def ConvolveWithPSF_2D(image_frame, gauss_kernel_rad):
    """
    convolves a 2d image with a gaussian kernel by multipliying in fourierspace
    """

    PSF_Type = "Gauss"
    if PSF_Type == "Gauss":
        image_frame_filtered = np.real(np.fft.ifft2(ndimage.fourier_gaussian(np.fft.fft2(image_frame), sigma=[gauss_kernel_rad  ,gauss_kernel_rad])))

    
    return image_frame_filtered



def ConvolveWithPSF_3D(rawframes_np, gauss_kernel_rad, DoParallel = True):
    """
    convolves a 3d image with a gaussian kernel. Select seriell or parallel type of 2D execution
    """

    #get number of frames for iteration    
    num_frames = rawframes_np.shape[0] 

    if DoParallel == False:
        nd.logger.warning("OLD METHOD - Do it seriell")

        # create space to save it in
        rawframes_filtered = np.zeros_like(rawframes_np)
        
        nd.logger.debug("Do FT of: %s frames. That might take a while", num_frames)
        for loop_frames in range(num_frames):
            rawframes_filtered[loop_frames,:,:] = ConvolveWithPSF_2D(rawframes_np[loop_frames,:,:], gauss_kernel_rad)
            
            # show the progres every 100 frames            
            if np.mod(loop_frames,100) == 0:
                print("Number of frames done: ", loop_frames)
              
    else:
        nd.logger.info("Do it parallel")
        
        # setup and execute parallel processing of the filterung
        num_cores = multiprocessing.cpu_count()
        nd.logger.info("Number of parallel cores: %s", num_cores)
        
        inputs = range(num_frames)
    
        rawframes_filtered_list = Parallel(n_jobs=num_cores, verbose = 5)(delayed(ConvolveWithPSF_2D)(rawframes_np[loop_frame,:,:].copy(), gauss_kernel_rad) for loop_frame in inputs)
    
        # make it into a proper array
        rawframes_filtered = np.asarray(rawframes_filtered_list)
        
        nd.logger.info("Parallel finished")
        
    
    return rawframes_filtered




def ConvolveWithPSF_Parameter(PSF_Type, settings):
    # set parameters depending on PSF type.
    
    if PSF_Type == "Gauss":
        # Standard Gaussian PSF
        # estimate PSF by experimental settings
        if settings["PreProcessing"]["KernelSize"] == 'auto':        
            settings["PreProcessing"]["KernelSize"] = nd.ParameterEstimation.SigmaPSF(settings)
        
        # set PSF Kernel
        gauss_kernel_rad = settings["PreProcessing"]["KernelSize"]
            
        nd.logger.info("Gauss Kernel in px: %.2f", gauss_kernel_rad)
    
    elif PSF_Type == "Airy":
        # CalcAiryDisc(settings, rawframes_np[0,:,:])
        gauss_kernel_rad = 0    
        nd.logger.error("RF: Implements the AIRY DISC")
        # Calculate the Airy disc for the given experimental parameters
        # this is for data with little or now aberrations
    
     
    json_path = settings["File"]["json"]   
    nd.handle_data.WriteJson(json_path, settings)
     
    return gauss_kernel_rad


def ConvolveWithPSF_Main(rawframes_np, settings, ShowFirstFrame = False, ShowColorBar = True, ExternalSlider = False, PlotIt = True):  
    # Convolve the rawdata by the PSF to reduce noise while maintaining the signal. This is the main function
    
    nd.logger.info('Enhance SNR by convolving image with PSF: starting...')

    #gets the gaussian kernel size for the convolution- other types not implemented yet     
    PSF_Type = "Gauss"
    gauss_kernel_rad = ConvolveWithPSF_Parameter(PSF_Type, settings)

    ImageIs2D = (rawframes_np.ndim == 2)
    # save for later
    if PlotIt == True:
        if ImageIs2D:
            show_im = rawframes_np[:,0:500]
        else:
            show_im = rawframes_np[0,:,0:500]

    if ImageIs2D:
        rawframes_filtered = ConvolveWithPSF_2D(rawframes_np, gauss_kernel_rad)
        
    else:
        #3D case requires efficent looping over the frames
        
        # rawframes_filtered = ConvolveWithPSF_3D(rawframes_np, gauss_kernel_rad, DoParallel = True)
        rawframes_np = ConvolveWithPSF_3D(rawframes_np, gauss_kernel_rad, DoParallel = True)
                

    # Do some plotting if requries  
    if (ExternalSlider == False) and (ShowFirstFrame == True):
        if ImageIs2D:
            disp_data = rawframes_np
        else:
            disp_data = rawframes_np[0,:,:]
            
        nd.visualize.Plot2DImage(disp_data,title = "Filtered image", xlabel = "[Px]", ylabel = "[Px]", ShowColorBar = ShowColorBar)

    if PlotIt == True:
        nd.visualize.Plot2DImage(show_im, title = "Raw Image (convolved by PSF) (x=[0:500])", xlabel = "x [Px]", ylabel = "y [Px]", ShowColorBar = False)

    nd.logger.info('Enhance SNR by convolving image with PSF: ...finished')
    
    return rawframes_np, settings



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



# def IdealDType(rawframes_np, settings):
#     print("select ideal data type - starting")
#     # remove empty bitdepth
#     rawframes_np = rawframes_np / settings["Exp"]["bit-depth-fac"]
        
#     min_value = np.min(rawframes_np) #check later if negative
#     max_value = np.max(np.abs(rawframes_np))

#     max_int8 = np.floor((2**8-1)/2)
#     max_int16 = np.floor((2**16-1)/2)
#     max_int32 = np.floor((2**32-1)/2)

#     max_uint8 = 2**8-1
#     max_uint16 = 2**16-1

#     if min_value < 0:
#         # SIGNED dtype needed
        
#         # check if 8,16 or 32 bit are required
#         if max_value <= max_int8:
#             #scale to maximum value to make most of the data if float is not accepted =/
#             rawframes_np = np.round(rawframes_np / max_value * max_int8)
#             rawframes_np = rawframes_np.astype("int8")
#             nd.logger.info("DType: int8")
            
#         elif max_value <= max_int16:
#             rawframes_np = np.round(rawframes_np / max_value * max_int16)
#             rawframes_np = rawframes_np.astype("int16")
#             nd.logger.info("DType: int16")
            
#         elif max_value <= max_int32:
#             rawframes_np = np.round(rawframes_np / max_value * max_int32)
#             rawframes_np = rawframes_np.astype("int32")
#             nd.logger.info("DType: int32")

#     else:
#         # UNSIGNED dtype possible
#         if max_value <= max_uint8:
#             rawframes_np = np.round(rawframes_np / max_value * max_uint8)
#             rawframes_np = rawframes_np.astype("uint8")
#             nd.logger.info("DType: uint8")
            
#         elif max_value <= max_uint16:
#             rawframes_np = np.round(rawframes_np / max_value * max_uint16)
#             rawframes_np = rawframes_np.astype("uint16")
#             nd.logger.info("DType: uint16")
     
#     print("select ideal data type - finished")
            
#     return rawframes_np