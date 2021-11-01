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
    4 - ENHANCE SNR: Convolve image with PSF- maintains signal, while reducing noise
    

    Parameters
    ----------
    rawframes_np : TYPE
        rawimage.
    ParameterJsonFile : TYPE
        DESCRIPTION.

    Returns
    -------
    rawframes_np : TYPE
        processed raw image.
    static_background : TYPE
        background image.

    """
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    

    # required float instead of int and make negative value (bg subtraction) possible
    nd.logger.info("convert image into float32 for following image processing")
    rawframes_np = np.float32(rawframes_np)
    

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


    # 4 - ENHANCE SNR
    if settings["PreProcessing"]["EnhanceSNR"] == 1:            
        rawframes_np, settings = ConvolveWithPSF_Main(rawframes_np, settings)   
    else:
        nd.logger.info('Image SNR not enhanced by a gaussian average')
 
    
    # Transform to int dtype, because trackpy requires this
    # uint16 or int 16 is good compromise from precision and memory
    rawframes_np, settings = MakeInt16(rawframes_np, settings)


    # save the settings
    nd.handle_data.WriteJson(ParameterJsonFile, settings)

    return rawframes_np, static_background



def MakeInt16(rawframes_np, settings):
    """
    make a uint16 or int16 dtype out of float

    Parameters
    ----------
    rawframes_np : TYPE
        DESCRIPTION.
    settings : TYPE
        DESCRIPTION.

    Returns
    -------
    rawframes_np : TYPE
        DESCRIPTION.
    settings : TYPE
        DESCRIPTION.

    """

    nd.logger.info("Convert image to integer DType: starting...")
    
    min_value = np.min(rawframes_np) #check later if negative
    max_value = np.max(np.abs(rawframes_np))

    max_uint16 = 2**16-1
    max_int16 = np.floor(max_uint16/2)
    
    if min_value < 0:
        # SIGNED dtype needed
        img_scale_fac = max_int16/max_value
        
        # faster than standard multiplication
        rawframes_np = np.multiply(rawframes_np, img_scale_fac, out = rawframes_np)
        
        rawframes_np  = nd.handle_data.MakeIntParallel(rawframes_np, "int16")

        
        nd.logger.info("DType: int16")
    else :
        # UNSIGNED dtype possible
        img_scale_fac = max_uint16/max_value
        rawframes_np = np.multiply(rawframes_np, img_scale_fac, out = rawframes_np)
        
        rawframes_np  = nd.handle_data.MakeIntParallel(rawframes_np, "uint16")
        
        nd.logger.info("DType: uint16")
        

    nd.logger.info("Convert image to integer DType ...finished")    

    return rawframes_np, settings

                

def SubtractCameraOffset(rawframes_np, settings, PlotIt = True):
    """
    Calculates and subtracted the camera offset

    Parameters
    ----------
    rawframes_np : TYPE
        DESCRIPTION.
    settings : TYPE
        DESCRIPTION.
    PlotIt : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    rawframes_np : TYPE
        DESCRIPTION.

    """
    
    nd.logger.info('Remove constant camera background: starting...')
    
    # calculates the minimum of all pixel counts. Assumption:
    # this is the total offset
    offsetCount=np.min(rawframes_np) 
    
    # I'm now subtracting the offset (only the offset, not the full background) from the complete data. Assumption:
    # Whenever there is a change in intensity, e.g. by fluctuations in incoupling,the lightsource etc., this affects mututally background and signal
    rawframes_np = np.subtract(rawframes_np, offsetCount, out = rawframes_np)
    
    nd.logger.info("Camera offset is: %s", offsetCount)
    
    if PlotIt == True:
        # show rawimage
        nd.visualize.Plot2DImage(rawframes_np[0,:,0:500], title = "Raw Image (x=[0:500])", xlabel = "x [Px]", ylabel = "y [Px]", ShowColorBar = False)
    
    
    nd.logger.info('Remove constant camera background: ...finished')  
    
    return rawframes_np



def RemoveLaserfluctuation(rawframes_np, settings, PlotIt = True):
    """
    Calculates and undoes laser fluctuations.
    This is based on the idea that the total brightness in each image should be constant and is only changed by the incoming laser intensity

    Parameters
    ----------
    rawframes_np : TYPE
        DESCRIPTION.
    settings : TYPE
        DESCRIPTION.
    PlotIt : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    rawframes_np : TYPE
        DESCRIPTION.

    """
    
    nd.logger.info('Removing laser fluctuations: starting...')
    nd.logger.warning("WARNING - this needs a roughly constant amount of particles in the object!")
    
    Laserfluctuation_Show = settings["Plot"]['Laserfluctuation']    
    
    # Mean-counts of a given frame
    tot_intensity, rel_intensity = nd.handle_data.total_intensity(rawframes_np, Laserfluctuation_Show)
    
    # normalize rawimages
    rawframes_np = np.divide(rawframes_np, rel_intensity[:, None, None], out = rawframes_np)


    if Laserfluctuation_Show == True:
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "Intensity_Fluctuations", settings, data = rel_intensity, data_header = "Intensity Fluctuations")
    
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

    1 - sort the values in each pixel along time
    2 - grab the middle percentile from 30% to 70%. Do a mean here.
    

    Parameters
    ----------
    rawframes_np_loop : TYPE
        DESCRIPTION.

    Returns
    -------
    static_background : TYPE
        DESCRIPTION.

    """
    
    min_percentile = int(0.3*rawframes_np_loop.shape[0])
    max_percentile = int(0.7*rawframes_np_loop.shape[0])
    
    static_background = np.mean(np.sort(rawframes_np_loop,axis = 0)[min_percentile:max_percentile,:],axis = 0)         
    
    return static_background
 
           

def Remove_StaticBackground(rawframes_np, settings, Background_Show = False, ShowColorBar = True):
    """
    Calculates and removes static background

    Parameters
    ----------
    rawframes_np : TYPE
        DESCRIPTION.
    settings : TYPE
        DESCRIPTION.
    Background_Show : Boolean, optional
        Show the background image. The default is False.
    ShowColorBar : Boolean, optional
        Show a colorbar of the displayed background image. The default is True.

    Returns
    -------
    rawframes_np : TYPE
        DESCRIPTION.
    static_background : TYPE
        DESCRIPTION.

    """
    
    nd.logger.info('Remove static background: starting...')
    Background_Show = settings["Plot"]['Background']

    '''
    Subtracting back-ground and take out points that are constantly bright
    '''

    # calculate the background - this is done in parallel (loop over the lines) due to the large data
    # prepare multiprocessing (parallel computing)
    num_cores = multiprocessing.cpu_count()
    nd.logger.info("Do that parallel. Number of cores: %s", num_cores)    
    
    num_lines = rawframes_np.shape[1]
    
    inputs = range(num_lines)

    num_verbose = nd.handle_data.GetNumberVerbose()

    # execute background estimation in parallel for each line
    static_background_list = Parallel(n_jobs=num_cores, verbose = num_verbose)(delayed(StaticBackground_Mean)(rawframes_np[:,loop_line,:].copy()) for loop_line in inputs)

    # get output back to proper array
    static_background = np.asarray(static_background_list)

    # Remove the background
    rawframes_np = np.subtract(rawframes_np, static_background, out = rawframes_np)


    if Background_Show == True:
        nd.visualize.Plot2DImage(static_background,title = "Background image", xlabel = "[Px]", ylabel = "[Px]", ShowColorBar = ShowColorBar)
                
    
    nd.logger.info('Remove static background: ...finished')
    
    return rawframes_np, static_background
    


def ConvolveWithPSF_2D(image_frame, gauss_kernel_rad):
    """
    convolves a 2d image with a gaussian kernel by multipliying in fourierspace    

    Parameters
    ----------
    image_frame : TYPE
        DESCRIPTION.
    gauss_kernel_rad : TYPE
        gaussian kernel size in px.

    Returns
    -------
    image_frame_filtered : TYPE
        DESCRIPTION.

    """


    image_frame_filtered = np.real(np.fft.ifft2(ndimage.fourier_gaussian(np.fft.fft2(image_frame), sigma=[gauss_kernel_rad, gauss_kernel_rad])))

    
    return image_frame_filtered



def ConvolveWithPSF_3D(rawframes_np, gauss_kernel_rad):
    """
    Convolves a 3d image with a gaussian kernel. Do this in parallel by looping over each frame    

    Parameters
    ----------
    rawframes_np : TYPE
        DESCRIPTION.
    gauss_kernel_rad : TYPE
        gaussian kernel size in px.

    Returns
    -------
    None.

    """

    nd.logger.info("Do it parallel")

    #get number of frames for iteration    
    num_frames = rawframes_np.shape[0] 
    
    # setup and execute parallel processing of the filterung
    num_cores = multiprocessing.cpu_count()
    nd.logger.info("Number of parallel cores: %s", num_cores)
    
    inputs = range(num_frames)

    rawframes_filtered_list = Parallel(n_jobs=num_cores, verbose = 5)(delayed(ConvolveWithPSF_2D)(rawframes_np[loop_frame,:,:].copy(), gauss_kernel_rad) for loop_frame in inputs)
        
    nd.logger.info("Collect parallel results.")

    # make it into a proper array
    rawframes_filtered = np.asarray(rawframes_filtered_list)
    
    nd.logger.info("Parallel finished")
        
    
    return rawframes_filtered



def ConvolveWithPSF_Main(rawframes_np, settings, ShowFirstFrame = False, ShowColorBar = True, PlotIt = True):  
    """
    Convolve the rawdata by the PSF to reduce noise while maintaining the signal. This is the main function

    Parameters
    ----------
    rawframes_np : TYPE
        DESCRIPTION.
    settings : TYPE
        DESCRIPTION.
    ShowFirstFrame : TYPE, optional
        Show Convoled Image. The default is False.
    ShowColorBar : TYPE, optional
        Display Colorbar. The default is True.
    PlotIt : TYPE, optional
        Show Raw image. The default is True.

    Returns
    -------
    rawframes_np : TYPE
        convolved image
    settings : TYPE
        DESCRIPTION.

    """

    
    nd.logger.info('Enhance SNR by convolving image with PSF: starting...')

    #gets the gaussian kernel size for the convolution- other types not implemented yet     
    if settings["PreProcessing"]["KernelSize"] == 'auto':        
        settings["PreProcessing"]["KernelSize"] = nd.ParameterEstimation.SigmaPSF(settings)
    
    gauss_kernel_rad = settings["PreProcessing"]["KernelSize"]
    
    ImageIs2D = (rawframes_np.ndim == 2)

    # save for later
    if PlotIt == True:
        if ImageIs2D:
            show_im = rawframes_np[:,0:500]
        else:
            show_im = rawframes_np[0,:,0:500]

    # here comes the convolution (overwrite existing data to save memory)
    if ImageIs2D:
        rawframes_np = ConvolveWithPSF_2D(rawframes_np, gauss_kernel_rad)
        
    else:
        rawframes_np = ConvolveWithPSF_3D(rawframes_np, gauss_kernel_rad)
                

    # Do some plotting if requries  
    if ShowFirstFrame == True:
        if ImageIs2D:
            disp_data = rawframes_np
        else:
            disp_data = rawframes_np[0,:,:]
            
        nd.visualize.Plot2DImage(disp_data,title = "Filtered image", xlabel = "[Px]", ylabel = "[Px]", ShowColorBar = ShowColorBar)

    if PlotIt == True:
        nd.visualize.Plot2DImage(show_im, title = "Raw Image (convolved by PSF) (x=[0:500])", xlabel = "x [Px]", ylabel = "y [Px]", ShowColorBar = False)

    nd.logger.info('Enhance SNR by convolving image with PSF: ...finished')
    
    return rawframes_np, settings

