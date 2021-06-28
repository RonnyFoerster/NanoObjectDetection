# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:12:14 2020

@author: foersterronny
"""
import NanoObjectDetection as nd
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace as bp #debugger
from scipy import ndimage
from joblib import Parallel, delayed
import multiprocessing
from scipy.ndimage import label, generate_binary_structure
import trackpy as tp
import scipy.constants
import time

def GaussianKernel(sigma, fac = 6, x_size = None,y_size = None):
    #https://martin-thoma.com/zero-mean-normalized-cross-correlation/
    #https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-79.php

    # get the size of the kernel if not given
    if x_size == None:
        x_size = int(np.ceil(fac*2*sigma))
    
    if y_size == None:
        y_size = int(np.ceil(fac*2*sigma))
    
    #kernel must be odd for symmetrie
    if np.mod(x_size,2) == 0:
        nd.logger.debug("x_size must be odd; x_size + 1")
        x_size = x_size + 1        
    if np.mod(y_size,2) == 0:
        nd.logger.debug("y_size must be odd; y_size + 1")        
        y_size = y_size + 1
      
    # radius of the kernel in px
    x_lim = int((x_size-1)/2)
    y_lim = int((y_size-1)/2)
        
    # calculate the normal distribution
    x, y = np.meshgrid(np.linspace(-x_lim, x_lim, x_size), np.linspace(-y_lim, y_lim, y_size))
    r = np.sqrt(x*x+y*y)    

    g = np.exp(-( r**2 / ( 2.0 * sigma**2 ) ) )    
    
    # normalize to sum 1
    g = g / np.sum(g)
    
    return g



def zncc(img1, img2):
    """ calculate Zero Mean Normalized Cross-Correlation 
    as presented in 
    https://martin-thoma.com/zero-mean-normalized-cross-correlation/, 27.07.2020
    
    img1, img2:     grayscale images (or any matrices) of the same size
    """
    # Zero Mean Normalized Cross-Correlation
    # https://martin-thoma.com/zero-mean-normalized-cross-correlation/, 27.07.2020
    img1_mean = np.mean(img1)
    img1_std  = np.sqrt(np.mean((img1 - img1_mean)**2))
    
    img2_mean = np.mean(img2)
    img2_std  = np.sqrt(np.mean((img2 - img2_mean)**2))
    
    if (img1_std * img2_std) != 0:
        zncc = np.mean((img1 - img1_mean) * (img2 - img2_mean)) / (img1_std * img2_std)
    else:
        zncc = 0
    
#    zncc = np.mean((img1 - img1_mean) * (img2 - avg2))/(img1_std * stdDeviation2)
    
    return zncc



def SigmaPSF(settings, method = "new"):
    #estimate best sigma
    
    if method == "new":
        NA = settings["Exp"]["NA"]
        lambda_nm = settings["Exp"]["lambda"]
        sigma_nm = nd.Theory.SigmaPSF(NA, lambda_nm)
    
    elif method == "old":
        #https://en.wikipedia.org/wiki/Numerical_aperture
        NA = settings["Exp"]["NA"]
        n  = settings["Exp"]["n_immersion"]
            
        # fnumber
        N = 1/(2*np.tan(np.arcsin(NA / n)))
        
        # approx PSF by gaussian
        # https://en.wikipedia.org/wiki/Airy_disk
        lambda_nm = settings["Exp"]["lambda"]
        sigma_nm = 0.45 * lambda_nm * N
        
    else:
        nd.logger.error("Input value of method must be new or old")
    
    
    sigma_um = sigma_nm / 1000
    sigma_px = sigma_um / settings["Exp"]["Microns_per_pixel"]
 
    return sigma_px   



def DiameterForTrackpy(settings):   
    
    NA = settings["Exp"]["NA"]
    lambda_nm = settings["Exp"]["lambda"]
    
    # radius_nm = nd.Theory.RayleighPSF(NA, lambda_nm)
    sigma_nm = nd.Theory.SigmaPSF(NA, lambda_nm)
    
    #2 sigma contains 95% of the photons. Rayleigh ring does not work properly
    radius_nm = 2*sigma_nm 
    
    radius_um = radius_nm / 1000
    radius_px = radius_um / settings["Exp"]["Microns_per_pixel"]
    diameter_px = 2 * radius_px
    
    if settings["PreProcessing"]["EnhanceSNR"] == 1:
        #gauss convolved by gauss of same size to enhance SNR. Result is a gauss with twice the variance
        diameter_px = np.sqrt(2) * diameter_px
    
    # must be int
    diameter_tp = int(np.ceil(diameter_px))
    
    if (diameter_tp%2) == 0:
        # value must be odd for trackpy
        diameter_tp = diameter_tp + 1
        
    nd.logger.debug("2-Sigma intervall (0.95 of the energy) within a radius of : %.2f px", radius_px)
    nd.logger.debug("Diameter for trackpy: %s px", diameter_tp)
    
    
    return diameter_tp



def MinmassAndDiameterMain(img1_raw, img1, ParameterJsonFile, NumShowPlots = 1, DoDiameter = False):

    """
    Estimate the minmass parameter trackpy requires to locate particles
    1 - Make an intensity independent feature finding by a zero-normalized cross correlation (ZNCC)
    This is computational very demanding
    2 - Optimize trackpy parameters such, that it obtains the same result than ZNCC
    showPlots: number of frames that are shown
    """    
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    # define search area (hot pixels need to be avoided if the background is low and the image was SNR enhanced by lowpass filtering)
    # estimate where the channel is
    if settings["PreProcessing"]['EnhanceSNR'] == 1:
        # lowpassed filtered image - only search where background is not zero (water trace)
        mychannel = FindChannel(img1_raw)
    else:
        # search everywhere
        mychannel = np.ones_like(img1_raw[0,:,:])
    
    # select several frames to make the parameter estimation with
    num_frames = settings["Help"]["TryFrames"]
    
    if num_frames == "auto":
        _, num_particles_zncc, _ = FindParticleByZNCC(settings, img1, mychannel, [0], OneFrameOnly=True)
        
        if num_particles_zncc == 0:
            nd.logger.warning("Zero normalized cross correlation (ZNCC) does not find any particle.")
        
        # check how many frames are needed to get 500 particles
        wanted_particles = 200
        num_frames = int(np.ceil(wanted_particles / num_particles_zncc)) + 1
        
        if num_frames > 50: #otherwise computation time is to long
            nd.logger.warning("Not many partiles found in each frame. To limit computation time, number of frames is limited to 20.")
            num_frames = 50
        
        nd.logger.info("Estimated number of tested frames: %i", num_frames)
        
        settings["Help"]["TryFrames"] = num_frames 
    
    use_frames = (np.round(np.linspace(0,img1.shape[0]-1, num_frames))).astype(int)
    
    img_in_zncc = img1[use_frames,:,:]
    
    plt.figure()
    plt.imshow(img_in_zncc[0,:,:])
    plt.title("img in zncc")
    
    # RUN THE PARTICLE FINDING ROUTINE BY ZNCC
    img_zncc, num_particles_zncc, pos_particles = FindParticleByZNCC(settings, img_in_zncc, mychannel, use_frames)

    plt.figure()
    plt.imshow(img_zncc[0,:,:])
    plt.title("zncc")

    # CLIP NEGATIVE VALUE
    img_in_tp = img_in_zncc.copy()
    if settings["PreProcessing"]["ClipNegativeValue"] == 1:
        nd.logger.info('Set negative pixel values to 0: staring...')
        nd.logger.warning("Ronny does not love clipping.")
        img_in_tp[img_in_tp < 0] = 0
        nd.logger.info('Set negative pixel values to 0: ...finished')
    else:
        nd.logger.info("Negative values in image kept")


    settings = OptimizeMinmassInTrackpyMain(settings, img_in_tp, num_particles_zncc, pos_particles, DoDiameter)

    
    # plot the stuff - optionally
    for ii, loop_frames in enumerate(use_frames[:NumShowPlots]):
        use_img1 = img_in_zncc[ii,:,:]
        use_img_zncc = img_zncc[ii,:,:]
        use_pos_particles = pos_particles[pos_particles[:,2] == loop_frames, 0:2]
        
        if len(use_pos_particles) > 0:
            plt.figure()
            PlotImageProcessing(use_img1, use_img_zncc, use_pos_particles)

    nd.handle_data.WriteJson(ParameterJsonFile, settings)  

    return settings



def FindParticleByZNCC(settings, img_in_zncc, search_area, use_frames, OneFrameOnly = True):
    """
    Find the particles by a zero-normalized cross correlation
    This function looks for pattern not for intensity
    """
    
    img_zncc = np.zeros_like(img_in_zncc, dtype = 'float')
    
    num_frames = len(use_frames)
    
    # Find Particle by ZNNC
    num_particles_zncc = 0
    for ii, loop_frames in enumerate(use_frames):
        # to only print the first parallel loop later
        if ii == 0:
            # just make it a bit higher, because the other loops are set to 0
            num_verbose = nd.handle_data.GetNumberVerbose() + 3
        else:
            num_verbose = 0
        
        nd.logger.debug("Do cross-correlation. Frame %s from %s. Frame number: %s", (ii+1), num_frames, loop_frames)
        
        # select image of the loop
        use_img = img_in_zncc[ii,:,:].copy()
        
        # only allow particles where the channel is estimated to be
        use_img[search_area == 0] = 0
        
        pos_particles_loop, num_particles_zncc_loop, img_zncc[ii,:,:] = FindParticlesByZNCC(use_img, settings, num_verbose)
        
        nd.logger.info("Cross-correlation (%s / %s): Frame: %s; Located particles: %i", (ii+1), num_frames, loop_frames, num_particles_zncc_loop)
        # nd.logger.info("Located particles: %i", num_particles_zncc_loop)
        
        # configure frame saving format
        save_frames = np.tile(loop_frames,[num_particles_zncc_loop,1])
        pos_particles_loop = np.concatenate((pos_particles_loop, save_frames), axis = 1)
        
        num_particles_zncc = num_particles_zncc + num_particles_zncc_loop
        
        if loop_frames == 0: #first run
            pos_particles = pos_particles_loop
        else:
            pos_particles = np.concatenate((pos_particles, pos_particles_loop), axis = 0)

    # number of particles that would be nice to have
    ideal_min_particles = 200
    
    #recommend more frames
    num_particles_zncc_auto = num_particles_zncc + 1
    
    ideal_min_frames = (ideal_min_particles/num_particles_zncc_auto) * num_frames
    ideal_min_frames = int(np.round(ideal_min_frames)+1)

    if OneFrameOnly == False:        
        # plot error only if multiple frames are given, otherwise it is for parameter estimation
        if num_particles_zncc < 50:
            nd.logger.error("Less than 50 particles found in the given frames. Maybe enhance TryFrames in the settings to %i", ideal_min_frames)
            
            time.sleep(10) # give 10 seconds of sleep to show this message properly
    
        elif num_particles_zncc < 200:
            nd.logger.warning("Less than 200 particles found in the given frames. Maybe enhance TryFrames in the settings to %i", ideal_min_frames)
        
            time.sleep(10) # give 10 seconds of sleep to show this message properly

    return img_zncc, num_particles_zncc, pos_particles


def AskBestDiameter(QuestionForUser):
    """ ask if user is satisfied
    """
    valid_answer = False
    while valid_answer == False:
        answer = input(QuestionForUser + ' (y/n) :')
        if answer in ('y','n') == False:
            print("Warning: press y or n")
        else:
            valid_answer = True
            if answer == 'y':
                UserSatisfied = True
            else:
                UserSatisfied = False
    return UserSatisfied


def FindParticlesByZNCC(img1, settings, num_verbose = 5):
    """
    Find Particles by zero normalized cross-correclation
    """
       
    #check if raw data is convolved by PSF to reduce noise
    ImgConvolvedWithPSF = settings["PreProcessing"]["EnhanceSNR"]
    
    #if so - convolve the rawimage with the PSF
    if ImgConvolvedWithPSF == True:
        gauss_kernel_rad = settings["PreProcessing"]["KernelSize"]
        img1_in = nd.PreProcessing.ConvolveWithPSF_2D(img1, gauss_kernel_rad)        
    else:
        img1_in = img1
        
    # clip negative values and set datatype to uint16
    img1[img1 < 0] = 0
    img1 = np.uint16(img1)
      
    # calculate zero normalized crosscorrelation of image and psf    
    img_zncc = CorrelateImgAndPSF(img1_in, settings, num_verbose)
    
    # find objects in zncc
    correl_min = 0.60
    correl_min = 0.70
    correl_min = settings["PreProcessing"]["ZNCC_min"]
    nd.logger.info("Threshold zncc at %.2f", correl_min)
       
    # get positions of located spots and number of located particles
    pos_particles, num_particles_zncc = FindParticles(img_zncc, correl_min)

    return pos_particles, num_particles_zncc, img_zncc


def CorrelateImgAndPSF(img1, settings, num_verbose = 5):
    # estimated the minmass for trackpy
    # img1 is the image that is tested
    # if the image is convolved with the PSF to enhance the SNR, than img1 should be the convolved image

    # print("Correlate Img and PSF: Start")
        
    #get sigma of PSF
    sigma = SigmaPSF(settings)
    
    ImgConvolvedWithPSF = settings["PreProcessing"]["EnhanceSNR"]
    
    # create the gaussian kernel
    if ImgConvolvedWithPSF == True:
        # if rawdata is convolved with PSF than imaged point scatteres are smeared
        sigma_after_conv = sigma * np.sqrt(2)
        gauss_kernel = GaussianKernel(sigma_after_conv, fac = 10)
        
    else:
        gauss_kernel = GaussianKernel(sigma)        
    
    # Correlation cannot be done on the edge of the image
    # u and v have a min and max value which describes the frame
    u_min = np.int((gauss_kernel.shape[0]-1)/2)
    v_min = np.int((gauss_kernel.shape[1]-1)/2)
    
    u_max = img1.shape[0]-1-u_min
    v_max = img1.shape[1]-1-v_min
    
    # number of correlation element in one direction
    n = u_min
    
    # the zero normalied cross correlation function has a few things that can be precomputed
#    kernel_stdDeviation = getStandardDeviation(gauss_kernel, n, n, n)
#    kernel_avg = getAverage(gauss_kernel, n, n, n)
    
    # here comes the result in
    img_zncc = np.zeros_like(img1, dtype = 'float32')
                   
    def zncc_one_line(img1, kernel, loop_u, n):
        img_zncc_loop = np.zeros([img1.shape[1]], dtype = 'float32')
        y_min = loop_u - n
        y_max = loop_u + n
        img1_roi_y = img1[y_min: y_max+1,:]
          
        for loop_v in range(v_min, v_max+1):
            
            x_min = loop_v - n
            x_max = loop_v + n
            img1_roi = img1_roi_y[:, x_min: x_max+1]
            
            # img_zncc_loop[loop_v] = zncc(img1_roi, kernel, loop_u, loop_v, n)
            img_zncc_loop[loop_v] = zncc(img1_roi, kernel)
            
        return img_zncc_loop
    
    # number of cores the parallel computing is distributed over    
    num_cores = multiprocessing.cpu_count()
    
    # loop range
    # each line of the image is done separately - parallel
    inputs = range(u_min, u_max+1)   
      
    # parallel zncc
    img_zncc_list = Parallel(n_jobs=num_cores, verbose = num_verbose)(delayed(zncc_one_line)(img1.copy(), gauss_kernel, loop_u, n) for loop_u in inputs)
           
    # resulting list to array
    img_zncc_roi = np.asarray(img_zncc_list)
    
    # place the result in the middle of the predefined result
    # otherwise is the result shifted by u_min and v_min
    img_zncc[u_min:u_max+1:] = img_zncc_roi

    # print("Correlate Img and PSF: Finished")

    return img_zncc



def Convolution_2D(img1, im2):
    return np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(img1) * np.fft.fft2(im2))))



def FindParticles(img_zncc, correl_min):
    # find the particles insice the zero normalized cross correlation
    
    #threshold the zncc
    area_with_particle = img_zncc > correl_min

    # form region of areas to find middle of each localized particle    
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html#scipy.ndimage.label    
    #8ther neighborhood
    s_8 = generate_binary_structure(2,2)
    
    # form regions out of found thresholded ncc
    labeled_array, num_features = ndimage.label(area_with_particle , structure = s_8)
    
    # predefine position of identified particles
    pos_particles = np.zeros([num_features, 2])
    
    # go through all the found particles and average their area as position
    for loop_particles in range(1, num_features+1):
        [y_part_area, x_part_area] = np.where(labeled_array == loop_particles)
        
        pos_particles[loop_particles-1,:] = [np.int(y_part_area.mean()), np.int(x_part_area.mean())]
    
    return pos_particles, num_features
  


def FindChannel(rawframes_super):
    # returns a boolean saying where the channel (and thus the particle) is
    
    nd.logger.info("Find the channel: starting...")
    
    # # a channel can only be, where 90% of the time light is detected (water bg signal)
    # mychannel_raw = np.percentile(rawframes_super, q = 90, axis = 0) != 0

    nd.logger.debug("Do the median filter along time dimension")
    num_cores = multiprocessing.cpu_count()
    nd.logger.info("Do that parallel. Number of cores: %s", num_cores)    
    
    num_lines = rawframes_super.shape[1]
    
    inputs = range(num_lines)

    num_verbose = nd.handle_data.GetNumberVerbose()

    # execute background estimation in parallel for each line
    mychannel_raw_list = Parallel(n_jobs=num_cores, verbose = num_verbose)(delayed(np.median)(rawframes_super[:,loop_line,:].copy(), axis = 0) for loop_line in inputs)

    # list to numpy
    mychannel_raw = np.asarray(mychannel_raw_list)
    
    # a channel can only be, where 50% of the time light is detected (water bg signal)
    mychannel_raw = (mychannel_raw != 0)
    
    
    #remove salt and pepper
    nd.logger.debug("Remove Salt and Pepper noise")
    mychannel_no_sp = scipy.ndimage.morphology.binary_opening(mychannel_raw, iterations = 2)
       
    # fill holes and area close to the edge of the channel where little water is scattered
    nd.logger.debug("Do Dilation for hole closing")
    mychannel = scipy.ndimage.morphology.binary_dilation(mychannel_no_sp, iterations = 15)
    
    plt.figure()
    plt.imshow(mychannel)
    
    nd.logger.info("Find the channel: ...finished")    
    
    return mychannel



def OptimizeMinmassInTrackpyMain(settings, img1, num_particles_zncc, pos_particles, DoDiameter):
    # prepare calling OptimizeMinmassInTrackpy
   
    # load settings
    diameter = settings["Find"]["tp_diameter"]
    separation = settings["Find"]["tp_separation"]
    percentile = settings["Find"]["tp_percentile"]
    ImgConvolvedWithPSF = settings["PreProcessing"]["EnhanceSNR"]
        
    # Trackpy does bandpass filtering as "preprocessing". If the rawdata is already convolved by the PSF this additional bandpass does not make any sense. Switch of the preprocessing if rawdata is already convolved by the PSF
    DoPreProcessing = (ImgConvolvedWithPSF == False)
    
 
    if DoDiameter == False:
        nd.logger.info("Optimize trackpy parameter MINMASS: starting...")
        
        minmass, num_particles_trackpy, not_needed = OptimizeMinmassInTrackpy(img1, diameter, separation, num_particles_zncc, pos_particles, minmass_start = 1, DoPreProcessing = DoPreProcessing, percentile = percentile)
    
        settings["Find"]["tp_minmass"] = minmass
    
        nd.logger.info("Optimize trackpy parameter MINMASS: ...finished")
    
    else:
        nd.logger.info("Optimize trackpy parameter MINMASS and DIAMETER: starting...")
        
        # try several diameters. ideal job for doing this in parallal
        num_cores = multiprocessing.cpu_count()
        nd.logger.info("This is done in parallel (Number of cores: %s)", num_cores)
        
        # set the diameter worth trying
        # theoreical value of diameter
        diam_theory = nd.ParameterEstimation.DiameterForTrackpy(settings)
        
        # minium diameter
        min_diam = diam_theory - 2
        if min_diam < 3:
            min_diam = 3
        
        # maximum diameter
        max_diam = diam_theory + 12 + 1
        
        inputs = range(min_diam, max_diam, 2)
    
        num_verbose = nd.handle_data.GetNumberVerbose()
        
        # Do it parallel
        output_list = Parallel(n_jobs=num_cores, verbose=num_verbose)(delayed(OptimizeMinmassInTrackpy)(img1, loop_diameter, separation, num_particles_zncc, pos_particles, minmass_start = 1, DoPreProcessing = DoPreProcessing, percentile = percentile, DoLog = False) for loop_diameter in inputs)

        
        for ii,jj in enumerate(inputs):
            tp.subpx_bias(output_list[ii][2])
            plt.title("y - diameter = %i" %(jj))
    
        #Pause sothat python can show the plots before waiting for an input
        plt.pause(3)
    
        valid_input = False
        
        # ask the user which diameter is best
        while valid_input == False:
            nd.logger.info("The histogramm should be flat. They should not have a dip in the middle! ")
            
            try:
                diameter = input("Which diameter is best [dtype interger]: ")        
                diameter = int(diameter)
                
                # position of best diameter inside range
                range_pos = int(np.where(np.asarray(inputs) == diameter)[0])
        
                # select ideal minmass from list
                minmass = output_list[range_pos][0]
                
                valid_input = True
                    
            except:
                nd.logger.error("Try again. Dtype must be int and one of the available frames from %i to %i", inputs[0], inputs[-1])
    
        # save result
        settings["Find"]["tp_minmass"] = minmass
        settings["Find"]["tp_diameter"] = diameter
    
        nd.logger.info("Minmass for trackpy estimate to: %i", minmass)
        nd.logger.info("Diameter for trackpy estimate to: %i", diameter)

        nd.logger.info("Optimize trackpy parameter MINMASS and DIAMETER: ...finished")

        #make the plotting of some internal functions in case of debug
        if nd.logger.getEffectiveLevel() < 20:
            nd.logger.debug("Plot some additional stuff for information")
            OptimizeMinmassInTrackpy(img1, diameter, separation, num_particles_zncc, pos_particles, minmass_start = 1, DoPreProcessing = DoPreProcessing, percentile = percentile, DoLog = True)
        else:
            nd.logger.info("If you want to see some extra plots, change to the debug mode")
        
    return settings


def OptimizeMinmassInTrackpy(img1, diameter, separation, num_particles_zncc, pos_particles, minmass_start = 1, DoPreProcessing = True, percentile = 64, DoLog = True):
    """
    the particles are found accurately by zncc, which is accurate but time consuming
    trackpy is faster but needs proper threshold to find particles - minmass
    start with a low threshold and increase it till the found particles by zncc are lost
    
    DoLog is required, because this function is sometimes called in parallel, where the logging fails
    """
    
    #start value
    minmass = minmass_start
    
    # First Particle that NCC has but not trackpy
    First_wrong_assignment = True
    
    # First iteration
    first_iteration = True
    
    # loop exiting variable
    stop_optimizing = False
    
    # maximum distance between position of zncc and trackpy
    if type(diameter)==int:
        max_distance = diameter / 2 
    else:
        max_distance = diameter[0] / 2 # assume that diameters for x and y direction are equal
    
    # save the optimal minmass
    minmass_optimum = 0
    
    # save the history
    right_found_save = []
    wrong_found_save = []
    # wrong_to_right_save = []
    minmass_save = []
    
    if DoLog: nd.logger.info("Separation: %s", separation)
    
    percentile = 0
    if DoLog: nd.logger.info("percentile: %s", percentile)
    
    # switch logger ouf for this optimization
    if nd.logger.getEffectiveLevel() >= 20:
        # Switch the logging of for the moment
        tp.quiet(suppress=True)
    
    count_loops = 0
    
    # run the following till the optimization is aborted
    while stop_optimizing == False:     
        # here comes trackpy.
        # Trackpy is not running in parallel mode, since it loads quite long and we have only one frame here
        
        if count_loops%10 == 0:
            #plot every ten iterations
            if DoLog: nd.logger.info("Iteration: %s with minmass: %s", count_loops, minmass)
        else:
            if DoLog: nd.logger.debug("Iteration: %s with minmass: %s", count_loops, minmass)  
            
        count_loops = count_loops + 1    
        
        
        
        output = tp.batch(img1, diameter, minmass = minmass, separation = separation, max_iterations = 10, preprocess = DoPreProcessing, percentile = percentile)
        # num of found particles by trackpy
        num_particles_trackpy = len(output)

        if first_iteration == True:
            # sanity check
            if num_particles_trackpy < num_particles_zncc:
                stop_optimizing = True
                if DoLog: nd.logger.error("Trackpy finds too few particles. Possible reasons: \n - start value of minmass is too low (unlikely) \n - specimen is too dense/too highly concentrated \n - Percentile filter gives you problems in tp.batch or tp.locate.")
                raise ValueError("")
        

        # reset counters
        # right_found: particle is found in ncc and trackpy. Location mismatch is smaller than diameter
        right_found = 0
        
        # wrong_found: particle only in zncc or trackpy found. This is not good
        wrong_found = 0
               
        # if there are more particles found by trackpy than in zncc. The difference adds to the number of wrong_found particles. If zncc finds more than trackpy, they are counted later, when a localized particle in zncc does not have a corresponding point in trackpy.
        
        if num_particles_trackpy > num_particles_zncc:
            num_particle_only_trackpy_finds = num_particles_trackpy - num_particles_zncc
        else:
            num_particle_only_trackpy_finds = 0
 
        if DoLog: nd.logger.debug("Found particles (trackpy): %s", num_particles_trackpy)
        if DoLog: nd.logger.debug("Found particles (zncc): %s", num_particles_zncc)
        
        if num_particles_trackpy > (5 * num_particles_zncc):
            # if far too many particles are found the threshold must be increased significantly
            if DoLog: nd.logger.debug("5 times more feactures than expected. Enhance threshold!")
            
            # + 1 is required to ensure that minmass is increasing, although the value might be small
            minmass = np.int(minmass * 1.5) + 1
            
        elif num_particles_trackpy > (2 * num_particles_zncc):
            if DoLog: nd.logger.debug("2 times more feactures than expected. Enhance threshold!")
            minmass = np.int(minmass * 1.2) + 1
            
        else:
            # trackpy and znnc have similar results. so make some find tuning in small steps
            # check for every particle found by zncc if trackpy finds a particle too, withing the diameter
            for id_part, pos in enumerate(pos_particles):
                pos_y, pos_x, frame = pos
                
                # choose correct frame
                output_frame = output[output.frame == frame]
                
                # get distance to each particle found by trackpy
                dist_each_det_particle = np.hypot(output_frame.y-pos_y, output_frame.x-pos_x)
                
                # get the nearest
                closest_agreement = np.min(dist_each_det_particle)
                
                # check if closer than the maximum allowed distance 
                
                if closest_agreement > max_distance:
                    # particle is to far away. That is not good. So a particle is wrong assigned
                    wrong_found = wrong_found + 1
                    
                    # show position of first wrong particle. If more are plotted the console is just overfull
                    if First_wrong_assignment  == True:
                        First_wrong_assignment  = False
                        if DoLog: nd.logger.debug("Particle found in ZNCC but not with trackpy")
                        if DoLog: nd.logger.debug("Problem with particle: %s", id_part)
                        if DoLog: nd.logger.debug("Position: %s", pos)
                        if DoLog: nd.logger.debug("Closest point: %s", closest_agreement)
                    
                else:
                    # This is what you want. Particle found by zncc and trackpy within a neighborhood.
                    # right found + 1
                    right_found = right_found + 1
                 
            # add number of particles trackpy finds to much
            wrong_found = wrong_found + num_particle_only_trackpy_finds
            
            # get the ratio of wrong to right assignments. This should be as small as possible
            if right_found > 0:
                wrong_to_right =  wrong_found / right_found
            else:
                wrong_to_right  = np.inf
            
            #save numbers for plotting later on

            right_found_save = np.append(right_found_save, right_found)
            wrong_found_save = np.append(wrong_found_save, wrong_found)
            # wrong_to_right_save = np.append(wrong_to_right_save, wrong_to_right)
            minmass_save = np.append(minmass_save, minmass)
            
            if DoLog: nd.logger.debug("right_found: %s", right_found)
            if DoLog: nd.logger.debug("wrong_found: %s", wrong_found)
            if DoLog: nd.logger.debug("Wrong to right assignment: %s", wrong_to_right)
            
            if DoLog: nd.logger.debug("Still optimizing...")

            # check how value is changing
#            if Wrong_to_right > Wrong_to_right_optimum:
            if  (wrong_found == 0) | (num_particles_trackpy < 0.8 * num_particles_zncc):  
                if DoLog: nd.logger.debug("TrackPy finds much less particles than the zncc")
                
                #value increasing so abort loop
                stop_optimizing = True
                
                plt.figure()
                plt.plot(minmass_save, right_found_save, '.-', label = 'right found')
                plt.plot(minmass_save, wrong_found_save, '.-', label = 'wrong found')
                plt.xlabel("Minmass")
                plt.ylabel("Number particles")                
                plt.legend()
                
                plt.figure()
                wrong_to_right_save = wrong_found_save / right_found_save
                plt.plot(minmass_save, wrong_to_right_save, '.-')
                plt.xlabel("Minmass")
                plt.ylabel("Wrong-to-right")
                
                pos_minmass_optimum = np.where(wrong_to_right_save == np.min(wrong_to_right_save))[0][0]
                minmass_optimum = minmass_save[pos_minmass_optimum]

                
            # enhance minmass for next iteration
            minmass = np.int(minmass * 1.02) + 10
        
        first_iteration = False
    
    #leave a bit of space to not work at the threshold
    minmass_optimum = np.int(minmass_optimum * 0.90)
    if DoLog: nd.logger.info("Optimized Minmass threshold is: %s", minmass_optimum)

    obj_all = tp.batch(img1, diameter, minmass = minmass_optimum, separation = diameter, max_iterations = 10, preprocess = DoPreProcessing)
      
    if nd.logger.getEffectiveLevel() >= 20:
        # Switch the logging back on
        tp.quiet(suppress=False)
    
    # num of found particles by trackpy
    num_particles_trackpy = len(obj_all)


    return minmass_optimum, num_particles_trackpy, obj_all



def PlotImageProcessing(img, img_zncc, pos_particles):
    
    
    fig,ax = plt.subplots(nrows = 4, ncols=1, sharex=True, sharey=True)
    # plt.subplot(4, 1, 1, sharex=True, sharey=True)
    # plt.imshow(np.abs(img), cmap = 'gray')
    ax[0].imshow(img, cmap = 'gray')
    ax[0].set_title("image in")
    
    #avoid negative values
    img = img - np.min(img)
    
    # ax[0].subplot(4, 1, 2)
    ax[1].imshow(img**0.3, cmap = 'gray')
    ax[1].set_title("image in (gamma = 0.3)")
    
    # plt.subplot(4, 1, 3)
    ax[2].imshow(np.abs(img_zncc), cmap = 'jet')
    ax[2].set_title("zero normalized cross correlation")
    
    # plt.subplot(4, 1, 4)
    ax[3].scatter(pos_particles[:,1], pos_particles[:,0])
    ax[3].set_title("identified particles")
    ax[3].axis("scaled")
    ax[3].set_ylim([img_zncc.shape[0],0])
    ax[3].set_xlim([0,img_zncc.shape[1]])



def SaltAndPepperKernel(sigma, fac = 6, x_size = None,y_size = None):
#https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-79.php
    import numpy as np
    
    if x_size == None:
        x_size = np.ceil(fac*2*sigma)
    
    if y_size == None:
        y_size = np.ceil(fac*2*sigma)
    
    if np.mod(x_size,2) == 0:
        print("x_size must be odd; x_size + 1")
        x_size = x_size + 1        
    if np.mod(y_size,2) == 0:
        print("y_size must be odd; y_size + 1")        
        y_size = y_size + 1
       
    x_lim = (x_size-1)/2
    y_lim = (y_size-1)/2
        
    x, y = np.meshgrid(np.linspace(-x_lim, x_lim, x_size), np.linspace(-y_lim, y_lim, y_size))
    g = 1+np.sqrt(x*x+y*y)    

    g[g > 1] = 0
    g = g / np.sum(g)
    
    return g



def FindMaxDisplacementTrackpy(ParameterJsonFile, GuessLowestDiameter_nm = None):
    """
    Estimate how much a particle moves maximum between two frames.
    Leads to: Minimum distance (separation) of two points in the locating procedure
    Leads to: Maximum displacment in the nearest neighbour linking
    """
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    temp_water = settings["Exp"]["Temperature"]
    visc_water = settings["Exp"]["Viscosity"]
    Dark_frame  = settings["Link"]["Dark time"]

    # ask for the smallest expected diameter, which sets the largest expected diffusion
    if GuessLowestDiameter_nm == None:
        GuessLowestDiameter_nm = settings["Help"]["GuessLowestDiameter_nm"]
        
        if GuessLowestDiameter_nm  == "unknown":
            GuessLowestDiameter_nm = int(input("What is the lower limit of diameter (in nm) you expect?\n"))
    
    nd.logger.info("Expected lowest diameter: %.0f nm", GuessLowestDiameter_nm)
    
    settings["Help"]["GuessLowestDiameter_nm"] = GuessLowestDiameter_nm
    GuessLowestDiameter_m  = GuessLowestDiameter_nm / 1e9
    
    # Estimate max diffusion
    MaxDiffusion_squm = DiameterToDiffusion(temp_water,visc_water,GuessLowestDiameter_m)
    
    nd.logger.info("Maximum expected diffusion in squm per second: %s", np.round(MaxDiffusion_squm,2))
    
    MaxDiffusion_sqpx = MaxDiffusion_squm / (settings["Exp"]["Microns_per_pixel"]**2)
    # Think about sigma of the diffusion probability is sqrt(2Dt)

    t = 1/settings["Exp"]["fps"]
    
    #consider that a particle can vanish for number of frames Dark_time
    t_max = t * (1+Dark_frame)
    
    sigma_diff_px = np.sqrt(2*MaxDiffusion_sqpx*t_max )

    # look into Förster2020
    # Estimate the maximum displacement ONE particle undergoes between two frames (maybe more due to dark frames if switched on), sothat linking by nearest neighbor works
    # 5 sigma is 1 in 1.74 million (or sth like this) that particle does not leave this area
    Max_displacement = 5 * sigma_diff_px 

    # trackpy require integer
    Max_displacement = int(np.ceil(Max_displacement))

    # one is added because a bit of drift is always in
    Max_displacement = Max_displacement + 1

    nd.logger.info("The distance a particle can maximal move (and identified as the same one) >Max displacement< is set to: %s", Max_displacement)
    settings["Link"]["Max displacement"] = Max_displacement 


    # Estimate the distance TWO particle must be apart in order to successfully link them in the next frame without interchanging them.
    # 7sigma leads to 1 in a million of mixing up two particle by nearest neighbor linking
    Min_Separation_diff = 7 * sigma_diff_px
    
    # sigma of the PSF  
    sigma_PSF_nm = nd.Theory.SigmaPSF(settings["Exp"]["NA"], settings["Exp"]["lambda"])
    sigma_PSF_px = sigma_PSF_nm / (settings["Exp"]["Microns_per_pixel"]*1000)

    # seperate two gaussian by 6 sigma (meaning 3 sigma on each particle (3 sigma = beyond 99.7%))
    Min_Separation_PSF = 6 * sigma_PSF_px

    Min_Separation = int(np.ceil(Min_Separation_diff + Min_Separation_PSF))


    nd.logger.info("The minimum distance between two located particles >Separation data< is set to: %s", Min_Separation)
    settings["Find"]["tp_separation"] = Min_Separation 


    nd.handle_data.WriteJson(ParameterJsonFile, settings)
    
    return Min_Separation, Max_displacement

    

def DiameterToDiffusion(temp_water,visc_water,diameter):
    """ calculate diffusion coefficient of a sphere with given diameter
    """
    
    const_Boltz = scipy.constants.Boltzmann
    pi = scipy.constants.pi
    
    diffusion = (2*const_Boltz*temp_water/(6*pi *visc_water)) / diameter
    
    return diffusion



def Drift(ParameterJsonFile, num_particles_per_frame):
    """
    Calculates how many frames are requires to get a good estimation of the drift
    """
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)

    # the drift can be estimated better if more particles are found and more trajectories are formed
    # averaging over many frames leads to more datapoints and thus to a better estimation
    # on the other hand drift changes - so averaging over many time frames reduces the temporal resolution
    
    # I assume that 1000 particles need to be averaged to separte drift from random motion
    required_particles = 100

    #average_frames is applied in tp.drift. It is the number of >additional< follwing frames a drift is calculated. Meaning if a frame has 80 particles, it needs 2 frames to have more than 100 particles to average about. These two frame is the current and 1 addition one. That's why floor is used.
    average_frames = int(np.floor(required_particles/num_particles_per_frame))

    settings["Drift"]["Drift smoothing frames"] = average_frames

    print("The drift correction is done by averaging over: ", average_frames, " frames")

    nd.handle_data.WriteJson(ParameterJsonFile, settings)

    return average_frames


def MaxRelIntensityJump(ParameterJsonFile):
    """
    predicts the maximal expected scattering intensity jump of a particle between two frames
    """
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    if settings["Fiber"]["Speckle"] == 1:
        nd.logger.error("Intensity jump not predictable. Use empirical value (0.5) instead.")
        max_rel_jump = 0.5
        
    else:
    
        # minmal diameter to estimate the maximal diffusion
        diameter_nm = settings["Help"]["GuessLowestDiameter_nm"]
        diameter = diameter_nm * 1E-9
        
        temp_water = settings["Exp"]["Temperature"]
        visc_water = settings["Exp"]["Viscosity"]
        
        diff = nd.Theory.StokesEinsteinEquation(diameter, temp_water = temp_water, visc_water = visc_water)
        
        # calcualte the sigma of the dissplacement
        dt = 1/settings["Exp"]["fps"]
        
        sigma_um = np.sqrt(2*diff*dt) * 1E6
        
        # 6 times the sigma happens once in 500 Mio. If this limit is exceeded, the intensity jump cannot be explained by diffusion only (5 is just 1:15000)
        # but maybe 6 is very unlucky, so i use 5
        dr_um = 5*sigma_um
        nd.logger.info("Maximum expected displacement (5-sigma): %.2f um", dr_um)
        
        # get the fiber parameters
        r_fiber_nm = settings["Fiber"]["TubeDiameter_nm"]/2
        r_fiber_um = r_fiber_nm / 1E3
        
        # setup the channel for the short simulation
        r_step_um = 0.1
        r_num_steps = int((2*r_fiber_um)/r_step_um)
        r = np.linspace(-r_fiber_um, r_fiber_um, 100)
        
        # relativ intensity of the mode
        I0 = 1
        
        # sigma of the gaussian mode
        w = settings["Fiber"]["Waist"]
        
        # here come the mode
        I = I0*np.e**(-(r/w)**2)
        
        # maximum displacment in pixels
        dr_px = int(dr_um/r_step_um)
        I_start = I[dr_px:]
        I_end = I[:-dr_px]
        
        # calculate relative intensity change
        dI = I_start - I_end
        
        # I_offset = (I_start + I_end)/2
        # I_offset = np.sqrt(I_start * I_end)
        I_offset = np.min([I_start, I_end], axis = 0)
        
        rel_dI = np.abs(dI/I_offset)
        
        plt.plot(r[dr_px:], rel_dI)
        
        # get the maximum value and save it
        max_rel_jump = np.max(rel_dI)
        max_rel_jump = np.round(max_rel_jump,3)

    settings["Split"]["Max rel median intensity step"] = max_rel_jump 


    nd.handle_data.WriteJson(ParameterJsonFile, settings)

    return max_rel_jump




## this is the zncc loop, which is in funciton format for parallel computing
#def zncc_one_line(img1, img2, stdDeviation2, avg2, loop_u, n):
#    img_zncc_loop = np.zeros([img1.shape[1]], dtype = 'float32')
#    for loop_v in range(v_min, v_max+1):
#        img_zncc_loop[loop_v] = zncc(img1, img2, stdDeviation2, avg2, loop_u, loop_v, n)
#        
#    return img_zncc_loop
#    
#def zncc(img1, img2, stdDeviation2, avg2, u1, v1, n):
#    #https://martin-thoma.com/zero-mean-normalized-cross-correlation/
#
#    stdDeviation1 = getStandardDeviation(img1, u1, v1, n)
#    avg1 = getAverage(img1, u1, v1, n)
#
#    s = 0
#    for i in range(-n, n+1):
#        for j in range(-n, n+1):
#            s += (img1[u1+i][v1+j] - avg1)*(img2[n+i][n+j] - avg2)
#    return float(s)/((2*n+1)**2 * stdDeviation1 * stdDeviation2)
#    
#def MeanOfSubarray(image,kernel_diam):
#    kernel = np.zeros(image.shape, dtype = 'float32')
#    
#    #pm is plus minus
#    kernel_pm = np.int((kernel_diam-1)/2)
#    
#    kernel_area = kernel_diam**2
#    
#    # x and y correct and not switched?
#    mid_y = np.int(np.ceil(kernel.shape[0]/2))
#    mid_x = np.int(np.ceil(kernel.shape[1]/2))
#    
#    #+1 because of sometimes retarded python
#    kernel[mid_y-kernel_pm:mid_y+kernel_pm+1, mid_x-kernel_pm:mid_x+kernel_pm+1] = 1/kernel_area
#
#    mean_subarray = Convolution_2D(image, kernel)
#    
#    return mean_subarray