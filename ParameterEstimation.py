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
        print("x_size must be odd; x_size + 1")
        x_size = x_size + 1        
    if np.mod(y_size,2) == 0:
        print("y_size must be odd; y_size + 1")        
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
    
    zncc = np.mean((img1 - img1_mean) * (img2 - img2_mean)) / (img1_std * img2_std)
#    zncc = np.mean((img1 - img1_mean) * (img2 - avg2))/(img1_std * stdDeviation2)
    
    return zncc



def EstimageSigmaPSF(settings):
    #estimate best sigma
    #https://en.wikipedia.org/wiki/Numerical_aperture
    NA = settings["Exp"]["NA"]
    n  = settings["Exp"]["n_immersion"]
    
    # fnumber
    N = 1/(2*np.tan(np.arcsin(NA / n)))
    
    # approx PSF by gaussian
    # https://en.wikipedia.org/wiki/Airy_disk
    lambda_nm = settings["Exp"]["lambda"]
    sigma_nm = 0.45 * lambda_nm * N
    sigma_um = sigma_nm / 1000
    sigma_px = sigma_um / settings["Exp"]["Microns_per_pixel"]
    
    return sigma_px   



def EstimateMinmassMain(img1, settings):
    img1 = img1[0,:,:]
       
    #check if raw data is convolved by PSF to reduce noise
    ImgConvolvedWithPSF = settings["PreProcessing"]["EnhanceSNR"]
    
    #if so - convolve the rawimage with the PSF
    if ImgConvolvedWithPSF == True:
        img1_in = nd.PreProcessing.ConvolveWithPSF(img1, settings)        
    else:
        img1_in = img1
        
    
    img1[img1 < 0] = 0
    img1 = np.uint16(img1)
      
    # calculate zero normalized crosscorrelation of image and psf    
    img_zncc = CorrelateImgAndPSF(img1_in, settings)
    
    
    # find objects in zncc
    correl_min = 0.60
    
    # get positions of located spots and number of located particles
    pos_particles, num_particles_zncc = FindParticles(img_zncc, correl_min)

    # load diameter from settings
    diameter = settings["Find"]["Estimated particle size"]
    
    # load separation from settings
    separation = settings["Find"]["Separation data"]
    
    # load the percentile filter value
    percentile = settings["Find"]["PercentileThreshold"]    
    
    # Trackpy does bandpass filtering as "preprocessing". If the rawdata is already convolved by the PSF this additional bandpass does not make any sense. Switch of the preprocessing if rawdata is already convolved by the PSF
    DoPreProcessing = (ImgConvolvedWithPSF == False)
    
    # optimize the minmass in trackpy, sothat the results of ncc and trackpy agree best
    minmass, num_particles_trackpy = OptimizeMinmassInTrackpy(img1, diameter, separation, num_particles_zncc, pos_particles, minmass_start = 1, DoPreProcessing = DoPreProcessing, percentile = percentile)
        
    # plot the stuff
    PlotImageProcessing(img1_in, img_zncc, pos_particles)
    
    return minmass, num_particles_trackpy



def CorrelateImgAndPSF(img1, settings):
    # estimated the minmass for trackpy
    # img1 is the image that is tested
    # if the image is convolved with the PSF to enhance the SNR, than img1 should be the convolved image

    print("Correlate Img and PSF: Start")
        
    #get sigma of PSF
    sigma = EstimageSigmaPSF(settings)
    
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
    img_zncc_list = Parallel(n_jobs=num_cores)(delayed(zncc_one_line)(img1.copy(), gauss_kernel, loop_u, n) for loop_u in inputs)
           
    # resulting list to array
    img_zncc_roi = np.asarray(img_zncc_list)
    
    # place the result in the middle of the predefined result
    # otherwise is the result shifted by u_min and v_min
    img_zncc[u_min:u_max+1:] = img_zncc_roi

    print("Correlate Img and PSF: Finished")

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
  
    

def EstimateDiameterForTrackpy(settings, ImgConvolvedWithPSF = True):   
    
    #theoretical sigma of the PSF
    sigma = EstimageSigmaPSF(settings)
    
    # create the gaussian kernel
    if ImgConvolvedWithPSF == True:
        # if rawdata is convolved with PSF than imaged point scatteres are smeared
        sigma = sigma * np.sqrt(2)

    #2,5 sigma is 99% of the intensity - visibile diameter
    #sigma is the radius so times two to get the diameter
    diameter = 2.5 * 2 * sigma
    
    #get odd integer (rather to large than to small)
    diameter  = np.int(np.ceil(diameter))
    
    if np.mod(diameter,2) == 0:
        diameter = diameter + 1
    
    print("\n Estimated diameter: ", diameter)
    
    return diameter
          
    

def OptimizeMinmassInTrackpy(img1, diameter, separation, num_particles_zncc, pos_particles, minmass_start = 1, DoPreProcessing = True, percentile = 64):
    # the particles are found accurately by zncc, which is accurate but time consuming
    # trackpy is faster but needs proper threshold to find particles - minmass
    # start with a low threshold and increase it till the found particles by zncc are lost
    
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
    
    # optimal value of wrong to right (the value to minimize) and the corresponding minmass
    wrong_to_right_optimum = np.inf
    minmass_optimum = 0
    
    # save the history
    wrong_to_right_save = []
    minmass_save = []
    
    print("separation: ", separation)
    
    # run the following till the optimization is aborted
    while stop_optimizing == False:       
        # here comes trackpy.
        # Trackpy is not running in parallel mode, since it loads quite long and we have only one frame here
        
        output = tp.locate(img1, diameter, minmass = minmass, separation = separation, max_iterations = 10, preprocess = DoPreProcessing, percentile = percentile)
        # num of found particles by trackpy
        num_particles_trackpy = len(output)

        if first_iteration == True:
            # sanity check
            if num_particles_trackpy < num_particles_zncc:
                stop_optimizing = True
                raise ValueError("Trackpy finds to few particles. Reasons: \n Start value of minmass is to low (unlikely) \n Specimen is too dense/ too high concentrated.")
        
#        wrong_found = num_found_particle - num_features
#        print("Wrong to right assignment: ", wrong_found / num_features)
        
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
        
        
        if num_particles_trackpy > (5 * num_particles_zncc):
            # if far to many particles are found the threshold must be increased significantly
            print("far too many features. enhance threshold")
            
            # + 1 is required to ensure that minmass is increasing, although the value might be small
            minmass = np.int(minmass * 1.5) + 1
            
        else:
            # trackpy and znnc have similar results. so make some find tuning in small steps
            # check for every particle found by zncc if trackpy finds a particle too, withing the diameter
            for id_part, pos in enumerate(pos_particles):
                pos_y, pos_x = pos
                
                # get distance to each particle found by trackpy
                dist_each_det_particle = np.hypot(output.y-pos_y, output.x-pos_x)
                
                # get the nearest
                closest_agreement = np.min(dist_each_det_particle)
                
                # check if closer than the maximum allowed distance 
                
                if closest_agreement > max_distance:
                    # particle is to far away. That is not good. So a particle is wrong assigned
                    wrong_found = wrong_found + 1
                    
                    # show position of first wrong particle. If more are plotted the console is just overfull
                    if First_wrong_assignment  == True:
                        First_wrong_assignment  = False
                        print("Particle found in ZNCC but not with trackpy")
                        print("Problem with particle: ", id_part)
                        print("Position: ", pos)
                        print("Closest point: ", closest_agreement)
                    
                else:
                    # This is what you want. Particle found by zncc and trackpy within a neighborhood.
                    # right found + 1
                    right_found = right_found + 1
                 
            # add number of particles trackpy finds to much
            wrong_found = wrong_found + num_particle_only_trackpy_finds
            
            # get the ratio of wrong to right assignments. This should be as small as possible
            wrong_to_right =  wrong_found / right_found
            
            wrong_to_right_save = np.append(wrong_to_right_save, wrong_to_right)
            minmass_save = np.append(minmass_save, minmass)
            
                
            print("\n minmass: ", minmass)
            
            print("Found particles (trackpy): ", num_particles_trackpy)
            print("Found particles (zncc): ", num_particles_zncc)
            
            print("right_found: ", right_found)
            print("wrong_found: ", wrong_found)
            print("Wrong to right assignment: ", wrong_to_right)
            
            print("Still optimizing...")

            # check how value is changing
#            if Wrong_to_right > Wrong_to_right_optimum:
            if num_particles_trackpy < 0.8 * num_particles_zncc:  
                print("TrackPy finds much less particles than the zncc")
                
                #value increasing so abort loop
                stop_optimizing = True
                
                plt.figure()
                plt.plot(minmass_save, wrong_to_right_save)
                plt.xlabel("Minmass")
                plt.ylabel("Wrong-to-right")
                plt.figure()
                
                pos_minmass_optimum = np.where(wrong_to_right_save == np.min(wrong_to_right_save))[0][0]
                minmass_optimum = minmass_save[pos_minmass_optimum]

                
            # enhance minmass for next iteration
            minmass = np.int(minmass * 1.01) + 1
        
        first_iteration = False
    
    #leave a bit of space to not work at the threshold
    minmass_optimum = np.int(minmass_optimum * 0.90)
    print("\n Optimized Minmass threshold is: ", minmass_optimum, "\n")

    output = tp.locate(img1, diameter, minmass = minmass_optimum, separation = diameter, max_iterations = 10, preprocess = DoPreProcessing)
        
    # num of found particles by trackpy
    num_particles_trackpy = len(output)

    return minmass_optimum, num_particles_trackpy



def PlotImageProcessing(img, img_zncc, pos_particles):
    
    plt.subplot(3, 1, 1)
    plt.imshow(np.abs(img)**(0.5), cmap = 'gray')
    plt.title("image in")
    
    plt.subplot(3, 1, 2)
    plt.imshow(np.abs(img_zncc), cmap = 'jet')
    plt.title("zero normalized cross correlation")
    
    plt.subplot(3, 1, 3)
    plt.scatter(pos_particles[:,1], pos_particles[:,0])
    plt.title("identified particles")
    plt.axis("scaled")
    plt.gca().set_ylim([img_zncc.shape[0],0])
    plt.gca().set_xlim([0,img_zncc.shape[1]])



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
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    temp_water = settings["Exp"]["Temperature"]
    visc_water = settings["Exp"]["Viscosity"]
    Dark_frame  = settings["Link"]["Dark time"]

    if GuessLowestDiameter_nm == None:
        GuessLowestDiameter_nm = int(input("What is the lower limit of diameter (in nm) you expect?\n"))
    
    
    settings["Help"]["GuessLowestDiameter_nm"] = GuessLowestDiameter_nm
    GuessLowestDiameter_m  = GuessLowestDiameter_nm / 1e9
    
    # Estimate max diffusion
    MaxDiffusion_squm = DiameterToDiffusion(temp_water,visc_water,GuessLowestDiameter_m)
    
    print("Maximum expected diffusion in squm per second:", np.round(MaxDiffusion_squm,2))
    
    MaxDiffusion_sqpx = MaxDiffusion_squm / (settings["Exp"]["Microns_per_pixel"]**2)
    # Think about sigma of the diffusion probability is sqrt(2Dt)

    t = 1/settings["Exp"]["fps"]
    
    #consider that a particle can vanish for number of frames Dark_time
    t_max = t * (1+Dark_frame)
    
    sigma_px = np.sqrt(2*MaxDiffusion_sqpx*t_max )

    # look into Förster2020
    # 5 sigma is 1 in 1.74 million (or sth like this) that particle does not leave this area
    Max_displacement = 5 * sigma_px

    # trackpy require integer
    Max_displacement = int(np.ceil(Max_displacement))

    # one is added because a bit of drift is always in
    Max_displacement = Max_displacement + 1

    # 7sigma leads to 1 in a million of mixing up two particle by nearest neighbor linking
    Min_Separation = 7 * sigma_px
    Min_Separation = int(np.ceil(Min_Separation))
    # one is added because a bit of drift is always in
    Min_Separation = Min_Separation + 1
    

    print("\n The distance a particle can maximal move (and identified as the same one) >Max displacement< is set to: ", Max_displacement)
    settings["Link"]["Max displacement"] = Max_displacement 

    print("\n The minimum distance between two located particles >Separation data< is set to: ", Min_Separation )
    settings["Find"]["Separation data"] = Min_Separation 


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
    settings = nd.handle_data.ReadJson(ParameterJsonFile)

    # the drift can be estimated better if more particles are found and more trajectories are formed
    # averaging over many frames leads to more datapoints and thus to a better estimation
    # on the other hand drift changes - so averaging over many time frames reduces the temporal resolution
    
    # I assume that 100 particles need to be averaged to separte drift from random motion
    required_particles = 100

    #average_frames is applied in tp.drift. It is the number of >additional< follwing frames a drift is calculated. Meaning if a frame has 80 particles, it needs 2 frames to have more than 100 particles to average about. These two frame is the current and 1 addition one. That's why floor is used.
    average_frames = int(np.floor(required_particles/num_particles_per_frame))

    settings["Drift"]["Drift smoothing frames"] = average_frames

    print("The drift correction is done by averaging over: ", average_frames, " frames")

    nd.handle_data.WriteJson(ParameterJsonFile, settings)


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