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


#%%
def GaussianKernel(sigma, fac = 6, x_size = None,y_size = None):
    #https://martin-thoma.com/zero-mean-normalized-cross-correlation/
    #https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-79.php

    # get the size of the kernel if not given
    if x_size == None:
        x_size = np.ceil(fac*2*sigma)
    
    if y_size == None:
        y_size = np.ceil(fac*2*sigma)
    
    #kernel must be odd for symmetrie
    if np.mod(x_size,2) == 0:
        print("x_size must be odd; x_size + 1")
        x_size = x_size + 1        
    if np.mod(y_size,2) == 0:
        print("y_size must be odd; y_size + 1")        
        y_size = y_size + 1
      
    # radius of the kernel in px
    x_lim = (x_size-1)/2
    y_lim = (y_size-1)/2
        
    # calculate the normal distribution
    x, y = np.meshgrid(np.linspace(-x_lim, x_lim, x_size), np.linspace(-y_lim, y_lim, y_size))
    r = np.sqrt(x*x+y*y)    

    g = np.exp(-( r**2 / ( 2.0 * sigma**2 ) ) )    
    
    # normalize to sum 1
    g = g / np.sum(g)
    
    return g


def getAverage(img, u, v, n):
    #https://martin-thoma.com/zero-mean-normalized-cross-correlation/
    """img as a square matrix of numbers"""
    s = 0
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            s += img[u+i][v+j]
    return float(s)/(2*n+1)**2


def getStandardDeviation(img, u, v, n):
    #https://martin-thoma.com/zero-mean-normalized-cross-correlation/
    s = 0
    avg = getAverage(img, u, v, n)
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            s += (img[u+i][v+j] - avg)**2
    return (s**0.5)/(2*n+1)


def zncc(img1, img2, stdDeviation2, avg2, u1, v1, n):
    #https://martin-thoma.com/zero-mean-normalized-cross-correlation/

    stdDeviation1 = getStandardDeviation(img1, u1, v1, n)
    avg1 = getAverage(img1, u1, v1, n)

    s = 0
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            s += (img1[u1+i][v1+j] - avg1)*(img2[n+i][n+j] - avg2)
    return float(s)/((2*n+1)**2 * stdDeviation1 * stdDeviation2)


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
    
    return sigma_px   



def EstimateMinmassMain(img1, settings):
    
    img1 = rawframes_pre[0,:,:]
    
    img1_conv = nd.PreProcessing.ConvolveWithPSF(img1, settings)

    # calculate zero normalized crosscorrelation of image and psf    
    img_zncc = CorrelateImgAndPSF(img1, settings)

    # find objects in zncc
    correl_min = 0.8
    pos_particles, num_features = FindParticles(img_zncc, correl_min)

    #estimate the minmass
    minmass = OptimizeMinmassInTrackpy(img1, diameter, num_features, pos_particles, minmass_start = 10)
        
    # plot the stuff
    PlotImageProcessing(img1_conv, img_zncc, pos_particles)
    
    return minmass

def CorrelateImgAndPSF(img1, settings, ImgConvolvedWithPSF = True):
    # estimated the minmass for trackpy
    # img1 is the image that is tested
    # if the image is convolved with the PSF to enhance the SNR, than img1 should be the convolved image

    print("Correlate Img and PSF: Start")
        
    #get sigma of PSF
    sigma = EstimageSigmaPSF(settings)
    
    # create the gaussian kernel
    if ImgConvolvedWithPSF == True:
        # if rawdata is convolved with PSF than imaged point scatteres are smeared
        sigma_after_conv = sigma * np.sqrt(2)
        gauss_kernel = GaussianKernel(sigma_after_conv)
        
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
    kernel_stdDeviation = getStandardDeviation(gauss_kernel, n, n, n)
    kernel_avg = getAverage(gauss_kernel, n, n, n)
    
    # here comes the result in
    img_zncc = np.zeros_like(img1, dtype = 'float32')
        
        
    # this is the zncc loop, which is in funciton format for parallel computing
    def zncc_one_line(img1, img2, stdDeviation2, avg2, loop_u, n):
        img_zncc_loop = np.zeros([img1.shape[1]], dtype = 'float32')
        for loop_v in range(v_min, v_max+1):
            img_zncc_loop[loop_v] = zncc(img1, img2, stdDeviation2, avg2, loop_u, loop_v, n)
            
        return img_zncc_loop
    
    # number of cores the parallel computing is distributed over    
    num_cores = multiprocessing.cpu_count()
    
    # loop range
    # each line of the image is done separately - parallel
    inputs = range(u_min, u_max+1)   
      
    # parallel zncc
    img_zncc_list = Parallel(n_jobs=num_cores)(delayed(zncc_one_line)(img1.copy(), gauss_kernel, kernel_stdDeviation, kernel_avg, loop_u, n) for loop_u in inputs)
           
    # resulting list to array
    img_zncc_roi = np.asarray(img_zncc_list)
    
    # place the result in the middle of the predefined result
    # otherwise is the result shifted by u_min and v_min
    img_zncc[u_min:u_max+1:] = img_zncc_roi

    print("Correlate Img and PSF: Finished")

    return img_zncc



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
    
    print("estimated diameter: ", diameter)
    
    return diameter
    
      
    

def OptimizeMinmassInTrackpy(img1, diameter, num_features, pos_particles, minmass_start = 10):
    # the particles are found accurately by zncc, which is accurate but time consuming
    # trackpy is faster but needs proper threshold to find particles - minmass
    # start with a low threshold and increase it till the found particles by zncc are lost
    
    #start value
    minmass = minmass_start
    
    #true as long as all particles are found
    all_particles_in = True
    
    # loop exiting variable
    stop_optimizing = False
    
    # maximum distance between position of zncc and trackpy
    max_distance = diameter / 2
    
    
    while stop_optimizing == False:    
        print("minmass: ", minmass)
        print("separation: ", diameter)
    
        #here comes trackpy
        output = tp.batch(np.expand_dims(img1,axis = 0), diameter, minmass = minmass, separation = diameter, max_iterations = 10, processes = 'auto')
        
        num_found_particle = len(output)
    
        print("Found particles (trackpy): ", num_found_particle)
        print("Found particles (zncc): ", num_features)
        
        
        if num_found_particle > (10 * num_features):
            # if far to many particles are found the threshold must be increased significantly
            print("far too many features. enhance threshold")
            minmass = np.int(minmass * 1.5) + 1
            
            
        else:
            # check to distance between found particle of trackpy and zncc
            for id_part, pos in enumerate(pos_particles):
                pos_y, pos_x = pos
                
                # check if pos is somewhere in the output
                dist_each_det_particle = np.hypot(output.y-pos_y, output.x-pos_x)
                closest_agreement = np.min(dist_each_det_particle)
                
                if closest_agreement > max_distance:
                    # if there should be a particle (zncc is accurate), but trackpy does not find one
                    # the iteration stops
                    all_particles_in = False
                    stop_optimizing = True
                    print("Particle found in ZNCC but not with trackpy")
                    print("Problem with particle: ", id_part)
                    print("Position: ", pos)
                    print("Closest point: ", closest_agreement)
            
            if all_particles_in == True:
                if num_found_particle == num_features:
                    # stop if trackpy and zncc lead to same result
                    print("all particles found and no disturbing objects detected. Stop optimization.")
                    stop_optimizing = True
                    
                else:
                    # enhance minmass slightly, if all particle of znnc are found by trackpy, but other stuff to
                    print("all particles found but still disturbing stuff inside")
                    minmass = np.int(minmass * 1.05) + 1
    
    #leave a bit of space to not work at the threshold
    minmass = np.int(minmass * 0.9)
    print("Optimized Minmass threshold is: ", minmass)

    return minmass



def PlotImageProcessing(img, img_zncc, pos_particles):
    
    plt.subplot(3, 1, 1)
    plt.imshow(np.abs(img)**(0.5), cmap = 'gray')
    plt.title("image in")
    
    plt.subplot(3, 1, 2)
    plt.imshow(np.abs(img_zncc), cmap = 'jet')
    plt.title("zero normalized cross correlation)
    
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
