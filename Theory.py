# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 08:44:12 2020

@author: foersterronny

collection of standard equations of physics, unit conversions etc.
"""

import matplotlib.pyplot as plt
import numpy as np
from pdb import set_trace as bp #debugger
import psf
import multiprocessing
import NanoObjectDetection as nd
from joblib import Parallel, delayed
from scipy.constants import Boltzmann as k_b
from scipy.constants import pi as pi
from scipy.constants import speed_of_light as c


def SigmaPSF(NA, mylambda):
    # Zhang et al 2007
    sigma = 0.21 * mylambda/ NA
    
    return sigma

    
def LocalError(sigma_psf, diffusion, t_exp, photons):
    ep = np.sqrt((sigma_psf**2 + diffusion*t_exp)/photons) 

    return ep

def CRLB(N,x):
    # Number of frames
    # x reduced localization precision
    
    # Theoretical minimum values of eps = std(D)/D
    # (Michalet & Berglund, 2012)
    
    eps = np.sqrt(2/(N-1) * (1+2*np.sqrt(1+2*x)))
    
    return eps


def StokesEinsteinEquation(diff = None, diam = None, temp_water = 295, visc_water = 9.5E-16):
    """ solves the Stokes-Einstein equation either for the diffusion coefficient or the 
    hydrodynamic diameter of a particle
    
    diff:   diffusion coefficient [m^2/s]
    diam:   particle diameter [m]    
    """
    
    # into SI unit mPa*s
    visc_water = visc_water * 1E12
    
    if (diff == None) and (diam == None):
        raise ValueError('Either diffusion or diameter must be >NONE<')
    
    if diff == None:
        #calc diffusion
        radius = diam / 2
        my_return = (k_b*temp_water)/(6*pi*visc_water * radius) # [um^2/s]
        
    elif diam == None:
        radius = (k_b*temp_water)/(6*pi *visc_water * diff) # [um]
        my_return = radius * 2
    else:
        raise ValueError('Either diffusion or diameter must be >NONE<')
        
    return my_return



def PulseEnergy2CW(E_pulse, rep_rate):
    """
    E_pulse:    pulse power [W]
    rep_rate:   Repeatition rate [Hz]
    
    https://www.thorlabs.com/images/tabimages/Laser_Pulses_Power_Energy_Equations.pdf
    """
    
    # average power
    P_avg = E_pulse * rep_rate
    
    return P_avg
    


def zncc(image, kernel, size):
    """ ZNCC = Zero Mean Normalized Cross-Correlation, ...??
    
    https://martin-thoma.com/zero-mean-normalized-cross-correlation/, 27.07.2020
    """
    img_zncc = np.zeros_like(image, dtype = 'float32')
    
    mid_x = np.int((kernel.shape[0]-1)/2)
    mid_y = np.int((kernel.shape[1]-1)/2)
    
    kernel = kernel[mid_x-size : mid_x+size+1, mid_y-size : mid_y+size+1]  
          
    u_min = np.int((kernel.shape[0]-1)/2)
    v_min = np.int((kernel.shape[1]-1)/2)
    
    u_max = image.shape[0]-1-u_min
    v_max = image.shape[1]-1-v_min
    
    def zncc_one_line(img1, kernel, loop_u, n):
        print("Test")
        img_zncc_loop = np.zeros([img1.shape[1]], dtype = 'float32')
        y_min = loop_u - n
        y_max = loop_u + n
        img1_roi_y = img1[y_min: y_max+1,:]
          
        for loop_v in range(v_min, v_max+1):
            
            x_min = loop_v - n
            x_max = loop_v + n
            img1_roi = img1_roi_y[:, x_min: x_max+1]
            
            img_zncc_loop[loop_v] = nd.ParameterEstimation.zncc(img1_roi, kernel)
            
            
        return img_zncc_loop
    
    # number of cores the parallel computing is distributed over    
    num_cores = multiprocessing.cpu_count()
    
    # loop range
    # each line of the image is done separately - parallel
    inputs = range(u_min, u_max+1)   
      
    
    # parallel zncc
    img_zncc_list = Parallel(n_jobs=num_cores)(delayed(zncc_one_line)(image.copy(), kernel, loop_u, size) for loop_u in inputs)
           
    # resulting list to array
    img_zncc_roi = np.asarray(img_zncc_list)
    
    # place the result in the middle of the predefined result
    # otherwise is the result shifted by u_min and v_min
    img_zncc[u_min:u_max+1:] = img_zncc_roi


    return img_zncc
    


def PSF(NA= 1.2, n = 1.46, sampling_z = None, shape_z = None):
    """ create an out of focus PSF
    """
    args = {
    'shape': (128, 128),  # number of samples in z and r direction
    'dims': (5.0, 5.0),   # size in z and r direction in micrometers
    'ex_wavelen': 488.0,  # excitation wavelength in nanometers
    'em_wavelen': 532.0,  # emission wavelength in nanometers
    'num_aperture': NA,
    'refr_index': n,
    'magnification': 1.0,
#    'pinhole_radius': 0.05,  # in micrometers
#    'pinhole_shape': 'square',
    }

    
    if shape_z != None:
        args["shape"] = [shape_z, args["shape"][1]]
        
    abbe_r = args["em_wavelen"]/(2*args["num_aperture"])
    sampling_r = abbe_r / 2.2
    dim_r = sampling_r * args['shape'][1] / 1000
    #1.2 for a bit of oversampling
    
    if sampling_z == None:
        alpha = np.arcsin(args["num_aperture"]/args["refr_index"])
        abbe_z = args["em_wavelen"]/(1-np.cos(alpha))
        sampling_z = abbe_z / 2.2
        
    dim_z = sampling_z * args['shape'][0] / 1000
    
    
    args["dims"] = [dim_z, dim_r]
    
    obsvol = psf.PSF(psf.ISOTROPIC | psf.WIDEFIELD, **args)
    empsf = obsvol.empsf

#    empsf.slice(100)
    
#    #def Main():
#    params = {
#       'figure.figsize': [8, 6],
#       'legend.fontsize': 12,
#       'text.usetex': False,
#       'ytick.labelsize': 10,
#       'ytick.direction': 'out',
#       'xtick.labelsize': 20,
#       'xtick.direction': 'out',
#       'font.size': 10,
#       }
##    mpl.rcParams.update(params)
#    
#    
#    title_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
#      'verticalalignment':'bottom'} # Bottom vertical alignment for more space
#    axis_font = {'fontname':'Arial', 'size':'18'}
    
#    plt.imshow(empsf.slice(100))

    args["sampling"] = [sampling_z, sampling_r]

    return empsf, args
    
    

def RFT2FT(image_in):
    """ [insert description here]
    """
    mirror_x = np.flip(image_in, axis = 0)
    mirror_x = mirror_x[:-1,:] #remove last line otherwise double
    
    image_1 = np.concatenate((mirror_x , image_in), axis = 0)

    mirror_y = np.flip(image_1, axis = 1)
    mirror_y = mirror_y[:,:-1] #remove last line otherwise double
    
    image_out = np.concatenate((mirror_y , image_1), axis = 1)

    return image_out
 


def MyFftConvolve(im1,im2):
    """ [insert description here]
    """
    from numpy.fft import fftn as fftn
    from numpy.fft import ifftn as ifftn
    
    convolved = np.real(ifftn(fftn(im1) * fftn(im2)))
    
    convolved  = np.fft.fftshift(convolved)
    
    return convolved   



def DeconRLTV(image, psf, lambda_tv = 0, num_iterations = 10):
    """ [insert description here]
    """
    # from scipy.signal import fftconvolve as fft_conv
    # import scipy
    #https://www.weizmann.ac.il/mcb/ZviKam/Papers/72.%20MRT_Paris.pdf
    
    # image = plt.imshow(np.matlib.repmat(image,3,3))
    
    result = np.zeros_like(image) + 1
    
    num_iter = 1
    use_mode = 'valid'
    
    while num_iter <= num_iterations:
        # print("number iteration: ", num_iter)
        num_iter += 1

        # rl_1 = image / (fft_conv(result, psf, mode = use_mode))
        rl_1 = image / (MyFftConvolve(result, psf))
        # np.real(np.fft.ifftn(np.fft.fftn(rl_1) * np.fft.fftn(psf)))
        
        # rl_2 = fft_conv(rl_1, psf, mode = use_mode)
        rl_2 = MyFftConvolve(rl_1, psf)
        
        if lambda_tv != 0:
            # nabla_result = np.gradient(image)
            # abs_nabla_result = np.hypot(nabla_result[0], nabla_result[1])
        
            # nabla_result[0] = nabla_result[0] / abs_nabla_result
            # nabla_result[1] = nabla_result[1] / abs_nabla_result
        
            # div_result = np.gradient(nabla_result[0], axis = 0) + \
            #     np.gradient(nabla_result[1], axis = 1)
                
            
            div_result = (result == np.max(result))
            
            tv = result / (1-lambda_tv*div_result)
            
            result = rl_2 * tv
        
        else:
            result = rl_2 * result
           
    
        result = np.abs(result)
        
        result[np.isnan(result)] = 0
    
    print("use other penalty???")
    
    final_pos = np.squeeze(np.asarray(np.where(result == np.max(result))))
    
    print("bead at: ", final_pos)
    
    return result



def DeconRL3D(image, psf3d, lambda_tv = 0, num_iterations = 10):
    """ [insert description here]
    """
    # from scipy.signal import fftconvolve as fft_conv
    # import scipy
    #https://www.weizmann.ac.il/mcb/ZviKam/Papers/72.%20MRT_Paris.pdf
    
    # image = plt.imshow(np.matlib.repmat(image,3,3))
    
    num_z = 9
    psf3d = psf3d.slice()[0:num_z,:,:]
    
    result = np.zeros_like(image) + 1
    
    num_iter = 1
    use_mode = 'valid'
    
    while num_iter <= num_iterations:
        # print("number iteration: ", num_iter)
        num_iter += 1

        # rl_1 = image / (fft_conv(result, psf, mode = use_mode))
        rl_1 = image / (MyFftConvolve(result, psf))
        # np.real(np.fft.ifftn(np.fft.fftn(rl_1) * np.fft.fftn(psf)))
        
        # rl_2 = fft_conv(rl_1, psf, mode = use_mode)
        rl_2 = MyFftConvolve(rl_1, psf)
        
        if lambda_tv != 0:
            # nabla_result = np.gradient(image)
            # abs_nabla_result = np.hypot(nabla_result[0], nabla_result[1])
        
            # nabla_result[0] = nabla_result[0] / abs_nabla_result
            # nabla_result[1] = nabla_result[1] / abs_nabla_result
        
            # div_result = np.gradient(nabla_result[0], axis = 0) + \
            #     np.gradient(nabla_result[1], axis = 1)
                
            
            div_result = (result == np.max(result))
            
            tv = result / (1-lambda_tv*div_result)
            
            result = rl_2 * tv
        
        else:
            result = rl_2 * result
           
    
        result = np.abs(result)
        
        result[np.isnan(result)] = 0
    
    print("use other penalty???")
    
    final_pos = np.squeeze(np.asarray(np.where(result == np.max(result))))
    
    print("bead at: ", final_pos)
    
    return result



def IntensityInFiber(P, radius, Mode = 'Peak' ):
    """ calculate intensity profile inside a fiber channel with circular cross section
    
    P_illu      power of illumination beam [W]
    radius      radius of channel [m]
    Mode        calculate Intensity at the "Peak" or assume "FlatTop" Profile
    """

    A = np.pi * radius **2
    if Mode == 'Peak':
        # peak intensity of gaussian beam
        I_illu = 2*P/A
    elif Mode == 'FlatTop':
        I_illu = P/A
    else:
        print("Mode must be either >>Peak<< OR >>FlatTop<<")
        
    return I_illu



def LensMakerEquation(R1,R2,d,n_media,n_glass):
    """ calculate focal distance of a lens
    https://de.wikipedia.org/wiki/Linsenschleiferformel, 27.07.2020
    
    R1, R2:     radii of curvature of the spherical surfaces
    d:          thickness of the lens
    n_*:        refr. index of lens material and surroundings
    """
    D = (n_glass-n_media)/n_media * (1/R1 - 1/R2 + (n_glass-n_media)*d/(n_glass*R1*R2))
    f = 1/D
    
    return f



def LensEquation(f,g):
    """ calculate image distance from focal length and object distance ("thin lens formula")
    """
    b_inv = 1/f - 1/g
    b = 1/b_inv
    
    return b



def RadiationForce(I, C_scat, C_abs, n_media = 1.333):   
    """ calculate the radiation force onto a scattering sphere
    
    Literature:
    https://reader.elsevier.com/reader/sd/pii/0030401895007539?token=48F2795599992EB11281DD1C2A50B58FC6C5F2614C90590B9700CD737B0B9C8E94F2BB8A17F74D0E6087FF3B7EF5EF49
    https://github.com/scottprahl/miepython/blob/master/doc/01_basics.ipynb
    
    lambda_nm:  wavelength of the incident light
    d_nm:       sphere's diameter
    P_W:        incident power
    A_sqm:      beam/channel cross sectional area
    material:   sphere's material
    
    "Optical trapping of metallic Rayleigh particles"
    "Radiation forces on a dielectric sphere in the Rayleigh scattering regime"
    """
    
    #Eq 11 in Eq 10
    #not quite sure here
    F_scat = C_scat * n_media/c * I
    F_abs = C_abs * n_media/c * I


    return F_scat
