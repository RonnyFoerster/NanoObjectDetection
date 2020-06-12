# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 08:44:12 2020

@author: foersterronny
"""

import matplotlib.pyplot as plt
import numpy as np

# HERE ARE ALL STANDARD EQUATIONS IN PHYSICS LIKE CONVERSIONS ETC.

def PulseEnergy2CW(E_pulse, rep_rate):
    """
    E_pulse: pulse power [W]
    rep_rate: Repeatition rate [Hz]
    
    https://www.thorlabs.com/images/tabimages/Laser_Pulses_Power_Energy_Equations.pdf
    """
    
    # average power
    P_avg = E_pulse * rep_rate
    
    return P_avg
    
    

def PSF(NA= 1.2, n = 1.46, sampling_z = None, shape_z = None):
    import psf
    # create an out of focus PSF
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
    
    #def Main():
    params = {
       'figure.figsize': [8, 6],
       'legend.fontsize': 12,
       'text.usetex': False,
       'ytick.labelsize': 10,
       'ytick.direction': 'out',
       'xtick.labelsize': 20,
       'xtick.direction': 'out',
       'font.size': 10,
       }
#    mpl.rcParams.update(params)
    
    
    title_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
      'verticalalignment':'bottom'} # Bottom vertical alignment for more space
    axis_font = {'fontname':'Arial', 'size':'18'}
    
#    plt.imshow(empsf.slice(100))

    args["sampling"] = [sampling_z, sampling_r]

    return empsf, args
    
    

def RFT2FT(image_in):
    mirror_x = np.flip(image_in, axis = 0)
    mirror_x = mirror_x[:-1,:] #remove last line otherwise double
    
    image_1 = np.concatenate((mirror_x , image_in), axis = 0)

    mirror_y = np.flip(image_1, axis = 1)
    mirror_y = mirror_y[:,:-1] #remove last line otherwise double
    
    image_out = np.concatenate((mirror_y , image_1), axis = 1)

    
    return image_out
    