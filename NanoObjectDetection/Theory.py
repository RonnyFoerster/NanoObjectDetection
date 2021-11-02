# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 08:44:12 2020

@author: foersterronny

collection of standard equations of physics, unit conversions etc.
"""


import numpy as np

import multiprocessing
import NanoObjectDetection as nd
from joblib import Parallel, delayed
from scipy.constants import Boltzmann as k_b
from scipy.constants import pi as pi
from scipy.constants import speed_of_light as c



def SigmaPSF(NA, mylambda):
    """
    Approximates the std of a PSF

    Parameters
    ----------
    NA : TYPE
        DESCRIPTION.
    mylambda : TYPE
        DESCRIPTION.

    Returns
    -------
    sigma : TYPE
        DESCRIPTION.

    """
    # Zhang et al 2007
    sigma = 0.21 * mylambda/ NA
    
    return sigma
    

def LocalErrorStatic(sigma_psf, photons):
    """
    Static Localization Error given by PSF FWHM and the number of Photons

    Parameters
    ----------
    sigma_psf : TYPE
        DESCRIPTION.
    photons : TYPE
        DESCRIPTION.

    Returns
    -------
    ep : TYPE
        DESCRIPTION.

    """
    ep = sigma_psf / np.sqrt(photons) 

    return ep



def CRLB(N,x):
    """
    compute the Cramer-Rao lower bound from experimental parameters
    according to Michalet & Berglund 2012, eq. (12)

    Parameters
    ----------
    N : TYPE
        number of frames
    x : TYPE
        reduced localization precision.

    Returns
    -------
    eps : TYPE
        relative error on diffusion (or size, resp.), i.e. eps = std(D)/D.

    """
    eps = np.sqrt( 2/(N-1) * (1 + 2*np.sqrt(1+2*x)) )
    
    return eps



def RedXOutOfTheory(diffusion, expTime, lagtime, NA, wavelength, photons):
    """
    compute reduced square localization error from theory
    
    cf. Michalet&Berglund 2012    

    Parameters
    ----------
    diffusion : TYPE
        DESCRIPTION.
    expTime : TYPE
        DESCRIPTION.
    lagtime : TYPE
        DESCRIPTION.
    NA : TYPE
        DESCRIPTION.
    wavelength : TYPE
        DESCRIPTION.
    photons : TYPE
        DESCRIPTION.

    Returns
    -------
    red_x : TYPE
        DESCRIPTION.

    """
    
    # motion blurr coefficient (case of uniform exposure; paragraph behind eq.(5))
    R = 1/6 * expTime/lagtime
    
    # standard dev. of a Gaussian approximation of the microscope point-spread function,
    # i.e. resolution of the optical system
    s0 = SigmaPSF(NA, wavelength/1000)
    # static localization error
    sigma0 = LocalErrorStatic(s0, photons)
    
    # dynamic localization error
    sigma = sigma0 * (1 + diffusion*expTime/s0**2)**0.5 # eq.(1)
    
    red_x = sigma**2/(diffusion*lagtime) - 2*R # eq.(4)
    
    return red_x



def StokesEinsteinEquation(diff = None, diam = None, temp_water = 295, visc_water = 9.5E-16):
    """
    solves the Stokes-Einstein equation either for the diffusion coefficient or the 
    hydrodynamic diameter of a particle    

    Parameters
    ----------
    diff : TYPE, optional
        diffusion coefficient [m^2/s]. The default is None.
    diam : TYPE, optional
        particle diameter [m]    . The default is None.
    temp_water : TYPE, optional
        DESCRIPTION. The default is 295.
    visc_water : TYPE, optional
        DESCRIPTION. The default is 9.5E-16.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    my_return : TYPE
        DESCRIPTION.  
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



# def RadiationForce(I, C_scat, C_abs, n_media = 1.333):   
#     """ 
#     THIS IS SWITCHED OFF BECAUSE THE CALLING FUNCTION DOES NOT WORK DUE TO PROBLEMS WITH MIEPYTHON
# calculate the radiation force onto a scattering sphere

    
#     Literature:
#     https://reader.elsevier.com/reader/sd/pii/0030401895007539?token=48F2795599992EB11281DD1C2A50B58FC6C5F2614C90590B9700CD737B0B9C8E94F2BB8A17F74D0E6087FF3B7EF5EF49
#     https://github.com/scottprahl/miepython/blob/master/doc/01_basics.ipynb
    
#     lambda_nm:  wavelength of the incident light
#     d_nm:       sphere's diameter
#     P_W:        incident power
#     A_sqm:      beam/channel cross sectional area
#     material:   sphere's material
    
#     "Optical trapping of metallic Rayleigh particles"
#     "Radiation forces on a dielectric sphere in the Rayleigh scattering regime"
#     """
    
#     #Eq 11 in Eq 10
#     #not quite sure here
#     F_scat = C_scat * n_media/c * I
#     F_abs = C_abs * n_media/c * I


#     return F_scat
