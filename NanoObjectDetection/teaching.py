# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:11:51 2020

Some functions to demonstrate a few effects that happen in measurement, analysis, etc.

@author: foersterronny
"""

import numpy as np
import matplotlib.pyplot as plt
import NanoObjectDetection as nd

def TruthVsResponse(mean, fhwm_truth, sigma_meas, num_particles):
    """
    how a uniform distributed specimen is measured by a different response function

    Parameters
    ----------
    mean : TYPE
        Mean of true data.
    fhwm_truth : TYPE
        fwhm of data (data is uniform distributed) .
    sigma_meas : TYPE
        std of measuring process.
    num_particles : TYPE
        number of evalued particles.

    Returns
    -------
    None.

    """
    
    
    my_min = mean - fhwm_truth/2
    my_max = mean + fhwm_truth/2
    
    num_bins = int(np.ceil(np.sqrt(num_particles)))
    
    truth = np.random.uniform(low = my_min, high=my_max, size = num_particles)
    
    measurement = np.random.normal(loc = truth, scale = sigma_meas)

    inv_measurement = 1/measurement

    fig, (ax1_truth, ax2_meas, ax3_inv_meas) = plt.subplots(3, 1, figsize=(10,12))
    ax1_truth.hist(truth, bins = num_bins)
    ax1_truth.set_title("Truth; uniform with FWHM: " + str(fhwm_truth), fontsize = 16)
    ax1_truth.set_ylabel("occurance", fontsize = 14)
    ax1_truth.set_xlabel("Value", fontsize = 14)

    ax2_meas.hist(measurement, bins = num_bins)
    ax2_meas.set_title("Measurment: sigma gaussian response function : " + str(sigma_meas), fontsize = 16)
    ax2_meas.set_ylabel("Measure Value", fontsize = 14)
    ax2_meas.set_xlabel("Value", fontsize = 14)


    ax3_inv_meas.hist(inv_measurement, bins = num_bins)
    ax3_inv_meas.set_title("Inverse of the measurment", fontsize = 16)
    ax3_inv_meas.set_ylabel("Inverse Value", fontsize = 14)
    ax3_inv_meas.set_xlabel("Value", fontsize = 14)

    ax1_truth.set_xlim(ax2_meas.get_xlim())
    fig.tight_layout()



def Correl(a,b):
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    b = (b - np.mean(b)) / (np.std(b))
    c = np.correlate(a, b, 'same')
    
    return c


def ZnccLevel():
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    import numpy as np 
    
    Photons = 1000
    bg = 100
    
    #initialize a normal distribution with frozen in mean=-1, std. dev.= 1
    Gauss = norm(loc = 5, scale = 0.5)
    
    x = np.arange(0, 10, .1)
    
    PSF = Gauss.pdf(x)
    PSF = PSF / np.max(PSF)
    I_image = PSF * Photons + bg
    
    I_image = np.random.poisson(I_image)
    
    #plot the pdfs of these normal distributions 
    plt.plot(x, I_image)
    
    cc = nd.teaching.Correl(PSF, I_image)
    plt.plot(x, cc)
    
    

