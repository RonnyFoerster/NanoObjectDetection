# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:11:51 2020

@author: foersterronny
"""

import numpy as np
import matplotlib.pyplot as plt
import NanoObjectDetection as nd

def TruthVsResponse(mean, fhwm_truth, sigma_meas, num_particles):
    #how a uniform distributed specimen is measured by a different response function
    my_min = mean - fhwm_truth/2
    my_max = mean + fhwm_truth/2
    
    num_bins = int(np.ceil(np.sqrt(num_particles)))
    
    truth = np.random.uniform(low = my_min, high=my_max, size = num_particles)
#    truth = np.random.normal(loc = mean, scale = fhwm_truth, size = num_particles)
    
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



def PhotonForceOverDiameterAndN():
    d = np.linspace(1,100,10) * 1E-9
    use_material = ["gold", "polystyrene", "DNA"]
    density_material = [19320, 1000, 1700]

    my_lambda = 532
    P = 10E-3
    radius = 25E-6
    
    I = nd.Theory.IntensityInFiber(P, radius, Mode = 'Peak')
    
    scatt_crosssection_sqnm = np.zeros([d.shape[0], len(use_material)])
    abs_crosssection_sqnm = np.zeros_like(scatt_crosssection_sqnm)
    photon_pressure = np.zeros_like(scatt_crosssection_sqnm)
    acceleration = np.zeros_like(scatt_crosssection_sqnm)
    
    
    
    for ii, loop_d in enumerate(d):
        for jj, loop_material in enumerate(use_material):
            C_scat, C_abs = nd.Simulation.CalcCrossSection(loop_d, material = loop_material , at_lambda_nm = my_lambda, do_print = False)
            scatt_crosssection_sqnm[ii,jj] = C_scat
            abs_crosssection_sqnm[ii,jj]   = C_abs
            
            photon_pressure[ii,jj] = nd.Theory.RadiationForce(I, C_scat*1E-18, C_abs*1E-18, n_media = 1.333)
           
    #get acceleration
    V = 4/3*np.pi * (d/2)**3
    
    for ii, loop_density in enumerate(density_material):
        m = V * loop_density
        acceleration[:,ii] = photon_pressure[:, ii] / m
   
         
    fig, (ax_c_scat, ax_c_abs, ax_force, ax_accel) = plt.subplots(4, sharex=True, gridspec_kw={'hspace': 0.1}, figsize=(9,12))
    
    d_nm = d * 1E9
        
    my_fontsize = 18
    
    ax_c_scat.loglog(d_nm, scatt_crosssection_sqnm, '.-')
    ax_c_scat.set_ylabel(r"$\sigma_{scat}\ [nm^2]$", fontsize = my_fontsize)
    ax_c_scat.legend(use_material)
    ax_c_scat.grid(which = 'both', axis = 'x')
    ax_c_scat.grid(which = 'major', axis = 'y')

    ax_c_abs.loglog(d_nm, abs_crosssection_sqnm, '.-')
    ax_c_abs.set_ylabel(r"$\sigma_{abs}\ [nm^2]$", fontsize = my_fontsize)
    ax_c_abs.grid(which = 'both', axis = 'x')
    ax_c_abs.grid(which = 'major', axis = 'y')
    
    ax_force.loglog(d_nm, photon_pressure, '.-')
    ax_force.set_ylabel(r"$F_{photon}\ [N]$", fontsize = my_fontsize)
    ax_force.grid(which = 'both', axis = 'x')
    ax_force.grid(which = 'major', axis = 'y')

    ax_accel.loglog(d_nm, acceleration, '.-')
    ax_accel.set_xlabel("d [nm]", fontsize = my_fontsize)
    ax_accel.set_ylabel(r"$a_{photon}\ [\frac{m}{s^2}]$", fontsize = my_fontsize)
    ax_accel.set_xlim([np.min(d_nm),np.max(d_nm)])
    ax_accel.grid(which = 'both', axis = 'x')
    ax_accel.grid(which = 'major', axis = 'y')

    fig.suptitle("P={:.3f}W, lambda={:.0f}nm, waiste={:.1f}um".format(P, my_lambda, radius*1E6), \
                 fontsize = my_fontsize)


