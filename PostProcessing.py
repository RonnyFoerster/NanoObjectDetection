# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 14:53:27 2021

@author: foersterronny
"""

import numpy as np # Library for array-manipulation
import pandas as pd # Library for DataFrame Handling
import matplotlib.pyplot as plt # Libraries for plotting
import NanoObjectDetection as nd

def ForceUltraUniformParticle(sizes_df_lin, ShowPlot = False):
    """
    Removes particles that do not fit into the Ultra Uniform Particle Hypothesis
    """
    nd.logger.info("Drop particles that do not fit into a single ultra uniform particle hypothesis.")
    
    
    data = sizes_df_lin[["particle", "diameter", "diffusion", "valid frames", "rawmass_max"]]
    data = data.set_index("particle")
    data = data.sort_values(["valid frames"])
    
    # data = data[data.diameter < 100]
    
    roll_area = 100
    my_s = 7
    
    # def MyStd(data, mean):
        # return np.mean((data-mean)**2)
    
    MyStd = lambda x: np.sqrt(np.mean((x - np.median(x))**2))
    
    # def MyStd(*args):
    #     return np.mean((x - np.median(x))**2)
    
    ii = 0
    
    z_max = 2.576
    # 1 in 1000 events false positive: 1/(1-scipy.special.erf(3.2905/np.sqrt(2)))
    # 1 in 100 events false positive: 1/(1-scipy.special.erf(2.576/np.sqrt(2)))
    
    Done = False

    
    if ShowPlot == True:
        fig, ax = plt.subplots(2,2, sharex = True, sharey=True)
    
    # true particle have a DIAMETER within a Gaussian distribution
    while Done == False:
        nd.logger.debug("length: %.0f", len(data))
        mean = data["diameter"].median()
        
        data.loc[:,"std"] = data.diameter.rolling(roll_area, min_periods = 1, center = True).apply(MyStd)
        
        nd.logger.debug("mean: %.0f", mean)
        
        data.loc[:,"z"] = np.abs((data["diameter"] - mean) / data["std"])
        
        if ii == 0:
            if ShowPlot == True:
                min_z = 0
                max_z = data["z"].max()
                im00 = ax[0,0].scatter(data["valid frames"], data.diameter, s = my_s, c = data["z"], vmin = min_z, vmax = max_z)
                cbar00 = fig.colorbar(im00, ax = ax[0,0])
                cbar00.set_label("z-score")
                
                min_b = 0
                max_b = data["rawmass_max"].max()
                im01 = ax[0,1].scatter(data["valid frames"], data.diameter, s = my_s, c = data["rawmass_max"], vmin = min_b, vmax = max_b, cmap = 'jet')
                cbar01 = fig.colorbar(im01, ax = ax[0,1])
                cbar01.set_label("Max Brightness")
        
        if np.max(data["z"]) < z_max:
            Done = True    
        
        data = data[data["z"]<z_max]
        
        ii = ii + 1
    
    
    # true particle have a DIFFUSION within a Gaussian distribution
    Done = False
    
    while Done == False:
        nd.logger.debug("length: %.0f", len(data))
        mean = data["diffusion"].median()
        
        data.loc[:,"std"] = data.diffusion.rolling(roll_area, min_periods = 1, center = True).apply(MyStd)
        
        nd.logger.debug("mean: %.0f", mean)
        
        data.loc[:,"z"] = np.abs((data["diffusion"] - mean) / data["std"])
        
        
        if np.max(data["z"]) < z_max:
            Done = True    
        
        data = data[data["z"]<z_max]
        
        ii = ii + 1    
    
    
    
    # give back initial variable of good particles
    particle_id = data.index.unique()
    sizes_df_lin = sizes_df_lin[sizes_df_lin.particle.isin(particle_id)]
    
    if ShowPlot == True:
        im10 = ax[1,0].scatter(data["valid frames"], data.diameter, s = my_s, c = data["z"], vmin = min_z, vmax = max_z)
        cbar10 = fig.colorbar(im00, ax = ax[1,0])
        cbar10.set_label("z-score")
        
        
        im11 = ax[1,1].scatter(data["valid frames"], data.diameter, s = my_s, c = data["rawmass_max"], vmin = min_b, vmax = max_b, cmap = 'jet')
        cbar11 = fig.colorbar(im01, ax = ax[1,1])
        cbar11.set_label("Max Brightness")
        
        
        for i, row in enumerate(ax):
            for j, my_ax in enumerate(row):
                my_ax.set_xlabel("Valid Frames")
                my_ax.set_ylabel("Diameter [nm]")
                my_ax.set_yscale("log")
                my_ax.set_xscale("log")
                
                
    return sizes_df_lin