# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 14:53:27 2021

@author: foersterronny
"""

import numpy as np # Library for array-manipulation
import matplotlib.pyplot as plt # Libraries for plotting
import NanoObjectDetection as nd


def ForceUltraUniformParticle(settings, sizes_df_lin, ShowPlot = False, save_z_score = False):
    """
    Removes particles that do not fit into the Ultra Uniform Particle Hypothesis
    """
    nd.logger.info("Drop particles that do not fit into a single ultra uniform particle hypothesis.")
    
    # select data that is required
    data = sizes_df_lin[["particle", "diameter", "diffusion", "valid frames", "rawmass_max"]]
    data = data.set_index("particle")
    
    # sort the data after valid frames - so sort them by precision!
    data = data.sort_values(["valid frames"])
    
    data_orig = data.copy()
    
    # rolling parameter
    roll_area = 100
    
    # size of scatterer in plot
    my_s = 7
    
    # define standard deviation - this is used for a rolling average later
    MyStd = lambda x: np.sqrt(np.mean((x - np.median(x))**2))

    CreatePlot = True
    
    # max z-score
    z_max = 2.576
    # 1 in 1000 events false positive: 1/(1-scipy.special.erf(3.2905/np.sqrt(2)))
    
    # Boolean that says if the remaining particles fullfill the Ultra Uniform Particle Hypothesis 
    UniFormDiameter = False

    
    if ShowPlot == True:
        # create 2x2 subplot if wanted
        fig, ax = plt.subplots(2,2, sharex = True, sharey=True)
    
    # true particle have a DIAMETER within a Gaussian distribution
    while UniFormDiameter == False:
        nd.logger.debug("length: %.0f", len(data))
        
        # estimate the mean as the MEDIAN (remove outliers) of the Uniform Particle distribution
        mean = data["diameter"].median()
        
        nd.logger.debug("mean: %.0f", mean)
        
        # estimated std of the data at each specific traj length
        data.loc[:,"std"] = data.diameter.rolling(roll_area, min_periods = 1, center = True).apply(MyStd)

        # get the z-score of each particles diameter
        data.loc[:,"z"] = np.abs((data["diameter"] - mean) / data["std"])
        
        if (CreatePlot == True) & (ShowPlot == True):
            # plot things at the beginning
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
            
            CreatePlot = False
            
            if save_z_score  == True:
                nd.handle_data.pandas2csv(data, settings["Plot"]["SaveFolder"], "z_score_before")
        
        if np.max(data["z"]) < z_max:
            # if all z-score are below the threshold the uniform particle hypothesis is true
            UniFormDiameter = True    
            
        else:
            # remove all particles beyond the maximum z-score
            data = data[data["z"] < z_max]

    
    
    # true particle have a DIFFUSION within a Gaussian distribution
    UniFormDiffusion = False
    
    # do the same thing for the diffusion now!
    while UniFormDiffusion == False:
        nd.logger.debug("length: %.0f", len(data))
        mean = data["diffusion"].median()
        nd.logger.debug("mean: %.0f", mean)
        
        data.loc[:,"std"] = data.diffusion.rolling(roll_area, min_periods = 1, center = True).apply(MyStd)
          
        data.loc[:,"z"] = np.abs((data["diffusion"] - mean) / data["std"])
        
        if np.max(data["z"]) < z_max:
            # if all z-score are below the threshold the uniform particle hypothesis is true
            UniFormDiffusion = True    
        
        else:
            data = data[data["z"] < z_max]

    
    
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
                
                
        my_title = "Force-Ultra-uniform"
        nd.visualize.export(settings["Plot"]["SaveFolder"], my_title, settings, data = data_orig)
         
        if save_z_score  == True:
            nd.handle_data.pandas2csv(data, settings["Plot"]["SaveFolder"], "z_score_after")
                
    return sizes_df_lin