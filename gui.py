# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:35:59 2020

@author: foersterronny
"""
import numpy as np # library for array-manipulation
import pandas as pd # library for DataFrame handling
import trackpy as tp # trackpy offers all tools needed for the analysis of diffusing particles
import seaborn as sns # allows boxplots to be put in one graph
#sns.reset_orig()
import math # offering some maths functions
import matplotlib
#matplotlib.rcParams['text.usetex'] = False
#matplotlib.rcParams['text.latex.unicode'] = False
import matplotlib.pyplot as plt # libraries for plotting
from matplotlib import animation # allows to create animated plots and videos from them
import json
import sys
import datetime
from pdb import set_trace as bp #debugger
import scipy
import os.path

import NanoObjectDetection as nd

# In[]





def AnimateDiameterAndRawData_Big2(rawframes, static_background, rawframes_pre, sizes_df_lin, traj, ParameterJsonFile): 
    from matplotlib.gridspec import GridSpec
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
        
    fps = settings["Exp"]["fps"]
    
    my_gamma = settings["Animation"]["gamma"]
    microns_per_pixel = settings["Exp"]["Microns_per_pixel"]
    frames_tot = settings["Animation"]["frames_tot"]
    num_points_pdf = 100

    # Diameter plot
    histogramm_min = settings["Plot"]["Histogramm_min"]
    histogramm_max = settings["Plot"]["Histogramm_max"]
    
    diam_grid = np.linspace(histogramm_min,histogramm_max,1000)
    diam_grid_inv = 1/diam_grid

    # here comes the font sizes
    global my_font_size
    my_font_size = 16
    my_font_size_title = 22

    global prob_inv_diam_sum
    prob_inv_diam_sum = np.zeros(num_points_pdf) 


    # get min and max particle id
    part_id_min = np.min(traj.particle)
    part_id_max = np.max(traj.particle)
    
    # get min and max diameter
    diam_max = np.round(np.max(sizes_df_lin.diameter) + 5,-1)
    diam_min = np.round(np.min(sizes_df_lin.diameter) - 5,-1)

    # get particle id of TRAJECTORIES in the roi
#    particle_id_traj = traj[(traj.x > 800) & (traj.x < 1800)].particle.unique()
    particle_id_traj = traj.particle.unique()

    #get traj in ROI
    traj_roi = traj[np.isin(traj.particle, particle_id_traj)]
    
    frame = 0
    # get trajectory of particles in current frame
    traj_roi_history = GetTrajHistory(frame, traj_roi)

    # get position and diameter of evaluated particles
    pos_roi, sizes_df_lin_frame = GetPosEvaluated(frame, traj_roi, sizes_df_lin)
  
    # design the subplot
    fig = plt.figure(figsize = [25, 13], constrained_layout=True)

#    gs = GridSpec(6, 3, figure=fig, width_ratios = [0.5,0.5,0.2])
    gs = GridSpec(12, 5, figure=fig, width_ratios = [1]*2 + [0.15]*3, height_ratios = [1/2]*(2*3) + [1]*2 + [1.5] + [0.5]*3)

    ax_raw = fig.add_subplot(gs[0:2, 0:2], aspect = "equal") # raw image
    ax_bg = fig.add_subplot(gs[2:4, 0:2], aspect = "equal", sharex = ax_raw, sharey = ax_raw)  # background 
    ax_pp = fig.add_subplot(gs[4:6, 0:2], aspect = "equal", sharex = ax_raw, sharey = ax_raw)  # post-processed image
    ax_traj = fig.add_subplot(gs[6, 0:2], aspect = "equal", sharex = ax_raw, sharey = ax_raw) # trajectories
    ax_eval = fig.add_subplot(gs[7, 0:2], aspect = "equal", sharex = ax_raw, sharey = ax_raw) # particles colour in diameter
    
    ax_hist = fig.add_subplot(gs[8, 0]) # histogram of current frame
    ax_hist_cum = fig.add_subplot(gs[8, 1], sharex = ax_hist, sharey = ax_hist) # summed histogram

    # axis for the colorbars / legends 
    c_ax_raw = plt.subplot(gs[0, 2:5])
    c_ax_bg = plt.subplot(gs[2, 2:5])
    c_ax_pp = plt.subplot(gs[4, 2:5])
    c_ax_traj = plt.subplot(gs[6, 2:5])
    c_ax_eval = plt.subplot(gs[7, 2:5])
    
    # axis for min, max and gamma values
    ax_raw_min = plt.subplot(gs[1, 2])
    ax_raw_max = plt.subplot(gs[1, 3])
    ax_raw_g = plt.subplot(gs[1, 4])
    
    ax_bg_min = plt.subplot(gs[3, 2])
    ax_bg_max = plt.subplot(gs[3, 3])
    ax_bg_g = plt.subplot(gs[3, 4])
    
    ax_pp_min = plt.subplot(gs[5, 2])
    ax_pp_max = plt.subplot(gs[5, 3])
    ax_pp_g = plt.subplot(gs[5, 4])
    
    
    #here come the sliders
    slider_frame_ax = plt.subplot(gs[9, 0:2])
    slider_x_min_ax = plt.subplot(gs[10, 0])
    slider_x_max_ax = plt.subplot(gs[11, 0])
    slider_y_min_ax = plt.subplot(gs[10, 1])
    slider_y_max_ax = plt.subplot(gs[11, 1])
    

    # plot the stuff
    import matplotlib.colors as colors
    raw_image = ax_raw.imshow(rawframes[0,:,:], cmap = 'gray', norm=colors.PowerNorm(gamma=my_gamma), animated = True, vmin = 0, vmax = np.max(rawframes))
    
    bg_image = ax_bg.imshow(static_background, cmap = 'gray', norm=colors.PowerNorm(gamma=my_gamma), animated = True, vmin = 0, vmax = np.max(static_background))

    pp_image = ax_pp.imshow(rawframes_pre[0,:,:], cmap = 'gray', norm=colors.PowerNorm(gamma=my_gamma), animated = True, vmin = np.min(rawframes_pre), vmax = np.max(rawframes_pre))
    
    ax_scatter_traj = ax_traj.scatter(traj_roi_history.x, traj_roi_history.y, s = 3, c = traj_roi_history.particle, cmap = 'gist_ncar', alpha = 1, vmin = 0,  vmax = part_id_max)
    
#    ax_scatter_traj = ax_traj.scatter(traj_roi_history.x, traj_roi_history.y, s = 3, c = traj_roi_history.particle, cmap = 'gist_ncar', alpha = 1, vmin=part_id_min, vmax=part_id_max)

    ax_scatter_diam = ax_eval.scatter(pos_roi.x, pos_roi.y, c = sizes_df_lin_frame.diameter, cmap='gist_ncar', vmin=diam_min, vmax=diam_max)   


    # add titles and labels
    ax_raw.set_title('raw-data', fontsize = my_font_size_title)
    ax_raw.set_ylabel('y-Position [px]', fontsize = my_font_size)    

    ax_bg.set_title('Background and stationary particles', fontsize = my_font_size_title)
    ax_bg.set_ylabel('y-Position [px]', fontsize = my_font_size)    

    ax_pp.set_title('Processed image', fontsize = my_font_size_title)
    ax_pp.set_ylabel('y-Position [px]', fontsize = my_font_size)    

    ax_traj.set_title('trajectory', fontsize = my_font_size_title)
    ax_traj.set_ylabel('y-Position [px]', fontsize = my_font_size)   
  
    ax_eval.set_title('Diameter of each particle', fontsize = my_font_size_title)
    ax_eval.set_ylabel('y-Position [px]', fontsize = my_font_size)    
    ax_eval.set_xlabel('x-Position [px]', fontsize = my_font_size)
    
    
    # COLORBARS
    from matplotlib.colorbar import Colorbar     
            
    cb_raw = Colorbar(ax = c_ax_raw, mappable = raw_image, orientation = 'horizontal', ticklocation = 'top')
    cb_raw.set_label("Brightness", fontsize = my_font_size)
  
    cb_bg = Colorbar(ax = c_ax_bg, mappable = bg_image, orientation = 'horizontal', ticklocation = 'top')
    cb_bg.set_label("Brightness", fontsize = my_font_size)
    
    cb_pp = Colorbar(ax = c_ax_pp, mappable = pp_image, orientation = 'horizontal', ticklocation = 'top')
    cb_pp.set_label("Brightness", fontsize = my_font_size)
    
    cb_traj = Colorbar(ax = c_ax_traj, mappable = ax_scatter_traj, orientation = 'horizontal', ticklocation = 'top')
    cb_traj.set_label("Particle ID", fontsize = my_font_size) 
    
    cb_eval = Colorbar(ax = c_ax_eval, mappable = ax_scatter_diam, orientation = 'horizontal', ticklocation = 'top')
    cb_eval.set_label("Diameter [nm]", fontsize = my_font_size)

    from matplotlib import ticker
    cb_raw.locator = ticker.MaxNLocator(nbins=3)
    cb_raw.update_ticks()

    cb_bg.locator = ticker.MaxNLocator(nbins=3)
    cb_bg.update_ticks()

    cb_pp.locator = ticker.MaxNLocator(nbins=3)
    cb_pp.update_ticks()

    cb_traj.locator = ticker.MaxNLocator(nbins=5)
    cb_traj.update_ticks()

    cb_eval.locator = ticker.MaxNLocator(nbins=5)
    cb_eval.update_ticks()    

    # Here come the two histograms
    line_diam_frame, = ax_hist.plot(diam_grid, np.zeros_like(diam_grid))
    line_diam_sum, = ax_hist_cum.plot(diam_grid, np.zeros_like(diam_grid)) 
    
    # label and title
    ax_hist.set_xlabel('Diameter [nm]', fontsize = my_font_size)    
    ax_hist.set_ylabel('Occurance', fontsize = my_font_size)
    ax_hist.set_title("Live Histogram", fontsize = my_font_size_title)

    ax_hist_cum.set_xlabel('Diameter [nm]', fontsize = my_font_size)    
    ax_hist_cum.set_ylabel('Occurance', fontsize = my_font_size)
    ax_hist_cum.set_title("Cummulated Histogram", fontsize = my_font_size_title)
    
    
    # limits
    ax_hist.set_xlim([histogramm_min, histogramm_max])
    ax_hist.set_ylim([0, 1.1])
    ax_hist.set_yticks([])
    ax_hist.tick_params(direction = 'out')


    # Global PDF
    inv_diam,inv_diam_std = nd.CalcDiameter.InvDiameter(sizes_df_lin, settings)
    
    prob_inv_diam = np.zeros_like(diam_grid_inv)
          
    for index, (loop_mean, loop_std, weight) in enumerate(zip(inv_diam,inv_diam_std, sizes_df_lin["traj length"])):
        #loop through all evaluated partices in that roi and frame
        
        #calc probability density function (PDF)
        my_pdf = scipy.stats.norm(loop_mean,loop_std).pdf(diam_grid_inv)

        # normalized nad weight
        my_pdf = my_pdf / np.sum(my_pdf) * weight
        
        #add up all PDFs
        prob_inv_diam = prob_inv_diam + my_pdf

    #normalized to 1
    prob_inv_diam_show = prob_inv_diam / np.max(prob_inv_diam)    
    
    line_diam_sum.set_ydata(prob_inv_diam_show)



    
    def animate(frame, x_min, x_max, y_min, y_max, UpdateFrame):
        global ColorbarDone
        print("\nframe", frame)
        print("x_min", x_min)
        print("x_max", x_max)
        print("y_min", y_min)
        print("y_max", y_max)
        print("Update Frame", UpdateFrame)
        
        # select new frame if required
        if UpdateFrame == True:
            rawframes_frame = rawframes[frame,:,:]
            rawframes_pp_frame = rawframes_pre[frame,:,:]
            
            raw_image.set_data(rawframes_frame)
            bg_image.set_data(static_background)
            pp_image.set_data(rawframes_pp_frame)
        
        # SET AXES
        ax_raw.set_xlim([x_min, x_max])
        ax_raw.set_ylim([y_min, y_max])
    
        ax_raw.tick_params(direction = 'out')
    
        # make the labels in um
        num_x_ticks = 5
        num_y_ticks = 3
        
        x_ticks_px = np.round(np.linspace(x_min,x_max,num_x_ticks, dtype = 'int'),-2)
        y_ticks_px = np.round(np.linspace(y_min,y_max,num_y_ticks, dtype = 'int'),-1)
        
        ax_raw.set_xticks(x_ticks_px)
        ax_raw.set_xticklabels(x_ticks_px)
        
        ax_raw.set_yticks(y_ticks_px)
        ax_raw.set_yticklabels(y_ticks_px)        
        
              
        # get particle id of TRAJECTORIES in the roi
        particle_id_traj = traj[(traj.x > x_min) & (traj.x < x_max)].particle.unique()
    
        #get traj in ROI
        traj_roi = traj[traj.particle.isin(particle_id_traj)]
        
        # get trajectory of particles in current frame
        traj_roi_history = GetTrajHistory(frame, traj_roi)

        # get position and diameter of evaluated particles
        pos_roi, sizes_df_lin_roi_frame = GetPosEvaluated(frame, traj_roi, sizes_df_lin)


        #update figure
        time_ms = frame * (1/fps) * 1000
        time_ms = np.round(time_ms,1)

        fig.suptitle('frame: ' + str(frame) + '; time: ' + str(time_ms) + ' ms', fontsize = my_font_size_title)

        ax_scatter_traj.set_offsets(np.transpose(np.asarray([traj_roi_history.x.values, traj_roi_history.y.values])))
        ax_scatter_traj.set_array(traj_roi_history.particle)

        ax_scatter_diam.set_offsets(np.transpose(np.asarray([pos_roi.x.values, pos_roi.y.values])))
        ax_scatter_diam.set_array(sizes_df_lin_roi_frame.diameter) 

        
        
        
        ## DIAMETER PDF FROM PREVIOUS POINTS
        sizes_df_lin_roi_frame = sizes_df_lin_roi_frame.sort_index()
        
        # get inverse diameter which is normal distributed (proportional to the diffusion)
        inv_diam,inv_diam_std = nd.CalcDiameter.InvDiameter(sizes_df_lin_roi_frame, settings)

        # probability of inverse diameter
        prob_inv_diam = np.zeros_like(diam_grid_inv)
              
        for index, (loop_mean, loop_std) in enumerate(zip(inv_diam,inv_diam_std)):
            #loop through all evaluated partices in that roi and frame
    
            #calc probability density function (PDF)
            my_pdf = scipy.stats.norm(loop_mean,loop_std).pdf(diam_grid_inv)
    
            # normalized
            my_pdf = my_pdf / np.sum(my_pdf)
            
            #add up all PDFs
            prob_inv_diam = prob_inv_diam + my_pdf    
                    
        #normalized to 1
        if np.max(prob_inv_diam) > 0:
            prob_inv_diam_show = prob_inv_diam / np.max(prob_inv_diam)
        else:
            prob_inv_diam_show = prob_inv_diam 

        line_diam_frame.set_ydata(prob_inv_diam_show)
        
        
        
#        
#    
#
#        ## ACCUMULATED DIAMETER PDF 
#        #normalized to 1
#        prob_inv_diam_sum_show = prob_inv_diam_sum / np.max(prob_inv_diam_sum)
#
#        line_diam_sum.set_ydata(prob_inv_diam_sum_show)

        print("Animation updated")

        return raw_image
    
    # Functions for Update the ROI and brightness
    def UpdateFrame(val):
        UpdateFrame = True
        UpdateAnimation(UpdateFrame, val)
        
    def UpdateROI(val):
        UpdateFrame = False
        UpdateAnimation(UpdateFrame, val)   
        
    
    def UpdateAnimation(UpdateROI, val):
        frame = int(slider_frame.val)
        x_min = int(slider_x_min.val)
        x_max = int(slider_x_max.val)
        y_min = int(slider_y_min.val)
        y_max = int(slider_y_max.val)
        
        animate(frame, x_min, x_max, y_min, y_max, UpdateROI)
        plt.draw()
    
    def UpdateColorbarRawimage(stuff):
        v_min = np.int(textbox_raw_min.text)
        v_max = np.int(textbox_raw_max.text)
        my_gamma = np.double(textbox_raw_g.text)
                
        print("\n v_min = ", v_min)
        print("v_max = ", v_max)
        print("gamma = ", my_gamma)
        
        raw_image.set_norm(colors.PowerNorm(gamma = my_gamma))
        raw_image.set_clim([v_min, v_max])
        
        cb_raw = Colorbar(ax = c_ax_raw, mappable = raw_image, orientation = 'horizontal', ticklocation = 'top')
        cb_raw.locator = ticker.MaxNLocator(nbins=3)
        cb_raw.update_ticks()
    
    def UpdateColorbarBg(stuff):
        v_min = np.int(textbox_bg_min.text)
        v_max = np.int(textbox_bg_max.text)
        my_gamma = np.double(textbox_bg_g.text)
                
        print("\n v_min = ", v_min)
        print("v_max = ", v_max)
        print("gamma = ", my_gamma)
        
        bg_image.set_norm(colors.PowerNorm(gamma = my_gamma))
        bg_image.set_clim([v_min, v_max])
        
        cb_bg = Colorbar(ax = c_ax_bg, mappable = bg_image, orientation = 'horizontal', ticklocation = 'top')
        cb_bg.locator = ticker.MaxNLocator(nbins=3)
        cb_bg.update_ticks()
        
    def UpdateColorbarPP(stuff):
        v_min = np.int(textbox_pp_min.text)
        v_max = np.int(textbox_pp_max.text)
        my_gamma = np.double(textbox_pp_g.text)
                
        print("\n v_min = ", v_min)
        print("v_max = ", v_max)
        print("gamma = ", my_gamma)
        
        pp_image.set_norm(colors.PowerNorm(gamma = my_gamma))
        pp_image.set_clim([v_min, v_max])
        
        cb_pp = Colorbar(ax = c_ax_pp, mappable = pp_image, orientation = 'horizontal', ticklocation = 'top')
        cb_pp.locator = ticker.MaxNLocator(nbins=3)
        cb_pp.update_ticks()
#    anim = animation.FuncAnimation(fig, animate, init_func=init, frames = 100, interval=100, blit=True, repeat=False)
    
#    anim = animation.FuncAnimation(fig, animate, frames = 1000, interval = 10, repeat = False)
    
    min_frame = int(traj_roi.frame.min())
    max_frame = int(traj_roi.frame.max())
    show_frames = np.linspace(min_frame, max_frame,frames_tot , dtype = 'int')
    
    
    Do_Save = True



    
    # HERE COME THE TEXTBOXES AND SLIDERS 
    from matplotlib.widgets import Slider, Button, TextBox
    #raw image
    textbox_raw_min = TextBox(ax_raw_min , "min: ", initial = str(np.min(rawframes[0,:,:])), hovercolor = "y")    
    textbox_raw_min.on_submit(UpdateColorbarRawimage)
    
    textbox_raw_max = TextBox(ax_raw_max , "max: ", initial = str(np.max(rawframes[0,:,:])), hovercolor = "y")
    textbox_raw_max.on_submit(UpdateColorbarRawimage)
    
    textbox_raw_g = TextBox(ax_raw_g , "gamma: ", initial = str(my_gamma), hovercolor = "y")
    textbox_raw_g.on_submit(UpdateColorbarRawimage)

    #bg
    textbox_bg_min = TextBox(ax_bg_min , "min: ", initial = str(np.int(np.min(static_background))), hovercolor = "y")    
    textbox_bg_min.on_submit(UpdateColorbarBg)
    
    textbox_bg_max = TextBox(ax_bg_max , "max: ", initial = str(np.int(np.max(static_background))), hovercolor = "y")
    textbox_bg_max.on_submit(UpdateColorbarBg)
    
    textbox_bg_g = TextBox(ax_bg_g , "gamma: ", initial = str(my_gamma), hovercolor = "y")
    textbox_bg_g.on_submit(UpdateColorbarBg)


    #preproccesd (pp)
    textbox_pp_min = TextBox(ax_pp_min , "min: ", initial = str(np.int(np.min(rawframes_pre[0,:,:]))), hovercolor = "y")    
    textbox_pp_min.on_submit(UpdateColorbarPP)
    
    textbox_pp_max = TextBox(ax_pp_max , "max: ", initial = str(np.int(np.max(rawframes_pre[0,:,:]))), hovercolor = "y")
    textbox_pp_max.on_submit(UpdateColorbarPP)
    
    textbox_pp_g = TextBox(ax_pp_g , "gamma: ", initial = str(my_gamma), hovercolor = "y")
    textbox_pp_g.on_submit(UpdateColorbarPP)
    



    # sliders
    frame_max = rawframes.shape[0] - 1
    x_max_max = rawframes.shape[2] - 1
    y_max_max = rawframes.shape[1] - 1
    

    slider_frame = Slider(slider_frame_ax , "Frame: ", valmin = 0, valmax = frame_max, valinit=0, valstep=1)
    slider_x_min = Slider(slider_x_min_ax , "x_min: ", valmin = 0, valmax = x_max_max, valinit=0, valstep=1)
    slider_x_max = Slider(slider_x_max_ax , "x_max: ", valmin = 0, valmax = x_max_max, valinit=x_max_max, valstep=1, slidermin=slider_x_min)
    slider_y_min = Slider(slider_y_min_ax , "y_min: ", valmin = 0, valmax = y_max_max, valinit=0, valstep=1)
    slider_y_max = Slider(slider_y_max_ax , "y_max: ", valmin = 0, valmax = y_max_max, valinit=y_max_max, valstep=1, slidermin = slider_y_min)
     
    
    slider_frame.on_changed(UpdateFrame)
    slider_x_min.on_changed(UpdateROI)
    slider_x_max.on_changed(UpdateROI)
    slider_y_min.on_changed(UpdateROI)
    slider_y_max.on_changed(UpdateROI)

    
    
    plt.show()
    
#    if Do_Save == True:
#        anim = animation.FuncAnimation(fig, animate, frames = show_frames, init_func=init, interval = 0, repeat = False)
#        anim.save('200204_2.html', writer = 'html', fps=1)
#    
#    else:
#        anim = animation.FuncAnimation(fig, animate, frames = show_frames, init_func=init, interval = 5, repeat = True)
    
#    return anim

    print("animation loaded")

    return




def GetTrajHistory(frame, traj_roi):
    # returns the entire history pf trajectories for particle which exist in the current frame
    # get particles in the frame
    id_particle_frame = list(traj_roi[traj_roi.frame == frame].particle.values)

    # select those trajectories, which exist in the current frame
    traj_roi_frame = traj_roi[traj_roi.particle.isin(id_particle_frame)]
    
    # select trajectores from the history
    traj_roi_history = traj_roi_frame[traj_roi_frame.frame <= frame]    
    
    return traj_roi_history


def GetPosEvaluated(frame, traj_roi, sizes_df_lin):
    #return the current position of particles which are successfully evaluated, and in the current ROI and frame
    traj_roi_frame = traj_roi[traj_roi.frame == frame]

    # id of present particles
    id_particle_frame =list(traj_roi_frame.particle.values)

    # make it compareable to size_df_lin
    pos_roi = traj_roi_frame.set_index(["particle"])
    pos_roi = pos_roi.sort_index()
    
    # select only particles which are evaluated in current frame and roi
    # that does not mean that all trajectories are evaluated
    sizes_df_lin_roi_frame = sizes_df_lin[sizes_df_lin.true_particle.isin(id_particle_frame)]
    
    # id of evaluated particles - remove unevaluated trajectories
    id_particle_eval = sizes_df_lin_roi_frame.true_particle.unique()
    
    sizes_df_lin_roi_frame = sizes_df_lin_roi_frame.sort_index()
    
    #trajectory of evaluated particles
    pos_roi = pos_roi.loc[id_particle_eval]
    pos_roi = pos_roi.sort_index()

    
    return pos_roi, sizes_df_lin_roi_frame