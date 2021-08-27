# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 11:18:16 2020

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

    


def GetTrajHistory(frame, traj_roi):
    """ returns the entire history of trajectories for particles which exist in the current frame
    """
    # get particles in the frame
    id_particle_frame = list(traj_roi[traj_roi.frame == frame].particle.values)

    # select those trajectories, which exist in the current frame
    traj_roi_frame = traj_roi[traj_roi.particle.isin(id_particle_frame)]
    
    # select trajectories from the history
    traj_roi_history = traj_roi_frame[traj_roi_frame.frame <= frame]    
    
    return traj_roi_history



def GetPosEvaluated(frame, traj_roi, sizes_df_lin):
    """ return the current position of particles which are successfully evaluated, and in the current ROI and frame """
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



def AnimateProcessedRawData(ParameterJsonFile, rawframes_rot, t4_cutted, t6_final, sizes_df_lin, sizes_df_lin_rolling):
    import NanoObjectDetection as nd
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    t4_cutted_show = t4_cutted.copy()
    t6_final_show = t6_final.copy()
    
    
    fig = plt.figure(figsize=(5,14))
    
    
    ax00 = plt.subplot2grid((5,1), (0,0))
    ax10 = plt.subplot2grid((5,1), (1,0), sharex=ax00, sharey=ax00)
    ax20 = plt.subplot2grid((5,1), (2,0), sharex=ax00, sharey=ax00)
    ax30 = plt.subplot2grid((5,1), (3,0), sharex=ax00, sharey=ax00)
    ax40 = plt.subplot2grid((5,1), (4,0), sharex=ax00, sharey=ax00)
    ax40.axis('off')
    
    plt.subplots_adjust(hspace=0.7)
    
      
    time_text = ax40.text(0.1, 0.1, '', transform=ax40.transAxes, fontsize = 16, zorder=0)  
    
    skip_image_fac = settings["Time"]["skip_image_fac"]
    fps = 992
    effective_fps = fps / skip_image_fac
        
    
    start_frame = 1000
    
    #    raw_im  = ax0.imshow(rawframes_rot[my_frame,:,:])
    raw_im  = ax00.imshow(rawframes_rot[start_frame,:,:], cmap='gray')
    ax00.set_title("Raw image")
    
    particles_detected = t4_cutted_show[t4_cutted_show.frame == start_frame]
    particles_detected_im = ax10.scatter(particles_detected.x,particles_detected.y,c = 'w')
    ax10.set_title("Identified particles")
    ax10.set_facecolor((0, 0, 0))
    
    particles_analysed = t6_final_show[t6_final_show.frame == start_frame]
    particles_analysed_im = ax20.scatter(particles_analysed.x,particles_analysed.y)    
    ax20.set_title("Analyzed particles (with Diameter)")
    
    sizes_df_lin_frame = sizes_df_lin[sizes_df_lin.index.isin(particles_analysed["particle"].values)]
    
    particles_analysed_im.set_array(sizes_df_lin_frame["diameter"])
    
    min_diam = np.floor(np.min(sizes_df_lin["diameter"]))
    max_diam = np.ceil(np.max(sizes_df_lin["diameter"]))
    
    particles_analysed_im.set_clim(min_diam,max_diam)
    #    plt.colorbar(particles_analysed_im, ax=ax[2,1])
    
    
    particles_analysed_r_im = ax30.scatter(particles_analysed.x,particles_analysed.y)    
    ax30.set_title("Time-resolved diameter") 
    
    particles_analysed_r_im.set_array(sizes_df_lin_frame["diameter"])
    
    min_diam = np.floor(np.min(sizes_df_lin["diameter"]))
    max_diam = np.ceil(np.max(sizes_df_lin["diameter"]))
    
    particles_analysed_r_im.set_clim(min_diam,max_diam)
    my_colbar = plt.colorbar(particles_analysed_r_im, ax=ax40, orientation='horizontal')
    my_colbar.set_label("Diameter [nm]")
    
    
    #do all the axes
    # scale them
    ax00.axis("scaled")
    ax10.axis("scaled")
    ax20.axis("scaled")
    ax30.axis("scaled")
    
    # set borders
    ax00.set_ylim([0, rawframes_rot.shape[1]])
    ax00.set_xlim([0, rawframes_rot.shape[2]])
    
    plt.subplots_adjust(hspace=.0)
    ax30.set_xlabel("Position fiber direction [px]")
    
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Position [px]")
    
    
    # place a text box in upper left in axes coords
    
    #    textstr = "frame no.: %.0f" %(start_frame, )
    #    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    #    info_text = ax01.text(0.05, 0.95, textstr, transform=ax01.transAxes, fontsize=14,
    #        verticalalignment='top', bbox=props)
    
    def init():
        raw_im.set_array(rawframes_rot[start_frame,:,:])
        
        particles_detected = t4_cutted_show[t4_cutted_show.frame == start_frame]
        particles_detected_im.set_offsets(np.asarray(particles_detected[["x","y"]]))
        
        particles_analysed = t6_final_show[t6_final_show.frame == start_frame]
        particles_analysed_im.set_offsets(np.asarray(particles_analysed[["x","y"]])) 
        
        particles_analysed_r_im.set_offsets(np.asarray(particles_analysed[["x","y"]])) 
        
    #        textstr = "frame no.: %.0f" %(0)
    #        info_text.set_text(textstr)
        
        time_text.set_text('')
        
        return [raw_im, particles_detected_im, particles_analysed_im, particles_analysed_r_im, time_text]
    
    
    max_frame = round(len(rawframes_rot)/skip_image_fac)
    
    def animate(i):
#         start = time.time()
        i = i * skip_image_fac
        
        print("Progress %.0f of 100 " %(i/ max_frame))
        
        current_time_ms = round(i/ fps * 1000)
        
        time_text.set_text('%.0f ms' % current_time_ms)
            
        # raw image   
        raw_im.set_array(rawframes_rot[i,:,:])
        
        # tracked
        particles_detected = t4_cutted_show[t4_cutted_show.frame == i]
        particles_detected_im.set_offsets(np.asarray(particles_detected[["x","y"]]))
            
        # analyzed
        particles_analysed = t6_final_show[t6_final_show.frame == i]
        particles_analysed = particles_analysed.sort_values("particle")
        particles_analysed_im.set_offsets(np.asarray(particles_analysed[["x","y"]])) 
       
        # Set colors..
        sizes_df_lin_frame = sizes_df_lin[sizes_df_lin.index.isin(particles_analysed["particle"].values)]
        sizes_df_lin_frame = sizes_df_lin_frame.sort_values("particle")
        
        particles_analysed_im.set_array(sizes_df_lin_frame["diameter"])
        
        #evaluated
        particles_analysed_r_im.set_offsets(np.asarray(particles_analysed[["x","y"]])) 
       
        # Set colors..
        sizes_df_lin_rolling_frame = sizes_df_lin_rolling \
        [(sizes_df_lin_rolling["particle"].isin(particles_analysed["particle"].values)) & (sizes_df_lin_rolling.index == i)]
        
        sizes_df_lin_rolling_frame = sizes_df_lin_rolling_frame.sort_values("particle")
        
        particles_analysed_r_im.set_array(sizes_df_lin_rolling_frame["diameter"])
        
    #    end = time.time()
           
    #    print("processing time for frame {}: {}".format(i, end-start))
        
        return [raw_im, particles_detected_im, particles_analysed_im, particles_analysed_r_im, time_text]
    
    
    #    ani = animation.FuncAnimation(fig, animate, init_func=init, interval = 10, blit=True)
    print("start animating")
    ani = animation.FuncAnimation(fig, animate, interval = 1/effective_fps, frames = max_frame, repeat = 0, blit=True)  
    print("animating finished")
    
    
    save_folder_name = settings["Plot"]["SaveFolder"]
    save_image_name = "animate_results"
    
    my_dir_name = '%s\\{date:%y%m%d}\\'.format( date=datetime.datetime.now()) %save_folder_name
    
    try:
        os.stat(my_dir_name)
    except:
        os.mkdir(my_dir_name) 
    
    time_string = '{date:%H_%M_%S}'.format( date=datetime.datetime.now())
    
    file_name_image = '%s_%s.gif'.format( date=datetime.datetime.now()) %(time_string, save_image_name)
    
    entire_path_image = my_dir_name +  file_name_image
    
    
    print("start saving")
#    ani.save(entire_path_image, writer=PillowWriter(fps = fps))
    print("THIS IS NOT WORKING WITH PILLOW ANYMORE")
    
    print('Animation saved at: {}'.format(my_dir_name))
    
    plt.show()
    
    
    
def AnimateDiameterAndRawData(rawframes_rot, sizes_df_lin, t6_final, settings, DoScatter=True, DoText=False): 
    
    id_particle = sizes_df_lin.true_particle.unique()
        
#    fig, (ax_raw, ax_eval, ax_hist) = plt.subplots(3, sharex=True, sharey=True, figsize = [18,8])
    fig, (ax_raw, ax_eval, ax_hist) = plt.subplots(3, figsize = [18, 10], constrained_layout=True)
    
    raw_image = ax_raw.imshow(rawframes_rot[0,:,:], cmap = 'gray', animated = True)
#    diameter_image  = ax_eval.plot()
  
    ax_raw.set_title('raw-data')
    ax_raw.set_ylabel('y-Position [Pixel]')    
    ax_raw.set_xlabel('x-Position [Pixel]')
    

    
    diam_max = np.round(np.max(sizes_df_lin.diameter) + 5,-1)
    diam_min = np.round(np.min(sizes_df_lin.diameter) - 5,-1)
    
    
    fps = settings["Exp"]["fps"]
   
    
    def animate(i):
        print("animate frame: ", i)
#        background.clear()
        raw_image.set_data(rawframes_rot[i,:,:])
        
        time_ms = i * (1/fps) * 1000
        time_ms = np.round(time_ms,1)
        
        ax_raw.set_title('frame: ' + str(i) + '; time: ' + str(time_ms) + ' ms')
    
        
        t6_frame = t6_final[t6_final.frame == i]
        t6_frame = t6_frame[t6_frame.particle.isin(id_particle)]
        
        t6_frame = t6_frame.set_index(["particle"])
        t6_frame = t6_frame.sort_index()
        
        ax_eval.clear()
        ax_hist.clear()
      
        sizes_df_lin_frame = sizes_df_lin[sizes_df_lin.true_particle.isin(t6_frame.index)]
        sizes_df_lin_frame = sizes_df_lin_frame.sort_index()
            
        
        if DoScatter == True:           
            ax_scatter = ax_eval.scatter(t6_frame.x, t6_frame.y, c = sizes_df_lin_frame.diameter, cmap='jet', vmin=diam_min, vmax=diam_max)
            ax_eval.set_xlim(ax_raw.get_xlim())
            ax_eval.set_ylim(ax_raw.get_ylim())
            
            ax_eval.set_ylabel('y-Position [Pixel]')    
            ax_eval.set_xlabel('x-Position [Pixel]')
            
            if i == 2:
                fig.colorbar(ax_scatter, ax = ax_eval)

        # histogram
        num_bins = 10
        show_diameters = sizes_df_lin_frame.diameter
        
        histogramm_min = diam_min
        histogramm_max = diam_max
        
        
        diam_grid = np.linspace(histogramm_min,histogramm_max,100)
        diam_grid_inv = 1/diam_grid
        
        inv_diam,inv_diam_std = nd.CalcDiameter.InvDiameter(sizes_df_lin_frame, settings)
        
        prob_inv_diam = np.zeros_like(diam_grid_inv)
        for index, (loop_mean, loop_std) in enumerate(zip(inv_diam,inv_diam_std)):
#            print("mean_diam_part = ", 1 / loop_mean)
    
            my_pdf = scipy.stats.norm(loop_mean,loop_std).pdf(diam_grid_inv)
    
            my_pdf = my_pdf / np.sum(my_pdf)
            
            prob_inv_diam = prob_inv_diam + my_pdf    
        
        ax_hist.plot(diam_grid, prob_inv_diam)
        
        ax_hist.set_xlim([histogramm_min, histogramm_max])
        ax_hist.set_ylim([0, 1.1*np.max(prob_inv_diam)])
 
        ax_hist.set_xlabel('Diameter [nm]')    
        ax_hist.set_ylabel('Occurance')
        ax_hist.set_yticks([])
         
#        title = settings["Plot"]["Title"]
#        xlabel = "Diameter [nm]"
#        ylabel = "Probability"
#        x_lim = [histogramm_min, histogramm_max]
#        y_lim = [0, 1.1*np.max(prob_inv_diam)]
    #    sns.reset_orig()
        
        

            
            
#            if DoText == True:
#                str_diam = float(np.round(diameter,1))
#                str_diam_error = float(np.round(diameter_error,1))
#                loop_text = str(str_diam) + ' +- ' + str(str_diam_error)
#            
#                ax_eval.text(x_pos, y_pos, loop_text, fontsize=10)
##                ax_eval.text(x_pos, y_pos, diameter, fontsize=15)

        return raw_image
    
    
#    anim = animation.FuncAnimation(fig, animate, init_func=init, frames = 100, interval=100, blit=True, repeat=False)
    
    anim = animation.FuncAnimation(fig, animate, frames = 1000, interval = 10, repeat = False)
    
#    anim.save('the_movie_1.html', writer = 'html', fps=100)
    
    return anim



def AnimateDiameterAndRawData_Big(rawframes_rot, sizes_df_lin, traj, settings): 
    from matplotlib.gridspec import GridSpec
    my_gamma = 0.7

    num_points_pdf = 100

    microns_per_pixel = settings["Exp"]["Microns_per_pixel"]

    global my_font_size
    my_font_size = 16
    my_font_size_title = 22

    global ColorbarDone
    ColorbarDone = False

    global prob_inv_diam_sum
    prob_inv_diam_sum = np.zeros(num_points_pdf)
    
    id_particle = sizes_df_lin.true_particle.unique()
    
    traj = traj[traj.particle.isin(id_particle)].copy()
      
    #make the particles id from 1 to N
    particles_id = sizes_df_lin.particle.unique()
    
    print(sizes_df_lin.particle.unique())
    
    for new_id,old_id in enumerate(id_particle):
        sizes_df_lin.loc[sizes_df_lin.particle == old_id, "particle"] = new_id
        traj.loc[traj.particle == old_id, "particle"] = new_id
    
    print(sizes_df_lin.particle.unique())
    
#    fig, (ax_raw, ax_eval, ax_hist) = plt.subplots(3, sharex=True, sharey=True, figsize = [18,8])
    fig = plt.figure(figsize = [18, 9], constrained_layout=True)
    gs = GridSpec(4, 3, figure=fig)

#    fig, ax_tot = plt.subplots(4,2, figsize = [18, 10], constrained_layout=True)
    ax_raw = fig.add_subplot(gs[0, :])
#    ax_traj = fig.add_subplot(gs[1, :], sharex = ax_raw, sharey = ax_raw)
#    ax_eval = fig.add_subplot(gs[2, :], sharex = ax_raw, sharey = ax_raw)

    ax_traj = fig.add_subplot(gs[1, :], aspect = "equal")
    ax_eval = fig.add_subplot(gs[2, :], aspect = "equal")
    
    ax_hist = fig.add_subplot(gs[3, 0])
    ax_hist_cum = fig.add_subplot(gs[3, 1])
    ax_hist_cum_w = fig.add_subplot(gs[3, 2])
    
    
    x_min = settings["Animation"]["x_min"]
    x_max = settings["Animation"]["x_max"]
    y_min = settings["Animation"]["y_min"]
    y_max = settings["Animation"]["y_max"]
    
    if y_max >= rawframes_rot.shape[1]:
        nd.logger.warning("y_max larger than: %.0f", rawframes_rot.shape[1]-1)
        
    if x_max >= rawframes_rot.shape[2]:
        nd.logger.warning("x_max larger than: %.0f", rawframes_rot.shape[2]-1)
    
    rawframes_rot = rawframes_rot[:, y_min:y_max, x_min:x_max]
   
    import matplotlib.colors as colors
#    raw_image = ax_raw.imshow(rawframes_rot[0,:,:]**my_gamma, cmap = 'gray', animated = True)
    raw_image = ax_raw.imshow(rawframes_rot[0,:,:], cmap = 'gray', norm=colors.PowerNorm(gamma=my_gamma), animated = True, vmin = np.min(rawframes_rot), vmax = np.max(rawframes_rot))
#    raw_image = ax_raw.imshow(rawframes_rot[0,:,:], cmap = 'PuBu_r', animated = True)
#    diameter_image  = ax_eval.plot()
  
    ax_raw.set_title('Preprocessed image (section)', fontsize = my_font_size_title)
    ax_raw.set_ylabel('x [um]', fontsize = my_font_size)    
    ax_raw.set_xlabel('z [um]', fontsize = my_font_size)
    


    ax_raw.set_xlim([x_min, x_max])
    ax_raw.set_ylim([y_min, y_max])
    
    x_max_um = x_max * microns_per_pixel
    y_min_um = y_min * microns_per_pixel
    y_max_um = y_max * microns_per_pixel

    part_id_min = np.min(traj.particle)
    part_id_max = np.max(traj.particle)
    
    diam_max = np.round(np.max(sizes_df_lin.diameter) + 5,-1)
    diam_min = np.round(np.min(sizes_df_lin.diameter) - 5,-1)
    # diam_range = diam_max - diam_min
    bin_width = 2
    diam_bins = int(np.round((diam_max - diam_min)/bin_width))
    # diam_min = 1
        
    fps = settings["Exp"]["fps"]

    
    #plot trajectory
    
    
    def init():
        prob_inv_diam_sum[:] = 0
    
    def animate(i):
        global prob_inv_diam_sum
        global ColorbarDone
                
        print("animate frame: ", i)
#        background.clear()
        raw_image.set_data(rawframes_rot[i,:,:])
        
        time_ms = i * (1/fps) * 1000
        time_ms = np.round(time_ms,1)
        
#        ax_raw.set_title('frame: ' + str(i) + '; time: ' + str(time_ms) + ' ms')
        fig.suptitle('frame: ' + str(i) + '; time: ' + str(time_ms) + ' ms', fontsize = my_font_size_title)
    
        ## HERE COMES THE TRAJECTORY
        #particles in frame
        id_particle_frame = list(traj[traj.frame == i].particle)
        
        t_hist = traj[traj.frame <= i]
        t_hist = t_hist[t_hist.particle.isin(id_particle_frame)]
#        ax_scatter = ax_traj.scatter(t_hist.x, t_hist.y, s = 3, c = t_hist.particle, cmap='prism', vmin=part_id_min, vmax=diam_max)
        ax_traj.clear()
        ax_scatter_traj = ax_traj.scatter(t_hist.x, t_hist.y, s = 3, c = t_hist.particle, cmap = 'gist_ncar', alpha = 0.5, vmin = part_id_min,  vmax = part_id_max)
        
        ax_traj.set_xlim(ax_raw.get_xlim())
        ax_traj.set_ylim(ax_raw.get_ylim())
        ax_traj.set_title('Evaluated trajectories', fontsize = my_font_size_title)
        ax_traj.set_ylabel('x [um]', fontsize = my_font_size)    
        ax_traj.set_xlabel('z [um]', fontsize = my_font_size)

        ## HERE COMES THE DIAMETER
        t6_frame = traj[traj.frame == i]
        t6_frame = t6_frame[t6_frame.particle.isin(id_particle)]
        
        t6_frame = t6_frame.set_index(["particle"])
        t6_frame = t6_frame.sort_index()
        
        ax_eval.clear()
        ax_hist.clear()
        ax_hist_cum.clear()
        ax_hist_cum_w.clear()
      
        sizes_df_lin_frame = sizes_df_lin[sizes_df_lin.true_particle.isin(t6_frame.index)]
        sizes_df_lin_frame = sizes_df_lin_frame.sort_index()
                    
        sizes_df_lin_cum = sizes_df_lin[sizes_df_lin["first frame"] <= i]
         
        ax_scatter_diam = ax_eval.scatter(t6_frame.x, t6_frame.y, c = sizes_df_lin_frame.diameter, cmap='jet', vmin=diam_min, vmax=diam_max)
 
        ax_raw.tick_params(direction = 'out')
        ax_traj.tick_params(direction = 'out')
        ax_eval.tick_params(direction = 'out')
        
        ax_eval.set_title('Diameter', fontsize = my_font_size_title)
        ax_eval.set_ylabel('x [um]', fontsize = my_font_size)    
        ax_eval.set_xlabel('z [um]', fontsize = my_font_size)
        
       
        ax_eval.set_xlim(ax_raw.get_xlim())
        ax_eval.set_ylim(ax_raw.get_ylim())


        if ColorbarDone == False:
            print("Insert Colorbar")
            cb_raw = fig.colorbar(raw_image , ax = ax_raw)
            cb_raw.set_label("Brightness", fontsize = my_font_size)
            
            cb_traj = fig.colorbar(ax_scatter_traj , ax = ax_traj)
            cb_traj.set_label("Particle ID", fontsize = my_font_size)
            
            cb_eval = fig.colorbar(ax_scatter_diam , ax = ax_eval)
            cb_eval.set_label("Diameter [nm]", fontsize = my_font_size)
            
            ColorbarDone = True
  
        # make the labels in um
        num_x_ticks = 5
        x_ticks_px = np.linspace(0,x_max,num_x_ticks)
        
        x_ticks_um = np.array(np.round(x_ticks_px * microns_per_pixel,-1), dtype = 'int')
        
        x_ticks_px = x_ticks_um / microns_per_pixel
        
        ax_raw.set_xticks(x_ticks_px)
        ax_raw.set_xticklabels(x_ticks_um)
        
        ax_raw.set_yticks([y_min,y_max])
        ax_raw.set_yticklabels([int(y_min_um),int(y_max_um)])

        ax_traj.set_xticks(x_ticks_px)
        ax_traj.set_xticklabels(x_ticks_um)
        
        ax_traj.set_yticks([y_min,y_max])
        ax_traj.set_yticklabels([int(y_min_um),int(y_max_um)])
        
        ax_eval.set_xticks(x_ticks_px)
        ax_eval.set_xticklabels(x_ticks_um)
        
        ax_eval.set_yticks([y_min,y_max])
        ax_eval.set_yticklabels([int(y_min_um),int(y_max_um)])
        

        ## DIAMETER PDF FROM PREVIOUS POINTS
        sizes_df_lin_before_frame = sizes_df_lin_frame[i > sizes_df_lin_frame["first frame"] + settings["Link"]["Min_tracking_frames"]]
        
        histogramm_min = diam_min
        histogramm_max = diam_max
        
        
        # diam_grid = np.linspace(histogramm_min,histogramm_max,100)
        # diam_grid_inv = 1/diam_grid
        
        # #inv_diam,inv_diam_std = nd.CalcDiameter.InvDiameter(sizes_df_lin_before_frame, settings)
        # inv_diam,inv_diam_std = nd.statistics.InvDiameter(sizes_df_lin_before_frame, settings)
        
        # prob_inv_diam = np.zeros_like(diam_grid_inv)
        
#        if ('prob_inv_diam_sum' in locals()) == False:
#            global prob_inv_diam_sum
#            print("create cum sum varialbe")
#            prob_inv_diam_sum = np.zeros_like(diam_grid_inv)
       
        
#         for index, (loop_mean, loop_std) in enumerate(zip(inv_diam,inv_diam_std)):
# #            print("mean_diam_part = ", 1 / loop_mean)
    
#             my_pdf = scipy.stats.norm(loop_mean,loop_std).pdf(diam_grid_inv)
    
#             my_pdf = my_pdf / np.sum(my_pdf)
            
#             prob_inv_diam = prob_inv_diam + my_pdf    
            
#             # accumulate        
#         prob_inv_diam_sum = prob_inv_diam_sum + prob_inv_diam
        
        # ax_hist.plot(diam_grid, prob_inv_diam)
        ax_hist.hist(sizes_df_lin_frame.diameter, bins = diam_bins, range = (diam_min, diam_max), edgecolor='black')
        
        ax_hist.set_xlim([histogramm_min, histogramm_max])
        # ax_hist.set_ylim([0, 1.1*np.max(prob_inv_diam)+0.01])
 
        ax_hist.set_xlabel('Diameter [nm]', fontsize = my_font_size)    
        ax_hist.set_ylabel('Occurance', fontsize = my_font_size)
        ax_hist.set_title("Histogram - frame", fontsize = my_font_size_title)
        ax_hist.set_yticks([])
 

        ## ACCUMULATED DIAMETER PDF 
        # ax_hist_cum.plot(diam_grid, prob_inv_diam_sum)
        ax_hist_cum.hist(sizes_df_lin_cum.diameter, bins = diam_bins, range = (diam_min, diam_max), edgecolor='black')
        
        ax_hist_cum.set_xlim([histogramm_min, histogramm_max])
        # ax_hist_cum.set_ylim([0, 1.1*np.max(prob_inv_diam_sum)+0.01])
 
        ax_hist_cum.set_xlabel('Diameter [nm]', fontsize = my_font_size)    
        ax_hist_cum.set_ylabel('Occurance', fontsize = my_font_size)
        ax_hist_cum.set_title("Cum. histogram - unweighted", fontsize = my_font_size_title)
        ax_hist_cum.set_yticks([])

        ax_hist_cum.tick_params(direction = 'out')
        ax_hist.tick_params(direction = 'out')
        
        traj_i = i - sizes_df_lin_cum["first frame"] + 1
        traj_i [traj_i  < 0] = 0
        
        traj_over = traj_i  > sizes_df_lin_cum["traj length"]
        traj_i[traj_over] = sizes_df_lin_cum.loc[traj_over, "traj length"]
        
        
        ax_hist_cum_w.hist(sizes_df_lin_cum.diameter, bins = diam_bins, range = (diam_min, diam_max), weights = traj_i, edgecolor='black')
        
        ax_hist_cum_w.set_xlim([histogramm_min, histogramm_max])
        # ax_hist_cum.set_ylim([0, 1.1*np.max(prob_inv_diam_sum)+0.01])
 
        ax_hist_cum_w.set_xlabel('Diameter [nm]', fontsize = my_font_size)    
        ax_hist_cum_w.set_ylabel('Occurance', fontsize = my_font_size)
        ax_hist_cum_w.set_title("Cum. histogram - weighted", fontsize = my_font_size_title)
        ax_hist_cum_w.set_yticks([])

        ax_hist_cum_w.tick_params(direction = 'out')
        
        
        
        if 1 == 0:
            # print limits
            print("ax_raw:", ax_raw.get_xlim())
            print("ax_traj:", ax_traj.get_xlim())
            print("ax_eval:", ax_eval.get_xlim())



        return raw_image
    
    
#    anim = animation.FuncAnimation(fig, animate, init_func=init, frames = 100, interval=100, blit=True, repeat=False)
    
#    anim = animation.FuncAnimation(fig, animate, frames = 1000, interval = 10, repeat = False)
    
    frames_tot = settings["Animation"]["frames_tot"]
    show_frames = np.linspace(int(traj.frame.min()),int(traj.frame.max()),frames_tot , dtype = 'int')
    
##    anim = animation.FuncAnimation(fig, animate, frames = [1,10,50,100,200,300,400,500,600,700,800,900], init_func=init, interval = 500, repeat = True)
    
    Do_Save = True
    
    if Do_Save == True:
        anim = animation.FuncAnimation(fig, animate, frames = show_frames, init_func=init, interval = 5, repeat = False)
        anim.save('200204_2.html', writer = 'html', fps=1)
    
    else:
        anim = animation.FuncAnimation(fig, animate, frames = show_frames, init_func=init, interval = 5, repeat = True)
    
    return anim



#def GetTrajOfFrame(frame, traj_roi, sizes_df_lin):
#    print("UNUSED AND OLD")
#    # get entire trajectory of particle which are in the current frame
#    # get particles in the frame
#    id_particle_frame = list(traj_roi[traj_roi.frame == frame].particle.values)
#    
#    # select trajectores from the history
#    traj_roi_history = traj_roi[traj_roi.frame <= frame]
#    
#    # select those trajectories, which exist in the current frame
#    traj_roi_history = traj_roi_history[traj_roi_history.particle.isin(id_particle_frame)]
#        
#    
#    # get summary of current frame (sizes_df_lin_frame)
#    # trajectory of current frame
#    traj_roi_frame = traj_roi[traj_roi.frame == frame]
#    
#    # make it compareable to size_df_lin
#    traj_roi_frame = traj_roi_frame.set_index(["particle"])
#    traj_roi_frame = traj_roi_frame.sort_index()
#    
#    sizes_df_lin_roi = sizes_df_lin[np.isin(sizes_df_lin.true_particle, particle_id_traj)]
#    particle_id_final = sizes_df_lin_roi.true_particle.unique()
#    
#    sizes_df_lin_frame = sizes_df_lin[sizes_df_lin.true_particle.isin(traj_roi_frame.index)]
#    sizes_df_lin_frame = sizes_df_lin_frame.sort_index()
#    
#    #trajectory of evaluated particles
#    traj_roi_eval = traj_roi[np.isin(traj_roi.particle, particle_id_final)]
#    
#    return traj_roi_frame, traj_roi_history, sizes_df_lin_frame



def AnimateDiameterAndRawData_Big2(rawframes, static_background, rawframes_pre, 
                                   sizes_df_lin, traj, ParameterJsonFile): 
    from matplotlib.gridspec import GridSpec
    import time
    
    # sys.exit("AnimateDiameterAndRawData_Big2 has moved to animate.py! Please change if you see this")
    
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
    # inv_diam,inv_diam_std = nd.CalcDiameter.InvDiameter(sizes_df_lin, settings)
    inv_diam,inv_diam_std = nd.statistics.InvDiameter(sizes_df_lin, settings)
    
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
        
        print("\n\n New Frame")
        print("frame", frame)
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
        

#        ## ACCUMULATED DIAMETER PDF 
#        #normalized to 1
#        prob_inv_diam_sum_show = prob_inv_diam_sum / np.max(prob_inv_diam_sum)
#
#        line_diam_sum.set_ydata(prob_inv_diam_sum_show)

        print("Animation updated \n")

        return raw_image
    
    
#    anim = animation.FuncAnimation(fig, animate, init_func=init, frames = 100, interval=100, blit=True, repeat=False)
    
#    anim = animation.FuncAnimation(fig, animate, frames = 1000, interval = 10, repeat = False)
    
    min_frame = int(traj_roi.frame.min())
    max_frame = int(traj_roi.frame.max())
    show_frames = np.linspace(min_frame, max_frame,frames_tot , dtype = 'int')
    
    
    Do_Save = True

    def UpdateColorbarRawimage(stuff):
        bp()
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

    
    # HERE COME THE TEXTBOXES AND SLIDERS 
    from matplotlib.widgets import Slider, TextBox
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
     
    
    def UpdateFrame(val):
        UpdateFrame = True
        UpdateAnimation(UpdateFrame, val)
        
    def UpdateROI(val):
        UpdateFrame = False
        
        start = time.perf_counter()
        print("\n \n start: ", start)
        
        UpdateAnimation(UpdateFrame, val)   
        
        end = time.perf_counter()
        print("end : ", end)
        print("Diff Time: ", end - start)
        
        
    
    def UpdateAnimation(UpdateROI, val):
        frame = int(slider_frame.val)
        x_min = int(slider_x_min.val)
        x_max = int(slider_x_max.val)
        y_min = int(slider_y_min.val)
        y_max = int(slider_y_max.val)
        
        animate(frame, x_min, x_max, y_min, y_max, UpdateROI)
        plt.draw()
    
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

    return



def AnimateDiameterAndRawData_temporal(rawframes_rot, sizes_df_lin_rolling):  
    
    fig, (ax0, ax5) = plt.subplots(2, sharex=True, sharey=True)
#    scat5 = ax5.scatter([], [], s=60)

    im_data=rawframes_rot

    background0 = ax0.imshow(im_data, animated=True)
    background5 = ax5.imshow(im_data, animated=True)
    fig.subplots_adjust(hspace=0)
    
    def init():
#        scat5.set_offsets([])
#        im_data0=rawframes_before_filtering[0]
        im_data=rawframes_rot
        background0.set_data(im_data)
        background5.set_data(im_data)
        ax0.set_ylabel('raw-data')
#        ax5.set_ylabel('size-analyzed particles')    
#        ax5.set_xlabel('x-Position [Pixel]')
        return background0, background5, scat5,
    
    def animate(i):
        ax0.clear()
#        ax5.clear()
#        scat5 = ax5.scatter(t_long.x[t_long.frame==i].values, t_long.y[t_long.frame==i].values, s=70, 
#                            alpha = 0.4, c=t_long.particle[t_long.frame==i].values % 20)
#        im_data0=rawframes_before_filtering[i]
        im_data=rawframes_rot[i]
        background0 = ax0.imshow(im_data0, animated=True)
        background5 = ax5.imshow(im_data, animated=True)
        return background0, background5, scat5
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=t_long.frame.nunique(), 
                                   interval=1000, blit=True, repeat=False)
    
    return anim


def AnimateStationaryParticles(rawframes_np):
    # Display as video
    plt.figure()
    plt.imshow(rawframes_np[0])
    show_stationary = t1_fix[t1_fix.frame == 0]
    plt.plot(show_stationary.x,show_stationary.y,'rx')
    
    fig, (ax5) = plt.subplots(1, sharex=True, sharey=True)
    scat5 = ax5.scatter([], [], s=60)
    im_data=rawframes[0]
    background5 = ax5.imshow(im_data, animated=True)
    fig.subplots_adjust(hspace=0)
    
    
    def init():
        scat5.set_offsets([])
        im_data=rawframes[0]
        background5.set_data(im_data)
        ax5.set_ylabel('fixed particles')    
        ax5.set_xlabel('x-Position [Pixel]')
        return background5, scat5,
    
    def animate(i):
        ax5.clear()
        scat5 = ax5.scatter(t1_fix.x[t1_fix.frame==i].values, t1_fix.y[t1_fix.frame==i].values, 
                            s=70, alpha = 0.7, c=t1_fix.particle[t1_fix.frame==i].values % 20)
        im_data=rawframes[i]
        background5 = ax5.imshow(im_data, animated=True)
        return background5, scat5,
    
    #frames=t.frame.nunique()
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=t1_fix.frame.nunique(), 
                                   interval=50, blit=True, repeat=False)