# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 11:18:16 2020

@author: foersterronny
"""

import numpy as np # library for array-manipulation
import matplotlib.pyplot as plt # libraries for plotting
from matplotlib import animation # allows to create animated plots and videos from them
import os.path

import NanoObjectDetection as nd
from matplotlib.gridspec import GridSpec
    
import matplotlib.colors as colors

# In[Here is the function]

def AnimateDiameterAndRawData_Big(rawframes_rot, sizes_df_lin, traj, ParameterJsonFile, DoSave = False): 
       
    # read the settings
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    # dont let sizes_df_lin overwrite the original pandas
    sizes_df_lin = sizes_df_lin.copy()
    
    # standard gamma for higher visibility of dim features
    my_gamma = settings["Animation"]["gamma"]

    microns_per_pixel = settings["Exp"]["Microns_per_pixel"]

    # define some global variable to have them inside the animation
    global my_font_size
    my_font_size = 16
    my_font_size_title = 22

    global ColorbarDone
    ColorbarDone = False
    
    #make the particles id from 1 to N sothat the animation can show the particles ID without jumps
    id_particle = sizes_df_lin.particle.unique()
    
    # get the trajectories of the particles that are evaluated and create a column that can hold the retrieved diameter
    traj = traj[traj.particle.isin(id_particle)][["frame", "particle", "x", "y"]].copy()
      
    traj["diameter"] = 0

    
    nd.logger.debug("Insert the retrieved diameter into the trajectory and sort the trajectories")
    
    for new_id, old_id in enumerate(id_particle):
        nd.logger.debug("Old parameter ID: %.0f", old_id)
        nd.logger.debug("New parameter ID: %.0f", new_id)
        traj.loc[traj.particle == old_id, "diameter"] = sizes_df_lin.loc[sizes_df_lin.particle == old_id, "diameter"].values[0]
        
        sizes_df_lin.loc[sizes_df_lin.particle == old_id, "particle"] = new_id
        
        traj.loc[traj.particle == old_id, "particle"] = new_id

    # create the figure and the subplots
    fig = plt.figure(figsize = [16, 8], constrained_layout=True)
    gs = GridSpec(4, 3, figure=fig)

    
     # rawimage
    ax_raw = fig.add_subplot(gs[0, :])
    # trajectory
    ax_traj = fig.add_subplot(gs[1, :], aspect = "equal")
    # particle position with retrieved diameter
    ax_eval = fig.add_subplot(gs[2, :], aspect = "equal")
    
    # histogramm of frame
    ax_hist = fig.add_subplot(gs[3, 0])
    # cummulated
    ax_hist_cum = fig.add_subplot(gs[3, 1])
    # cummulated weighted with traj length
    ax_hist_cum_w = fig.add_subplot(gs[3, 2])
    
    # set the ROI
    x_min = settings["Animation"]["x_min"]
    x_max = settings["Animation"]["x_max"]
    y_min = settings["Animation"]["y_min"]
    y_max = settings["Animation"]["y_max"]
    
    if x_min == "auto":
        x_min = 0
    if x_max == "auto":
        x_max = rawframes_rot.shape[0]-1
    if y_min == "auto":
        y_min = 0
    if y_max == "auto":
        y_max = rawframes_rot.shape[1]-1
    
    
    if y_max >= rawframes_rot.shape[1]:
        nd.logger.warning("y_max larger than: %.0f", rawframes_rot.shape[1]-1)
        
    if x_max >= rawframes_rot.shape[2]:
        nd.logger.warning("x_max larger than: %.0f", rawframes_rot.shape[2]-1)
    
    # select the ROI you want to show
    rawframes_rot = rawframes_rot[:, y_min:y_max, x_min:x_max]
   
    # subtract the offset from the trajectories for consisting plotting
    traj.x = traj.x - x_min
    traj.y = traj.y - y_min
   
    y_max = y_max - y_min
    x_max = x_max - x_min
    y_min = 0
    x_min = 0
    
    # here come the raw image
    raw_image = ax_raw.imshow(rawframes_rot[0,:,:], cmap = 'gray', norm=colors.PowerNorm(gamma=my_gamma), animated = True, vmin = np.min(rawframes_rot), vmax = np.max(rawframes_rot))
  
    # label the axis
    ax_raw.set_title('Preprocessed image (section)', fontsize = my_font_size_title)
    ax_raw.set_ylabel('x [um]', fontsize = my_font_size)    
    ax_raw.set_xlabel('z [um]', fontsize = my_font_size)
    
    ax_raw.set_xlim([x_min, x_max])
    ax_raw.set_ylim([y_min, y_max])
    
    y_min_um = y_min * microns_per_pixel
    y_max_um = y_max * microns_per_pixel

    part_id_min = np.min(traj.particle)
    part_id_max = np.max(traj.particle)
    
    diam_max = np.round(np.max(sizes_df_lin.diameter) + 5,-1)
    diam_min = np.round(np.min(sizes_df_lin.diameter) - 5,-1)
    # diam_range = diam_max - diam_min
    
    bin_width = settings["Plot"]["Histogramm_Bin_size_nm"]
    diam_bins = int(np.round((diam_max - diam_min)/bin_width))
    # diam_min = 1
        
    fps = settings["Exp"]["fps"]


    def init():
        nd.logger.info("Init")
    
    def animate(i):

        global ColorbarDone
                
        print("animate frame: ", i)

        # update raw image
        raw_image.set_data(rawframes_rot[i,:,:])
        
        # make the tick outside
        ax_raw.tick_params(direction = 'out')
        
        # update displayed time
        time_ms = i * (1/fps) * 1000
        time_ms = np.round(time_ms,1)
        fig.suptitle('frame: ' + str(i) + '; time: ' + str(time_ms) + ' ms', fontsize = my_font_size_title)
    
        ## HERE COMES THE TRAJECTORY
        # particles in current frame
        id_particle_frame = list(traj[traj.frame == i].particle)
        
        # select all trajectories that are presend in the current frame but only that part of the trajectory that was already measured (no future events, but past) 
        t_hist = traj[traj.frame <= i]
        t_hist = t_hist[t_hist.particle.isin(id_particle_frame)]

        # plot the trajectories
        ax_traj.clear()        

        ax_scatter_traj = ax_traj.scatter(t_hist.x, t_hist.y, s = 3, c = t_hist.particle, cmap = 'gist_ncar', alpha = 0.5, vmin = part_id_min,  vmax = part_id_max)
        
        ax_traj.set_xlim(ax_raw.get_xlim())
        ax_traj.set_ylim(ax_raw.get_ylim())
        ax_traj.set_title('Evaluated trajectories', fontsize = my_font_size_title)
        ax_traj.set_ylabel('x [um]', fontsize = my_font_size)    
        ax_traj.set_xlabel('z [um]', fontsize = my_font_size)
        ax_traj.tick_params(direction = 'out')

        ## HERE COMES THE DIAMETER
        t6_frame = traj[traj.frame == i]
        t6_frame = t6_frame[t6_frame.particle.isin(id_particle_frame)]
        
        t6_frame = t6_frame.set_index(["particle"])
        t6_frame = t6_frame.sort_index()
        
        # plot the diameter
        ax_eval.clear()

        ax_scatter_diam = ax_eval.scatter(t6_frame.x, t6_frame.y, c = t6_frame.diameter, cmap='jet', vmin=diam_min, vmax=diam_max)
 
                
        # label
        ax_eval.set_title('Diameter', fontsize = my_font_size_title)
        ax_eval.set_ylabel('x [um]', fontsize = my_font_size)    
        ax_eval.set_xlabel('z [um]', fontsize = my_font_size)
        ax_eval.tick_params(direction = 'out')
       
        ax_eval.set_xlim(ax_raw.get_xlim())
        ax_eval.set_ylim(ax_raw.get_ylim())

        # make the colorbar legend one time
        if ColorbarDone == False:
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
           
        
        # Now update the histogramms
        ax_hist.clear()
        ax_hist_cum.clear()
        ax_hist_cum_w.clear()
        
        histogramm_min = diam_min
        histogramm_max = diam_max
        
        sizes_df_lin_frame = sizes_df_lin[sizes_df_lin.particle.isin(id_particle_frame)]
        
        sizes_df_lin_frame = sizes_df_lin_frame.sort_index()
        
        # https://stackoverflow.com/questions/23061657/plot-histogram-with-colors-taken-from-colormap
        n, bins, patches = ax_hist.hist(sizes_df_lin_frame.diameter, bins = diam_bins, range = (diam_min, diam_max), edgecolor='black')
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        cm = plt.cm.get_cmap('jet')
        
        col = bin_centers - min(bin_centers)
        col /= max(col)
        
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))
        
        # make the labels
        ax_hist.set_xlim([histogramm_min, histogramm_max])
        ax_hist.set_xlabel('Diameter [nm]', fontsize = my_font_size)    
        ax_hist.set_ylabel('Occurance', fontsize = my_font_size)
        ax_hist.set_title("Histogram - frame", fontsize = my_font_size_title)
        ax_hist.set_yticks([])
 

        ## ACCUMULATED DIAMETER PDF 
        sizes_df_lin_cum = sizes_df_lin[sizes_df_lin["first frame"] <= i]
        
        n, bins, patches = ax_hist_cum.hist(sizes_df_lin_cum.diameter, bins = diam_bins, range = (diam_min, diam_max), edgecolor='black')
        
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        cm = plt.cm.get_cmap('jet')
        
        col = bin_centers - min(bin_centers)
        col /= max(col)
        
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))
        
        
        ax_hist_cum.set_xlim([histogramm_min, histogramm_max])
 
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
        
        
        n, bins, patches = ax_hist_cum_w.hist(sizes_df_lin_cum.diameter, bins = diam_bins, range = (diam_min, diam_max), weights = traj_i, edgecolor='black')
        
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        cm = plt.cm.get_cmap('jet')
        
        col = bin_centers - min(bin_centers)
        col /= max(col)
        
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))
        
        ax_hist_cum_w.set_xlim([histogramm_min, histogramm_max])
 
        ax_hist_cum_w.set_xlabel('Diameter [nm]', fontsize = my_font_size)    
        ax_hist_cum_w.set_ylabel('Occurance', fontsize = my_font_size)
        ax_hist_cum_w.set_title("Cum. histogram - weighted", fontsize = my_font_size_title)
        ax_hist_cum_w.set_yticks([])

        ax_hist_cum_w.tick_params(direction = 'out')
        
        return raw_image
    
    
    # total number of frames that are animated (stretched from first to last)
    frames_tot = settings["Animation"]["frames_tot"]
    f_min = int(traj.frame.min())
    if f_min == 0:
        f_min = 1
    f_max = int(traj.frame.max())

    # have more frames at the beginning to see image processing better
    accelerate = settings["Animation"]["accelerate"]
    
    # define frames that are animated
    show_frames = np.linspace(0,1,frames_tot)**accelerate
    show_frames = show_frames  * (f_max - f_min)
    show_frames = show_frames  + f_min
    show_frames = np.round(show_frames).astype("int")  

    
    if DoSave == True:
        anim = animation.FuncAnimation(fig, animate, frames = show_frames, init_func=init, interval = 5, repeat = False)        
        
        nd.logger.info("Save the animation ...starting")
        
        # plt.animation can only save in the current directory. so we have to change it (and than go back)
        current_directory = os.getcwd()
        
        save_folder = settings["Plot"]["SaveFolder"]
        os.chdir(save_folder)
        
        anim.save("animation.html", writer = 'html', fps=10)
        
        # go back to old path
        os.chdir(current_directory)
        
        nd.logger.info("Save the animation ...finished")
    
    else:
        anim = animation.FuncAnimation(fig, animate, frames = show_frames, init_func=init, interval = 5, repeat = True)
    
    return anim



