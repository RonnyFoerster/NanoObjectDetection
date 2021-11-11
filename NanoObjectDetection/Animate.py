# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 11:18:16 2020

@author: foersterronny
"""

import numpy as np # library for array-manipulation
import matplotlib.pyplot as plt # libraries for plotting
from matplotlib import animation # allows to create animated plots and videos from them
import datetime
from pdb import set_trace as bp #debugger
import scipy
import os.path

import NanoObjectDetection as nd
from matplotlib.gridspec import GridSpec
    
# In[Here is the function]

def AnimateDiameterAndRawData_Big(rawframes_rot, sizes_df_lin, traj, ParameterJsonFile): 
       
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    # dont let sizes_df_lin overwrite the original pandas
    sizes_df_lin = sizes_df_lin.copy()
    
    my_gamma = 0.7

    num_points_pdf = 100

    microns_per_pixel = settings["Exp"]["Microns_per_pixel"]

    global my_font_size
    my_font_size = 16
    my_font_size_title = 22

    global ColorbarDone
    ColorbarDone = False
    
    id_particle = sizes_df_lin.particle.unique()
    
    traj = traj[traj.particle.isin(id_particle)][["frame", "particle", "x", "y"]].copy()
      
    traj["diameter"] = 0
    
    #make the particles id from 1 to N
    particles_id = sizes_df_lin.particle.unique()
    
    nd.logger.debug("Insert the retrieved diameter into the trajectory")
    
    for new_id, old_id in enumerate(particles_id):
        nd.logger.debug("Old parameter ID: %.0f", old_id)
        nd.logger.debug("New parameter ID: %.0f", new_id)
        traj.loc[traj.particle == old_id, "diameter"] = sizes_df_lin.loc[sizes_df_lin.particle == old_id, "diameter"].values[0]
        
        sizes_df_lin.loc[sizes_df_lin.particle == old_id, "particle"] = new_id
        
        traj.loc[traj.particle == old_id, "particle"] = new_id

    
    fig = plt.figure(figsize = [16, 8], constrained_layout=True)
    gs = GridSpec(4, 3, figure=fig)


    ax_raw = fig.add_subplot(gs[0, :])

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
   
    traj.x = traj.x - x_min
    traj.y = traj.y - y_min
   
    y_max = y_max - y_min
    x_max = x_max - x_min
    y_min = 0
    x_min = 0
    
   
    import matplotlib.colors as colors

    raw_image = ax_raw.imshow(rawframes_rot[0,:,:], cmap = 'gray', norm=colors.PowerNorm(gamma=my_gamma), animated = True, vmin = np.min(rawframes_rot), vmax = np.max(rawframes_rot))
  
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
    bin_width = 1
    diam_bins = int(np.round((diam_max - diam_min)/bin_width))
    # diam_min = 1
        
    fps = settings["Exp"]["fps"]

    
    #plot trajectory
    
    
    def init():
        nd.logger.info("Init")
    
    def animate(i):

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
        t6_frame = t6_frame[t6_frame.particle.isin(id_particle_frame)]
        
        t6_frame = t6_frame.set_index(["particle"])
        t6_frame = t6_frame.sort_index()
        
        ax_eval.clear()
        ax_hist.clear()
        ax_hist_cum.clear()
        ax_hist_cum_w.clear()
      
        sizes_df_lin_frame = sizes_df_lin[sizes_df_lin.particle.isin(id_particle_frame)]
        
        sizes_df_lin_frame = sizes_df_lin_frame.sort_index()
                    
        sizes_df_lin_cum = sizes_df_lin[sizes_df_lin["first frame"] <= i]
         
        # ax_scatter_diam = ax_eval.scatter(t6_frame.x, t6_frame.y, c = sizes_df_lin_frame.diameter, cmap='jet', vmin=diam_min, vmax=diam_max)
        ax_scatter_diam = ax_eval.scatter(t6_frame.x, t6_frame.y, c = t6_frame.diameter, cmap='jet', vmin=diam_min, vmax=diam_max)
 
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
                
        histogramm_min = diam_min
        histogramm_max = diam_max
        
        # https://stackoverflow.com/questions/23061657/plot-histogram-with-colors-taken-from-colormap
        # ax_hist.plot(diam_grid, prob_inv_diam)
        n, bins, patches = ax_hist.hist(sizes_df_lin_frame.diameter, bins = diam_bins, range = (diam_min, diam_max), edgecolor='black')
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        cm = plt.cm.get_cmap('jet')
        
        col = bin_centers - min(bin_centers)
        col /= max(col)
        
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))
        
        ax_hist.set_xlim([histogramm_min, histogramm_max])
        # ax_hist.set_ylim([0, 1.1*np.max(prob_inv_diam)+0.01])
 
        ax_hist.set_xlabel('Diameter [nm]', fontsize = my_font_size)    
        ax_hist.set_ylabel('Occurance', fontsize = my_font_size)
        ax_hist.set_title("Histogram - frame", fontsize = my_font_size_title)
        ax_hist.set_yticks([])
 

        ## ACCUMULATED DIAMETER PDF 
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
    
    
#    anim = animation.FuncAnimation(fig, animate, init_func=init, frames = 100, interval=100, blit=True, repeat=False)
    
#    anim = animation.FuncAnimation(fig, animate, frames = 1000, interval = 10, repeat = False)
    
    frames_tot = settings["Animation"]["frames_tot"]
    f_min = int(traj.frame.min())
    if f_min == 0:
        f_min = 1
    f_max = int(traj.frame.max())
    # show_frames = np.linspace(f_min, f_max, frames_tot , dtype = 'int')
    # show_frames = np.logspace(np.log10(f_min), np.log10(f_max), frames_tot , dtype = 'int')
    
    accelerate = settings["Animation"]["accelerate"]
    
    show_frames = np.linspace(0,1,frames_tot)**accelerate
    show_frames = show_frames  * (f_max - f_min)
    show_frames = show_frames  + f_min
    show_frames = np.round(show_frames).astype("int")  
    

    
##    anim = animation.FuncAnimation(fig, animate, frames = [1,10,50,100,200,300,400,500,600,700,800,900], init_func=init, interval = 500, repeat = True)
    
    Do_Save = True
    
    if Do_Save == True:
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



