# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 12:23:39 2019

@author: foersterronny
"""

# In[0]:
# coding: utf-8
"""
Analyzis of Gold-Particle data for ARHCF paper

Created on 20181001 by Stefan Weidlich (SW)
Based on previous skript of Ronny Förster (RF) and SW

Target here: Implement only routines as they'll be used in paper

Modifications:
181020, SW: Cleaning up of code
Amongst others: Python v3.5 implementation deleted. Now working with 3.6 and above only
181025, SW: Adjustment of tracking-parameters and structuring of header
--> Realized that 32-bit parameters for tracking lead to unsufficient pixel-precision for 64 bit-version
181026, SW: Taking out of log-tracking. --> Not needed

******************************************************************************
Importing neccessary libraries
"""
#from __future__ import division, unicode_literals, print_function # For compatibility with Python 2 and 3
import numpy as np # Library for array-manipulation
import pandas as pd # Library for DataFrame Handling
import trackpy as tp # trackpy offers all tools needed for the analysis of diffusing particles
import seaborn as sns # allows boxplots to be put in one graph
import math # offering some maths functions
import matplotlib.pyplot as plt # Libraries for plotting
from matplotlib import animation # Allows to create animated plots and videos from them
import json
import sys
import datetime
import os
from PIL import Image
from pdb import set_trace as bp #debugger

"""
******************************************************************************
Some settings
"""
pd.set_option('display.max_columns',20) # I'm just using this to tell my Spyder-console to allow me to 
# see up to 20 columns of a dataframe (instead of a few only by default)


# In[]

def update_progress(job_title, progress):
    length = 50 # modify this to change the length
    block = int(round(length*progress))
    msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100))
    if progress >= 1: msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()


def GetPlotParameters(settings):
    
    with open(settings["Gui"]['SaveProperties']) as json_file:
        catch_all = json.load(json_file)
        
    params = catch_all["params"]
    title_font = catch_all["title_font"]
    axis_font = catch_all["axis_font"]
    
    return params, title_font, axis_font
    


def export(save_folder_name, save_image_name, settings):    
#    plt.show()
    my_dir_name = '%s\\{date:%y%m%d}\\'.format( date=datetime.datetime.now()) %save_folder_name
    
    try:
        os.stat(my_dir_name)
    except:
        os.mkdir(my_dir_name) 
    
    time_string = '{date:%H_%M_%S}'.format( date=datetime.datetime.now())
    
    file_name_image = '%s_%s.png'.format( date=datetime.datetime.now()) %(time_string, save_image_name)
    
    entire_path_image = my_dir_name +  file_name_image
    
    plt.savefig(entire_path_image, dpi=400, bbox_inches='tight')
    print('Figure saved')    

    #here comes the json parameter file 
    file_name_json = '%s_%s.json'.format( date=datetime.datetime.now()) %(time_string, save_image_name)
    entire_path_json = my_dir_name +  file_name_json
    
    settings["Exp"]["path_setting"] = entire_path_json
    
    with open(entire_path_json, 'w') as outfile:
        json.dump(settings, outfile, sort_keys=True, indent=4)
 
    Image.open(entire_path_image).show()
    
    return settings


def IntMedianFit(t1_gapless, my_particle = -1):
    if my_particle == -1:
        #get the first occuring particle
        # can be different from 0 or 1 because they might have been thrown away due to beeing stumps
        # show the maximum relative step for each particle
        my_particle = t1_gapless.iloc[0].particle
        
#    show_curve = t1_gapless[t1_gapless['particle'] == my_particle][['mass','mass_smooth','rel_step']]
    show_curve = t1_gapless[t1_gapless['particle'] == my_particle][['mass','mass_smooth']]
    plt.figure()
    plt.plot(show_curve.mass, 'x:')
    plt.plot(show_curve.mass_smooth, '-')
    plt.title("Intensity Analysis of each Particle")
    plt.legend(['Mass', 'Median Filter'])
    plt.xlabel("Frame")
    plt.ylabel("Mass")
    plt.ylim(0)
    
    
def MaxIntFluctuationPerBead(t1_gapless): 
    beads_property = pd.DataFrame(columns=['max_step']);
    
    beads_property['max_step'] = t1_gapless.groupby('particle')['rel_step'].max()
    
    # show the maximum relative step for each particle
    plt.figure()
    plt.plot(beads_property,'x')
    plt.title("Intensity Analysis of each Particle")
    plt.xlabel("particle")
    plt.ylabel("Maximum relative step")

    
def CutTrajectorieAtStep(t1_gapless, particle_to_split, max_rel_median_intensity_step):  
    
    #Display the intensity curvature of particles that need to split

    
    show_with_fit = t1_gapless[t1_gapless.particle == particle_to_split]
    show_no_fit= t1_gapless[t1_gapless.particle == particle_to_split]
    
    # do not show fitted data on existing sampling point
    show_with_fit.loc[show_no_fit.index, 'mass'] = math.nan
    
    plt.figure()
    #top plot with intensity curve
    ax1 = plt.subplot(2, 1, 1)
    
    # four curves in one plot
    plt.plot(show_no_fit.mass,'.',color='black')
    plt.plot(show_with_fit.mass, '.',color='orange')
    plt.plot(show_with_fit.mass_smooth, '-',color='black')
    #plt.plot(show_no_fit.mass_smooth,':',color='blue')
    
    plt.title("Intensity of Particle No. %i" %particle_to_split)
    #plt.legend(['Fitted Mass', 'Fitted Median Filter','Mass','Median Filter'])
    plt.legend(['Mass','Interpolated Mass', 'Fitted Median Filter'])
    plt.xlabel("Frame")
    plt.ylabel("Mass")
    plt.ylim(0)
    plt.xlim(show_with_fit.index[0],show_with_fit.index[-1])
    ax1.xaxis.grid()

    #bottom plot with slope curve
    ax2 = plt.subplot(2, 1, 2)
    plt.plot(show_with_fit.rel_step, ':.',color='black')
    
    # show threshold
    my_xlim = ax2.get_xlim()
    ax2.hlines(y=max_rel_median_intensity_step, xmin=my_xlim[0], xmax=my_xlim[1], linewidth=2, color='red')
        
    plt.title("Slope of Intensity of Particle No. %i" %particle_to_split)
    plt.legend(['Slope Fitted Mass','Threshold'])
    plt.xlabel("Frame")
    plt.ylabel("Mass")
    plt.ylim(0,2*max_rel_median_intensity_step)
    plt.xlim(show_with_fit.index[0],show_with_fit.index[-1])
    ax2.xaxis.grid()
    
   
    
def AnimateProcessedRawDate(rawframes_ROI, t_long):
    rawframes_before_filtering=rawframes_ROI
    
    fig, (ax0, ax5) = plt.subplots(2, sharex=True, sharey=True)
    scat5 = ax5.scatter([], [], s=60)
    im_data0=rawframes_before_filtering[0]
    #im_data=rawframes[0]
    im_data=rawframes_ROI[0]
    background0 = ax0.imshow(im_data0, animated=True)
    background5 = ax5.imshow(im_data, animated=True)
    fig.subplots_adjust(hspace=0)
    
    def init():
        scat5.set_offsets([])
        im_data0=rawframes_before_filtering[0]
        im_data=rawframes_ROI[0]
        background0.set_data(im_data0)
        background5.set_data(im_data)
        ax0.set_ylabel('raw-data')
        ax5.set_ylabel('size-analyzed particles')    
        ax5.set_xlabel('x-Position [Pixel]')
        return background0, background5, scat5,
    
    def animate(i):
        ax0.clear()
        ax5.clear()
        scat5 = ax5.scatter(t_long.x[t_long.frame==i].values, t_long.y[t_long.frame==i].values, s=70, 
                            alpha = 0.4, c=t_long.particle[t_long.frame==i].values % 20)
        im_data0=rawframes_before_filtering[i]
        im_data=rawframes_ROI[i]
        background0 = ax0.imshow(im_data0, animated=True)
        background5 = ax5.imshow(im_data, animated=True)
        return background0, background5, scat5
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=t_long.frame.nunique(), 
                                   interval=1000, blit=True, repeat=False)
    
    return anim
    
    
    
def DriftAvgSpeed():
    # show average speed
    #ax = plt.figure()
    total_drift[['y','x']].plot( marker='.')
    plt.title('Drift in laminar flow ')
    plt.ylabel('Drift [px]')
    plt.xlabel('Lateral position in fibre [px]')
    plt.legend(['y', 'x (flow-direction)'])
        
        
def DriftTimeDevelopment():
    # SW 180701: There is some issue with the plot in the block beneath!
    # show time developement
    fig, ax = plt.subplots()
    calc_drift.groupby(['y_range'])[['x']].plot(ax=ax)
    ax.legend(np.linspace(1,number_blocks,number_blocks,dtype='int'))
    plt.title('Drift in laminar flow ')
    plt.ylabel('Drift [px]')
    plt.xlabel('Time frame')
    
    
def DriftFalseColorMapFlow():
    
    # show false colour map from flow
    # https://matplotlib.org/gallery/images_contours_and_fields/pcolormesh_levels.html
    fig, ax = plt.subplots()
    disp_calc_drift = pd.DataFrame()
    for x in range(0,number_blocks):
        disp_calc_drift[str(x)] = calc_drift[calc_drift.y_range == y_range[x]].x
    #heatmap = ax.pcolormesh(y_range,disp_calc_drift.index.values,disp_calc_drift, cmap=plt.cm.hsv, alpha=1)
    #heatmap = ax.contourf(y_range,disp_calc_drift.index.values,disp_calc_drift, 50, cmap="jet", alpha=1, origin='lower')
    heatmap = ax.contourf(disp_calc_drift.index.values,y_range,disp_calc_drift.transpose(), 50, cmap="jet", alpha=1, origin='lower')
    plt.title('y-dependend flow')
    plt.xlabel('Frame')
    plt.ylabel('Lateral position in fibre [px]')
    plt.colorbar(mappable=heatmap, orientation='vertical', label='Drift along fiber direction')
      
    
def DriftVectors():
        
    # DRIFT VECTORS
    # https://matplotlib.org/gallery/images_contours_and_fields/pcolormesh_levels.html
    fig, ax = plt.subplots()
    disp_calc_drift = pd.DataFrame()
    for x in range(0,number_blocks):
        disp_calc_drift[str(x)] = calc_drift[calc_drift.y_range == y_range[x]].x
    
    start_frame = min(disp_calc_drift.index)
    end_frame = max(disp_calc_drift.index)
    #step = 100;
    
    sampling_grid_y_sub = 1;
    sampling_grid_frame = 100;
    
    my_frames = np.arange(start_frame,end_frame, sampling_grid_frame);
    my_ysub = np.arange(1,number_blocks, sampling_grid_y_sub);
    
    drift_grid = disp_calc_drift.loc[my_frames];
    drift_vec = drift_grid.diff()
    drift_vec = drift_vec.iloc[1:]
    my_frames = my_frames[1:]
    drift_vec_np = np.array(drift_vec)
    
    arrow_x = -drift_vec_np[:,my_ysub]
    arrow_y = arrow_x.copy()
    arrow_y[:,:] = 0
    
    q = plt.quiver(my_frames, y_range[my_ysub], np.matrix.transpose(arrow_x), arrow_y)
          
    qk = plt.quiverkey(q, 0.3, 0.9, 2, '', labelpos='E', coordinates='figure')
    
    plt.title('Horizontal drift between frame')
    plt.xlabel('frame')
    plt.ylabel('Lateral position in core [px]')
    
    
#        heatmap = ax.contourf(disp_calc_drift.index.values,y_range,disp_calc_drift.transpose(), 50, cmap="jet", alpha=1, origin='lower')
#        plt.title('y-dependend flow')
#        plt.xlabel('Frame')
#        plt.ylabel('Lateral position in fibre [px]')
#        plt.colorbar(mappable=heatmap, orientation='vertical', label='Drift along fiber direction')
    #    
        

def DriftFalseColorMapSpeed():
    # show false colour map from SPEED
    # https://matplotlib.org/gallery/images_contours_and_fields/pcolormesh_levels.html
    fig, ax = plt.subplots()
    disp_calc_drift = pd.DataFrame()
    for x in range(0,number_blocks):
        disp_calc_drift[str(x)] = calc_drift[calc_drift.y_range == y_range[x]].velocity_x
    #heatmap = ax.pcolormesh(y_range,disp_calc_drift.index.values,disp_calc_drift, cmap=plt.cm.hsv, alpha=1)
    #heatmap = ax.contourf(y_range,disp_calc_drift.index.values,disp_calc_drift, 50, cmap="jet", alpha=1, origin='lower')
    heatmap = ax.contourf(disp_calc_drift.index.values,y_range,disp_calc_drift.transpose(), 50, cmap="Greys", alpha=1, origin='lower')
    plt.title('Speed in fibre')
    plt.xlabel('Frame')
    plt.ylabel('Lateral position in fibre [px]')
    plt.colorbar(mappable=heatmap, orientation='vertical', label='Speed')


def DriftCorrectedTraj():
    # Show drift-corrected trajectories
    fig, ax = plt.subplots()
    tp.plot_traj(tm_sub)
    plt.title('Trajectories: Drift-Correction depending on laterial-position (y)')
    plt.xlabel('x-Position [px]')
    plt.ylabel('Lateral position in fibre (y) [px]')
    
    
def PlotGlobalDrift():
    d.plot() # plot the calculated drift
    
   
    
def MsdOverLagtime(lagt_direct, mean_displ_direct, mean_displ_fit_direct_lin, alpha_values = 0.1, alpha_fit = 0.3):
    plt.plot(lagt_direct, mean_displ_direct,'k-', alpha = alpha_values) # plotting msd-lag-time-tracks for all particles
    plt.plot(lagt_direct, mean_displ_fit_direct_lin, 'r', alpha = alpha_fit) # plotting the lin fits
    #ax.annotate(particleid, xy=(lagt_direct.max(), mean_displ_fit_direct_lin.max()))
    
    
def DiameterHistogramm(sizes_df_lin, binning, cutoff_size):
    fig, ax = plt.subplots()
    ax.set_xlim(0, cutoff_size)
    sns.distplot(sizes_df_lin.diameter[sizes_df_lin.diameter <= cutoff_size], 
                 bins=binning, rug=True, kde=False) # histogram of sizes, only taking into account 
    #those that are below threshold size as defined in the initial parameters
    plt.ylabel(r'absolute occurance')
    plt.xlabel('diameter [nm]')
    textstr = 'Amount of particles analyzed =%r' % len(sizes_df_lin)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props) 
    
    
    
def VarDiameterOverTracklength(sizes_df_lin, save_folder_name = r'Z:\Datenauswertung\19_ARHCF', save_image_name = 'errorpropagation_diameter_vs_trajectorie_length'):
    import nanoobject_detection as nd
    from nd.PlotProperties import axis_font, title_font
    # NO WORKING AT THE MOMENT BECAUSE ERROR IS NOT RIGHT YET
    
    import ronnys_tools as rt
#    from ronnys_tools.mpl_style import axis_font, title_font
    
    ax = plt.subplot(111)
    fig = plt.errorbar(sizes_df_lin['frames'],sizes_df_lin['diameter'], yerr = sizes_df_lin['diameter_std'],fmt='.k', alpha = 0.4, ecolor='orange')
    
    ax.set_yscale("log")
    ax.set_ylim(10,300)
    
    ax.set_yticks([10, 30, 100, 300])
    ax.set_yticklabels(['10', '30', '100', '300'])
    
    ax.set_ylabel('Diameter [nm]', **axis_font)
    
    ax.set_xscale("log")
    ax.set_xlim(30,1000)
    
    ax.set_xticks([30, 100, 300, 1000])
    ax.set_xticklabels(['30', '100', '300', '1000'])
    
    ax.set_xlabel('Trajectorie duration [frames]', **axis_font)
    
    ax.grid(which='major', linestyle='-')
    ax.grid(which='minor', linestyle=':')
    
    

    
    #rt.mpl_style.export(save_folder_name, save_image_name)
    
    
def VarDiameterOverMinTracklength(settings, sizes_df_lin, obj_all, num_bins_traj_length = 100, min_traj_length = 0, max_traj_length = 1000):
    import nanoobject_detection as nd
    from nd.PlotProperties import axis_font, title_font
          
    diameter_by_min_trajectorie = pd.DataFrame(np.zeros((max_traj_length, 6)),columns=['min_trajectorie_length', 'particles_min_length', 'tot_traj_length', 'rel_traj_length', 'mean_diam_min', 'std_diam_min'])
    
    
    #    ideal_tot_traj_length = len(f)
    ideal_tot_traj_length = len(obj_all)
    
    min_tracking_frames = np.int(settings["Processing"]["min_tracking_frames"])
    
    for loop_min_traj_length in range(min_tracking_frames,max_traj_length):       
        diameter_by_min_trajectorie.loc[loop_min_traj_length]['min_trajectorie_length'] = loop_min_traj_length       
    
        
        diameter_with_min_trajectorie = sizes_df_lin[(sizes_df_lin['frames'] >= loop_min_traj_length)]
        
        diameter_by_min_trajectorie.loc[loop_min_traj_length]['mean_diam_min'] = np.mean(diameter_with_min_trajectorie['diameter'])
        diameter_by_min_trajectorie.loc[loop_min_traj_length]['std_diam_min'] = np.std(diameter_with_min_trajectorie['diameter'])
        diameter_by_min_trajectorie.loc[loop_min_traj_length]['particles_min_length'] = len(diameter_with_min_trajectorie['diameter'])
    
        diameter_by_min_trajectorie.loc[loop_min_traj_length]['tot_traj_length'] = np.sum(diameter_with_min_trajectorie["frames"])
        diameter_by_min_trajectorie.loc[loop_min_traj_length]['rel_traj_length'] = diameter_by_min_trajectorie.loc[loop_min_traj_length]['tot_traj_length'] / ideal_tot_traj_length
    
    
    # remove empty rows before min tracking frames value
    diameter_by_min_trajectorie = diameter_by_min_trajectorie.loc[min_tracking_frames:]
    
    
    # remove empty rows where no particle is anymore because none is so long as required
    diameter_by_min_trajectorie = diameter_by_min_trajectorie[np.isnan(diameter_by_min_trajectorie['std_diam_min']) == False]
    
    mean_plus_sigma = diameter_by_min_trajectorie['mean_diam_min'] + diameter_by_min_trajectorie['std_diam_min']
    mean_minus_sigma = diameter_by_min_trajectorie['mean_diam_min'] - diameter_by_min_trajectorie['std_diam_min']
    
     
    
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex = True)
    
    
    ax1.loglog(diameter_by_min_trajectorie['min_trajectorie_length'], diameter_by_min_trajectorie['mean_diam_min'], label='')
    ax1.set_ylabel('Diameter of particles [nm]', **axis_font)
    
    ax1.fill_between(diameter_by_min_trajectorie['min_trajectorie_length'], mean_plus_sigma, mean_minus_sigma, alpha=.25, label='1-sigma interval')
    ax1.set_ylim([10, 100])
    ax1.set_xlim([10,1000])
    
    ax1.tick_params(axis = 'y')
    
    ax1.grid(which='major', linestyle='-')
    ax1.grid(which='minor', linestyle=':')
    
    ax1.legend()
    
    
    
    ax2.loglog(diameter_by_min_trajectorie['min_trajectorie_length'], diameter_by_min_trajectorie['std_diam_min'], label='')
    ax2.set_ylabel('Std of diameter [nm]', **axis_font)
    
    ax2.set_ylim([1, 300])
    
    ax2.tick_params(axis = 'y')
    
    ax2.grid(which='major', linestyle='-')
    ax2.grid(which='minor', linestyle=':')
    
    
    
    ax3.loglog(diameter_by_min_trajectorie['min_trajectorie_length'], diameter_by_min_trajectorie['rel_traj_length'],'-r', label='Ratio of analyzed particles')
    ax3.set_ylabel('Tracked Particles', **axis_font)
    ax3.tick_params(axis = 'y')
    
    ax3.set_xlabel('Minimum Trajectorie Time [frames]', **axis_font)
    
    ax3.grid(which='major', linestyle='-')
    ax3.grid(which='minor', linestyle=':')
    
    ax3.set_ylim([0.005,1,])
    

    
    
    save_folder_name = r'Z:\Datenauswertung\19_ARHCF'
    save_image_name = 'statistic_diameter_vs_trajectorie_length'
    
    nd.mpl_style.export(save_folder_name, save_image_name)
    
    
    
    
    # Crop image to the region of interest only (if needed re-define cropping parameters above):
    def crop(img): 
        x_min = x_min_global
        x_max = x_max_global
        y_min = y_min_global
        y_max = y_max_global 
        #return img[y_min:y_max,x_min:x_max,0] # ATTENTION: This form is used for images that are stored as 2d-data (*tif)!
        return img[y_min:y_max,x_min:x_max] # ATTENTION: Use this form if not using 2d-data (e.g. *.bmp)!
    #rawframes = pims.ImageSequence(data_folder_name + '\\' + '*.' + data_file_extension, process_func=crop)
    # ATTENTION: Data get's cut into a smaller part here above. When data should be rotated, it's better to do this first and then cut



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
    
    
    
    
    