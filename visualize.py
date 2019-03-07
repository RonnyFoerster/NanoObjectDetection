# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 12:23:39 2019

@author: Ronny Förster und Stefan Weidlich
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
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt # Libraries for plotting
from matplotlib import animation # Allows to create animated plots and videos from them
import json
import sys
import datetime
from PIL import Image
from pdb import set_trace as bp #debugger

"""
******************************************************************************
Some settings
"""
pd.set_option('display.max_columns',20) # I'm just using this to tell my Spyder-console to allow me to 
# see up to 20 columns of a dataframe (instead of a few only by default)


# In[]
def SetXYLim(x_min_max = None, y_min_max = None):
    if (x_min_max is None) == False:
        plt.xlim([x_min_max[0], x_min_max[1]])

    if (y_min_max is None) == False:
        plt.ylim([y_min_max[0], y_min_max[1]])

def Plot1DPlot(plot_np,title, xlabel, ylabel):
    from NanoObjectDetection.PlotProperties import axis_font, title_font
    
    plt.figure()
    plt.plot(plot_np)
    plt.title(title, **title_font)
    plt.xlabel(xlabel, **axis_font)
    plt.ylabel(ylabel, **axis_font)


def Plot2DPlot(x_np,y_np,title = None, xlabel = None, ylabel = None, myalpha = 1):
    from NanoObjectDetection.PlotProperties import axis_font, title_font
    
    plt.figure()
    plt.plot(x_np,y_np, ':x', alpha = myalpha)
    plt.title(title, **title_font)
    plt.xlabel(xlabel, **axis_font)
    plt.ylabel(ylabel, **axis_font)


def Plot2DScatter(x_np,y_np,title = None, xlabel = None, ylabel = None, myalpha = 1,
                  x_min_max = None, y_min_max = None):
    from NanoObjectDetection.PlotProperties import axis_font, title_font

    plt.figure()
    plt.scatter(x_np,y_np, alpha = myalpha, linewidths  = 0)
    plt.title(title, **title_font)
    plt.xlabel(xlabel, **axis_font)
    plt.ylabel(ylabel, **axis_font)

    SetXYLim(x_min_max, y_min_max)



def Plot2DImage(array_np,title, xlabel, ylabel):
    from NanoObjectDetection.PlotProperties import axis_font, title_font
    
    plt.figure()
    plt.imshow(array_np, cmap='gray')
    plt.colorbar()
    plt.title(title, **title_font)
    plt.xlabel(xlabel, **axis_font)
    plt.ylabel(ylabel, **axis_font)
    
    



def PlotDiameters(ParameterJsonFile, sizes_df_lin, any_successful_check, histogramm_min = None, histogramm_max = None, Histogramm_min_max_auto = None, binning = None):
    import NanoObjectDetection as nd
    
    if any_successful_check == False:
        print("NO PARTICLE WAS MEASURED LONG ENOUGH FOR A GOOD STATISTIC !")
    else:
        settings = nd.handle_data.ReadJson(ParameterJsonFile)
        
        show_hist, save_hist = settings["Plot"]["Histogramm_Show"], settings["Plot"]["Histogramm_Save"]
        if show_hist or save_hist:
            DiameterHistogramm(ParameterJsonFile, sizes_df_lin)

        
        show_diam_traj, save_diam_traj = settings["Plot"]["DiamOverTraj_Show"], settings["Plot"]["DiamOverTraj_Save"]
        if show_diam_traj or save_diam_traj:
            DiamerterOverTrajLenght(ParameterJsonFile, sizes_df_lin, show_diam_traj, save_diam_traj)




def DiamerterOverTrajLenght(ParameterJsonFile, sizes_df_lin, show_plot = None, save_plot = None):
    import NanoObjectDetection as nd
    from NanoObjectDetection.PlotProperties import axis_font, title_font

    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    Histogramm_min_max_auto = settings["Plot"]["Histogramm_min_max_auto"]
    
    if Histogramm_min_max_auto == 1:
            histogramm_min = np.round(np.min(sizes_df_lin.diameter) - 5, -1)
            histogramm_max = np.round(np.max(sizes_df_lin.diameter) + 5, -1)
            
    histogramm_min, settings = nd.handle_data.SpecificValueOrSettings(histogramm_min, settings, "Plot", "Histogramm_min")
    histogramm_max, settings = nd.handle_data.SpecificValueOrSettings(histogramm_max, settings, "Plot", "Histogramm_max")
    
    my_title = "Particles diameter over its tracking time"
    my_ylabel = "Diameter [nm]"
    my_xlabel = "Trajectory length [frames]"
    
    plot_diameter = sizes_df_lin["diameter"]
    plot_traj_length = sizes_df_lin["traj_length"]
    
    x_min_max = nd.handle_data.Get_min_max_round(plot_traj_length, 2)
    x_min_max[0] = 0
    
    y_min_max = nd.handle_data.Get_min_max_round(plot_diameter,1)
    
    Plot2DScatter(plot_traj_length, plot_diameter, title = my_title, xlabel = my_xlabel, ylabel = my_ylabel,
                  myalpha = 0.6, x_min_max = x_min_max, y_min_max = y_min_max)
 
    if save_plot == True:
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "DiameterOverTrajLength",
                                       settings, data = sizes_df_lin)



def NumberOfBinsAuto(mydata, average_heigt = 4):
    number_of_points = len(mydata)
    
    bins = int(np.ceil(number_of_points / average_heigt))
    
    return bins



def DiameterHistogramm(ParameterJsonFile, sizes_df_lin, histogramm_min = None, histogramm_max = None, Histogramm_min_max_auto = None, binning = None):
    import NanoObjectDetection as nd
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
        
    Histogramm_Show = settings["Plot"]['Histogramm_Show']
    Histogramm_Save = settings["Plot"]['Histogramm_Save']
    
    if settings["Plot"]["Histogramm_Bins_Auto"] == 1:
        settings["Plot"]["Histogramm_Bins"] = NumberOfBinsAuto(sizes_df_lin)
 
    binning = settings["Plot"]["Histogramm_Bins"]
    
    
    Histogramm_min_max_auto = settings["Plot"]["Histogramm_min_max_auto"]
    if Histogramm_min_max_auto == 1:

        histogramm_min = np.round(np.min(sizes_df_lin.diameter) - 5, -1)
        histogramm_max = np.round(np.max(sizes_df_lin.diameter) + 5, -1)
        
    histogramm_min, settings = nd.handle_data.SpecificValueOrSettings(histogramm_min, settings, "Plot", "Histogramm_min")
    histogramm_max, settings = nd.handle_data.SpecificValueOrSettings(histogramm_max, settings, "Plot", "Histogramm_max")

    
    
    if Histogramm_Save == True:
        Histogramm_Show = True
    
    if Histogramm_Show == True:
#            print(sizes_df_lin)#
        xlabel = 'Diameter [nm]'
        ylabel = 'Absolute occurance'
        title = 'Amount of particles analyzed =%r' % len(sizes_df_lin)

        nd.visualize.PlotDiameterHistogramm(sizes_df_lin, binning, histogramm_min, histogramm_max, title, xlabel, ylabel)
  
        if Histogramm_Save == True:
            settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "Diameter Histogramm", settings,
                                           data = sizes_df_lin)
        
        
    nd.handle_data.WriteJson(ParameterJsonFile, settings) 


def GetMeanStdMedian(data):
    my_mean   = np.mean(data)
    my_std    = np.std(data)
    my_median = np.median(data)

    return my_mean, my_std, my_median    
    

def PlotDiameterHistogramm(sizes_df_lin, binning, min_size = 0, cutoff_size = 10000, title = '', xlabel = '', ylabel = ''):
    from NanoObjectDetection.PlotProperties import axis_font, title_font
#    plt.figure()
    fig, ax = plt.subplots()
 
    diameters = sizes_df_lin.diameter
    show_diameters = diameters[(diameters >= min_size) & (diameters <= cutoff_size)]
    # histogram of sizes, only taking into account 
    sns.distplot(show_diameters, bins=binning, rug=True, kde=False) 
    #those that are below threshold size as defined in the initial parameters
    plt.rc('text', usetex=True)
    plt.title(title, **title_font)
    #   plt.ylabel(r'absolute occurance')
    plt.ylabel(ylabel, **axis_font)
    plt.ylabel(ylabel, **axis_font)
    plt.xlabel(xlabel, **axis_font)
    plt.grid(True)
 
    ax.set_xlim(min_size, cutoff_size)

    # infobox
    my_mean, my_std, my_median = GetMeanStdMedian(sizes_df_lin.diameter)
    
    textstr = '\n'.join((
    r'$\mu=%.1f$ nm' % (my_mean, ),
    r'$\sigma=%.1f$ nm' % (my_std, ),
    r'$\mathrm{median}=%.1f$ nm' % (my_median, )))
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
#    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
#        , bbox=props)
    
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, **axis_font, verticalalignment='top', bbox=props)
    
    
#    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#    ax.text(0.05, 0.95, title, transform=ax.transAxes, fontsize=14,
#            verticalalignment='top', bbox=props) 




def update_progress(job_title, progress):
    length = 50 # modify this to change the length
    block = int(round(length*progress))
    msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100))
    if progress >= 1: msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()


def GetPlotParameters(settings):
    
    with open(settings["Plot"]['SaveProperties']) as json_file:
        catch_all = json.load(json_file)
        
    params = catch_all["params"]
    title_font = catch_all["title_font"]
    axis_font = catch_all["axis_font"]
    
    return params, title_font, axis_font
    



def export(save_folder_name, save_image_name, settings = None, use_dpi = None, data = None, data_header = None,
           save_json = 1, save_data2cs = 1):    
    import os.path
    import NanoObjectDetection as nd
    if settings is None:
        if use_dpi is None:
            sys.exit("need settings or dpi!")
    else:
        use_dpi, settings = nd.handle_data.SpecificValueOrSettings(use_dpi,settings, "Plot", "dpi")
        save_json, settings = nd.handle_data.SpecificValueOrSettings(save_json,settings, "Plot", "save_json")
        save_data2cs, settings = nd.handle_data.SpecificValueOrSettings(save_data2cs,settings, "Plot", "save_data2cs")
        
    use_dpi = int(use_dpi)
    
    
    my_dir_name = '%s\\{date:%y%m%d}\\'.format( date=datetime.datetime.now()) %save_folder_name

    try:
        os.stat(my_dir_name)
    except:
        os.mkdir(my_dir_name) 
    
    time_string = '{date:%H_%M_%S}'.format( date=datetime.datetime.now())
    
    file_name_image = '%s_%s.png'.format( date=datetime.datetime.now()) %(time_string, save_image_name)
    
    entire_path_image = my_dir_name +  file_name_image
    
    plt.savefig(entire_path_image, dpi= use_dpi, bbox_inches='tight')
    print('Figure saved')    

    #here comes the json parameter file 
    file_name_json = '%s_%s.json'.format( date=datetime.datetime.now()) %(time_string, save_image_name)
    entire_path_json = my_dir_name +  file_name_json
    
    if settings is None:
        settings = 0
    else:
        settings["Plot"]["SaveProperties"] = entire_path_json
    
        with open(entire_path_json, 'w') as outfile:
            json.dump(settings, outfile, sort_keys=True, indent=4)
     
    Image.open(entire_path_image).show()
    if (data is None) == False:
            #here comes the data file 
            data_name_csv = '%s_%s.csv'.format( date=datetime.datetime.now()) %(time_string, save_image_name)
            entire_path_csv = my_dir_name +  data_name_csv
            if type(data) == np.ndarray:
                np.savetxt(entire_path_csv, data, delimiter="," , fmt='%.10e', header = data_header)
            else:
                data.to_csv(entire_path_csv, index = False)
            
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
    
    
def PlotGlobalDrift(d):
    d.plot() # plot the calculated drift
    
   
    
def MsdOverLagtime(lagt_direct, mean_displ_direct, mean_displ_fit_direct_lin, collect_data = None, alpha_values = 0.1, alpha_fit = 0.3):
    from NanoObjectDetection.PlotProperties import axis_font, title_font
    plt.plot(lagt_direct, mean_displ_direct,'k-', alpha = alpha_values) # plotting msd-lag-time-tracks for all particles
    plt.plot(lagt_direct, mean_displ_fit_direct_lin, 'r', alpha = alpha_fit) # plotting the lin fits
    #ax.annotate(particleid, xy=(lagt_direct.max(), mean_displ_fit_direct_lin.max()))
    plt.title("MSD fit", **title_font)
    plt.ylabel("MSD $[\mu m^2]$", **axis_font)
    plt.xlabel("Lagtime [s]", **axis_font)



    
    
    
def VarDiameterOverTracklength(sizes_df_lin, save_folder_name = r'Z:\Datenauswertung\19_ARHCF', save_image_name = 'errorpropagation_diameter_vs_trajectorie_length'):
    import NanoObjectDetection as nd
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
    import NanoObjectDetection as nd
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
    
    
    
    
    