# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 12:23:39 2019

@author: Ronny Förster, Stefan Weidlich, Mona Nissen
"""

# In[0]:
# coding: utf-8

"""
collection of plotting functions for visualization


******************************************************************************
Importing neccessary libraries
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

## unused for the moment:
#import time
#from matplotlib.animation import FuncAnimation, PillowWriter
#from matplotlib.animation import FuncAnimation



# In[]
def SetXYLim(x_min_max = None, y_min_max = None):
    """ set axes limits to fixed values """
    
    if (x_min_max is None) == False:
        plt.xlim([x_min_max[0], x_min_max[1]])

    if (y_min_max is None) == False:
        plt.ylim([y_min_max[0], y_min_max[1]])



def Plot1DPlot(plot_np,title = None, xlabel = None, ylabel = None, settings = None):
    """ plot 1D-data in standardized format as line plot """
    
    import NanoObjectDetection as nd
    from NanoObjectDetection.PlotProperties import axis_font, title_font
    
    plt.figure()
    plt.plot(plot_np)
    plt.title(title, **title_font)
    plt.xlabel(xlabel, **axis_font)
    plt.ylabel(ylabel, **axis_font)
    
    
    if settings != None:
        nd.visualize.export(settings["Plot"]["SaveFolder"], "Diameter Histogramm", settings, data = plot_np)



def Plot2DPlot(x_np,y_np,title = None, xlabel = None, ylabel = None, myalpha = 1, mymarker = 'x', mylinestyle  = ':', x_lim = None, y_lim = None, y_ticks = None, semilogx = False):
    """ plot 2D-data in standardized format as line plot """
    
    from NanoObjectDetection.PlotProperties import axis_font, title_font, params
#    sns.reset_orig()
    
    plt.style.use(params)
    
    plt.figure()
    if semilogx == False:
        plt.plot(x_np,y_np, marker = mymarker, linestyle  = mylinestyle, alpha = myalpha)
    else:
        plt.semilogx(x_np,y_np, marker = mymarker, linestyle  = mylinestyle, alpha = myalpha)
        import matplotlib.ticker

        ax = plt.gca()
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
        
    plt.title(title, **title_font)
    plt.xlabel(xlabel, **axis_font)
    plt.ylabel(ylabel, **axis_font)

#    matplotlib.rcParams.update(params)

    

    if x_lim != None:
        plt.xlim(x_lim)
        
    if y_lim != None:
        plt.ylim(y_lim)
                
    if y_ticks != None:
        frame1 = plt.gca() 
        frame1.axes.yaxis.set_ticks(y_ticks)


    nd.PlotProperties
    plt.style.use(params)


def Plot2DScatter(x_np,y_np,title = None, xlabel = None, ylabel = None, myalpha = 1,
                  x_min_max = None, y_min_max = None):
    """ plot 2D-data in standardized format as individual, scattered points """
    
    from NanoObjectDetection.PlotProperties import axis_font, title_font

    plt.figure()
    plt.scatter(x_np,y_np, alpha = myalpha, linewidths  = 0)
    plt.title(title, **title_font)
    plt.xlabel(xlabel, **axis_font)
    plt.ylabel(ylabel, **axis_font)

    SetXYLim(x_min_max, y_min_max)



def Plot2DImage(array_np,title = None, xlabel = None, ylabel = None):
    """ plot image in standardized format """
    
    from NanoObjectDetection.PlotProperties import axis_font, title_font, params

    
    plt.figure()
    plt.imshow(array_np, cmap='gray')
    plt.colorbar()
    plt.title(title, **title_font)
    plt.xlabel(xlabel, **axis_font)
    plt.ylabel(ylabel, **axis_font)
    
    

def PlotDiameters(ParameterJsonFile, sizes_df_lin, any_successful_check, histogramm_min = None, histogramm_max = None, Histogramm_min_max_auto = None, binning = None):
    """ plot and/or save diameter histogram (or other statistics) for analyzed particles """
    
    import NanoObjectDetection as nd

    if any_successful_check == False:
        print("NO PARTICLE WAS MEASURED LONG ENOUGH FOR A GOOD STATISTIC !")
    else:
        settings = nd.handle_data.ReadJson(ParameterJsonFile)
        
        show_hist, save_hist = settings["Plot"]["Histogramm_Show"], settings["Plot"]["Histogramm_Save"]
        if show_hist or save_hist:
            DiameterHistogramm(ParameterJsonFile, sizes_df_lin)


        show_pdf, save_pdf = settings["Plot"]["DiameterPDF_Show"], settings["Plot"]["DiameterPDF_Save"]
        if show_pdf or save_pdf:
            DiameterPDF(ParameterJsonFile, sizes_df_lin)

        
        show_diam_traj, save_diam_traj = settings["Plot"]["DiamOverTraj_Show"], settings["Plot"]["DiamOverTraj_Save"]
        if show_diam_traj or save_diam_traj:
            DiamerterOverTrajLenght(ParameterJsonFile, sizes_df_lin, show_diam_traj, save_diam_traj)


        show_hist_time, save_hist_time = settings["Plot"]["Histogramm_time_Show"], settings["Plot"]["Histogramm_time_Save"]
        if show_hist_time or save_hist_time:
            DiameterHistogrammTime(ParameterJsonFile, sizes_df_lin)
#            PlotDiameter2DHistogramm(ParameterJsonFile, sizes_df_lin, show_hist_time, save_hist_time)


        show_correl, save_correl = settings["Plot"]["Correlation_Show"], settings["Plot"]["Correlation_Save"]
        if show_correl or save_correl:
            Correlation(ParameterJsonFile, sizes_df_lin, show_correl, save_correl)
            
            
        show_pearson, save_pearson = settings["Plot"]["Pearson_Show"], settings["Plot"]["Pearson_Save"]
        if show_pearson or save_pearson:
            Pearson(ParameterJsonFile, sizes_df_lin, show_pearson, save_pearson)



def corrfunc(x, y, **kws):
    """ auxiliary function for applying Pearson statistics on (diameter) data """
    
    pearson, _ = scipy.stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("p = {:.2f}".format(pearson),
                xy=(.1, .9), xycoords=ax.transAxes)
    


def Correlation(ParameterJsonFile, sizes_df_lin, show_plot = None, save_plot = None):
    """ produces grid plot for the investigation of particle data correlation
    
    for more information, see
    https://stackoverflow.com/questions/30942577/seaborn-correlation-coefficient-on-pairgrid
    """
    import NanoObjectDetection as nd
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    sns.set(style="white")
#    mygraph = sns.pairplot(sizes_df_lin[["diameter", "mass","signal"]])
    
    mycol = list(sizes_df_lin.columns)
    #remove diamter std because it is note done yet
#    mycol = [x for x in mycol if x!= "diameter std"]

    mygraph = sns.pairplot(sizes_df_lin[mycol])
    
    mygraph.map_lower(corrfunc)
    mygraph.map_upper(corrfunc)

    if save_plot == True:
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "Correlation",
                                       settings, data = sizes_df_lin)
        
    plt.show()



def Pearson(ParameterJsonFile, sizes_df_lin, show_plot = None, save_plot = None):
    import NanoObjectDetection as nd
    from NanoObjectDetection.PlotProperties import axis_font
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    sns.set_context("paper", rc={"font.size":int(axis_font["size"]),
                                 "axes.titlesize":8,
                                 "axes.labelsize":100})
                            
    mycorr = sizes_df_lin.corr()
    
    plt.figure()
    sns.set(font_scale=0.6)
#    mygraph = sns.heatmap(np.abs(mycorr), annot=True, vmin=0, vmax=1)
    mygraph = sns.heatmap(mycorr, annot=True, vmin=-1, vmax=1, cmap="bwr", cbar_kws={'label': 'Pearson Coefficent'})
    
#    sns.set(font_scale=1.2)
#    mygraph = sns.clustermap(np.abs(mycorr), annot=True)
    

    if save_plot == True:
        print("Saving the Pearson heat map take a while. No idea why =/")
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "PearsonCoefficent",
                                       settings, data = sizes_df_lin)

    plt.show()
    


def ShowAllCorrel(sizes_df_lin,min_correl = 0.4):
    mycorr = sizes_df_lin.corr()
    mycorr_np = np.abs(mycorr.values)
    mycorr_high = (mycorr_np > min_correl) & (mycorr_np < 1)  
    
    # get position of high correlation
    index_corr_high = np.argwhere(mycorr_high)

    # removed mirror side
    index_corr_high = index_corr_high[index_corr_high[:,0] > index_corr_high[:,1]]
    
    mycol = sizes_df_lin.columns
    mycol_corr = mycol[index_corr_high]
    
    for num_loop, (col1, col2) in enumerate(zip(mycol_corr[:,0],mycol_corr[:,1])):
        print(col1,col2)
        
        Plot2DScatter(sizes_df_lin[col1],sizes_df_lin[col2],
                      title = "Correlated Parameters",
                      xlabel = col1, ylabel = col2, myalpha = 1)
#        plt.plot(,sizes_df_lin[col2],'x')
        
    
    plt.show()



def DiamerterOverTrajLenght(ParameterJsonFile, sizes_df_lin, show_plot = None, save_plot = None):
    """ plot (and save) calculated particle diameters vs. the number of frames
    where the individual particle is visible (in standardized format) """
        
    import NanoObjectDetection as nd

    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    Histogramm_min_max_auto = settings["Plot"]["Histogramm_min_max_auto"]
    
    if Histogramm_min_max_auto == 1:
        histogramm_min = np.round(np.min(sizes_df_lin.diameter) - 5, -1)
        histogramm_max = np.round(np.max(sizes_df_lin.diameter) + 5, -1)
    else:
        histogramm_min = None
        histogramm_max = None 

#    bp()
    
    histogramm_min, settings = nd.handle_data.SpecificValueOrSettings(histogramm_min, settings, "Plot", "Histogramm_min")
    histogramm_max, settings = nd.handle_data.SpecificValueOrSettings(histogramm_max, settings, "Plot", "Histogramm_max")
    
    my_title = "Particles diameter over its tracking time"
    my_ylabel = "Diameter [nm]"
    my_xlabel = "Trajectory length [frames]"
    
    plot_diameter = sizes_df_lin["diameter"]
    plot_traj_length = sizes_df_lin["traj length"]
    
    x_min_max = nd.handle_data.Get_min_max_round(plot_traj_length, 2)
    x_min_max[0] = 0
    
    y_min_max = nd.handle_data.Get_min_max_round(plot_diameter,1)
    
    Plot2DScatter(plot_traj_length, plot_diameter, title = my_title, xlabel = my_xlabel, ylabel = my_ylabel,
                  myalpha = 0.6, x_min_max = x_min_max, y_min_max = y_min_max)
 
    if save_plot == True:
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "DiameterOverTrajLength",
                                       settings, data = sizes_df_lin)





def DiameterHistogrammTime(ParameterJsonFile, sizes_df_lin, show_plot = None, save_plot = None, histogramm_min = None, histogramm_max = None, 
                           Histogramm_min_max_auto = None, binning = None):
    import NanoObjectDetection as nd

    settings = nd.handle_data.ReadJson(ParameterJsonFile)
        
    Histogramm_Show = settings["Plot"]['Histogramm_time_Show']
    Histogramm_Save = settings["Plot"]['Histogramm_time_Save']
    
    if settings["Plot"]["Histogramm_Bins_Auto"] == 1:
        settings["Plot"]["Histogramm_Bins"] = NumberOfBinsAuto(sizes_df_lin)
 
    binning = settings["Plot"]["Histogramm_Bins"]
    
    
    Histogramm_min_max_auto = settings["Plot"]["Histogramm_min_max_auto"]
    if Histogramm_min_max_auto == 1:

        histogramm_min = np.round(np.min(sizes_df_lin.diameter) - 5, -1)
        histogramm_max = np.round(np.max(sizes_df_lin.diameter) + 5, -1)
        
    histogramm_min, settings = nd.handle_data.SpecificValueOrSettings(histogramm_min, settings, "Plot", "Histogramm_min")
    histogramm_max, settings = nd.handle_data.SpecificValueOrSettings(histogramm_max, settings, "Plot", "Histogramm_max")
    
    
    xlabel = 'Diameter [nm]'
    ylabel = 'Absolute occurrence' # better: "Absolute number of occurrences"?
    title = 'Amount of particles analyzed = %r' % len(sizes_df_lin)

    nd.visualize.PlotDiameter2DHistogramm(sizes_df_lin, binning, histogramm_min, histogramm_max, title, xlabel, ylabel)
  
    if Histogramm_Save == True:
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "Diameter Histogramm", settings,
                                       data = sizes_df_lin)
        
        
    nd.handle_data.WriteJson(ParameterJsonFile, settings) 



def PlotDiameter2DHistogramm(sizes_df_lin, binning, histogramm_min = 0, histogramm_max = 10000, 
                             title = '', xlabel = '', ylabel = ''):
#    from NanoObjectDetection.PlotProperties import axis_font, title_font
    
#    import seaborn as sns
#    import numpy as np

    sns.set(style="darkgrid")
    
#    tips = sns.load_dataset("tips")
    
    time_points = int(sizes_df_lin["traj length"].sum())
    plot_data = np.zeros([time_points,2])
   
    num_eval_particles = sizes_df_lin.shape[0]
    
    first_element = True
    
    for ii in range(0,num_eval_particles):
        ii_save = sizes_df_lin.iloc[ii]
        
        ii_first_frame = int(ii_save["first frame"])
        ii_traj_length = int(ii_save["traj length"])
        ii_last_frame = ii_first_frame + ii_traj_length - 1
        ii_diameter = ii_save["diameter"]
        
        
        add_data_plot = np.zeros([ii_traj_length,2])
        
        add_data_plot[:,0] = np.linspace(ii_first_frame,ii_last_frame,ii_traj_length)
        add_data_plot[:,1] = ii_diameter
        
        if first_element == True:
            plot_data = add_data_plot
            first_element = False
        else:
            plot_data = np.concatenate((plot_data,add_data_plot), axis = 0)


    from matplotlib.ticker import NullFormatter
    
    # the random data
    diam = plot_data[:,1] #x
    time = plot_data[:,0] #y
    
    diam_max = np.max(diam)
    diam_min = np.min(diam)
    
    if 1==0:
        lim_diam_start = 0
        lim_diam_end = np.round(1.10*diam_max,-1)
    else:
        lim_diam_start = diam_min
        lim_diam_end = diam_max
    
    lim_time_start = np.min(time)
    lim_time_end = np.max(time)+1
    
    diam_bins_nm = 10
    diam_bins = int((lim_diam_end - lim_diam_start)/diam_bins_nm)
    time_bins = int(lim_time_end - lim_time_start + 0)
    
    nullfmt = NullFormatter()         # no labels
    
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    
    # start with a rectangular Figure
    plt.figure(figsize=(8, 8))
#    plt.figure(1, figsize=(8, 8))
    
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    
    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    
    # the scatter plot:
    axScatter.hist2d(diam, time, bins = [diam_bins, time_bins], cmap = "jet")
    axScatter.set_xlabel("Diameter [nm]")
    axScatter.set_ylabel("Time [frames]")
        
    axScatter.set_xlim((lim_diam_start, lim_diam_end))
    axScatter.set_ylim((lim_time_start, lim_time_end))
    
    axHistx.hist(diam, bins=diam_bins)
    axHistx.set_ylabel("Occurrence [a.u.]")
    
    time_hist = axHisty.hist(time, bins=time_bins, orientation='horizontal')
    axHisty.set_xlabel("Analyzed Particles")
    
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    
    plt.show()      


        
    
#    my_x = sizes_df_lin["diameter"].values
#    my_y = sizes_df_lin["traj length"].values
#    g = sns.jointplot(x = plot_data[:,1], y = plot_data[:,0], kind='kde')
    
    
    
#    g = sns.jointplot(x = my_x, y = my_y, kind="reg",
#                      xlim=(0, 60), ylim=(0, 12), color="m", height=7)
    
    
    
    
    
#    plt.figure()
#    fig, ax = plt.subplots()
# 
#    diameters = sizes_df_lin.diameter
#    show_diameters = diameters[(diameters >= min_size) & (diameters <= cutoff_size)]
#    # histogram of sizes, only taking into account 
#    sns.distplot(show_diameters, bins=binning, rug=True, kde=False) 
#    #those that are below threshold size as defined in the initial parameters
#    plt.rc('text', usetex=True)
#    plt.title(title, **title_font)
#    #   plt.ylabel(r'absolute occurrence')
#    plt.ylabel(ylabel, **axis_font)
#    plt.ylabel(ylabel, **axis_font)
#    plt.xlabel(xlabel, **axis_font)
#    plt.grid(True)
# 
#    ax.set_xlim(min_size, cutoff_size)
#
#    # infobox
#    my_mean, my_std, my_median = GetMeanStdMedian(sizes_df_lin.diameter)
#    
#    textstr = '\n'.join((
#    r'$\mu=%.1f$ nm' % (my_mean, ),
#    r'$\sigma=%.1f$ nm' % (my_std, ),
#    r'$\mathrm{median}=%.1f$ nm' % (my_median, )))
#    
#    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#    
##    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
##        , bbox=props)
#    
#    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, **axis_font, verticalalignment='top', bbox=props)
#    
#    
##    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
##    ax.text(0.05, 0.95, title, transform=ax.transAxes, fontsize=14,
##            verticalalignment='top', bbox=props) 





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

    xlabel = 'Diameter [nm]'
    ylabel = 'Absolute occurrence'
    title = 'Amount of particles analyzed = %r' % len(sizes_df_lin)

    values_hist = nd.visualize.PlotDiameterHistogramm(sizes_df_lin, binning, histogramm_min, histogramm_max, title, xlabel, ylabel)
    
    if settings["Plot"]["Histogramm_Fit_1_Particle"] == 1:
        max_hist = values_hist.max()
        # here comes the fit
    
        if 1 == 1:
            print("method 1")
#        diam_inv_mean, diam_inv_std = nd.CalcDiameter.StatisticOneParticle(sizes_df_lin)
            diam_inv_mean, diam_inv_std = nd.CalcDiameter.StatisticOneParticle(sizes_df_lin)
        else:
            print("NOT SURE IF THIS IS RIGHT")
            diam_mean, diam_mean_std =  nd.CalcDiameter.FitMeanDiameter(sizes_df_lin, settings)
            diam_inv_mean, diam_inv_std = nd.CalcDiameter.StatisticOneParticle(sizes_df_lin)
            diam_inv_mean = 1/diam_mean
            
        diam_grid = np.linspace(histogramm_min,histogramm_max,1000)
        diam_grid_inv = 1/diam_grid
        
        prob_diam_inv = scipy.stats.norm(diam_inv_mean,diam_inv_std).pdf(diam_grid_inv)
        prob_diam_inv = prob_diam_inv / prob_diam_inv.max() * max_hist
        prob_diam = 1 / prob_diam_inv 
        
    
        plt.plot(diam_grid,prob_diam_inv)

    
    if Histogramm_Save == True:
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "Diameter Histogramm", settings,
                                       data = sizes_df_lin)
        
        
    nd.handle_data.WriteJson(ParameterJsonFile, settings) 




def DiameterPDF(ParameterJsonFile, sizes_df_lin, histogramm_min = None, histogramm_max = None, Histogramm_min_max_auto = None, binning = None):
    import NanoObjectDetection as nd

    settings = nd.handle_data.ReadJson(ParameterJsonFile)
       
#    sizes_df_lin = sizes_df_lin[sizes_df_lin["diameter"] > 75]
    
    DiameterPDF_Show = settings["Plot"]['DiameterPDF_Show']
    DiameterPDF_Save = settings["Plot"]['DiameterPDF_Save']

    histogramm_min = settings["Plot"]['PDF_min']
    histogramm_max = settings["Plot"]['PDF_max']
        
    diam_inv_mean, diam_inv_std = nd.CalcDiameter.StatisticOneParticle(sizes_df_lin)
    diam_grid = np.linspace(histogramm_min,histogramm_max,10000)
    diam_grid_inv = 1/diam_grid
    
    
    inv_diam,inv_diam_std = nd.CalcDiameter.InvDiameter(sizes_df_lin, settings)
    
    prob_inv_diam = np.zeros_like(diam_grid_inv)
    for index, (loop_mean, loop_std) in enumerate(zip(inv_diam,inv_diam_std)):
        print("mean_diam_part = ", 1 / loop_mean)

        my_pdf = scipy.stats.norm(loop_mean,loop_std).pdf(diam_grid_inv)

        my_pdf = my_pdf / np.sum(my_pdf)
        
        prob_inv_diam = prob_inv_diam + my_pdf    

    # normalize
    prob_inv_diam = prob_inv_diam / np.sum(prob_inv_diam)

    diam_mean = np.mean(GetCI_Interval(prob_inv_diam, diam_grid, 0.001))
    
    lim_68CI = GetCI_Interval(prob_inv_diam, diam_grid, 0.68)
    diam_68CI = lim_68CI[1] - lim_68CI[0]
    
    lim_95CI = GetCI_Interval(prob_inv_diam, diam_grid, 0.95)
    diam_95CI = lim_95CI[1] - lim_95CI[0]
    
    num_trajectories = len(sizes_df_lin) 

    Histogramm_min_max_auto = settings["Plot"]["DiameterPDF_min_max_auto"]
    
    if Histogramm_min_max_auto == 1:
        histogramm_min, histogramm_max = GetCI_Interval(prob_inv_diam, diam_grid, 0.999)
        histogramm_min = 0
    else:
        histogramm_min, settings = nd.handle_data.SpecificValueOrSettings(PDF_min, settings, "Plot", "Histogramm_min")
        histogramm_max, settings = nd.handle_data.SpecificValueOrSettings(PDF_max, settings, "Plot", "Histogramm_max")
        

    title = str("Median: {:2.3g} nm; Trajectories{:3.0f}; \n CI68: [{:2.3g} : {:2.3g}] nm;  CI95: [{:2.3g} : {:2.3g}] nm".format(diam_mean, num_trajectories, lim_68CI[0], lim_68CI[1],lim_95CI[0], lim_95CI[1]))
    xlabel = "Diameter [nm]"
    ylabel = "Probability"
    x_lim = [histogramm_min, histogramm_max]
#    x_lim = [1, 1000]
    y_lim = [0, 1.1*np.max(prob_inv_diam)]
#    sns.reset_orig()

    
    Plot2DPlot(diam_grid, prob_inv_diam, title, xlabel, ylabel, mylinestyle = "-",  mymarker = "", x_lim = x_lim, y_lim = y_lim, y_ticks = [0], semilogx = False)

    print("\n\n mean diameter: ", np.round(np.mean(GetCI_Interval(prob_inv_diam, diam_grid, 0.001)),1))
    print("68CI Intervall: ", np.round(GetCI_Interval(prob_inv_diam, diam_grid, 0.68),1),"\n\n")


    
#    prob_diam_inv = scipy.stats.norm(diam_inv_mean,diam_inv_std).pdf(diam_grid_inv)
#    prob_diam_inv = prob_diam_inv / prob_diam_inv.max() * max_hist
#    prob_diam = 1 / prob_diam_inv 
    

#    plt.plot(diam_grid,prob_diam_inv)

    
    if DiameterPDF_Save == True:
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "Diameter Probability", settings,
                                       data = sizes_df_lin)
        
        
    nd.handle_data.WriteJson(ParameterJsonFile, settings)



def GetCI_Interval(probability, value, ratio_in_ci):
    cum_sum = np.cumsum(probability)

    cum_min = 0.5 - (ratio_in_ci/2)
    cum_max = 0.5 + (ratio_in_ci/2)
    
    pos_min = np.int(np.where(cum_sum > cum_min)[0][0])
    pos_max = np.int(np.where(cum_sum > cum_max)[0][0])
    
    value_min = value[pos_min]
    value_max = value[pos_max]
        
    return value_min,value_max

def GetMeanStdMedian(data):
    my_mean   = np.mean(data)
    my_std    = np.std(data)
    my_median = np.median(data)

    return my_mean, my_std, my_median    
    

def PlotDiameterHistogramm(sizes_df_lin, binning, histogramm_min = 0, histogramm_max = 10000, title = '', xlabel = '', ylabel = ''):
    from NanoObjectDetection.PlotProperties import axis_font, title_font
    import NanoObjectDetection as nd

#    plt.figure()
    fig, ax = plt.subplots()
    diameters = sizes_df_lin.diameter
    show_diameters = diameters[(diameters >= histogramm_min) & (diameters <= histogramm_max)]
    values_hist, positions_hist = np.histogram(sizes_df_lin["diameter"], bins = binning)
    # histogram of sizes, only taking into account 
    sns.distplot(show_diameters, bins=binning, rug=True, kde=False) 
    #those that are below threshold size as defined in the initial parameters
#    plt.rc('text', usetex=True)
    plt.rc('text', usetex=False)
    plt.title(title, **title_font)
    #   plt.ylabel(r'absolute occurrence')
    plt.ylabel(ylabel, **axis_font)
    plt.xlabel(xlabel, **axis_font)
    plt.grid(True)
 
    ax.set_xlim(histogramm_min, histogramm_max)

    # infobox
    _, _, my_median = GetMeanStdMedian(sizes_df_lin.diameter)
    diam_inv_mean, diam_inv_std = nd.CalcDiameter.StatisticOneParticle(sizes_df_lin)

    my_mean = 1/diam_inv_mean

    diam_68 = [1/(diam_inv_mean + 1*diam_inv_std), 1/(diam_inv_mean - 1*diam_inv_std)]
    diam_95 = [1/(diam_inv_mean + 2*diam_inv_std), 1/(diam_inv_mean - 2*diam_inv_std)]
#    diam_99 = [1/(diam_inv_mean + 3*diam_inv_std), 1/(diam_inv_mean - 3*diam_inv_std)]
    

    textstr = '\n'.join([
    r'$\mathrm{median}=  %.1f$ nm' % (my_median),
    r'$\mu = %.1f$ nm' % (my_mean),
    r'$1 \sigma = [%.1f; %.1f]$ nm' %(diam_68[0], diam_68[1]), 
    r'$2 \sigma = [%.1f; %.1f]$ nm' %(diam_95[0], diam_95[1]), ])

    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
#    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
#        , bbox=props)
    
    ax.text(0.60, 0.95, textstr, transform=ax.transAxes, **axis_font, verticalalignment='top', bbox=props)
    
    
#    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#    ax.text(0.05, 0.95, title, transform=ax.transAxes, fontsize=14,
#            verticalalignment='top', bbox=props) 
    return values_hist



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
        
    bp() 
    params = catch_all["params"]
    title_font = catch_all["title_font"]
    axis_font = catch_all["axis_font"]
    
    return params, title_font, axis_font
    



def export(save_folder_name, save_image_name, settings = None, use_dpi = None, data = None, data_header = None):    
    import NanoObjectDetection as nd
    if settings is None:
        if use_dpi is None:
            sys.exit("need settings or dpi!")
    else:
        use_dpi, settings = nd.handle_data.SpecificValueOrSettings(use_dpi,settings, "Plot", "dpi")
        save_json         = settings["Plot"]["save_json"]
        save_data2csv     = settings["Plot"]["save_data2csv"]
        
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
    print('Figure saved at: {}'.format(my_dir_name))

#    Image.open(entire_path_image).show()
    

    #here comes the json parameter file 
    if save_json == 1:
        file_name_json = '%s_%s.json'.format( date=datetime.datetime.now()) %(time_string, save_image_name)
        entire_path_json = my_dir_name +  file_name_json
        
        if settings is None:
            settings = 0
        else:
            settings["Plot"]["SaveProperties"] = entire_path_json
        
            with open(entire_path_json, 'w') as outfile:
                json.dump(settings, outfile, sort_keys=True, indent=4)

    #here comes the csv data file 
             
    if save_data2csv == 1:
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
    """ plot comparison of the median filter smoothed particle intensity ("mass") with original values """
    
    if my_particle == -1:
        # get the first occuring particle
        # (can be different from 0 or 1 because they might have been thrown away due to being stumps)
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
    """ show the maximum relative step for each particle """

    beads_property = pd.DataFrame(columns=['max_step']);
    
    beads_property['max_step'] = t1_gapless.groupby('particle')['rel_step'].max()
    
    # show the maximum relative step for each particle
    plt.figure()
    plt.plot(beads_property,'x')
    plt.title("Intensity Analysis of each Particle")
    plt.xlabel("particle")
    plt.ylabel("Maximum relative step")


    
def CutTrajectorieAtStep(t1_gapless, particle_to_split, max_rel_median_intensity_step):  
    """ display the intensity ("mass") vs. frame# of a particle trajectory that needs to be split """
    
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
    my_gamma = 2

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
    
    traj = traj[traj.particle.isin(id_particle)]
        
#    fig, (ax_raw, ax_eval, ax_hist) = plt.subplots(3, sharex=True, sharey=True, figsize = [18,8])
    fig = plt.figure(figsize = [22, 14], constrained_layout=True)
    gs = GridSpec(4, 2, figure=fig)

#    fig, ax_tot = plt.subplots(4,2, figsize = [18, 10], constrained_layout=True)
    ax_raw = fig.add_subplot(gs[0, :])
#    ax_traj = fig.add_subplot(gs[1, :], sharex = ax_raw, sharey = ax_raw)
#    ax_eval = fig.add_subplot(gs[2, :], sharex = ax_raw, sharey = ax_raw)

    ax_traj = fig.add_subplot(gs[1, :], aspect = "equal")
    ax_eval = fig.add_subplot(gs[2, :], aspect = "equal")
    
    ax_hist = fig.add_subplot(gs[3, 0])
    ax_hist_cum = fig.add_subplot(gs[3, 1])
    
   
    import matplotlib.colors as colors
#    raw_image = ax_raw.imshow(rawframes_rot[0,:,:]**my_gamma, cmap = 'gray', animated = True)
    raw_image = ax_raw.imshow(rawframes_rot[0,:,:], cmap = 'gray', norm=colors.PowerNorm(gamma=1/my_gamma), animated = True, vmin = np.min(rawframes_rot), vmax = np.max(rawframes_rot))
#    raw_image = ax_raw.imshow(rawframes_rot[0,:,:], cmap = 'PuBu_r', animated = True)
#    diameter_image  = ax_eval.plot()
  
    ax_raw.set_title('raw-data', fontsize = my_font_size_title)
    ax_raw.set_ylabel('y-Position [um]', fontsize = my_font_size)    
    ax_raw.set_xlabel('x-Position [um]', fontsize = my_font_size)
    
    Show_all = False
    if Show_all:
        [x_min, x_max, y_min, y_max] = [0, rawframes_rot.shape[2],0, rawframes_rot.shape[1]]        
        
    else:
        [x_min, x_max, y_min, y_max] = [0, rawframes_rot.shape[2], 30, 120]

    ax_raw.set_xlim([x_min, x_max])
    ax_raw.set_ylim([y_min, y_max])
    
    x_max_um = x_max * microns_per_pixel
    y_min_um = y_min * microns_per_pixel
    y_max_um = y_max * microns_per_pixel

    part_id_min = np.min(traj.particle)
    part_id_max = np.max(traj.particle)
    
    diam_max = np.round(np.max(sizes_df_lin.diameter) + 5,-1)
    diam_min = np.round(np.min(sizes_df_lin.diameter) - 5,-1)
        
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
        ax_traj.set_title('drift corrected trajectory', fontsize = my_font_size_title)
        ax_traj.set_ylabel('y-Position [um]', fontsize = my_font_size)    
        ax_traj.set_xlabel('x-Position [um]', fontsize = my_font_size)

        ## HERE COMES THE DIAMETER
        t6_frame = traj[traj.frame == i]
        t6_frame = t6_frame[t6_frame.particle.isin(id_particle)]
        
        t6_frame = t6_frame.set_index(["particle"])
        t6_frame = t6_frame.sort_index()
        
        ax_eval.clear()
        ax_hist.clear()
        ax_hist_cum.clear()
      
        sizes_df_lin_frame = sizes_df_lin[sizes_df_lin.true_particle.isin(t6_frame.index)]
        sizes_df_lin_frame = sizes_df_lin_frame.sort_index()
                    
         
        ax_scatter_diam = ax_eval.scatter(t6_frame.x, t6_frame.y, c = sizes_df_lin_frame.diameter, cmap='jet', vmin=diam_min, vmax=diam_max)
 
        ax_raw.tick_params(direction = 'out')
        ax_traj.tick_params(direction = 'out')
        ax_eval.tick_params(direction = 'out')
        
        ax_eval.set_title('Diameter of each particle', fontsize = my_font_size_title)
        ax_eval.set_ylabel('y-Position [um]', fontsize = my_font_size)    
        ax_eval.set_xlabel('x-Position [um]', fontsize = my_font_size)
        
       
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
        
        
        diam_grid = np.linspace(histogramm_min,histogramm_max,100)
        diam_grid_inv = 1/diam_grid
        
        inv_diam,inv_diam_std = nd.CalcDiameter.InvDiameter(sizes_df_lin_before_frame, settings)
        
        prob_inv_diam = np.zeros_like(diam_grid_inv)
        
#        if ('prob_inv_diam_sum' in locals()) == False:
#            global prob_inv_diam_sum
#            print("create cum sum varialbe")
#            prob_inv_diam_sum = np.zeros_like(diam_grid_inv)
       
        
        for index, (loop_mean, loop_std) in enumerate(zip(inv_diam,inv_diam_std)):
#            print("mean_diam_part = ", 1 / loop_mean)
    
            my_pdf = scipy.stats.norm(loop_mean,loop_std).pdf(diam_grid_inv)
    
            my_pdf = my_pdf / np.sum(my_pdf)
            
            prob_inv_diam = prob_inv_diam + my_pdf    
            
            # accumulate        
        prob_inv_diam_sum = prob_inv_diam_sum + prob_inv_diam
        
        ax_hist.plot(diam_grid, prob_inv_diam)
        
        ax_hist.set_xlim([histogramm_min, histogramm_max])
        ax_hist.set_ylim([0, 1.1*np.max(prob_inv_diam)+0.01])
 
        ax_hist.set_xlabel('Diameter [nm]', fontsize = my_font_size)    
        ax_hist.set_ylabel('Occurance', fontsize = my_font_size)
        ax_hist.set_title("Live Histogram", fontsize = my_font_size_title)
        ax_hist.set_yticks([])
 

        ## ACCUMULATED DIAMETER PDF 
        ax_hist_cum.plot(diam_grid, prob_inv_diam_sum)
        
        ax_hist_cum.set_xlim([histogramm_min, histogramm_max])
        ax_hist_cum.set_ylim([0, 1.1*np.max(prob_inv_diam_sum)+0.01])
 
        ax_hist_cum.set_xlabel('Diameter [nm]', fontsize = my_font_size)    
        ax_hist_cum.set_ylabel('Occurance', fontsize = my_font_size)
        ax_hist_cum.set_title("Cummulated Histogram", fontsize = my_font_size_title)
        ax_hist_cum.set_yticks([])

        ax_hist_cum.tick_params(direction = 'out')
        ax_hist.tick_params(direction = 'out')
        
        if 1 == 0:
            # print limits
            print("ax_raw:", ax_raw.get_xlim())
            print("ax_traj:", ax_traj.get_xlim())
            print("ax_eval:", ax_eval.get_xlim())



        return raw_image
    
    
#    anim = animation.FuncAnimation(fig, animate, init_func=init, frames = 100, interval=100, blit=True, repeat=False)
    
#    anim = animation.FuncAnimation(fig, animate, frames = 1000, interval = 10, repeat = False)
    
    frames_tot = 250
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




def AnimateDiameterAndRawData_Big2(rawframes, static_background, rawframes_pre, sizes_df_lin, traj, ParameterJsonFile): 
    from matplotlib.gridspec import GridSpec
    import time
    
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
    
    
def DriftFalseColorMapFlow(calc_drift, number_blocks, y_range):
    
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
    
   
    
def MsdOverLagtime(lagt_direct, mean_displ_direct, mean_displ_fit_direct_lin, collect_data = None, alpha_values = 0.5, alpha_fit = 0.3):
    from NanoObjectDetection.PlotProperties import axis_font, title_font

    plt.plot(lagt_direct, mean_displ_direct,'k.', alpha = alpha_values) # plotting msd-lag-time-tracks for all particles
    plt.plot(lagt_direct, mean_displ_fit_direct_lin, 'r-', alpha = alpha_fit) # plotting the lin fits
    #ax.annotate(particleid, xy=(lagt_direct.max(), mean_displ_fit_direct_lin.max()))
    plt.title("MSD fit", **title_font)
    plt.ylabel("MSD $[\mu m^2]$", **axis_font)
    plt.xlabel("Lagtime [s]", **axis_font)

    plt.xlim(0,np.max(lagt_direct) * 1.1)
    plt.ylim(0,plt.ylim()[1])


    
    
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
    
    
    
    
def HistogrammDiameterOverTime():
    import seaborn as sns
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import numpy as np
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
 
    binning = settings["Plot"]["Histogramm_Bins"]
    binning = 25
    
    Histogramm_min_max_auto = settings["Plot"]["Histogramm_min_max_auto"]
    if Histogramm_min_max_auto == 1:

        histogramm_min = np.round(np.min(sizes_df_lin.diameter) - 5, -1)
        histogramm_max = np.round(np.max(sizes_df_lin.diameter) + 5, -1)
        
    histogramm_min, settings = nd.handle_data.SpecificValueOrSettings(histogramm_min, settings, "Plot", "Histogramm_min")
    histogramm_max, settings = nd.handle_data.SpecificValueOrSettings(histogramm_max, settings, "Plot", "Histogramm_max")

    effective_fps = settings["MSD"]["effective_fps"]   
    
    test = sizes_df_lin[["diameter","first_frame","traj_length"]]
    diameter = np.asarray(test["diameter"])
    first_frame = np.asarray(test["first_frame"])
    traj_length = np.asarray(test["traj_length"])
    last_frame = first_frame + traj_length - 1
    
    diameter_time = np.repeat(diameter,traj_length.astype(int))
    my_frame = np.empty(diameter_time.shape)
    
    new_row = 0;
    for loop, value in enumerate(diameter):
        loop_first_frame = new_row
        loop_traj_length = np.int(traj_length[loop])
        loop_last_frame = loop_first_frame + loop_traj_length - 1
        
        frames_with_this_particle = np.linspace(first_frame[loop], last_frame[loop], traj_length[loop])
        
        my_frame[loop_first_frame : loop_last_frame + 1] = frames_with_this_particle
        
        new_row = loop_last_frame + 1
    
    my_time = my_frame / effective_fps
    

    from NanoObjectDetection.PlotProperties import axis_font, title_font
    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.unicode'] = True
    f = plt.figure(figsize=(10,10))
     

    
    range_frame_min = my_time.min()
    range_frame_max = my_time.max()
    
    range_diam_min = histogramm_min
    range_diam_max = histogramm_max
    
    
    ax3 = f.add_subplot(223)
    
    binning_diameter = binning
    binning_time = np.int((range_frame_max - range_frame_min)*effective_fps + 1)
    
    myhist = ax3.hist2d(diameter_time, my_time, cmap=cm.Greys, bins = [binning_diameter, binning_time], 
               range = [[range_diam_min, range_diam_max], [range_frame_min, range_frame_max]])
    ax3.set_xlabel("Diameter [nm]", **axis_font)
    ax3.set_ylabel("Time [s]", **axis_font)
#    ax3.grid("on")


    ax1 = f.add_subplot(221) 
    ax1.hist(diameter_time, bins = binning_diameter,
             range = [range_diam_min, range_diam_max])
    ax1.set_xlim([range_diam_min, range_diam_max])
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position("top")
    ax1.set_xlabel("Diameter [nm]", **axis_font)
    ax1.set_ylabel("counts [a.u.]", **axis_font)
#    ax1.grid("on")


    ax4 = f.add_subplot(224)
    ax4.hist(my_time, orientation="horizontal", histtype ='step', bins = binning_time,
             range = [range_frame_min, range_frame_max])
    ax4.set_ylim([range_frame_min, range_frame_max])
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")
    ax4.set_ylabel("Time [s]", **axis_font)
    ax4.set_xlabel('Number of Particles', **axis_font)
#    ax4.grid("on")
    
#    ax2 = f.add_subplot(222)
    

    
#    cbar = f.colorbar(ax1, cax=ax2)
#    ax2.colorbar(myhist[3], ax1=ax1)
    
#        # infobox
#    my_mean, my_std, my_median = nd.visualize.GetMeanStdMedian(sizes_df_lin.diameter)
#    
#    textstr = '\n'.join((
#    r'$\mu=%.1f$ nm' % (my_mean, ),
#    r'$\sigma=%.1f$ nm' % (my_std, ),
#    r'$\mathrm{median}=%.1f$ nm' % (my_median, )))
#
#    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#    
#    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, **axis_font, verticalalignment='top', bbox=props)


