# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 12:23:39 2019

@author: Ronny Förster, Stefan Weidlich, Mona Nissen
"""



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
import time

import NanoObjectDetection as nd

## unused for the moment:
#import time
#from matplotlib.animation import FuncAnimation, PillowWriter
#from matplotlib.animation import FuncAnimation


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
        nd.visualize.export(settings["Plot"]["SaveFolder"], "Diameter_Histogramm", settings, data = plot_np)



def Plot2DPlot(x_np,y_np,title = None, xlabel = None, ylabel = None, myalpha = 1, mymarker = 'x', mylinestyle  = ':', x_lim = None, y_lim = None, y_ticks = None, semilogx = False, FillArea = False, Color = None):
    """ plot 2D-data in standardized format as line plot """
    
    from NanoObjectDetection.PlotProperties import axis_font, title_font, params
#    sns.reset_orig()
    
    plt.style.use(params)
    
    plt.figure()
    if semilogx == False:
        if Color == None:
            plt.plot(x_np,y_np, marker = mymarker, linestyle  = mylinestyle, alpha = myalpha)
        else:
            plt.plot(x_np,y_np, marker = mymarker, linestyle  = mylinestyle, alpha = myalpha, color = Color)
            
        if FillArea == True:
            plt.fill_between(x_np,y_np, y2 = 0, color = Color, alpha=0.5)
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
    
    ax = plt.gca()
    return ax



def Plot2DScatter(x_np, y_np, c = None, title = None, xlabel = None, ylabel = None, myalpha = 1,
                  x_min_max = None, y_min_max = None, log = False, cmap = None, ShowLegend = False, ShowLegendTitle = None):
    """ plot 2D-data in standardized format as individual, scattered points """
    
    from NanoObjectDetection.PlotProperties import axis_font, title_font

    plt.figure()
    plt.scatter(x_np,y_np, c=c, cmap=cmap, vmin = 0, alpha = myalpha, linewidths  = 0)    
    plt.grid("on")
    plt.title(title, **title_font)
    plt.xlabel(xlabel, **axis_font)
    plt.ylabel(ylabel, **axis_font)

    if log == False:
        SetXYLim(x_min_max, y_min_max)
    else:
        ax = plt.gca()
        
        #make it log and look pretty
        ax.set_xscale('log')
        
        x_label = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]
        
        ax.set_xticks(x_label)
        
        #check which labels are around the values
        x_min = x_label[np.argwhere(x_label <= np.min(x_np))[-1,0]]
        x_max = x_label[np.argwhere(x_label >= np.max(x_np))[0,0]]
        plt.xlim([x_min, x_max])
        
        ax.set_yscale('log')
        y_label = [1, 2, 3, 4, 6, 10, 15, 20, 30, 40, 60, 80 ,100, 150, 200, 300, 400, 600, 800, 1000]
        ax.set_yticks(y_label)
        
        #check which labels are around the values
        y_min_value = np.min(y_np)
        y_max_value = np.max(y_np)
        
        if y_min_value > 1:
            y_min = y_label[np.argwhere(y_label <= np.min(y_np))[-1,0]]
        else:
            y_min = 1
            
        if y_max_value < 1000:
            y_max = y_label[np.argwhere(y_label >= np.max(y_np))[0,0]]
        else:
            y_max = 1000
        
        plt.ylim([y_min, y_max])
        
        
        
        # https://stackoverflow.com/questions/21920233/matplotlib-log-scale-tick-label-number-formatting
        from matplotlib.ticker import ScalarFormatter
        plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        plt.gca().yaxis.set_major_formatter(ScalarFormatter())

        

    if ShowLegend == True:
        cbar = plt.colorbar()
        if ShowLegendTitle == None:
            cbar.set_label("Brightness")
        else: 
            cbar.set_label(ShowLegendTitle)

def Plot2DImage(array_np,title = None, xlabel = None, ylabel = None, ShowColorBar = True):
    """ plot image in standardized format """
    
    from NanoObjectDetection.PlotProperties import axis_font, title_font, params

    
    plt.figure()
    plt.imshow(array_np, cmap='gray')
    
    if ShowColorBar == True:
        plt.colorbar()
    plt.title(title, **title_font)
    plt.xlabel(xlabel, **axis_font)
    plt.ylabel(ylabel, **axis_font)
    
    

def PlotDiameters(ParameterJsonFile, sizes_df_lin, any_successful_check, histogramm_min = None, histogramm_max = None, Histogramm_min_max_auto = None, binning = None, yEval = False):
    """ plot and/or save diameter histogram (or other statistics) for analyzed particles 
    (main function)
    """
    import NanoObjectDetection as nd
    
    # check for NaN values in DataFrame
    if pd.isna(sizes_df_lin).sum().sum() > 0:
        # drop rows with missing values (e.g. from invalid diameter calculations)
        sizes_df_lin = sizes_df_lin.dropna()
        print('ATTENTION: NaN values were detected in sizes_df_lin. Please check it carefully.\n'
              'In the following plots, rows with NaN values are ignored.\n\n')

    if any_successful_check == False:
        print("NO PARTICLE WAS MEASURED LONG ENOUGH FOR A GOOD STATISTIC !")
    else:
        settings = nd.handle_data.ReadJson(ParameterJsonFile)
        
        show_hist, save_hist = settings["Plot"]["Histogramm_Show"], settings["Plot"]["Histogramm_Save"]
        if show_hist or save_hist:
            do_weighting = settings["Plot"]["Histogramm_abs_or_rel"]
            if do_weighting in ["both", "abs"]:
                DiameterHistogramm(ParameterJsonFile, sizes_df_lin, weighting = False, yEval = yEval)
            if do_weighting in ["both", "rel"]:
                DiameterHistogramm(ParameterJsonFile, sizes_df_lin, weighting = True, yEval = yEval)


        show_pdf, save_pdf = settings["Plot"]["DiameterPDF_Show"], settings["Plot"]["DiameterPDF_Save"]
        if show_pdf or save_pdf:
            PlotDiameterPDF(ParameterJsonFile, sizes_df_lin, yEval = yEval)

        
        show_diam_traj, save_diam_traj = settings["Plot"]["DiamOverTraj_Show"], settings["Plot"]["DiamOverTraj_Save"]
        if show_diam_traj or save_diam_traj:
            color_type = settings["Plot"]["UseRawMass"]
            DiameterOverTrajLenght(ParameterJsonFile, sizes_df_lin.sort_values("rawmass_mean"), show_diam_traj, save_diam_traj, color_type = color_type, yEval = yEval)


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



def DiameterOverTrajLenght(ParameterJsonFile, sizes_df_lin, show_plot = None, save_plot = None, color_type = None, yEval = False):
    """ plot (and save) calculated particle diameters vs. the number of frames
    where the individual particle is visible (in standardized format) """
        
    import NanoObjectDetection as nd

    settings = nd.handle_data.ReadJson(ParameterJsonFile)
   
    plot_diameter = sizes_df_lin["diameter"]
    plot_traj_length = sizes_df_lin["valid frames"] 
   
    Histogramm_min_max_auto = settings["Plot"]["Histogramm_min_max_auto"]
    
    if Histogramm_min_max_auto == 1:
        histogramm_min = np.round(np.min(sizes_df_lin.diameter) - 5, -1)
        histogramm_max = np.round(np.max(sizes_df_lin.diameter) + 5, -1)
        y_min_max = nd.handle_data.Get_min_max_round(plot_diameter,1)
        
    else:
        histogramm_min = settings["Plot"]["Histogramm_min"]
        histogramm_max = settings["Plot"]["Histogramm_max"]
        y_min_max = [histogramm_min, histogramm_max]
    
    # histogramm_min, settings = nd.handle_data.SpecificValueOrSettings(histogramm_min, settings, "Plot", "Histogramm_min")
    # histogramm_max, settings = nd.handle_data.SpecificValueOrSettings(histogramm_max, settings, "Plot", "Histogramm_max")
    
    if yEval == True:
        my_title = "Trans. Particle size over tracking time"
    else:
        my_title = "Long. Particle size over tracking time"
        
    my_ylabel = "Diameter [nm]"
    my_xlabel = "Trajectory length [frames]"
    
    x_min_max = nd.handle_data.Get_min_max_round(plot_traj_length, 2)
    x_min_max[0] = 0
    
    if color_type == None:
        my_c = sizes_df_lin["rawmass_mean"]
    
    else:
        if color_type == "mean":
            my_c = sizes_df_lin["rawmass_mean"]
            ShowLegendTitle = "Mean Brightness"
                    
        elif color_type == "median":
            my_c = sizes_df_lin["rawmass_median"]
            ShowLegendTitle = "Median Brightness"
                    
        elif color_type == "max":
            my_c = sizes_df_lin["rawmass_max"]
            ShowLegendTitle = "Max Brightness"
    
    Plot2DScatter(plot_traj_length, plot_diameter, c = my_c, title = my_title, 
                  xlabel = my_xlabel, ylabel = my_ylabel,
                  myalpha = 0.9, x_min_max = x_min_max, y_min_max = y_min_max, log = True, cmap = 'jet', ShowLegend = True, ShowLegendTitle = ShowLegendTitle)
 
    if save_plot == True:
        if yEval == True:
            my_title = "DiameterOverTrajLength_trans"
        else:
            my_title = "DiameterOverTrajLength_long"
        
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], my_title, settings, data = sizes_df_lin, ShowPlot = show_plot)



def DiameterHistogrammTime(ParameterJsonFile, sizes_df_lin, show_plot = None, save_plot = None, histogramm_min = None, histogramm_max = None, Histogramm_min_max_auto = None, binning = None):
    """ plot (and save) temporal evolution of the diameter histogram """
    
    #import NanoObjectDetection as nd
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

    nd.visualize.PlotDiameter2DHistogramm(sizes_df_lin, binning, histogramm_min,
                                          histogramm_max, title, xlabel, ylabel)
  
    if Histogramm_Save == True:
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "Diameter_Histogramm_Time", settings, data = sizes_df_lin)
        
    nd.handle_data.WriteJson(ParameterJsonFile, settings) 



def PlotDiameterHistogramm(sizes, binning, histogramm_min = 0, histogramm_max = 10000, 
                           title = '', xlabel = '', ylabel = '', mycol='C0', my_range = None):
    """ plot a histogram of particles sizes

    Parameters
    ----------
    sizes : pandas.Series
    binning : 

    Returns
    -------
    values_hist : 
    ax : AxesSubplot object

    """
    from NanoObjectDetection.PlotProperties import axis_font, title_font
    # import NanoObjectDetection as nd

#    plt.figure()
    fig, ax = plt.subplots()
    diameters = sizes
    show_diameters = diameters[(diameters >= histogramm_min) & (diameters <= histogramm_max)]

    values_hist, positions_hist = np.histogram(sizes, bins = binning, range = my_range)
    # histogram of sizes, only taking into account 
    plt.hist(show_diameters, bins=binning, range = my_range, color=mycol) 

    # sns.distplot(show_diameters, bins=binning, rug=True, rug_kws={"alpha": 0.4}, kde=False,color=mycol) 
    #those that are below threshold size as defined in the initial parameters
#    plt.rc('text', usetex=True)
    plt.rc('text', usetex=False)
    plt.title(title, **title_font)
    #   plt.ylabel(r'absolute occurrence')
    plt.ylabel(ylabel, **axis_font)
    plt.xlabel(xlabel, **axis_font)
    plt.grid(True)
 
    ax.set_xlim(histogramm_min, histogramm_max)

    return values_hist, ax



def PlotDiameter2DHistogramm(sizes_df_lin, binning, histogramm_min = 0, histogramm_max = 10000, title = '', xlabel = '', ylabel = ''):
    """ plot histogram evolution over time
    
    The figure consists of 3 subplots: 
        - a color matrix indicating the number of particles measured with a
          certain size in each frame ('2D histogram')
        - a size histogram on top
        - a 'particles analyzed per frame' plot on the right
    """
    sns.set(style="darkgrid")
    
    time_points = int(sizes_df_lin["traj length"].sum()) # ... for all trajectories
    plot_data = np.zeros([time_points,2]) # create result matrix
    
    # get rid of these strange array-type index values
    sizes_df_lin = sizes_df_lin.set_index("true_particle")
    
    first_element = True # ... for 'plot_data' array initialization only
    
    # iterate over results' DataFrame 
    # NB: particle_data is pd.Series type (with previous column names as index)
    for particle_index, particle_data in sizes_df_lin.iterrows():
        ii_first_frame = int(particle_data["first frame"])
        ii_traj_length = int(particle_data["traj length"])
        ii_last_frame = ii_first_frame + ii_traj_length - 1
        ii_diameter = particle_data["diameter"]
        
        add_data_plot = np.zeros([ii_traj_length,2])
        
        # fill matrix with frames of appearance (col0) 
        # and diameter values (col1; const.)
        add_data_plot[:,0] = np.linspace(ii_first_frame,ii_last_frame,ii_traj_length)
        add_data_plot[:,1] = ii_diameter
        
        if first_element == True: # initialize results' array
            plot_data = add_data_plot
            first_element = False
        else: 
            # add data points to a large array
            # => matrix with frames of occurence and repeated diameter entries 
            #    (cf. 'weighting' option for the 1D histogram)
            plot_data = np.concatenate((plot_data,add_data_plot), axis = 0)

    from matplotlib.ticker import NullFormatter
    
    # the random data (MN2101: what do you want to say with this??)
    diam = plot_data[:,1] # x: repeated diameter values to count into the histogram
    time = plot_data[:,0] # y: frame numbers where analyzed particles appear
    
    # define plot limits and properties
    lim_diam_start = np.min(diam)
    lim_diam_end = np.max(diam)
    lim_time_start = np.min(time)
    lim_time_end = np.max(time)+1
    
    if binning == None:
        diam_bins_nm = 5
        diam_bins = int((lim_diam_end - lim_diam_start)/diam_bins_nm)
    else:
        diam_bins = binning
    
    time_bins = int(lim_time_end - lim_time_start + 0)
    
    nullfmt = NullFormatter()         # no labels
    
    # define the dimensions of different image sections/subplots
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    
    # start with a quadratic Figure
    plt.figure(figsize=(8, 8))
    # define specific axis handles for the 3 image sections
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    
    # delete ticks between the image sections
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    
    # plot the 2D histogram/'scatter plot' into the major image section
    axScatter.hist2d(diam, time, bins = [diam_bins, time_bins], cmap = "jet")
    axScatter.set_xlabel("Diameter [nm]")
    axScatter.set_ylabel("Time [frames]")
    axScatter.set_xlim((lim_diam_start, lim_diam_end))
    axScatter.set_ylim((lim_time_start, lim_time_end))
    
    # plot the 1D histograms alongside into the minor image sections (top+right)
    axHistx.hist(diam, bins=diam_bins)
    axHistx.set_ylabel("Occurrence [counts]")
    time_hist = axHisty.hist(time, bins=time_bins, orientation='horizontal')
    axHisty.set_xlabel("Analyzed particles")
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    
    # plt.show()      

    

def NumberOfBinsAuto(mydata, average_height = 4):
    number_of_points = len(mydata)
    
    bins = int(np.ceil(number_of_points / average_height))
    
    return bins



def DiameterHistogramm(ParameterJsonFile, sizes_df_lin, histogramm_min = None, 
                       histogramm_max = None, Histogramm_min_max_auto = None, 
                       binning = None, weighting=False, num_dist_max=2, fitInvSizes=True,
                       showInfobox=True, fitNdist=False, showICplot=False, showInvHist=False, yEval = False):
    """ wrapper for plotting a histogram of particles sizes, 
    optionally with statistical information or distribution model in sizes
    or inverse sizes space
    
    Parameters
    ----------
    weighting : count each track once or as often as it appears in all frames
    num_dist_max : maximum number of size components N considered in the distribution model
    fitInvSizes : optimize model for the inverse sizes
    showInfobox : add infobox with statistical information on the distribution
    fitNdist : fit Gaussian mixture model or simply plot an equivalent recipr. Gaussian
    showICplot : separately plot AIC and BIC over N for all considered models
    """
    
    if yEval == False:
        title = 'Amount of long. trajectories analyzed = %r' % len(sizes_df_lin)
    else:
        title = 'Amount of trans. trajectories analyzed = %r' % len(sizes_df_lin)
    
    if weighting==True:
        # repeat each size value as often as the particle appears in the data
        # e.g. sizes=[5,4,7] with tracklengths=[2,3,2]
        #      => sizes_weighted=[5,5,4,4,4,7,7]
        sizes = sizes_df_lin.diameter.repeat(np.array(sizes_df_lin['valid frames'],dtype='int'))
        title = title + ', weighted by track length'
    else:
        if type(sizes_df_lin) is pd.DataFrame:
            sizes = sizes_df_lin.diameter
        else:
            sizes = sizes_df_lin # should be np.array or pd.Series type here
        title = title + ', each track counted once'     
        
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    # read in plotting parameters
    Histogramm_Show = settings["Plot"]['Histogramm_Show']
    Histogramm_Save = settings["Plot"]['Histogramm_Save']
    
    min_diam = np.floor(np.min(sizes))
    max_diam = np.ceil(np.max(sizes))
    
    if settings["Plot"]["Histogramm_Bins_Auto"] == 1:
        settings["Plot"]["Histogramm_Bins"] = NumberOfBinsAuto(sizes_df_lin)
        my_range = (min_diam, max_diam)
        
    else:
        bin_nm = settings["Plot"]["Histogramm_Bin_size_nm"]
    
        # set min and maximum a multiple of bin_nm
        min_diam = np.floor(min_diam / bin_nm) * bin_nm
        max_diam = np.ceil(max_diam / bin_nm) * bin_nm
    
        bin_number = int(np.ceil((max_diam - min_diam) / bin_nm))
        settings["Plot"]["Histogramm_Bins"] = bin_number 
        
        my_range = (min_diam, min_diam + bin_number * bin_nm)
        
    if binning is None:
        #binning = np.arange(settings["Plot"]["Histogramm_Bins"])
        binning = settings["Plot"]["Histogramm_Bins"]
    if Histogramm_min_max_auto is None:
        Histogramm_min_max_auto = settings["Plot"]["Histogramm_min_max_auto"]
    if Histogramm_min_max_auto == 1:
        histogramm_min = np.round(np.min(sizes) - 5, -1)
        histogramm_max = np.round(np.max(sizes) + 5, -1)
        
    histogramm_min, settings = \
        nd.handle_data.SpecificValueOrSettings(histogramm_min, settings, "Plot", "Histogramm_min")
    histogramm_max, settings = \
        nd.handle_data.SpecificValueOrSettings(histogramm_max, settings, "Plot", "Histogramm_max")

    xlabel = 'Diameter [nm]'
    ylabel = 'Absolute occurrence'

    values_hist, ax = \
        nd.visualize.PlotDiameterHistogramm(sizes, binning, histogramm_min, 
                                            histogramm_max, title, xlabel, ylabel, my_range = my_range)
    if showInvHist:
        inv_diams = 1000/sizes # 1/um
        values_invHist, ax0 = PlotDiameterHistogramm(inv_diams, 40, histogramm_min = inv_diams.min(), 
                                                     histogramm_max = inv_diams.max(), 
                                                     xlabel='Inv. diameter [1/$\mu$m]', 
                                                     ylabel=ylabel, title=title, mycol='C3')
        max_invHist = values_invHist.max()
    else:
        ax0 = 'none'
        max_invHist = 1
    
    if (showInfobox==True) and (fitNdist==False):
        nd.visualize.PlotInfobox1N(ax, sizes)
    
    if settings["Plot"]["Histogramm_Fit_1_Particle"] == 1:
        if histogramm_min != 0:
            diam_grid = np.linspace(histogramm_min,histogramm_max,1000) # equidistant grid
        else:
            diam_grid = np.linspace(1E-5,histogramm_max,1000) # equidistant grid
            
        grid_stepsizes = (diam_grid[1]-diam_grid[0])*np.ones_like(diam_grid)
        max_hist = values_hist.max()
        
        if fitNdist == False: 
            nd.logger.info("Show equivalent (reciprocal) Gaussian in the plot.")
            # consider only one contributing particle size
            mean, median, CV = \
                nd.visualize.PlotReciprGauss1Size(ax, diam_grid, grid_stepsizes, 
                                                  max_hist, sizes, fitInvSizes)
            nd.logger.info("Parameters: mean={:.2f}, median={:.2f}, CV={:.2f}".format(mean,median,100*CV))
            
        else:
            nd.logger.info("Model the (inverse) sizes distribution as a mixture of Gaussians.")
            # consider between 1 and num_dist_max contributing particle sizes
            diam_means, CVs, weights, medians = \
                nd.visualize.PlotReciprGaussNSizes(ax, diam_grid, grid_stepsizes,
                                                   max_hist, sizes, fitInvSizes, 
                                                   num_dist_max=num_dist_max,
                                                   showICplot=showICplot, axInv=ax0,
                                                   max_hist_inv=max_invHist, 
                                                   showInvHist=showInvHist)
                # weights should be the same, no matter if inverse or not
                # CVs as well (at least approximately)
            
            if showInfobox==True:
                nd.visualize.PlotInfoboxMN(ax, diam_means, CVs, weights, medians)
            
    if Histogramm_Save == True:
        if yEval == True:
            if weighting == True:
                my_title = "Diameter_Histogramm-tran-weighted"
            else:
                my_title = "Diameter_Histogramm-tran-unweighted"
        else:
            if weighting == True:
                my_title = "Diameter_Histogramm-long-weighted"
            else:
                my_title = "Diameter_Histogramm-long-unweighted"
            
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], my_title, settings, data = sizes_df_lin, ShowPlot = Histogramm_Show)
        
        
    nd.handle_data.WriteJson(ParameterJsonFile, settings) 
    
    return ax, ax0



def PlotVlines(ax, PDF, grid, mean, median, CI68, CI95, mycolor):
    """ plot vertical lines to visualize mean, median and CI-intervals """
    ax.vlines([mean], 0, [PDF[np.isclose(grid,mean,rtol=0.002)].mean()],
              # transform=ax.get_xaxis_transform(), 
              colors='black', alpha=0.3)
    ax.vlines([median], 0, [PDF[np.where(grid==median)]],
              colors=mycolor, alpha=0.8)
    ax.vlines([CI68[0], CI68[1]], 0, 
              [PDF[np.where(grid==CI68[0])], PDF[np.where(grid==CI68[1])]],
              colors=mycolor, alpha=0.8, ls='--')
    ax.vlines([CI95[0], CI95[1]], 0, 
              [PDF[np.where(grid==CI95[0])], PDF[np.where(grid==CI95[1])]],
              colors=mycolor, alpha=0.8, ls=':')



def PlotDiameterPDF(ParameterJsonFile, sizes_df_lin, plotInSizesSpace=True, fitInInvSpace=True, invGridEquidist=True, PDF_min_max_auto = None, nComponentsFit=1, statsInfoTitle=True, showFitInfobox=True, useCRLB=True, fillplot=True, mycolor='C2', plotVlines=True, plotComponents=True, yEval = False):
    """ plot and optionally fit the diameter probability density function of 
    a particle ensemble as the sum of individual PDFs
    
    Parameters
    ----------
    plotInSizesSpace : visualize the PDF in sizes space (or in inverse sizes space)
    fitInInvSpace : optimize fits in inverse sizes space (or sizes space)
    invGridEquidist : choose an equidistant plotting grid in inverse sizes space
    nComponentsFit : fix number of components to be considered (1 or 2)
    statsInfoTitle : put statistical quantities in the plot title
    showFitInfobox : add extra info-box with fit results
    useCRLB : calculate PDF with CRLB theory (instead of Qian91)
    plotVlines : visualize mean, median etc. in the plot
    plotComponents : visualize individual (inv.) Gaussian contributions

    Returns
    -------
    PDF, grid, ax, fit
    """
    import NanoObjectDetection as nd
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    DiameterPDF_Show = settings["Plot"]['DiameterPDF_Show']
    DiameterPDF_Save = settings["Plot"]['DiameterPDF_Save']
    if PDF_min_max_auto is None:
        PDF_min_max_auto = settings["Plot"]["DiameterPDF_min_max_auto"]
    if PDF_min_max_auto == 0:
        PDF_min = settings["Plot"]['PDF_min'] 
        PDF_max = settings["Plot"]['PDF_max'] 
        
    # define numerical grid
    if invGridEquidist: # equidistant in inverse sizes space
        i_grid = np.linspace(1,1000,999*100+1) # 1/um
    else: # equidistant in sizes space
        e_grid = np.linspace(0.1,1000,10000) # nm
        i_grid = 1000/e_grid[::-1] # 1/um; grid in increasing order
    
    # get PDF in inverse sizes space
    PDF_i, i_grid_stepsizes = \
        nd.statistics.CalcInversePDF(i_grid, sizes_df_lin, settings, useCRLB)
    
    # get statistical quantities
    i_mean, _, i_median, i_CI68, i_CI95 = \
        nd.statistics.GetMeanStdMedianCIfromPDF(PDF_i, i_grid, i_grid_stepsizes)

    num_trajectories = len(sizes_df_lin) 
    
    if plotInSizesSpace: # convert grid and PDF to sizes space
        prefix = 'D'
        unit = 'nm'
        PDF_s, s_grid, s_grid_stepsizes = nd.statistics.ConvertPDFtoSizesSpace(PDF_i, i_grid)
        s_median = 1000/i_median
        s_CI68 = 1000/i_CI68[1], 1000/i_CI68[0]
        s_CI95 = 1000/i_CI95[1], 1000/i_CI95[0]
        # med = np.mean(nd.statistics.GetCI_Interval(PDF_i, i_grid, 0))
        # CI68 = nd.statistics.GetCI_Interval(PDF_i, i_grid, 0.68) # same values as above
        s_mean = sum(PDF_s*s_grid*s_grid_stepsizes) # definition of mean/expected value on continuous data
        # s_mean = 1000/i_mean # this is NOT the same value (!)
        
        mean, median, CI68, CI95 = s_mean, s_median, s_CI68, s_CI95
        PDF, grid, grid_stepsizes = PDF_s, s_grid, s_grid_stepsizes
    else:
        mycolor = 'C3'
        prefix = 'Inv. d'
        unit = '1/$\mu$m'
        
        mean, median, CI68, CI95 = i_mean, i_median, i_CI68, i_CI95
        PDF, grid, grid_stepsizes = PDF_i, i_grid, i_grid_stepsizes
        
    # set the x limits in the plot
    if PDF_min_max_auto == 1:
        PDF_min, PDF_max = nd.statistics.GetCI_Interval(PDF, grid, 0.999)
        PDF_min = int(np.floor(PDF_min))
        PDF_max = int(np.ceil(PDF_max))
        
    # define plot labeling and axes limits
    if statsInfoTitle:
        if yEval == True:
            title = str("{:.0f} trans. trajectories; mean: {:.2f}, median: {:.2f},\nCI68: [{:.2f}, {:.2f}], CI95: [{:.2f}, {:.2f}] ".format(num_trajectories,mean,median,*CI68,*CI95) + unit)
        else:
            title = str("{:.0f} long. trajectories; mean: {:.2f}, median: {:.2f},\nCI68: [{:.2f}, {:.2f}], CI95: [{:.2f}, {:.2f}] ".format(num_trajectories,mean,median,*CI68,*CI95) + unit)
    else:
        title = str("{:.0f} trajectories".format(num_trajectories))
    xlabel = prefix +"iameter ["+unit+"]"
    ylabel = "Probability [a.u.]"
    x_lim = [PDF_min, PDF_max]
    y_lim = [0, 1.1*np.max(PDF)]

    # plot it!
    ax = nd.visualize.Plot2DPlot(grid, PDF, title, xlabel, ylabel, mylinestyle = "-",  
                                 mymarker = "", x_lim = x_lim, y_lim = y_lim, y_ticks = [0,0.01], 
                                 semilogx = False, FillArea = fillplot, Color = mycolor)
    
    if plotVlines: # plot lines for mean, median and CI-intervals
        nd.visualize.PlotVlines(ax, PDF, grid, mean, median, CI68, CI95, mycolor)
    
    nd.logger.info("mean diameter: {:.3f}".format(mean))
    nd.logger.info("median diameter: {:.3f}".format(median))
    nd.logger.info("68CI interval: {:.3f}, {:.3f}".format(*CI68))
    nd.logger.info("95CI interval: {:.3f}, {:.3f}".format(*CI95))
    # nd.logger.info("... in "+unit)

    fit = 'undone'
    if settings["Plot"]["PDF_show_fit"] == 1:
        
        if nComponentsFit == 1: # consider only one contributing particle size   
        
            if not(plotInSizesSpace) and not(fitInInverseSpace):
                nd.logger.warning("Fitting a distribution in sizes space and plotting it in the inversed is not implemented.")
                nd.logger.info("No fit was computed.")
            else:
                nd.logger.info("Fit normal distribution to the PDF.")
                if (fitInInvSpace^plotInSizesSpace):# '^' is the exclusive 'or' operator ('xor')
                    # case that plotting and fitting is to happen in the same space
                    # => fit Gaussian directly
                    
                    fit, f_mean, f_std, residIntegral, R2 = \
                        nd.statistics.FitGaussian(PDF, grid, grid_stepsizes, mean)
                    _, _, f_median, f_CI68, _ = \
                        nd.statistics.GetMeanStdMedianCIfromPDF(fit, grid, grid_stepsizes)
                    f_CV = f_std/f_mean
                    
                    fitcolor='C0'
                        
                else: # (fitInInvSpace and plotInSizesSpace): 
                    # fit Gaussian in the inv. space and back-convert it
                        
                    # OLD VERSION
                    # # plot reciprocal Gaussian equivalent to the original sizes ensemble
                    # nd.visualize.PlotReciprGauss1Size(ax, inv_grid, max_PDF, sizes_df_lin)
                    
                    fit_i, fi_mean, fi_std, residIntegral, R2 = \
                        nd.statistics.FitGaussian(PDF_i, i_grid, i_grid_stepsizes, i_mean)
                    f_CV = fi_std/fi_mean
                        
                    fit, _, _ = nd.statistics.ConvertPDFtoSizesSpace(fit_i, i_grid)
                    f_mean, _, f_median, f_CI68, _ = \
                        nd.statistics.GetMeanStdMedianCIfromPDF(fit, grid, grid_stepsizes)
                    
                    fitcolor='C3'
                    
                ax.plot(grid, fit, color=fitcolor)
                
                if showFitInfobox:
                    nd.visualize.PlotInfoboxMN(ax,[f_mean],[f_CV],[1],[f_median],unit,residIntegral)
                
                nd.logger.info("Fit results: mean = {:.2f}, CV = {:.3f}".format(f_mean,f_CV))
                nd.logger.info("Fit quality: residual integral = {:.4f}, R**2 = {:.4f}".format(residIntegral,R2))
                    
                
        elif nComponentsFit == 2: # consider 2 contributing particles sizes
            if not(plotInSizesSpace) and not(fitInInvSpace):
                nd.logger.warning("Fitting a distribution in sizes space and plotting it in the inversed is not implemented.")
                nd.logger.info("No fit was computed.")
            else:                
                nd.logger.info("Fit sum of two normal distributions to the PDF.")
                from scipy.stats import norm
                
                if (fitInInvSpace^plotInSizesSpace):
                    # case that plotting and fitting is to happen in the same space
                    # => fit Gaussians directly
                    fit, residIntegral, R2, popt = \
                        nd.statistics.FitGaussianMixture(PDF, grid, grid_stepsizes, CI95[1])
                    
                    m1, m2, s1, s2, w = popt
                    f_means = [m1, m2]
                    f_medians = f_means # true for this symmetric case
                    f_CVs = [s1/m1, s2/m2]
                    fitcolor='C0'
                    
                    # compute the 2 individual contributions
                    if plotComponents:
                        fits = [w*norm(m1,s1).pdf(grid), (1-w)*norm(m2,s2).pdf(grid)]
                    
                else: # (fitInInvSpace and plotInSizesSpace): 
                    # fit Gaussian in the inv. space and back-convert it
                    fit_i, residIntegral, R2, popt_i = \
                        nd.statistics.FitGaussianMixture(PDF_i, i_grid, i_grid_stepsizes, i_CI95[1])
                    i_m1, i_m2, i_s1, i_s2, w = popt_i # weight stays unchanged
                    f_CVs = [i_s1/i_m1, i_s2/i_m2] # CV is a relative quantity => also unchanged (approx.)
                    
                    # convert sum of both components and plot it
                    fit, _, _ = nd.statistics.ConvertPDFtoSizesSpace(fit_i, i_grid)
                    fitcolor='C3'
                    
                    # compute individual Gaussian curves and convert them separately
                    fits_i = [norm(i_m1,i_s1).pdf(i_grid), norm(i_m2,i_s2).pdf(i_grid)]
                    fits = []
                    f_means, f_medians = [], []
                    ws = [w,1-w]
                    for f_i,weight in zip(fits_i,ws):
                        f, _, _ = nd.statistics.ConvertPDFtoSizesSpace(f_i, i_grid)
                        f_mean, _, f_median, _, _ = \
                            nd.statistics.GetMeanStdMedianCIfromPDF(f, grid, grid_stepsizes)
                        f_means.append(f_mean)
                        f_medians.append(f_median)
                        fits.append(weight*f)
                    
                # plot the fits
                ax.plot(grid, fit, color=fitcolor)
                if plotComponents:
                    for f in fits:
                        ax.plot(grid, f, color=fitcolor, ls='--')
                
                if showFitInfobox:
                    nd.visualize.PlotInfoboxMN(ax,f_means,f_CVs,(w,1-w),f_medians,unit,residIntegral)
                    
                nd.logger.info("Fit results: means = {:.2f}, {:.2f}; CVs = {:.1f},{:.1f} %; weights = {:.1f}, {:.1f} 5".format(f_means[0],f_means[1],100*f_CVs[0],100*f_CVs[1],w*100,(1-w)*100)) 
                nd.logger.info("Fit quality: residual integral = {:.4f}, R**2 = {:.4f}".format(residIntegral,R2))
            
                
        else:
            nd.logger.warning("Fitting more than 2 components is not implemented yet.")
            nd.logger.info("No fit was performed.")
        
        if not(type(fit) is str):
            if (residIntegral > 0.1):
                nd.logger.warning("Fit quality is not optimal (residual integral > 10%).")
                    
    
    if DiameterPDF_Save:
        save_folder_name = settings["Plot"]["SaveFolder"]
        
        if yEval == True:
            my_title = "Diameter_Probability_trans"
        else:
            my_title = "Diameter_Probability_long"
        
        settings = nd.visualize.export(save_folder_name, my_title, settings, data = sizes_df_lin, ShowPlot = DiameterPDF_Show)
        
        # data = np.transpose(np.asarray([grid, PDF]))
        # nd.visualize.save_plot_points(data, save_folder_name, 'Diameter_Probability_Data')
        
    nd.handle_data.WriteJson(ParameterJsonFile, settings)
    
    return PDF, grid, ax, fit
    
    
    
def myGauss(d,mean,std):
    return 1/((2*np.pi)**0.5*std)*np.exp(-(d-mean)**2/(2.*std**2))



def PlotReciprGauss1Size(ax, diam_grid, diam_grid_stepsizes, max_y, sizes, fitInvSizes):
    """ add (reciprocal) Gaussian function to a sizes histogram (or PDF)
    
    NB: No fit is computed here, only an equivalent distribution with the same
        mean and std as the original "sizes" values.
    
    if fitInvSizes==True:
        With mean and std of the inversed values, a Gaussian curve in the 
        reciprocal size space is computed, back-transformed and
        normalized to the maximum histogram/PDF value.
    else:
        Mean and std are taken directly from sizes data and Gaussian is plotted
        (NOT a reciprocal in this case).
    """
    # diam_grid_stepsize = diam_grid[1] - diam_grid[0] # equidistant grid (!)
    
    if fitInvSizes:
        mycolor = 'C3'
        
        diam_inv_mean, diam_inv_std, diam_inv_median, diam_inv_CI68, diam_inv_CI95 = \
            nd.statistics.StatisticOneParticle(sizes)
         
        diam_grid = diam_grid[diam_grid > 0]   
         
        diam_grid_inv = 1000/diam_grid # 1/um
        
        # max_non_inf = np.max(diam_grid_inv[np.isinf(diam_grid_inv) == False])
        # diam_grid_inv[np.isinf(diam_grid_inv)] = max_non_inf
        
        prob_diam_inv_1size = \
            scipy.stats.norm(diam_inv_mean,diam_inv_std).pdf(diam_grid_inv)
        
        # law of inverse fcts:
        prob_diam_1size = 1/(diam_grid**2) * prob_diam_inv_1size
        
        # normalize to integral=1
        prob_diam_1size = prob_diam_1size/sum(prob_diam_1size*diam_grid_stepsizes)
        
        # mean = 1000/diam_inv_mean # incorrect
        # mean = sum(prob_diam_1size*diam_grid)*diam_grid_stepsize
        mean = sum(prob_diam_1size*diam_grid*diam_grid_stepsizes)
        median = 1000/diam_inv_median
        CV = diam_inv_std/diam_inv_mean # from inverse space (!)
        # _, _, _, CI68, _ = \
        #     nd.statistics.GetMeanStdMedianCIfromPDF(prob_diam_1size, diam_grid,
        #                                             diam_grid_stepsize*np.ones_like(diam_grid))
        # print(CI68)
    else:
        mycolor = 'C2'
        mean, std, median = nd.statistics.GetMeanStdMedian(sizes)
        CV = std/mean
        
        prob_diam_1size = \
            scipy.stats.norm(mean,std).pdf(diam_grid)
            
    # scale it up to match histogram (or PDF) plot
    prob_diam_1size = prob_diam_1size / prob_diam_1size.max() * max_y
    
    # print('median={}, CI68=[{},{}]'.format(1000/diam_inv_median,1000/diam_inv_CI68[0],1000/diam_inv_CI68[1]))
    ax.plot(diam_grid,prob_diam_1size,color=mycolor)
    return mean, median, CV
    
    
    
def PlotReciprGaussNSizes(ax, diam_grid, grid_stepsizes, max_y, sizes, fitInvSizes,
                          num_dist_max=2, useAIC=False, showICplot=False, axInv='none', 
                          max_hist_inv=1, showInvHist=False):
    """ plot the (reciprocal) fcts of Gaussian fits on top of a histogram
        
    diam_grid : np.ndarray; plotting grid [nm]
    num_dist_max : int; max. number of size components to consider
    useAIC : boolean; if True, use AIC, if False, use BIC
    """
    # inv_diams = 1000/sizes # 1/um
    # diam_grid_inv = 1000/diam_grid # 1/um
    
    # use Gaussian mixture model (GMM) fitting to get best parameters
    if fitInvSizes:
        inv_diams = 1000/sizes # 1/um
        diam_grid_inv = 1000/diam_grid[::-1] # 1/um; in increasing order
        grid, data = diam_grid_inv, inv_diams
        # inv_diam_means, inv_diam_stds, inv_weights = \
        #     nd.statistics.StatisticDistribution(inv_diams, num_dist_max=num_dist_max,
        #                                         showICplot=showICplot, useAIC=useAIC)
        # if showInvHist:
        #     values_invHist, ax0 = PlotDiameterHistogramm(inv_diams, 40, histogramm_min = inv_diams.min(), 
        #                                                  histogramm_max = inv_diams.max(), 
        #                                                  xlabel = 'Inv. diameter [1/$\mu$m]', 
        #                                                  ylabel = 'Counts', mycol='C3')
    
    else:
        grid, data = diam_grid, sizes
        # diam_means, diam_stds, diam_weights = \
        #     nd.statistics.StatisticDistribution(sizes, #weighting=weighting,
        #                                         num_dist_max=num_dist_max,
        #                                         showICplot=showICplot, useAIC=useAIC)
    means, stds, weights = \
        nd.statistics.StatisticDistribution(data, num_dist_max=num_dist_max,
                                            showICplot=showICplot, useAIC=useAIC)
    CVs = stds/means
    # compute individual Gaussian functions from GMM fitted parameters 
    dist = np.array([weights[n]*myGauss(grid,means[n],stds[n]) 
                     for n in range(weights.size)])
    
    if fitInvSizes:
        if showInvHist:
            # calculate sum of all distributions
            dsum = dist.sum(axis=0)
            # normalize dsum to histogram/PDF max. value...
            normFactor = max_hist_inv / dsum.max()
            dsum = normFactor * dsum
            dist = normFactor * dist # and the individual distributions accordingly
            
            axInv.plot(grid,dist.transpose(),ls='--')
            axInv.plot(grid,dist.sum(axis=0),color='k')
            PlotInfoboxMN(axInv, means, CVs, weights, means, unit='1/$\mu$m', resInt='')
        
        # convert the individual PDFs back to sizes space
        pdfs = []
        for d,w in zip(dist,weights):
            pdf, grid, steps = nd.statistics.ConvertPDFtoSizesSpace(d, diam_grid_inv)
            pdfs.append(pdf) # all normalized to integral=1 (!)
        
        print('CVs from CI68')
        # means = 1000/means # MN: Misleading!
        means, stds, medians, CI68s, dist = [], [], [], [], []
        for p,pdf in enumerate(pdfs):
            mea, std, medi, CI68, _ = \
                nd.statistics.GetMeanStdMedianCIfromPDF(pdf, grid, steps)
                # nd.statistics.GetMeanStdMedianCIfromPDF(pdf, diam_grid, grid_stepsizes)
            means.append(mea)
            stds.append(std)
            medians.append(medi)
            CI68s.append(CI68)
            print((CI68[1]-CI68[0])/(2*mea))
            dist.append(pdf*weights[p]) # weight the individual PDFs to get correct sum
        means, medians, dist = np.array(means), np.array(medians), np.array(dist)
    else:
        medians = means # always true for fully symmetric fcts
        
    
    # calculate sum of all distributions
    dsum = dist.sum(axis=0)
    # normalize dsum to histogram/PDF max. value... (for similar-scale plotting)
    normFactor = max_y / dsum.max()
    dsum = normFactor * dsum
    dist = normFactor * dist # ... and the individual distributions accordingly
    
    ax.plot(grid,dist.transpose(),ls='--')
    ax.plot(grid,dist.sum(axis=0),color='k')
    
    # sort the parameters from lowest to highest mean value
    sortedIndices = means.argsort()#[::-1]
    means = means[sortedIndices]
    medians = medians[sortedIndices]
    # stds = stds[sortedIndices]
    CVs = CVs[sortedIndices]
    weights = weights[sortedIndices]
    
    return means, CVs, weights, medians



# def PlotGaussNSizes(ax, diam_grid, max_y, sizes, num_dist_max=2, weighting=False, useAIC=False, showICplot=False):
#     """ plot Gaussian fits on top of a histogram/PDF
    
#     probably a bit redundant... but at least it's working reliably
#     """    
#     # use Gaussian mixture model (GMM) fitting to get best parameters
#     diam_means, diam_stds, weights = \
#         nd.statistics.StatisticDistribution(sizes, weighting=weighting,
#                                             num_dist_max=num_dist_max,
#                                             showICplot=showICplot, useAIC=useAIC)
    
#     # compute individual Gaussian functions from GMM fitted parameters 
#     dist = np.array([weights[n]*myGauss(diam_grid,diam_means[n],diam_stds[n]) 
#                      for n in range(weights.size)])
    
#     # calculate sum of all distributions
#     dsum = dist.sum(axis=0)
#     # normalize dsum to histogram/PDF max. value...
#     normFactor = max_y / dsum.max()
#     dsum = normFactor * dsum
#     dist = normFactor * dist # and the individual distributions accordingly
    
#     ax.plot(diam_grid,dist.transpose(),ls='--')
#     ax.plot(diam_grid,dist.sum(axis=0),color='k')
    
#     return diam_means, diam_stds, weights



def PlotInfobox1N(ax, sizes):
    """ add textbox to a plot containing statistical information 
    on the size distribution, assuming one single contributing particle size
    
    NB: Quantiles are used here in order to give an idea of the sigma-intervals
        independent of any Gaussian fit.
    """
    from NanoObjectDetection.PlotProperties import axis_font
    
    diam_inv_mean, diam_inv_std, diam_inv_median, diam_inv_CI68, diam_inv_CI95 = \
        nd.statistics.StatisticOneParticle(sizes)
    my_median = 1000/diam_inv_median
    CV = diam_inv_std/diam_inv_mean
    diam_68 = 1000/diam_inv_CI68[::-1] # invert order
    diam_95 = 1000/diam_inv_CI95[::-1]
    # my_mean = 1000/diam_inv_mean # MN: This is misleading, I think
    my_mean = sizes.mean()
    nd.logger.warning("Instead of the converted mean of the inverse sizes, the mean sizes value is displayed in the infobox.")
    
# #   old version (uses the assumption of a Gaussian fct):
#     diam_68 = [1/(diam_inv_mean + 1*diam_inv_std), 1/(diam_inv_mean - 1*diam_inv_std)]
#     diam_95 = [1/(diam_inv_mean + 2*diam_inv_std), 1/(diam_inv_mean - 2*diam_inv_std)]
# #    diam_99 = [1/(diam_inv_mean + 3*diam_inv_std), 1/(diam_inv_mean - 3*diam_inv_std)]
    
    textstr = '\n'.join([
    r'$\mathrm{median}=  %.1f$ nm' % (my_median),
    # r'$\mu_{\mathrm{inv}} = %.1f$ nm' % (my_mean),
    r'$\mu = %.1f$ nm' % (my_mean),
    r'CV$_{\mathrm{inv}}$ = %.3f' % (CV),
    r'$1 \sigma_{\mathrm{q}} = [%.1f, %.1f]$ nm' %(diam_68[0], diam_68[1]), 
    r'$2 \sigma_{\mathrm{q}} = [%.1f, %.1f]$ nm' %(diam_95[0], diam_95[1]), ])
    
    props = dict(boxstyle='round', facecolor='honeydew', alpha=0.7)
    
#    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
#        , bbox=props)
    ax.text(0.55, 0.95, textstr, transform=ax.transAxes, 
            **axis_font, verticalalignment='top', bbox=props)#, va='center')
    
#    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#    ax.text(0.05, 0.95, title, transform=ax.transAxes, fontsize=14,
#            verticalalignment='top', bbox=props) 



def PlotInfoboxMN(ax, means, CVs, weights, medians, unit='nm', resInt=''):
    """ add textbox to a plot containing statistical information 
    on the size distribution, assuming one or several contributing particle sizes
    """    
    medtxt = ''
    mtext = ''
    stext = ''
    wtext = ''
    for med,m,CV,w in zip(medians,means,CVs,weights):
        medtxt += '{:.1f}, '.format(med)
        mtext += '{:.1f}, '.format(m)
        # try:
        #     stext += '[{:.1f},{:.1f}],'.format(*diam_CI68)
        #     sname = 'CI68'
        # except TypeError:
        #     stext += '{:.1f},'.format(diam_CI68)
        #     sname = '$\sigma$'
        stext += '{:.1f}, '.format(100*CV)
        wtext += '{:.1f}, '.format(100*w) # [%]
    
    textstr = '\n'.join([
        r'median = ['+medtxt[:-2]+'] '+unit,
        r'$\mu$ = ['+mtext[:-2]+'] '+unit, # '[:-1]' removes the last ','
        # sname+' = ['+stext[:-1]+'] '+unit, 
        'CV = ['+stext[:-2]+'] %',
        r'$\phi$ = ['+wtext[:-2]+'] %', ])
    
    if len(means)==1: # remove brackets if there's only 1 contribution
        textstr = textstr.replace('[','')
        textstr = textstr.replace(']','')
    
    if not(type(resInt) is str): # add fit quality if given
        textstr += '\nresid.integr. = {:.2f} %'.format(100*resInt)

    props = dict(boxstyle='round', facecolor='honeydew', alpha=0.7)
    
    # choose median from strongest contribution (= position of highest peak)
    mMax = medians[np.where(weights==weights.max())]
    if mMax < (sum(ax.get_xlim())/2): # highest peak left from plot center
        x_text = 0.65 - (0.05*len(weights)) # display text box on the right
    else:
        x_text = 0.05 # otherwise: on the left
    ax.text(x_text, 0.95, textstr, transform=ax.transAxes, 
            **{'fontname':'Arial', 'size':'15'}, #**axis_font, 
            verticalalignment='top', bbox=props)
    


def update_progress(job_title, progress):
    """ display a progress bar for an ongoing task
    """
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
    


def CreateFileAndFolderName(folder_name, file_name, d_type = 'png'):
    '''
    make a folder of the current date (if required)
    and file name (including data) to save stuff in there 
    '''
    
    if folder_name == "auto":
        nd.logger.error("Folder name not given and still auto. Specify folder or run CheckSystem.")
    
    dir_name = '%s\\{date:%y%m%d}\\'.format( date=datetime.datetime.now()) %folder_name

    if len(dir_name) > 259:
        print("\n directory name: \n", dir_name)
        raise ValueError ("directory name to long for windows!")

    try:
        os.stat(dir_name)
    except:
        os.mkdir(dir_name) 
    
    time_string = '{date:%H_%M_%S}'.format( date=datetime.datetime.now())
    
#    file_name_image = '%s_%s.png'.format( date=datetime.datetime.now()) %(time_string, file_name)
    file_name_image = '%s_%s'.format( date=datetime.datetime.now()) %(time_string, file_name)
    file_name_image = file_name_image + '.' + d_type
    
    entire_path = dir_name +  file_name_image

    if len(entire_path) > 259:
        print("\n entire path: \n", entire_path)
        raise ValueError ("full path to long name to long for windows!")

    return dir_name, entire_path, time_string


def save_plot_points(data, save_folder_name, save_csv_name):
    dir_name, entire_path, time_string = CreateFileAndFolderName(save_folder_name, save_csv_name, d_type = 'csv')
    
    np.savetxt(entire_path, data, delimiter = ',')



def export(save_folder_name, save_image_name, settings = None, use_dpi = None, data = None, data_header = None, ShowPlot = -1):    
    import NanoObjectDetection as nd
    if settings is None:
        if use_dpi is None:
            sys.exit("need settings or dpi!")
    else:
        use_dpi, settings = nd.handle_data.SpecificValueOrSettings(use_dpi,settings, "Plot", "dpi")
        save_json         = settings["Plot"]["save_json"]
        save_data2csv     = settings["Plot"]["save_data2csv"]
        
    use_dpi = int(use_dpi)
    
    my_dir_name, entire_path_image, time_string = CreateFileAndFolderName(save_folder_name, save_image_name)

    
    plt.savefig(entire_path_image, dpi= use_dpi, bbox_inches='tight')
    nd.logger.info('Figure saved at: {}'.format(my_dir_name))

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
         
     # if plot is only displayed in order save it, close it now
    if ShowPlot == False:
        nd.logger.info("Close plot since noone wants it")
        plt.close(plt.gcf())   
    elif ShowPlot == -1:
        nd.logger.info("You can prevent the plot to be shown if you just wanna save it in the export function")
    
    #   wait 1 second for unique label
    time.sleep(1)
         
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



def DriftAvgSpeed(drift):
    # show average speed
    #ax = plt.figure()
    drift.plot( marker='.')
    plt.title('Drift in laminar flow ')
    plt.ylabel('Drift [px]')
    plt.xlabel('Lateral position in fibre [px]')
    plt.legend(['y', 'x (flow-direction)'])
        
     
        
def DriftTimeDevelopment(drift, number_blocks):
    # SW 180701: There is some issue with the plot in the block beneath!
    # show time developement
    fig, ax = plt.subplots()
    drift.groupby(['y_range'])[['x']].plot(ax=ax)
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
     
    
    
def DriftVectors(calc_drift, number_blocks, y_range):
        
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
        

def DriftFalseColorMapSpeed(calc_drift, number_blocks, y_range):
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


def DriftCorrectedTraj(tm_sub):
    # Show drift-corrected trajectories
    fig, ax = plt.subplots()
    tp.plot_traj(tm_sub)
    plt.title('Trajectories: Drift-Correction depending on laterial-position (y)')
    plt.xlabel('x-Position [px]')
    plt.ylabel('Lateral position in fibre (y) [px]')
    
    
def PlotGlobalDrift(d,settings,save=True):
    my_font = 18
    plt.figure(figsize=(6,6))
    plt.plot(d)
    plt.title("accumulated drift", fontsize = my_font+2)
    plt.xlabel("frames", fontsize = my_font)
    plt.ylabel("drift [px]", fontsize = my_font)
    plt.legend(("transversal","along fiber"), fontsize = my_font-2)
    
    if save==True:
        save_folder_name = settings["Plot"]["SaveFolder"]
        settings = nd.visualize.export(save_folder_name, "Global_Drift", settings)
    
   
    
def MsdOverLagtime(lagt_direct, mean_displ_direct, mean_displ_fit_direct_lin, collect_data = None, alpha_values = 0.5, alpha_fit = 0.3):
    from NanoObjectDetection.PlotProperties import axis_font, title_font


    t_ms = lagt_direct * 1000

    plt.figure("MSD-Plot")

    # plotting msd-lag-time-tracks for all particles
    plt.plot(t_ms[:-1], mean_displ_direct,'k.', alpha = alpha_values) 
    
    # plotting the lin fits
    plt.plot(t_ms, mean_displ_fit_direct_lin, 'r-', alpha = alpha_fit) 
    
    #ax.annotate(particleid, xy=(lagt_direct.max(), mean_displ_fit_direct_lin.max()))
    plt.title("MSD fit", **title_font)
    plt.ylabel(r"MSD $[\mu m^2]$", **axis_font)
    plt.xlabel("Lagtime [ms]", **axis_font)

    
    
# def VarDiameterOverTracklength(sizes_df_lin, save_folder_name = r'Z:\Datenauswertung\19_ARHCF', save_image_name = 'errorpropagation_diameter_vs_trajectorie_length'):
#     import NanoObjectDetection as nd
#     from nd.PlotProperties import axis_font, title_font
#     # NO WORKING AT THE MOMENT BECAUSE ERROR IS NOT RIGHT YET
    
#     import ronnys_tools as rt
# #    from ronnys_tools.mpl_style import axis_font, title_font
    
#     ax = plt.subplot(111)
#     fig = plt.errorbar(sizes_df_lin['frames'],sizes_df_lin['diameter'], yerr = sizes_df_lin['diameter_std'],fmt='.k', alpha = 0.4, ecolor='orange')
    
#     ax.set_yscale("log")
#     ax.set_ylim(10,300)
    
#     ax.set_yticks([10, 30, 100, 300])
#     ax.set_yticklabels(['10', '30', '100', '300'])
    
#     ax.set_ylabel('Diameter [nm]', **axis_font)
    
#     ax.set_xscale("log")
#     ax.set_xlim(30,1000)
    
#     ax.set_xticks([30, 100, 300, 1000])
#     ax.set_xticklabels(['30', '100', '300', '1000'])
    
#     ax.set_xlabel('Trajectorie duration [frames]', **axis_font)
    
#     ax.grid(which='major', linestyle='-')
#     ax.grid(which='minor', linestyle=':')
    
    
#     #rt.mpl_style.export(save_folder_name, save_image_name)
    
    
def VarDiameterOverMinTracklength(settings, sizes_df_lin, obj_all, num_bins_traj_length = 100, min_traj_length = 0, max_traj_length = 1000):
    """ [insert description here] """
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
    
    
    
def HistogrammDiameterOverTime(sizes_df_lin,ParameterJsonFile):
    """ ?? obsolete ?? """
    import matplotlib.cm as cm
    from NanoObjectDetection.PlotProperties import axis_font, title_font
    
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
    

#    from NanoObjectDetection.PlotProperties import axis_font, title_font
#    import matplotlib
#    matplotlib.rcParams['text.usetex'] = True
#    matplotlib.rcParams['text.latex.unicode'] = True
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
#    my_mean, my_std, my_median = nd.statistics.GetMeanStdMedian(sizes_df_lin.diameter)
#    
#    textstr = '\n'.join((
#    r'$\mu=%.1f$ nm' % (my_mean, ),
#    r'$\sigma=%.1f$ nm' % (my_std, ),
#    r'$\mathrm{median}=%.1f$ nm' % (my_median, )))
#
#    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#    
#    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, **axis_font, verticalalignment='top', bbox=props)


def FitFiberToScreen(image):
    """
    The channel can hardly shown because it is very wide
    Convert it into a more quadratic shape
    """

    height = image.shape[0]
    width = image.shape[1]

    #ratio width to height
    image_ratio = width / height
    
    # cut into pieces to make it more quadrativ
    num_pieces = int(np.sqrt(image_ratio))
    
    # in order to stack them, the width need to be a multiple of num_pieces. Cut away the tail
    width_new = int(np.floor(width / num_pieces))
    width_cut = width_new * num_pieces
    image_cut = image[:,:width_cut]
    
    height_new = height * num_pieces
    
    image_new = np.concatenate(np.hsplit(image, num_pieces), axis = 0)
    
    plt.imshow(image_new)
    
#=============================================================================
#
#=============================================================================

def AnimateDiameterAndRawData_Big2(rawframes, static_background, rawframes_pre, 
                                   sizes_df_lin, traj, ParameterJsonFile):     
    sys.exit("AnimateDiameterAndRawData_Big2 has moved to animate.py! Please change if you see this")

def AnimateDiameterAndRawData_temporal(rawframes_rot, sizes_df_lin_rolling):
    sys.exit("AnimateDiameterAndRawData_temporal has moved to animate.py! Please change if you see this")    

def AnimateStationaryParticles(rawframes_np):
        sys.exit("AnimateStationaryParticles has moved to animate.py! Please change if you see this")
        
def AnimateProcessedRawData(ParameterJsonFile, rawframes_rot, t4_cutted, t6_final, sizes_df_lin, sizes_df_lin_rolling):
    sys.exit("AnimateProcessedRawData has moved to animate.py! Please change if you see this")
    
def AnimateDiameterAndRawData(rawframes_rot, sizes_df_lin, t6_final, settings, DoScatter=True, DoText=False): 
    sys.exit("AnimateDiameterAndRawData has moved to animate.py! Please change if you see this")

def AnimateDiameterAndRawData_Big(rawframes_rot, sizes_df_lin, traj, settings): 
    sys.exit("AnimateDiameterAndRawData_Big has moved to animate.py! Please change if you see this")

def DiameterPDF(ParameterJsonFile, sizes_df_lin, PDF_min_max_auto = None, 
                fitNdist=False, num_dist_max=2, useAIC=True,
                statsInfo=True, showInfobox=True, showICplot=False, 
                fillplot=True, mycolor='C2', plotStats=False, useCRLB=True):
    
    nd.logger.error('The function DiameterPDF has been replaced by PlotDiameterPDF.')
    nd.logger.warning('No PDF plotted.')
    
#     """ calculate and plot the diameter probability density function of 
#     a particle ensemble as the sum of individual PDFs
#     NB: each trajectory is considered individually, the tracklength determines
#         the PDF widths
    
#     assumption: 
#         relative error = std/mean either given by CRLB (if useCRLB==True) or by 
#         sqrt( 2*N_tmax/(3*N_f - N_tmax) ) 
#         with N_tmax : number of considered lagtimes
#              N_f : number of frames of the trajectory (=tracklength)

#     Parameters
#     ----------
#     fitNdist : boolean, optional
#         If True, a GMM is fitted to the data. The default is False.
#     num_dist_max : integer, optional
#         Maximum number of Gaussians to fit. Default is 2.

#     Returns
#     -------
#     prob_inv_diam
#     diam_grid
#     ax
#     """
#     # import NanoObjectDetection as nd
#     settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
#     DiameterPDF_Show = settings["Plot"]['DiameterPDF_Show']
#     DiameterPDF_Save = settings["Plot"]['DiameterPDF_Save']
    
    
#     # calculate mean and std of the ensemble (inverse!) and the grid for plotting
#     # THE INVERSE DIAMETER IS USED BECAUSE IT HAS A GAUSSIAN DISTRIBUTED ERROR; 
#     # WHILE THE DIAMETER ITSELF DOES NOT HAVE SUCH AN EASY PROBABILTY DENSITY FUNCTION (PDF)
#     # CALCULATE IT IN THE INVERSE REGIME FIRST; AND CALCULATE BACK TO DIAMETER SPACE
#     diam_inv_mean, diam_inv_std = nd.statistics.StatisticOneParticle(sizes_df_lin)
    
#     diam_grid = np.linspace(0.1,1000,10000) # prepare grid for plotting (equidistant)
#     # diam_grid = np.linspace(1,1000,20000) # prepare grid for plotting (equidistant)
#     diam_grid_stepsize = diam_grid[1] - diam_grid[0]
#     # diam_grid_inv = 1/diam_grid
    
#     # array that save the PDF
#     prob_diam = np.zeros_like(diam_grid)
    
#     # get mean and std of each inverse diameter
#     inv_diam, inv_diam_std = nd.statistics.InvDiameter(sizes_df_lin, settings, useCRLB) # 1/um
#     inv_diam = 0.001* inv_diam # 1/nm
#     inv_diam_std = 0.001*inv_diam_std # 1/nm
    
    
#     for index, (loop_mean, loop_std) in enumerate(zip(inv_diam,inv_diam_std)):
#         # law of inverse fcts
#         # Y = 1/X --> PDF_Y(Y)=1/(Y^2) * PDF_X(1/Y)
#         my_pdf = scipy.stats.norm(loop_mean,loop_std).pdf(1/diam_grid)
#         my_pdf = 1/(diam_grid**2) * my_pdf
        
#         # normalization to 1
#         my_pdf = my_pdf / np.sum(my_pdf) 
        
#         # sum it all up
#         prob_diam = prob_diam + my_pdf    
    
#     # normalize the summed pdf to integral=1
#     prob_diam = prob_diam / (np.sum(prob_diam) * diam_grid_stepsize)
    
#     # get statistical quantities
#     diam_median = np.median(nd.statistics.GetCI_Interval(prob_diam, diam_grid, 0))
#     # diam_mean = np.average(diam_grid, weights = prob_diam) # MN: this seems only justified on equidistant grids
#     diam_mean = sum(prob_diam*diam_grid)*diam_grid_stepsize
#     lim_68CI = nd.statistics.GetCI_Interval(prob_diam, diam_grid, 0.68) 
#     lim_95CI = nd.statistics.GetCI_Interval(prob_diam, diam_grid, 0.95)
#     num_trajectories = len(sizes_df_lin) 
        
    
#     # set the x limits in the plot
#     PDF_min_max_auto = settings["Plot"]["DiameterPDF_min_max_auto"]
#     if PDF_min_max_auto == 1:
#         PDF_min, PDF_max = nd.statistics.GetCI_Interval(prob_diam, diam_grid, 0.999)
#         # PDF_min, PDF_max = nd.statistics.GetCI_Interval(prob_inv_diam, diam_grid, 0.999)
#         PDF_min = int(np.floor(PDF_min))
#         PDF_max = int(np.ceil(PDF_max))
#     else:
#         PDF_min = settings["Plot"]['PDF_min'] 
#         PDF_max = settings["Plot"]['PDF_max'] 
        
        
#     # info box for the plot
#     if statsInfo==True:    
#         title = str("{:.0f} trajectories; mean: {:.2f}, median: {:.1f}, \n CI68: [{:.1f} : {:.1f}],  CI95: [{:.1f} : {:.1f}] nm".format(num_trajectories, diam_mean, diam_median, lim_68CI[0], lim_68CI[1],lim_95CI[0], lim_95CI[1]))
#     else:
#         title = str("Trajectories: {:.0f}".format(num_trajectories))
        
#     xlabel = "Diameter [nm]"
#     ylabel = "Probability [a.u.]"
#     x_lim = [PDF_min, PDF_max]
#     # y_lim = [0, 1.1*np.max(prob_inv_diam)]
#     y_lim = [0, 1.1*np.max(prob_diam)]
# #    sns.reset_orig()

#     # plot it!
#     ax = Plot2DPlot(diam_grid, prob_diam, title, xlabel, ylabel, mylinestyle = "-",  
#                     mymarker = "", x_lim = x_lim, y_lim = y_lim, y_ticks = [0,0.01], 
#                     semilogx = False, FillArea = fillplot, Color = mycolor)
    
#     if plotStats: # plot lines for mean, median and CI68-interval
#         ax.vlines([diam_mean, diam_median], 0, 1,
#                   transform=ax.get_xaxis_transform(), 
#                   colors=mycolor, alpha=0.8)
#         ax.vlines([lim_68CI[0], lim_68CI[1]], 0, 1, 
#                   transform=ax.get_xaxis_transform(), 
#                   colors=mycolor, alpha=0.6)
    
    
#     mean_diameter = np.round(np.mean(nd.statistics.GetCI_Interval(prob_diam, diam_grid, 0.001)),1)
#     CI_interval = np.round(nd.statistics.GetCI_Interval(prob_diam, diam_grid, 0.68),1)
    
#     nd.logger.info("mean diameter: %s", mean_diameter)
#     nd.logger.info("68CI interval: %s", CI_interval)

    
#     if settings["Plot"]["Histogramm_Fit_1_Particle"] == 1:
#         max_PDF = prob_diam.max()
        
#         if fitNdist == False: 
#             # consider only one contributing particle size   
#             nd.visualize.PlotReciprGauss1Size(ax, diam_grid, max_PDF, sizes_df_lin)
            
#         else: 
#             # consider multiple contributing particles sizes
            
#             # MN: THE FOLLOWING IS A BIT EXPERIMENTAL... PROBABLY NEEDS REVIEW
#             # reduce the grid from 10000 to 1000 values
#             diamsR = diam_grid[::10]
#             probsR = prob_diam[::10] * 6E5 # scale up to be able to use the values as integers
#             probsN = probsR.round()
#             # create (large!) array with repeated diameter values, cf. weighting scheme
#             sizesPDF = diamsR.repeat(np.array(probsN,dtype='int'))
            
#             diam_means, diam_stds, weights = \
#                 nd.visualize.PlotReciprGaussNSizes(ax, diamsR, max_PDF, sizesPDF,
#                                                    showICplot=showICplot, useAIC=useAIC,
#                                                    num_dist_max=num_dist_max)
            
#             if showInfobox==True:
#                 nd.visualize.PlotInfoboxMN(ax, diam_means, diam_stds, weights)


#     if DiameterPDF_Save == True:
#         save_folder_name = settings["Plot"]["SaveFolder"]
        
#         settings = nd.visualize.export(save_folder_name, "Diameter_Probability", settings,
#                                        data = sizes_df_lin, ShowPlot = DiameterPDF_Show)
        
#         data = np.transpose(np.asarray([diam_grid, prob_diam]))
#         nd.visualize.save_plot_points(data, save_folder_name, 'Diameter_Probability_Data')
        
#     nd.handle_data.WriteJson(ParameterJsonFile, settings)
    
#     return prob_diam, diam_grid, ax