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
sns.reset_orig()
import math # offering some maths functions
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt # Libraries for plotting
from matplotlib import animation # Allows to create animated plots and videos from them
import json
import sys
import datetime
from pdb import set_trace as bp #debugger
import scipy
import os.path

import time
from matplotlib.animation import FuncAnimation, PillowWriter


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

def Plot1DPlot(plot_np,title = None, xlabel = None, ylabel = None):
    from NanoObjectDetection.PlotProperties import axis_font, title_font
    
    plt.figure()
    plt.plot(plot_np)
    plt.title(title, **title_font)
    plt.xlabel(xlabel, **axis_font)
    plt.ylabel(ylabel, **axis_font)


def Plot2DPlot(x_np,y_np,title = None, xlabel = None, ylabel = None, myalpha = 1, mymarker = 'x', mylinestyle  = ':',
               x_lim = None):
    from NanoObjectDetection.PlotProperties import axis_font, title_font
    sns.reset_orig()
    
    plt.figure()
    plt.plot(x_np,y_np, marker = mymarker, linestyle  = mylinestyle, alpha = myalpha)
    plt.title(title, **title_font)
    plt.xlabel(xlabel, **axis_font)
    plt.ylabel(ylabel, **axis_font)

    if x_lim != None:
        plt.xlim(x_lim)

def Plot2DScatter(x_np,y_np,title = None, xlabel = None, ylabel = None, myalpha = 1,
                  x_min_max = None, y_min_max = None):
    from NanoObjectDetection.PlotProperties import axis_font, title_font

    plt.figure()
    plt.scatter(x_np,y_np, alpha = myalpha, linewidths  = 0)
    plt.title(title, **title_font)
    plt.xlabel(xlabel, **axis_font)
    plt.ylabel(ylabel, **axis_font)

    SetXYLim(x_min_max, y_min_max)



def Plot2DImage(array_np,title = None, xlabel = None, ylabel = None):
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


        show_pdf, save_pdf = settings["Plot"]["DiameterPDF_Show"], settings["Plot"]["DiameterPDF_Save"]
        if show_pdf or save_pdf:
            DiameterPDF(ParameterJsonFile, sizes_df_lin)

        
        show_diam_traj, save_diam_traj = settings["Plot"]["DiamOverTraj_Show"], settings["Plot"]["DiamOverTraj_Save"]
        if show_diam_traj or save_diam_traj:
            DiamerterOverTrajLenght(ParameterJsonFile, sizes_df_lin, show_diam_traj, save_diam_traj)


        show_hist_time, save_hist_time = settings["Plot"]["Histogramm_time_Show"], settings["Plot"]["Histogramm_time_Save"]
        if show_hist_time or save_hist_time:
            DiameterHistogrammTime(ParameterJsonFile, sizes_df_lin, show_hist_time, save_hist_time)


        show_correl, save_correl = settings["Plot"]["Correlation_Show"], settings["Plot"]["Correlation_Save"]
        if show_correl or save_correl:
            Correlation(ParameterJsonFile, sizes_df_lin, show_correl, save_correl)
            
            
        show_pearson, save_pearson = settings["Plot"]["Pearson_Show"], settings["Plot"]["Pearson_Save"]
        if show_pearson or save_pearson:
            Pearson(ParameterJsonFile, sizes_df_lin, show_pearson, save_pearson)




def Correlation(ParameterJsonFile, sizes_df_lin, show_plot = None, save_plot = None):
    """
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
    

def corrfunc(x, y, **kws):
    pearson, _ = scipy.stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("p = {:.2f}".format(pearson),
                xy=(.1, .9), xycoords=ax.transAxes)



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
    import NanoObjectDetection as nd

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
    plot_traj_length = sizes_df_lin["traj length"]
    
    x_min_max = nd.handle_data.Get_min_max_round(plot_traj_length, 2)
    x_min_max[0] = 0
    
    y_min_max = nd.handle_data.Get_min_max_round(plot_diameter,1)
    
    Plot2DScatter(plot_traj_length, plot_diameter, title = my_title, xlabel = my_xlabel, ylabel = my_ylabel,
                  myalpha = 0.6, x_min_max = x_min_max, y_min_max = y_min_max)
 
    if save_plot == True:
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "DiameterOverTrajLength",
                                       settings, data = sizes_df_lin)





def DiameterHistogrammTime(ParameterJsonFile, sizes_df_lin, histogramm_min = None, histogramm_max = None, 
                           Histogramm_min_max_auto = None, binning = None):
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
    ylabel = 'Absolute occurance'
    title = 'Amount of particles analyzed =%r' % len(sizes_df_lin)

    nd.visualize.PlotDiameter2DHistogramm(sizes_df_lin, binning, histogramm_min, histogramm_max, title, xlabel, ylabel)
  
    if Histogramm_Save == True:
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "Diameter Histogramm", settings,
                                       data = sizes_df_lin)
        
        
    nd.handle_data.WriteJson(ParameterJsonFile, settings) 


def PlotDiameter2DHistogramm(sizes_df_lin, binning, min_size = 0, cutoff_size = 10000, title = '', xlabel = '', ylabel = ''):
    from NanoObjectDetection.PlotProperties import axis_font, title_font
    
    import seaborn as sns
    sns.set(style="darkgrid")
    
    tips = sns.load_dataset("tips")
    g = sns.jointplot("total_bill", "tip", data=tips, kind="reg",
                      xlim=(0, 60), ylim=(0, 12), color="m", height=7)
    
    
    
    
    
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
#    #   plt.ylabel(r'absolute occurance')
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
    ylabel = 'Absolute occurance'
    title = 'Amount of particles analyzed =%r' % len(sizes_df_lin)

    values_hist = nd.visualize.PlotDiameterHistogramm(sizes_df_lin, binning, histogramm_min, histogramm_max, title, xlabel, ylabel)
    
    if settings["Plot"]["Histogramm_Fit_1_Particle"] == 1:
        max_hist = values_hist.max()
        # here comes the fit
    
        
        diam_inv_mean, diam_inv_std = nd.CalcDiameter.StatisticOneParticle(sizes_df_lin)
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
    
    xlabel = 'Diameter [nm]'
    ylabel = 'Probability'
    title = 'Amount of particles analyzed =%r' % len(sizes_df_lin)

    
    diam_inv_mean, diam_inv_std = nd.CalcDiameter.StatisticOneParticle(sizes_df_lin)
    diam_grid = np.linspace(histogramm_min,histogramm_max,10000)
    diam_grid_inv = 1/diam_grid
    
    
    inv_diam,inv_diam_std = nd.CalcDiameter.InvDiameter(sizes_df_lin)
    
    prob_inv_diam = np.zeros_like(diam_grid_inv)
    for index, (loop_mean, loop_std) in enumerate(zip(inv_diam,inv_diam_std)):
        my_pdf = scipy.stats.norm(loop_mean,loop_std).pdf(diam_grid_inv)

        my_pdf = my_pdf / np.sum(my_pdf)
        
        prob_inv_diam = prob_inv_diam + my_pdf
     
      
    title = settings["Plot"]["Title"]
    xlabel = "Diameter [nm]"
    ylabel = "Probability [a.u.]"
    x_lim = [histogramm_min, histogramm_max]
    sns.reset_orig()
    Plot2DPlot(diam_grid, prob_inv_diam, title, xlabel, ylabel, mylinestyle = "-",  mymarker = "", x_lim = x_lim)

    
    
#    prob_diam_inv = scipy.stats.norm(diam_inv_mean,diam_inv_std).pdf(diam_grid_inv)
#    prob_diam_inv = prob_diam_inv / prob_diam_inv.max() * max_hist
#    prob_diam = 1 / prob_diam_inv 
    

#    plt.plot(diam_grid,prob_diam_inv)

    
    if DiameterPDF_Save == True:
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "Diameter Probability", settings,
                                       data = sizes_df_lin)
        
        
    nd.handle_data.WriteJson(ParameterJsonFile, settings)



def GetMeanStdMedian(data):
    my_mean   = np.mean(data)
    my_std    = np.std(data)
    my_median = np.median(data)

    return my_mean, my_std, my_median    
    

def PlotDiameterHistogramm(sizes_df_lin, binning, min_size = 0, cutoff_size = 10000, title = '', xlabel = '', ylabel = ''):
    from NanoObjectDetection.PlotProperties import axis_font, title_font
    import NanoObjectDetection as nd
#    plt.figure()
    fig, ax = plt.subplots()
    diameters = sizes_df_lin.diameter
    show_diameters = diameters[(diameters >= min_size) & (diameters <= cutoff_size)]
    values_hist, positions_hist = np.histogram(sizes_df_lin["diameter"], bins = binning)
    # histogram of sizes, only taking into account 
    sns.distplot(show_diameters, bins=binning, rug=True, kde=False) 
    #those that are below threshold size as defined in the initial parameters
    plt.rc('text', usetex=True)
    plt.title(title, **title_font)
    #   plt.ylabel(r'absolute occurance')
    plt.ylabel(ylabel, **axis_font)
    plt.xlabel(xlabel, **axis_font)
    plt.grid(True)
 
    ax.set_xlim(min_size, cutoff_size)

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
    ani.save(entire_path_image, writer=PillowWriter(fps = fps))
    
    print('Animation saved at: {}'.format(my_dir_name))
    
    plt.show()
    


    
    
    
    

def AnimateDiameterAndRawData(rawframes_rot, sizes_df_lin_rolling):
    
    
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


