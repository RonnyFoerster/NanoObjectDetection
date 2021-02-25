# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 13:45:19 2020

@author: foersterronny
"""
import numpy as np # library for array-manipulation
import matplotlib.pyplot as plt # libraries for plotting
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from pdb import set_trace as bp #debugger
from joblib import Parallel, delayed
import multiprocessing

## In[]
def RollingMedianFilter_main(image, window = 11):
    num_cores = multiprocessing.cpu_count()
    
    num_lines = image.shape[1]
    
    inputs = range(num_lines)

    print("start median background filter - parallel")
    background_list = Parallel(n_jobs=num_cores)(delayed(RollingMedianFilter)(image[:,loop_line,:].copy(), window) for loop_line in inputs)
    print("finished median background filter - parallel")

    background = np.asarray(background_list)
    
    background = np.swapaxes(background, 0, 1)
    
    return background

#here is the file for playing with new functions, debugging and so on
def RollingMedianFilter(image, window = 11):
    print("start median background filter")
    
    # check if window is odd
    if (window%2) == 0:
        raise ValueError("window size has to be odd")
    
    # window in one direction
    a = int((window-1)/2)
    
    first_frame = a
    last_frame = image.shape[0] - a
    
    valid_frames = np.arange(first_frame, last_frame)
    
    background = np.zeros_like(image, dtype = "double")
    
    #frame that does not exceed the winodw limit
    for loop_frame in valid_frames:
        print("loop_frame : ", loop_frame)
        # print(loop_frame)
        # background[loop_frame, : , :] = np.median(image[loop_frame-a:loop_frame+a, :, :], axis = 0)
        
        image_loop = image[loop_frame-a:loop_frame+a, :]

        min_percentile = int(0.4*image_loop.shape[0])
        max_percentile = int(0.6*image_loop.shape[0])
            
        background[loop_frame, :] = np.mean(np.sort(image_loop,axis = 0)[min_percentile:max_percentile,:],axis = 0)
        
        # background[loop_frame, :] = np.median(image[loop_frame-a:loop_frame+a, :], axis = 0)
        
    # handle frames at the beginning
    # background[:first_frame, : , :] = background[first_frame, : , :]
    background[:first_frame, :] = background[first_frame, :]
    
    # handle frames at the end
    # background[last_frame:, : , :] = background[last_frame, : , :]
    background[last_frame:, :] = background[last_frame, :]
    
    
    return background



def NewBGFilter(img):
    #checks the background if the histogramm is normal distributed by kolmogorow
    import scipy
    
    img = img / 16
    
    size_y = img.shape[1]
    size_x = img.shape[2]
    
    img_test = np.zeros([size_y,size_x])
    img_mu  = np.zeros([size_y,size_x])
    img_std = np.zeros([size_y,size_x])
    
    for loop_y in range(0, size_y):
        print(loop_y)
        for loop_x in range(0, size_x):
            test = img[: ,loop_y, loop_x]
            mu, std = scipy.stats.norm.fit(test)
            [D_Kolmogorow, _] = scipy.stats.kstest(test, 'norm', args=(mu, std))
    
            img_test[loop_y, loop_x] = D_Kolmogorow
            img_mu[loop_y, loop_x] = mu
            img_std[loop_y, loop_x] = std
    
    return img_test, img_mu, img_std



def TestSlider():
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    
    x = list(range(0,11))
    y = [10] * 11
    
    fig, ax = plt.subplots()
    plt.subplots_adjust(left = 0.1, bottom = 0.35)
    p, = plt.plot(x,y, linewidth = 2, color = 'blue')
    
    plt.axis([0, 10, 0, 100])
    
    axSlider1 = plt.axes([0.1, 0.2, 0.8, 0.05])
    
    slder1 = Slider()
    
    
    plt.show()



def TryTextBox():
    from matplotlib.widgets import TextBox
    
    fig = plt.figure(figsize = [5, 5], constrained_layout=True)

    gs = GridSpec(2, 1, figure=fig)
    ax_gs = fig.add_subplot(gs[0, 0])
    
    textbox_raw_min = TextBox(ax_gs , "x min: ", initial = "10")    

    ax_plot = fig.add_subplot(gs[1, 0])
    t = np.arange(-2.0, 2.0, 0.001)
    s = t ** 2
    initial_text = "t ** 2"
    l, = ax_plot.plot(t, s, lw=2)


    def PrintASDF(text):
        ax_plot.set_xlim([0.5])
    
    textbox_raw_min.on_submit(PrintASDF)

    return textbox_raw_min


def TryTextBox2():
    #https://matplotlib.org/3.1.1/gallery/widgets/textbox.html
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import TextBox

    fig = plt.figure(figsize = [5, 5], constrained_layout=True)
    gs = GridSpec(2, 1, figure=fig)
    ax_plot = fig.add_subplot(gs[1, 0])
    
    plt.subplots_adjust(bottom=0.2)
    t = np.arange(-2.0, 2.0, 0.001)
    s = t ** 2
    initial_text = "t ** 2"
    l, = plt.plot(t, s, lw=2)
    
    
    def submit(text):
        t = np.arange(-2.0, 2.0, 0.001)
        print(text)
        ydata = eval(text)
        l.set_ydata(ydata)
        ax_plot.set_ylim(np.min(ydata), np.max(ydata))
        plt.draw()
    
    ax_gs = fig.add_subplot(gs[0, 0])
    textbox_raw_min = TextBox(ax_gs , "x min: ", initial = "10")    
    textbox_raw_min.on_submit(submit)
    
#    axbox = plt.axes([0.1, 0.05, 0.8, 0.075])
#    text_box = TextBox(axbox, 'Evaluate', initial=initial_text)
#    text_box.on_submit(submit)
    
    plt.show()

    return textbox_raw_min



""" 
###############################################################################
Mona's section 
###############################################################################
""" 
    
def DiameterOverTrajLengthColored(ParameterJsonFile, sizes_df_lin, 
                                  color_by='mass', use_log=False,
                                  save_plot = True):
    """ plot (and save) calculated particle diameters vs. the number of frames
    where the individual particle is visible (in standardized format) 
    and color it by a property of choice
    """
        
    import NanoObjectDetection as nd

    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    Histogramm_min_max_auto = settings["Plot"]["Histogramm_min_max_auto"]
    
    if Histogramm_min_max_auto == 1:
        histogramm_min = np.round(np.min(sizes_df_lin.diameter) - 5, -1)
        histogramm_max = np.round(np.max(sizes_df_lin.diameter) + 5, -1)
    else:
        histogramm_min = None
        histogramm_max = None 
    
    histogramm_min, settings = nd.handle_data.SpecificValueOrSettings(histogramm_min, settings, "Plot", "Histogramm_min")
    histogramm_max, settings = nd.handle_data.SpecificValueOrSettings(histogramm_max, settings, "Plot", "Histogramm_max")
    
    my_title = "Particle size over tracking time (colored by {}".format(color_by)
    if use_log==True:
        my_title = my_title + " in log scale)"
    else:
        my_title = my_title + ")"
    my_ylabel = "Diameter [nm]"
    my_xlabel = "Trajectory length [frames]"
    
    plot_diameter = sizes_df_lin["diameter"]
    plot_traj_length = sizes_df_lin["traj length"]
    
    if use_log==True:
        plot_color = np.log10(sizes_df_lin[color_by])
    else:
        plot_color = sizes_df_lin[color_by]
    
    x_min_max = nd.handle_data.Get_min_max_round(plot_traj_length,2)
    x_min_max[0] = 0
    y_min_max = nd.handle_data.Get_min_max_round(plot_diameter,1)
    
    
    plt.figure()
    plt.scatter(plot_traj_length, plot_diameter, c=plot_color, cmap='viridis')
    plt.title(my_title)
    plt.xlabel(my_xlabel)
    plt.ylabel(my_ylabel)
    plt.xlim([x_min_max[0], x_min_max[1]])
    plt.ylim([y_min_max[0], y_min_max[1]])
    plt.colorbar()

 
    if save_plot == True:
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "DiameterOverTrajLength",
                                       settings, data = sizes_df_lin)



def DiameterPDF_transparent(ParameterJsonFile, sizes_df_lin, histogramm_min = None, 
                            histogramm_max = None, Histogramm_min_max_auto = 0, 
                            binning = None):
    """ calculate and plot the diameter probability density function of a
    particle ensemble as the sum of individual PDFs - and plot the individual 
    PDFs as well!
    
    NB: each trajectory is considered individually, the tracklength determines
        the PDF widths
    
    assumption: 
        relative error = std/mean = sqrt( 2*N_tmax/(3*N_f - N_tmax) )
        with N_tmax : number of considered lagtimes
             N_f : number of frames of the trajectory (=tracklength)

    Parameters
    ----------
    ParameterJsonFile : TYPE
        DESCRIPTION.
    sizes_df_lin : TYPE
        DESCRIPTION.
    histogramm_min : TYPE, optional
        DESCRIPTION. The default is None.
    histogramm_max : TYPE, optional
        DESCRIPTION. The default is None.
    Histogramm_min_max_auto : TYPE, optional
        DESCRIPTION. The default is None.
    binning : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    prob_inv_diam
    diam_grid
    ax
    """
    import NanoObjectDetection as nd
    from scipy.stats import norm
    # sns.set(style="dark")
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    DiameterPDF_Show = settings["Plot"]['DiameterPDF_Show']
    DiameterPDF_Save = settings["Plot"]['DiameterPDF_Save']
    PDF_min = settings["Plot"]['PDF_min']
    PDF_max = settings["Plot"]['PDF_max']
    
    # calculate mean and std of the ensemble (inverse!) and the grid for plotting
    diam_inv_mean, diam_inv_std = nd.CalcDiameter.StatisticOneParticle(sizes_df_lin)
    diam_grid = np.linspace(PDF_min,PDF_max,10000) # prepare grid for plotting
    diam_grid_inv = 1/diam_grid
    
    # calculate inverse diameters and estimates of their std
    inv_diam,inv_diam_std = nd.CalcDiameter.InvDiameter(sizes_df_lin, settings)
    num_trajectories = len(sizes_df_lin) 
    
    fig,ax = plt.subplots()
    
    prob_inv_diam = np.zeros_like(diam_grid_inv)
    # loop over all individual diameters, calculate their PDFs and add them up
    for index, (loop_mean, loop_std) in enumerate(zip(inv_diam,inv_diam_std)):
        my_pdf = norm(loop_mean,loop_std).pdf(diam_grid_inv)
        my_pdf = my_pdf / np.sum(my_pdf) # normalization
        
        ax.fill_between(diam_grid,my_pdf,alpha=0.3)
        
        prob_inv_diam = prob_inv_diam + my_pdf    

    # normalize
    prob_inv_diam = prob_inv_diam / np.sum(prob_inv_diam)
    
    # instantiate a second axes that shares the same x-axis
    ax2 = ax.twinx()  
    mycol = 'tab:blue'
    ax2.plot(diam_grid,prob_inv_diam, color=mycol)
    
    # configure plot settings
    ax2.set_ylabel('Ensemble probability [a.u.]', color=mycol)
    ax2.tick_params(axis='y', labelcolor=mycol)
    
    if Histogramm_min_max_auto == 1:
        PDF_min, PDF_max = nd.visualize.GetCI_Interval(prob_inv_diam, diam_grid, 0.999)
        #histogramm_min = 0
    else:
        PDF_min, settings = nd.handle_data.SpecificValueOrSettings(PDF_min, settings, 
                                                                   "Plot", "PDF_min")
        PDF_max, settings = nd.handle_data.SpecificValueOrSettings(PDF_max, settings, 
                                                                   "Plot", "PDF_max")
    ax.set(title="Trajectories: {:3.0f}".format(num_trajectories),
           xlim = [PDF_min, PDF_max], xlabel = "Diameter [nm]", 
           ylabel = "Individual probabilities [a.u.]",
           ylim=[0, 11.5*np.max(prob_inv_diam)])
    ax2.set(ylim = [0, 1.05*np.max(prob_inv_diam)])
    
    # use scientific tick notation
    from matplotlib import ticker
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1)) 
    ax.yaxis.set_major_formatter(formatter) 
    ax2.yaxis.set_major_formatter(formatter) 
        
    fig.tight_layout()
    
    
    if DiameterPDF_Save == True:
        save_folder_name = settings["Plot"]["SaveFolder"]
        
        settings = nd.visualize.export(save_folder_name, "Diameter_Probability", settings,
                                       data = sizes_df_lin, ShowPlot = DiameterPDF_Show)
        
        data = np.transpose(np.asarray([diam_grid, prob_inv_diam]))
        nd.visualize.save_plot_points(data, save_folder_name, 'Diameter_Probability_Data')
        
    nd.handle_data.WriteJson(ParameterJsonFile, settings)
    
    return prob_inv_diam, diam_grid, ax



def calcInversePDF(grid, sizes_df_lin, settings, useCRLB):
    """ compute the probability density function (PDF) of the sizes ensemble 
    on a given grid in the inverse sizes space
    
    NB: Due to the Gaussian/normal distributed uncertainty in the diffusion space,
        the PDF is always first calculated in the inverse sizes space and then
        back-converted.
    """
    import NanoObjectDetection as nd
    from scipy.stats import norm
    
    grid_steps = abs(grid[:-1]-grid[1:])
    if grid_steps[0] > grid_steps[-1]: # duplicate smallest entry (either the first or the last)
        grid_steps = np.append(grid_steps, grid_steps[-1]) 
    else:
        grid_steps = np.append(grid_steps, grid_steps[0])
    
    # # numerical grid (equidistant)
    # inv_grid = np.linspace(1,1000,999*100+1) # 1/um
    # inv_grid_stepsize = inv_grid[1] - inv_grid[0] 
    
    # initialize array to save the PDF
    PDF_inv = np.zeros_like(grid)
    
    # get mean and std of each inverse diameter
    inv_diams, inv_diam_stds = nd.statistics.InvDiameter(sizes_df_lin, settings, useCRLB) # 1/um
    
    # calculate individual PDFs for each particle and sum them up
    for inv_d, inv_std in zip(inv_diams,inv_diam_stds):
        PDF_inv = PDF_inv + norm(inv_d,inv_std).pdf(grid)    
    
    # normalize to integral=1
    PDF_inv = PDF_inv/sum(PDF_inv * grid_steps) 
    
    return PDF_inv, grid_steps # 1/um



def convertPDFtoSizesSpace(PDF_inv, inv_grid):
    """ convert PDF on 1/um-grid back into one in sizes space on a nm-grid
    """
    grid = 1000/inv_grid # nm
    grid_steps = abs(grid[:-1]-grid[1:])
    if grid_steps[0] > grid_steps[-1]: # duplicate smallest entry (either the first or the last)
        grid_steps = np.append(grid_steps, grid_steps[-1]) 
    else:
        grid_steps = np.append(grid_steps, grid_steps[0])
    
    # law of inverse fcts: Y = 1/X --> PDF_Y(Y) = 1/(Y^2) * PDF_X(1/Y)
    PDF = 1/(grid**2) * PDF_inv
    
    # normalize to integral=1 again
    PDF = PDF/sum(PDF*grid_steps)
    
    return PDF, grid, grid_steps



def fitGaussian(PDF, grid, grid_steps, mean_start):
    """ calculate least-squares fit of a Gaussian function to a given PDF
    
    Assumption: PDF integrated over grid = 1
    """
    from scipy.optimize import curve_fit
    
    # define Gaussian fct (integral normalized to 1)
    def myGauss(d,mean,std):
        return 1/((2*np.pi)**0.5*std)*np.exp(-(d-mean)**2/(2.*std**2))
    
    popt, pcov = curve_fit(myGauss, grid, PDF, p0=[mean_start,1])
    
    fit = myGauss(grid, *popt) # values of the fit
    resid = PDF - fit # residuals
    residIntegral = sum(abs(resid)*grid_steps) # absolute resids integrated
    
    SQR = (resid**2).sum() # sum of squared residuals (SQR) = N*MSE (mean square error)
    # total sum of squares ("Gesamtstreuung") SQT
    SQT = sum((PDF - PDF.mean())**2) # = N*variance
    
    # coefficient of determination ("Bestimmheitsmaß") R^2 
    # (NB: def. here is the same as for the coefficient of efficiency in Legates1999)
    R2 = 1 - SQR/SQT # = 1 - MSE/variance
    
    mean, std = popt
    
    return fit, mean, std, residIntegral, R2



def fitGaussianMixture(PDF, grid, grid_steps):
    """ calculate least-squares fit of a superpositions of 2 Gaussian functions 
    to a given PDF
    """
    from scipy.optimize import curve_fit
    
    # define mixture of 2 Gaussian fcts (integral normalized to 1)
    def myGaussMix(d,m1,m2,s1,s2,w):
        gauss1 = w/((2*np.pi)**0.5*s1)*np.exp(-(d-m1)**2/(2.*s1**2))
        gauss2 = (1-w)/((2*np.pi)**0.5*s2)*np.exp(-(d-m2)**2/(2.*s2**2))
        return gauss1 + gauss2
    
    popt, pcov = curve_fit(myGaussMix, grid, PDF)#, p0=[mean_start,1])
    
    fit = myGaussMix(grid, *popt) # values of the fit
    resid = PDF - fit # residuals
    residIntegral = sum(abs(resid)*grid_steps) # absolute resids integrated
    
    SQR = (resid**2).sum() # sum of squared residuals (SQR) = N*MSE (mean square error)
    # total sum of squares ("Gesamtstreuung") SQT
    SQT = sum((PDF - PDF.mean())**2) # = N*variance
    
    # coefficient of determination ("Bestimmheitsmaß") R^2 
    # (NB: def. here is the same as for the coefficient of efficiency in Legates1999)
    R2 = 1 - SQR/SQT # = 1 - MSE/variance
    
    # mean, std = popt
    
    return fit, residIntegral, R2, popt



def PlotInfoboxFit(ax,residIntegral,R2,fit_mean,fit_median,fit_68CI):
    """ add textbox to a plot containing fit results,
    assuming one single contributing particle size
    """
    if fit_68CI[0] > fit_68CI[1]:
        fit_68CI = fit_68CI[1],fit_68CI[0] # change order
    
    textstr = '\n'.join([r'FIT:',
    r'$\mathrm{median}=  %.2f$ nm' % (fit_median),
    r'$\mu = %.2f$ nm' % (fit_mean),
    r'$1 \sigma = [%.2f, %.2f]$ nm' %(fit_68CI[0],fit_68CI[1]), 
    r'$R^2 = {:.3f},$ resid.int. ${:.1f}$%'.format(R2, 100*residIntegral), ])
    
    props = dict(boxstyle='round', facecolor='honeydew', alpha=0.7)
    
    ax.text(0.51, 0.95, textstr, transform=ax.transAxes, 
            fontsize=14, verticalalignment='top', bbox=props)



def PlotDiameterPDF(ParameterJsonFile, sizes_df_lin, plotInSizesSpace=True,
                    fitInInvSpace=True, invGridEquidist=True, PDF_min_max_auto = None, 
                    fitNdist=False, num_dist_max=2, statsInfoTitle=True, showFitInfobox=True, 
                    fillplot=True, mycolor='C3', plotVlines=False, useCRLB=True):
    """ calculate and plot the diameter probability density function of 
    a particle ensemble as the sum of individual PDFs
    NB: each trajectory is considered individually, the tracklength determines
        the PDF widths
        
    HERE WE CALCULATE TRUE FITS

    Parameters
    ----------
    fitNdist : boolean, optional
        If True, a GMM is fitted to the data. The default is False.
    num_dist_max : integer, optional
        Maximum number of Gaussians to fit. Default is 2.

    Returns
    -------
    prob_inv_diam
    diam_grid
    ax
    """
    import NanoObjectDetection as nd
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    DiameterPDF_Show = settings["Plot"]['DiameterPDF_Show']
    DiameterPDF_Save = settings["Plot"]['DiameterPDF_Save']
    
    # define numerical grid
    if invGridEquidist:
        inv_grid = np.linspace(1,1000,999*100+1) # 1/um
        # inv_grid_stepsize = inv_grid[1] - inv_grid[0] 
    else:
        s_grid = np.linspace(0.1,1000,10000) # nm
        inv_grid = 1000/s_grid[::-1] # 1/um; grid in increasing order
    
    # get inverse PDF
    PDF_inv, inv_grid_stepsizes = calcInversePDF(inv_grid, sizes_df_lin, settings, useCRLB)
    
    # get statistical quantities
    diam_mean, _, diam_median, lim_68CI, lim_95CI = \
        nd.statistics.GetMeanStdMedianCIfromPDF(PDF_inv, inv_grid, inv_grid_stepsizes)

    num_trajectories = len(sizes_df_lin) 
    
    if plotInSizesSpace: # otherwise, PDF is plotted on inverse grid
        mycolor = 'C2'
        prefix = 'D'
        unit = 'nm'
        PDF, grid, grid_stepsizes = PDF_inv, inv_grid, inv_grid_stepsizes # save for fitting later
        PDF_inv, inv_grid, inv_grid_stepsizes = convertPDFtoSizesSpace(PDF_inv, inv_grid)
        
        diam_median = 1000/diam_median
        lim_68CI = 1000/lim_68CI[1], 1000/lim_68CI[0]
        lim_95CI = 1000/lim_95CI[1], 1000/lim_95CI[0]
        # med = np.median(nd.statistics.GetCI_Interval(PDF_inv, inv_grid, 0))
        # CI68 = nd.statistics.GetCI_Interval(PDF_inv, inv_grid, 0.68) # same values as above
        diam_mean = sum(PDF_inv*inv_grid*inv_grid_stepsizes) # definition of mean/expected value on continuous data
        # diam_mean = 1000/diam_mean # this is NOT the same value (!)
    else:
        prefix = 'Inv. d'
        unit = '1/$\mu$m'
        
    # set the x limits in the plot
    if PDF_min_max_auto is None:
        PDF_min_max_auto = settings["Plot"]["DiameterPDF_min_max_auto"]
    if PDF_min_max_auto == 1:
        PDF_min, PDF_max = nd.statistics.GetCI_Interval(PDF_inv, inv_grid, 0.999)
        # PDF_min, PDF_max = nd.statistics.GetCI_Interval(prob_inv_diam, diam_grid, 0.999)
        PDF_min = int(np.floor(PDF_min))
        PDF_max = int(np.ceil(PDF_max))
        if plotInSizesSpace:
            PDF_min, PDF_max = PDF_max, PDF_min # switch
    else:
        PDF_min = settings["Plot"]['PDF_min'] 
        PDF_max = settings["Plot"]['PDF_max'] 
        
        
    # define plot labeling and axes limits
    if statsInfoTitle:    
        title = str("{:.0f} trajectories; mean: {:.2f}, median: {:.2f},\nCI68: [{:.2f}, {:.2f}], CI95: [{:.2f}, {:.2f}] ".format(num_trajectories,diam_mean,diam_median,*lim_68CI,*lim_95CI) + unit)
    else:
        title = str("{:.0f} trajectories".format(num_trajectories))
    xlabel = prefix +"iameter ["+unit+"]"
    ylabel = "Probability [a.u.]"
    x_lim = [PDF_min, PDF_max]
    # y_lim = [0, 1.1*np.max(prob_inv_diam)]
    y_lim = [0, 1.1*np.max(PDF_inv)]
#    sns.reset_orig()

    # plot it!
    ax = nd.visualize.Plot2DPlot(inv_grid, PDF_inv, title, xlabel, ylabel, mylinestyle = "-",  
                                 mymarker = "", x_lim = x_lim, y_lim = y_lim, y_ticks = [0,0.01], 
                                 semilogx = False, FillArea = fillplot, Color = mycolor)
    
    if plotVlines: # plot lines for mean, median and CI-intervals
        ax.vlines([diam_mean], 0, [PDF_inv[np.isclose(inv_grid,diam_mean,rtol=0.002)].mean()],
                  # transform=ax.get_xaxis_transform(), 
                  colors='black', alpha=0.3)
        ax.vlines([diam_median], 0, [PDF_inv[np.where(inv_grid==diam_median)]],
                  # transform=ax.get_xaxis_transform(), 
                  colors=mycolor, alpha=0.8)
        ax.vlines([lim_68CI[0], lim_68CI[1]], 0, 
                  [PDF_inv[np.where(inv_grid==lim_68CI[0])], PDF_inv[np.where(inv_grid==lim_68CI[1])]],
                  # transform=ax.get_xaxis_transform(), 
                  colors=mycolor, alpha=0.8, ls='--')
        ax.vlines([lim_95CI[0], lim_95CI[1]], 0, 
                  [PDF_inv[np.where(inv_grid==lim_95CI[0])], PDF_inv[np.where(inv_grid==lim_95CI[1])]],
                  # transform=ax.get_xaxis_transform(), 
                  colors=mycolor, alpha=0.8, ls=':')
    
    nd.logger.info("mean diameter: {:.3f}".format(diam_mean))
    nd.logger.info("median diameter: {:.3f}".format(diam_median))
    nd.logger.info("68CI interval: {:.3f}, {:.3f}".format(*lim_68CI))
    nd.logger.info("95CI interval: {:.3f}, {:.3f}".format(*lim_95CI))
    # nd.logger.info("... in "+unit)

    fit = 'undone'
    if settings["Plot"]["PDF_show_fit"] == 1:
        max_PDF = PDF_inv.max()
        
        if fitNdist == False: # consider only one contributing particle size   
            
            if (fitInInvSpace^plotInSizesSpace):# '^' is the exclusive 'or' operator ('xor')
                # case that plotting and fitting is to happen in the same space
                # => fit Gaussian directly
                
                fit, mean, std, residIntegral, R2 = \
                    nd.sandbox.fitGaussian(PDF_inv, inv_grid, inv_grid_stepsizes, diam_mean)
                _, _, fit_median, fit_68CI, _ = \
                    nd.statistics.GetMeanStdMedianCIfromPDF(fit, inv_grid, inv_grid_stepsizes)
                    
                ax.plot(inv_grid, fit)
                
                if showFitInfobox:
                    PlotInfoboxFit(ax,residIntegral,R2,mean,fit_median,fit_68CI)
                
                nd.logger.info("Fit results: mean = {:.2f}, std = {:.3f}".format(mean,std))
                nd.logger.info("Fit quality: residual integral = {:.4f}, R**2 = {:.4f}".format(residIntegral,R2))
                    
            elif (fitInInvSpace and plotInSizesSpace): # fit Gaussian in the inv. space and back-convert it
                    
                # OLD VERSION
                # # plot reciprocal Gaussian equivalent to the original sizes ensemble
                # nd.visualize.PlotReciprGauss1Size(ax, inv_grid, max_PDF, sizes_df_lin)
                
                fit, mean, std, residIntegral, R2 = \
                    nd.sandbox.fitGaussian(PDF, grid, grid_stepsizes, diam_mean)
                    
                fit, _, _ = convertPDFtoSizesSpace(fit, grid)
                fit_mean, _, fit_median, fit_68CI, _ = \
                    nd.statistics.GetMeanStdMedianCIfromPDF(fit, inv_grid, inv_grid_stepsizes)
                
                ax.plot(inv_grid, fit, 'C3')
                if showFitInfobox:
                    PlotInfoboxFit(ax,residIntegral,R2,fit_mean,fit_median,fit_68CI)
                
                nd.logger.info("Fit results: mean = {:.3f}, median = {:.3f}, CI68 = [{:.3f},{:.3f}]".format(fit_mean,fit_median,fit_68CI[0],fit_68CI[1]))
                nd.logger.info("Fit quality: residual integral = {:.4f}, R**2 = {:.4f}".format(residIntegral,R2))
                    
            else: 
                nd.logger.warning("Fitting a distribution in sizes space and plotting it in the inversed is not implemented.")
                nd.logger.info("No fit was performed.")
                        
                
            
        else: # consider multiple contributing particles sizes
            print('to do ...')
            
            
            # # MN: THE FOLLOWING IS A BIT EXPERIMENTAL... NEEDS REVIEW!!!
            # # reduce the grid from 10000 to 1000 values
            # diamsR = diam_grid[::10]
            # probsR = prob_diam[::10] * 6E5 # scale up to be able to use the values as integers
            # probsN = probsR.round()
            # # create (large!) array with repeated diameter values, cf. weighting scheme
            # sizesPDF = diamsR.repeat(np.array(probsN,dtype='int'))
            
            # diam_means, diam_stds, weights = \
            #     nd.visualize.PlotReciprGaussNSizes(ax, diamsR, max_PDF, sizesPDF,
            #                                        showICplot=showICplot, useAIC=useAIC,
            #                                        num_dist_max=num_dist_max)
            
            # if showFitInfobox:
            #     nd.visualize.PlotInfoboxMN(ax, diam_means, diam_stds, weights)
    
    if DiameterPDF_Save:
        save_folder_name = settings["Plot"]["SaveFolder"]
        
        settings = nd.visualize.export(save_folder_name, "Diameter_Probability", settings,
                                       data = sizes_df_lin, ShowPlot = DiameterPDF_Show)
        
        data = np.transpose(np.asarray([inv_grid, PDF_inv]))
        nd.visualize.save_plot_points(data, save_folder_name, 'Diameter_Probability_Data')
        
    nd.handle_data.WriteJson(ParameterJsonFile, settings)
    
    return PDF_inv, inv_grid, ax, fit






# def AnimateTracksOnRawData(t2_long,rawframes_ROI,settings,max_frm=100,frm_start=0,gamma=1.0):
#     """ animate trajectories on top of raw data
    
#     to do:
#         - compare with functions below...!!

#         - reduce ROI to actually necessary section
#         - cut history of tracks for frames < frm_start
#         - check if trajectory data and rawframes match (?)
#         - make starting frame free to choose
#         - implement gamma transform that does not blow up the file size by a factor of 10!!
#         - choose traj. color by length/mass/size/...?
#         - convert from px to um (optionally)
# """
#     rawframes_gam = rawframes_ROI.copy()
#     if gamma != 1.0:
#         rawframes_gam = 2**16 * (rawframes_gam/2**16)**gamma # for 16bit images
#         rawframes_gam = np.uint16(rawframes_gam)
        
#     fps = settings["Exp"]["fps"]    
#     # get rawframes' dimensions
#     _,y_len_raw,x_len_raw = rawframes_ROI.shape
    
#     # prepare the data
#     trajdata = t2_long.copy() 
#     frm0 = trajdata.frame.min()
#     frmMax = trajdata.frame.max()
#     amnt_f = frmMax - frm0 + 1
#     if amnt_f > max_frm:
#         amnt_f = max_frm
#         # track = track[track.frame <= amnt_f]  
    
#     # get trajectory dimensions for plotting (and add 15px to have it more centered)
#     xMin = int(trajdata.x.min() - 15)
#     if xMin < 0:
#         xMin = 0
#     xMax = int(trajdata.x.max() + 15)
#     if xMax > x_len_raw:
#         xMax = x_len_raw
#     x_len = xMax - xMin
    
#     # make DataFrame a bit slimmer
#     trajdata = trajdata.set_index(['frame'])
#     trajdata = trajdata.drop(columns=['mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep', 'abstime'])
    
#     # prepare plotting
#     fig,ax = plt.subplots(1,figsize=plt.figaspect(y_len_raw/x_len))
#     cm_prism = plt.get_cmap('prism')
    
#     raw_img = ax.imshow(rawframes_gam[frm0,:,xMin:xMax], cmap='gray', aspect="equal",
#                         animated=True)
#     ax.set(xlabel='x [px]', ylabel='y [px]',
#            ylim=[y_len_raw-1,0] ) # invert limits of y-axis!
#     ax_t = ax.twiny()
#     ax_t.set(xlim=[xMin,xMax-1])#,ylim=[70,0])
#     # initialize plots for all particles in the whole video
#     trajplots = [ax_t.plot([],[],'-',color=cm_prism(part_id),
#                            animated=True
#                            )[0] for part_id in trajdata.particle.unique()]
    
#     # heading = ax.annotate("", (5,10), animated=True,
#     #                       bbox=dict(boxstyle="round", fc="white", alpha=0.3, lw=0) )

#     # fig.tight_layout() 
    
#     # def init_tracks():
#     #     raw_img.set_data(rawframes_gam[0,:,:])
        
#     #     for tplot in trajplots:
#     #         tplot.set_data([],[])
        
#     #     # heading.set_text("")
#     #     # ax.set(title='frame: {}, time: {}s'.format(frm_start,time_start))
#     #     # return raw_img, trajplots #,heading
    
#     def update_frame(frmN, rawframes_gam, raw_img, trajdata, trajplots): #
#         frm = frmN + frm0 # if frm0!=0, we have to add it as offset
        
#         raw_img.set_data(rawframes_gam[frm,:,xMin:xMax])
        
#         # update all trajectory plots individually
#         for traj,tplot in zip(trajdata.groupby('particle'),trajplots): 
#             # get DataFrame of an individual particle
#             _,trajdf = traj # omit the particle ID
            
#             # check if current frame adds a trajectory point
#             if frm in trajdf.index: 
#                 tplot.set_data(trajdf.loc[:frm+1].x.values, trajdf.loc[:frm+1].y.values) 
        
#         time = 1000*frm/fps # ms
#         # heading.set_text("frame: {}\ntime: {:.1f} ms".format(frm,time))
#         ax.set_title("frame: {}, time: {:.1f} ms".format(frm,time))
        
#         return raw_img, trajplots#, heading 
    
#     traj_ani = FuncAnimation(fig, update_frame, #init_func=init_tracks, 
#                              frames=amnt_f, 
#                              fargs=(rawframes_gam, raw_img, trajdata, trajplots),
#                              interval=200, blit=False, repeat=False)
#     return traj_ani

# """ why are not all trajectories plotted? """


# # anim2 = AnimateTracksOnRawData(t2_long,rawframes_ROI,settings)
# # anim2.save('Au50_raw+tracks_1000frames.mp4')


def AnimateSingleTrackOnRawData(t_1particle,rawframes_ROI,settings,frm_max=200):#, gamma=0.5):
    """ animate a single trajectory on top of cropped (!) raw data
    
    to do:
        - crop the rawimage correctly (!!)
        - implement gamma transform that does not blow up the file size by a factor of 10!!
        - choose traj. color by length/mass/size/...?
    """
    trajdata = t_1particle.copy() 
    
    # check if only 1 particle is contained
    if trajdata.particle.nunique() != 1:
        print('Trajectories of more than 1 particle are contained. \nThe longest is selected for animation.')
        tlengths = trajdata.groupby("particle").size() # get the tracklengths of all contained particles
        part_ID = tlengths[tlengths==tlengths.max()].index[0] # select the longest track and get its ID
        trajdata = trajdata[trajdata.particle == part_ID] # filter out all other tracks
    
    rawfrms = rawframes_ROI.copy()
    amnt_f,_ = trajdata.shape
    if amnt_f > frm_max:
        amnt_f = frm_max # fix a max. number of frames for the video
    frm_0 = trajdata.frame.min() # get the first frame to be displayed
    fps = settings["Exp"]["fps"]

    # y is probably not necessary to be cropped
    # y_min = trajdata.y.min()
    # y_max = trajdata.y.max()
    # include some more pixels left and right to get a nice view of the whole particle
    x_crop0 = trajdata.x.min() - 20 
    x_crop1 = trajdata.x.max() + 20
    # check if one of the limits exceeds the rawframes dimensions
    if x_crop0 < 0:
        x_crop0 = 0
    if x_crop1 > rawfrms.shape[2]:
        x_crop1 = None # with this as index, also the last value of array will be included
    x_ext = x_crop1 - x_crop0    
    y_ext = rawfrms.shape[1] # image extend in y-direction

    
    """ to be continued... 
    
    attention: 2 different "ax" objects (= artists?!) probably become necessary here
    due to different x and y labeling for cropped rawframes
    
    or can this be handled via xlim, ylim??
    """
    # ax2 = ax1.twiny() 
    
    fig,ax = plt.subplots(1,figsize=plt.figaspect(y_ext/x_ext))
    ax.set_xlim(x_crop0,x_crop1)
    ax.set_ylim(y_ext-1,0) # invert for correct display!
    ax.set(xlabel='x [px]', ylabel='y [px]')
    # fig.tight_layout() 
    cm_viri = plt.get_cmap('viridis')
    
    # prepare the data
    trajdata = trajdata.set_index(['frame'])
    meanmass = trajdata.mass.mean()
    trajdata = trajdata.drop(columns=['mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep', 'abstime'])
    
    # raw_img = ax.imshow(rawfrms[frm_0,:,:], cmap='gray', aspect="equal",
    #                     animated=True)
    # raw_img = ax.imshow(rawfrms[frm_0,:,x_crop0:x_crop1], cmap='gray', aspect="equal",
    #                     animated=True)

    trajplot = ax.plot([],[],'-',color=cm_viri(meanmass), animated=True)[0]
    
    
    def init_track():
        # raw_img.set_data(rawfrms[frm_0,:,:])
        trajplot.set_data([],[])
        
    
    def update_frame(frm): #,trajdata): #,trajplots):
        # raw_img.set_data(rawfrms[frm,:,:])
        
        trajplot.set_data(trajdata.loc[:frm+1].x, trajdata.loc[:frm+1].y)
        
        time = 1000*frm/fps # ms
        # heading.set_text("frame: {}\ntime: {:.1f} ms".format(frm,time))
        ax.set_title("frame: {}, time: {:.1f} ms".format(frm,time))
        
        return trajplot #raw_img, trajplot
    
    traj_ani = FuncAnimation(fig, update_frame, init_func=init_track, 
                             frames=amnt_f, #fargs=(trajdata),#, trajplots),
                             interval=70, blit=False, repeat=False)
    return traj_ani



def AnimateTracksOnRawData(t2_long,rawframes_ROI,settings,frm_start=0):#, gamma=0.5):
    """ animate trajectories on top of raw data
    
    to do:
        - make starting frame free to choose
        - implement gamma transform that does not blow up the file size by a factor of 10!!
        - choose traj. color by length/mass/size/...?
"""
    rawframes_gam = rawframes_ROI.copy()
    # rawframes_gam = 2**16 * (rawframes_gam/2**16)**gamma # for 16bit images
    # rawframes_gam = np.uint16(rawframes_gam)
    
    amnt_f,y_len,x_len = rawframes_ROI.shape
    fps = settings["Exp"]["fps"]
    
    fig,ax = plt.subplots(1,figsize=plt.figaspect(y_len/x_len))
    cm_prism = plt.get_cmap('prism')
    
    # prepare the data
    trajdata = t2_long.copy() 
    trajdata = trajdata.set_index(['frame'])
    trajdata = trajdata.drop(columns=['mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep', 'abstime'])
    
    
    raw_img = ax.imshow(rawframes_gam[0,:,:], cmap='gray', aspect="equal",
                        animated=True)
    # heading = ax.annotate("", (5,10), animated=True,
    #                       bbox=dict(boxstyle="round", fc="white", alpha=0.3, lw=0) )
    # initialize plots for all particles in the whole video
    trajplots = [ax.plot([],[],'-',color=cm_prism(part_id),
                         animated=True
                         )[0] for part_id in trajdata.particle.unique()]
    
    ax.set(xlabel='x [px]', ylabel='y [px]',
           xlim=[0,x_len-1], ylim=[y_len-1,0] ) # invert limits of y-axis!
    # fig.tight_layout() 
    
    def init_tracks():
        raw_img.set_data(rawframes_gam[0,:,:])
        
        for tplot in trajplots:
            tplot.set_data([],[])
        
        # heading.set_text("")
        # ax.set(title='frame: {}, time: {}s'.format(frm_start,time_start))
        # return raw_img, trajplots #,heading
    
    def update_frame(frm): #,trajdata): #,trajplots):
        raw_img.set_data(rawframes_gam[frm,:,:])
        
        # update all trajectory plots individually
        for traj,tplot in zip(trajdata.groupby('particle'),trajplots): 
            # get DataFrame of an individual particle
            _,trajdf = traj # omit the particle ID
            
            # check if current frame adds a trajectory point
            if frm in trajdf.index: 
                tplot.set_data(trajdf.loc[:frm+1].x, trajdf.loc[:frm+1].y)
        
        time = 1000*frm/fps # ms
        # heading.set_text("frame: {}\ntime: {:.1f} ms".format(frm,time))
        ax.set_title("frame: {}, time: {:.1f} ms".format(frm,time))
        
        return raw_img, trajplots#, heading
    
    traj_ani = FuncAnimation(fig, update_frame, init_func=init_tracks, 
                             frames=amnt_f, #fargs=(trajdata),#, trajplots),
                             interval=70, blit=False, repeat=False)
    return traj_ani

# anim2 = AnimateTracksOnRawData(t2_long,rawframes_ROI,settings)
# anim2.save('Au50_raw+tracks_1000frames.mp4')


