# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 09:50:52 2021

@author: nissenmona, foersterronny


collection of functions for statistical information on the tracking data
(meant to tidy up visualize.py and CalcDiameter.py a bit...)
"""
import numpy as np
import pandas as pd
import NanoObjectDetection as nd
import matplotlib.pyplot as plt # libraries for plotting
from NanoObjectDetection.PlotProperties import axis_font, title_font, params


def GetCI_Interval(probability, grid, ratio_in_ci):
    """
    get the boundaries of the confidence interval (CI) of a probability density function (PDF)
    
    NB: This should work on both equidistant and non-equidistant grids.

    Parameters
    ----------
    probability : TYPE
        PDF (y) values.
    grid : TYPE
        x values.
    ratio_in_ci : TYPE
        confidence interval (CI).

    Returns
    -------
    value_min : TYPE
        DESCRIPTION.
    value_max : TYPE
        DESCRIPTION.
    """
    
    
    grid_steps = abs(grid[:-1]-grid[1:])
    if grid_steps[0] > grid_steps[-1]: # duplicate smallest entry (either the first or the last)
        grid_steps = np.append(grid_steps, grid_steps[-1]) 
    else:
        grid_steps = np.append(grid_steps, grid_steps[0])
    
    
    #cumulated probabilty sum
    cum_sum = np.cumsum(probability*grid_steps)
    
    # lower and upper limit of CI (staring at 50% of the cumulated probabilty sum)
    cum_min = 0.5 - (ratio_in_ci/2)
    cum_max = 0.5 + (ratio_in_ci/2)
    
    # find the position in the array of the argument
    pos_min = np.int(np.where(cum_sum > cum_min)[0][0])
    pos_max = np.int(np.where(cum_sum > cum_max)[0][0])
    
    # get the x values where the CI is
    value_min = grid[pos_min]
    value_max = grid[pos_max]
    
    return value_min,value_max



def GetMeanStdMedian(data):
    """
    Calculates mean, standarddeviation and median of an array
    """
    my_mean   = np.mean(data)
    my_std    = np.std(data)
    my_median = np.median(data)

    return my_mean, my_std, my_median   



def GetMeanStdMedianCIfromPDF(PDF, grid, grid_stepsizes):
    """ get statistical quantities from normalized (!) PDF """
    mean = sum(PDF*grid*grid_stepsizes)
    var = sum(PDF*grid**2*grid_stepsizes) - mean**2
    std = var**0.5 
    median = np.mean(nd.statistics.GetCI_Interval(PDF, grid, 0))
    CI68 = nd.statistics.GetCI_Interval(PDF, grid, 0.68) 
    CI95 = nd.statistics.GetCI_Interval(PDF, grid, 0.95) 
    return mean, std, median, CI68, CI95



def InvDiameter(sizes_df_lin, settings, useCRLB=True):
    """ calculates inverse diameter values and estimates of their stds
    
    NOTE: If diameters are in nm, inv. diameters are returned in 1/um (!).
    
    assumption: 
        relative error = std/mean = CRLB
        with N_tmax : number of considered lagtimes
        N_f : number of frames of the trajectory (=tracklength)
    
    Parameters
    ----------
    sizes_df_lin : pandas.DataFrame
        with "diameter" and "valid frames" or "traj length" columns
    settings : dict
        with ["MSD"]["lagtimes_max"] entry

    Returns
    -------
    inv_diam, inv_diam_std : both numpy.ndarray
    """
    
    inv_diam = 1000/sizes_df_lin["diameter"].values # numpy.ndarray
    
    if useCRLB==True:
        # CRLB is already included in the diffusion std calculation (!)
        rel_error = (sizes_df_lin["diffusion std"] / sizes_df_lin["diffusion"]).values
        
        # x = sizes_df_lin["red_x"] # pd.Series
        # rel_error = nd.Theory.CRLB(N_f, x) # RF 210113
    else:        
        # MN 210227: replace "traj length" by "valid frames"
        try:
            N_f = sizes_df_lin["valid frames"] # pd.Series 
        except KeyError:
            N_f = sizes_df_lin["traj length"]
            
        N_tmax = settings["MSD"]["lagtimes_max"] # int value
        # ... should only be used if "Amount lagtimes auto" = 0 (!!)
        
        # Qian 1991
        rel_error = np.sqrt((2*N_tmax)/(3*(N_f - N_tmax))) # pd.Series
        
        # # CRLB/Michalet 2012 for the ideal case red_x = 0
        # rel_error = np.sqrt( 6/(N_f-1))
        
    inv_diam_std = inv_diam * rel_error
    
    return inv_diam, inv_diam_std



def StatisticOneParticle(sizes):
    """ calculate the inverse statistical quantities of a particle diameter ensemble,
    assuming only one contributing size
    
    NOTE:
    mean diameter is ~1/(mean diffusion) not mean of all diameter values (!)
    => reciprocal Gaussian distribution

    Parameters
    ----------
    sizes : pandas.DataFrame with 'diameter' column 
            OR pandas.Series of float 
            OR np.array of float
            particle diameters
        
    Returns
    -------
    diam_inv_mean, diam_inv_std, diam_inv_median, diam_inv_CI68 : (tuple of) float
    """
    if type(sizes) is pd.DataFrame:
        sizes = sizes.diameter
    
    # percentage of values that is within 1 or 2 sigma (for the normal distr.)
    s1 = 0.682689492
    s2 = 0.954499736
    
    inv_sizes = 1000/sizes # 1/um
    diam_inv_mean, diam_inv_std, diam_inv_median = GetMeanStdMedian(inv_sizes)
    diam_inv_CI68 = np.array([np.quantile(inv_sizes,0.5-s1/2), np.quantile(inv_sizes,0.5+s1/2)])
    diam_inv_CI95 = np.array([np.quantile(inv_sizes,0.5-s2/2), np.quantile(inv_sizes,0.5+s2/2)])
    
    return diam_inv_mean, diam_inv_std, diam_inv_median, diam_inv_CI68, diam_inv_CI95
    
    

def StatisticDistribution(sizesInv, num_dist_max=10, showICplot=False, useAIC=False):
    """ compute the best fit of a mixture distribution to a sizes/inv.sizes ensemble
    
    ATTENTION: scikit learn python package needed!
    Gaussian mixture models for different number of mixture components are 
    calculated via the expectation maximization algorithm. 
    The model with the smallest AIC (Akaike information criterion) or BIC 
    (Bayesian information criterion) value is chosen for computing the 
    distribution parameters.

    NOTE: 
        Unlike in the "StatisticOneParticle" function, here the distribution 
        parameters are NOT computed for the inverse diameters. 
    
    Parameters
    ----------
    sizesInv: np.array; sizes or inv.sizes values
    num_dist_max: int
        maximum number of considered components N in the mixture
    showICplot: boolean
        plot a figure with AIC and BIC over N
    
    Returns
    -------
    means, stds, weights: np.arrays of length N_best
        fitting parameters in descending order of the mean values
    
    credits to Jake VanderPlas
    https://www.astroml.org/book_figures/chapter4/fig_GMM_1D.html (15.12.2020)
    """
    from sklearn.mixture import GaussianMixture
    
    # special format needed here...
    sizesInv = np.array(sizesInv,ndmin=2).transpose()
    
    N = np.arange(1,num_dist_max+1) # number of components for all models
    models = [None for i in range(len(N))]

    for i in range(len(N)):
        models[i] = GaussianMixture(N[i],covariance_type='spherical').fit(sizesInv) 
        # default is 'full', but this is only needed for data dimensions >1
       
    # https://en.wikipedia.org/wiki/Akaike_information_criterion
    AIC = [m.aic(sizesInv) for m in models]
    # https://en.wikipedia.org/wiki/Bayesian_information_criterion
    BIC = [m.bic(sizesInv) for m in models]
    ''' Wikipedia:
        "The formula for the Bayesian information criterion (BIC) is similar 
        to the formula for AIC, but with a different penalty for the number 
        of parameters. With AIC the penalty is 2k, whereas with BIC 
        the penalty is ln(n) k." 
        (k: number of fitting parameters,
         n: number of observation points in the sample) '''
    
    if showICplot==True:        
        plt.style.use(params)
        plt.figure()
        plt.plot(N, AIC, marker = 'x', linestyle  = '-')
        
        plt.title('Akaike (-) and Bayesian (--)', **title_font)
        plt.xlabel('number of components', **axis_font)
        plt.ylabel('information criterion', **axis_font)
        
        nd.PlotProperties
        plt.style.use(params)
        
        ax0 = plt.gca()
        
        ax0.plot(N, BIC, 'x--')
    
    if useAIC==True:
        minICindex = np.argmin(AIC)
    else:
        minICindex = np.argmin(BIC)
    M_best = models[minICindex] # choose model where AIC is smallest
    nd.logger.info('Number of components considered: {}'.format(N[minICindex]))

    means = M_best.means_.flatten()
    stds = (M_best.covariances_.flatten())**0.5
    weights = M_best.weights_
    
    # sort the parameters from lowest to highest mean value
    sortedIndices = means.argsort()#[::-1]
    means = means[sortedIndices]
    stds = stds[sortedIndices]
    weights = weights[sortedIndices]
    
    nd.logger.info('Number of iterations performed: {}'.format(M_best.n_iter_))
    
    return means, stds, weights



def CalcInversePDF(grid, sizes_df_lin, settings, useCRLB):
    """ compute the probability density function (PDF) of the sizes ensemble 
    on a given grid in the inverse sizes space
    
    NB: Due to the Gaussian/normal distributed uncertainty in the diffusion space,
        the PDF is always first calculated in the inverse sizes space and then
        back-converted.
        Each trajectory is considered individually, the tracklength determines
        the PDF widths (via CRLB or Qian91).
    """
    import NanoObjectDetection as nd
    from scipy.stats import norm
    
    grid_steps = abs(grid[:-1]-grid[1:])
    if grid_steps[0] > grid_steps[-1]: # duplicate smallest entry (either the first or the last)
        grid_steps = np.append(grid_steps, grid_steps[-1]) 
    else:
        grid_steps = np.append(grid_steps, grid_steps[0])
    
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



def ConvertPDFtoSizesSpace(PDF_inv, inv_grid):
    """ convert PDF on 1/um-grid back into one in sizes space on a nm-grid,
    including normalization to integral=1 and sorting the grid
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
    
    if grid[0] > grid[-1]: # invert order
        grid = grid[::-1]
        grid_steps = grid_steps[::-1]
        PDF = PDF[::-1]
    
    return PDF, grid, grid_steps



def FitGaussian(PDF, grid, grid_steps, mean_start):
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



def FitGaussianMixture(PDF, grid, grid_steps, mean_max):
    """ calculate least-squares fit of a superpositions of 2 Gaussian functions 
    to a given PDF
    """
    from scipy.optimize import curve_fit
    
    # define mixture of 2 Gaussian fcts (integral normalized to 1)
    def myGaussMix(d,m1,m2,s1,s2,w):
        gauss1 = w/((2*np.pi)**0.5*s1)*np.exp(-(d-m1)**2/(2.*s1**2))
        gauss2 = (1-w)/((2*np.pi)**0.5*s2)*np.exp(-(d-m2)**2/(2.*s2**2))
        return gauss1 + gauss2
    
    popt, pcov = curve_fit(myGaussMix, grid, PDF, 
                           bounds=(0, [mean_max, mean_max, mean_max/2, mean_max/2, 1]))#, p0=[mean_start,1])
    
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



