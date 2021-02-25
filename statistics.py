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

from pdb import set_trace as bp #debugger



def GetCI_Interval(probability, grid, ratio_in_ci):
    """ get the confidence interval (CI) of a probability density function (PDF)
    
    NB: This should work on both equidistant and non-equidistant grids.
    
    probability : PDF (y) values
    grid : x values
    ratio_in_ci : confidence interval (CI)
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
    my_mean   = np.mean(data)
    my_std    = np.std(data)
    my_median = np.median(data)

    return my_mean, my_std, my_median   



def GetMeanStdMedianCIfromPDF(PDF, grid, grid_stepsizes):
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
    
    inv_sizes = 1000/sizes # 1/um
    diam_inv_mean, diam_inv_std, diam_inv_median = GetMeanStdMedian(inv_sizes)
    diam_inv_CI68 = (np.quantile(inv_sizes,0.5-0.68/2), np.quantile(inv_sizes,0.5+0.68/2))
    
    return diam_inv_mean, diam_inv_std, diam_inv_median, diam_inv_CI68
    
    

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
        ax0 = nd.visualize.Plot2DPlot(N, AIC,
                                      title = 'Akaike (-) and Bayesian (--)', 
                                      xlabel = 'number of components', 
                                      ylabel = 'information criterion',
                                      mymarker = 'x', mylinestyle = '-')
        ax0.plot(N, BIC, 'x--')
    
    if useAIC==True:
        minICindex = np.argmin(AIC)
    else:
        minICindex = np.argmin(BIC)
    M_best = models[minICindex] # choose model where AIC is smallest
    print('Number of components considered: {}'.format(N[minICindex]))

    means = M_best.means_.flatten()
    stds = (M_best.covariances_.flatten())**0.5
    weights = M_best.weights_
    
    # sort the parameters from highest to lowest mean value
    sortedIndices = means.argsort()[::-1]
    means = means[sortedIndices]
    stds = stds[sortedIndices]
    weights = weights[sortedIndices]
    
    print('Number of iterations performed: {}'.format(M_best.n_iter_))
    
    return means, stds, weights



