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



def GetCI_Interval(probability, value, ratio_in_ci):
    """ get the confidence interval (CI) of a probability density function
    
    probability = PDF (y) values
    probability = x - values
    ratio_in_ci = confidence intercall (CI)
    """
    
    #cumulated probabilty sum
    cum_sum = np.cumsum(probability)
    
    # lower and upper limit of CI (staring at 50% of the cumulated probabilty sum)
    cum_min = 0.5 - (ratio_in_ci/2)
    cum_max = 0.5 + (ratio_in_ci/2)
    
    # find the position in the array of the argument
    pos_min = np.int(np.where(cum_sum > cum_min)[0][0])
    pos_max = np.int(np.where(cum_sum > cum_max)[0][0])
    
    # get the x values where the CI is
    value_min = value[pos_min]
    value_max = value[pos_max]
        
    return value_min,value_max



def GetMeanStdMedian(data):
    my_mean   = np.mean(data)
    my_std    = np.std(data)
    my_median = np.median(data)

    return my_mean, my_std, my_median   



def InvDiameter(sizes_df_lin, settings, useCRLB=True):
    """ calculates inverse diameter values and estimates of their stds
    
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
    
    inv_diam = 1/sizes_df_lin["diameter"].values # numpy.ndarray
    
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
    """ calculate the inverse mean and std of a particle diameter ensemble,
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
    diam_inv_mean, diam_inv_std : both float
    """
    if type(sizes) is pd.DataFrame:
        sizes = sizes.diameter
    
    diam_inv_mean = (1/sizes).mean()
    diam_inv_std  = (1/sizes).std()
    
    return diam_inv_mean, diam_inv_std
    
    

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



