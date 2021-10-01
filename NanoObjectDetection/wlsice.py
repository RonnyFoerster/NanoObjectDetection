

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Notation:
# N = number of sample points
# M = number of trajectories
# C = Covariance matrix of data

import sys
import numpy as np
import scipy.optimize

from pdb import set_trace as bp #debugger

f = 0
df = 0
d2f = 0

# string -> nil
def error(string, stop=True):
    "Auxiliary function for error printing "
    sys.stderr.write("- Error! " + string)
    if stop:
        sys.exit(2)

# function, function, function -> nil
def init(f_, df_, d2f_):
    "Set function to fit"
    global f
    global df
    global d2f

    f = f_
    df = df_
    d2f = d2f_


# np.array(2) -> np.array(1)
def computeMean(trajectories):
    "Compute mean of trajectories"

    M,N = np.shape(trajectories)
    y_mean = np.zeros(N)
    for i in range(N):
        y_mean[i] = np.sum(trajectories[:,i]) / M
    return y_mean


# np.array(2), np.array(1) -> np.array(2)
def makeY(trajectories, y_mean):
    """Take an M x N array of trajectories, and return an N x M matrix with:
    Y[n,m] = (y_n^(m) - \bar{y}_n)
    """

    M, N = np.shape(trajectories)

    # compute Y-matrix
    Y = np.zeros((N,M), dtype=np.float64)
    for m in range(M):
        Y[:,m] = np.subtract(trajectories[m,:], y_mean)

    return Y


# np.array(2) -> np.array(2)
def makeCovarianceFromY(Y):
    "Take a matrix of dim N x M and compute the covariance matrix"
    N, M = np.shape(Y)
    C = np.dot(Y,np.transpose(Y)) / (M - 1.0)
    return C


# np.array(2) -> np.array(2), np.array(1)
def makeCovarianceMatrix(trajectories):
    "Make covariance matrix from an MxN array of trajectories"

    M, N = np.shape(trajectories)
    y_mean = computeMean(trajectories)
    Y = makeY(trajectories, y_mean)
    C = makeCovarianceFromY(Y)

    return C, y_mean


# np.array(2),  np.array(3), np.array(2), np.array(2), np.array(1) -> np.array(1)
def errorEstimation(df, d2f, R, C, delta):
    """WLS-ICE error estimation, valid also for non-linear fitting. R could
    be R = inv(cov), or the diagonal of that, or some other symmetric
    matrix of our choosing. With N sampling points, and k parameters
    we have:

    df     is a  k x N      dimensional array
    d2f    is a  k x k x N  dimensional array
    delta  is a  N          dimensional array

    """
    q = len(df)
    N = len(R)

    first_term = np.zeros((q,q), dtype=np.float64)
    for a in range(q):
        for b in range(q):
            for i in range(N):
                for j in range(N):
                    first_term[a,b] = first_term[a,b] + 2 * d2f[a,b,i] * R[i,j] * delta[j]

    second_term = 2 * np.dot(np.dot(df,R),np.transpose(df))

    hessian = first_term + second_term

    H_inv = np.linalg.inv(hessian)
    RCR = np.dot(R,np.dot(C,R))

    error = np.zeros((q,q), dtype=np.float64)
    for a in range(q):
        for b in range(q):
            for c in range(q):
                for d in range(q):
                    dfRCRdf = np.dot(df[c],np.dot(RCR,np.transpose(df[d])))
                    error[a,b] = error[a,b] + 4 * H_inv[a,c] * dfRCRdf * H_inv[d,b]

    return(error.diagonal())


# np.array(1), np.array(1), np.array(1), np.array(2) -> number
def chi2(params, t, y, R):
    """ Compute chi^2 function to be minimized, based on input
    (y-f)*R*(y-f)
    This gives us our fitted parameters in params.
    (If R=diag(inv(Cov)) we get least square fit)
    """
    Y = f(t, params)
    delta = np.subtract(y, Y)
    ans = np.dot(np.dot(delta, R), delta)
    return ans


# np.array(1), np.array(1), np.array(1), np.array(2) -> np.array(1)
def chi2Jacobian(params, t, y, R):
    """Jacobian of chi^2 function, i.e.
    -df/dparams[i]*[C^(-1)]*(y-f) + (y-f)*[C^(-1)]*(-df/dparams[i])
    return np.array of result for each i
    """
    print("Params into jacobian:", params)
    dY = df(t, params)
    Y = f(t, params)
    ans = np.zeros(len(dY))
    for i in range(len(dY)):
        ans[i] = - 2 * np.dot(np.dot(dY[i], R), np.subtract(y, Y))

    print("Jacobian:\t%s\n" % ans)
    return ans


# np.array(1), np.array(1), np.array(2), np.array(2), np,array(1), string,  ->
#  tuple(np.array(1), np.array(1), number)
def minimize(t, y, R, C, guess_start, min_method):
    """Minimize the chi-square function, to find optimal parameters and
    their esitmated error for function f"""

    display = False
    if min_method == "bfgs":
        result = scipy.optimize.minimize(chi2, guess_start, args=(t, y, R), method='BFGS',
                                         jac=chi2Jacobian, options={'gtol': 1e-8, 'disp': display})
    elif min_method == "nm":
        result = scipy.optimize.minimize(chi2, guess_start, args=(t, y, R), method='Nelder-Mead',
                                         options={'xtol': 1e-8, 'disp': display})
    else:
        error("Unknown method: %s\n" % min_method)

    params = result.x
    err = errorEstimation(df(t,params), d2f(t,params), R, C, np.subtract(f(t,params),y))

    chi_value = chi2(params, t, y, R)
    sigma = np.sqrt(err)
    return (params, sigma, chi_value)


# np.array(1), np.array(2), string, np.array(1) -> tuple(np.array(1), np.array(1), number)
def fit(time, trajectories, guess, min_method='nm'):
    "Perform the correlated corrected least squares fit"
    M,N = np.shape(trajectories)

    C, y_mean = makeCovarianceMatrix(trajectories)
    y_sigma = np.sqrt(np.diag(C))

    # LS-ICE
    C = C / M
    bp()
    R = np.linalg.inv(np.diag(np.diag(C), 0))

    # Do the parameter fitting, return: [opt_parameters, errors, chi2_min]
    bp()
    p_opt, p_sigma, chi2_min = minimize(time, y_mean, R, C, guess, min_method)
    
    bp()
    return (p_opt, p_sigma, chi2_min)
