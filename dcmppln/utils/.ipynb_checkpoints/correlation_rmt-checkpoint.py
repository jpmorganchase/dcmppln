###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2024: Amazon Web Services, Inc. - Contributions from JPMC
#
###############################################################################
import math
import numpy as np
from dcmppln.utils.utils import timeit 
from joblib import Parallel, delayed


def marchenko_pastur_pdf(eigenvalue:np.ndarray, beta:float,sigma:float=1.0) -> np.ndarray:
    """Used to characteize the eigenvalues that are predominantly noise induced
    Parameters
    ----------
    eigenvalue:np.ndarray
        eignvalue of the covariance matrix
    beta: float
        number of rows/columns -> N_stocks/N_days
    sigma: float
        variance of the noise (between 0.0 and 1.0?)

    Returns
    -------
    pdf : np.ndarray
        distribution for each Stock            
    """
    lambda_plus = (sigma**2)*((1 + np.sqrt(beta)) ** 2)
    lambda_minus = (sigma**2)*((1 - np.sqrt(beta)) ** 2)
    pdf = np.sqrt(np.maximum(lambda_plus - eigenvalue, 0) * np.maximum(eigenvalue - lambda_minus, 0)) / (2 * np.pi * beta * eigenvalue*(sigma**2))
    return pdf 

def fit_marchenko_pastur(eigenvalues:np.ndarray, beta:float, q_fit:bool=True,sigma:float=0.5,steps=1000)-> float:
    """
    Fit the Marchenko-Pastur distribution to the given eigenvalues.
    Parameters:
    -----------
    eigenvalues: numpy.ndarray 
        The eigenvalues of the covariance to fit.
    beta: float 
        The ratio of the number of variables to the number of observations. (e.g. N_stocks/N_days)
    sigma: float 
        The variance of the noise.
    q_fit: bool
        True or False whether fit is required for best beta or not
    steps: int
        Number of steps to perform better fit        

    Returns
    --------
    lambda_min: float
        represent minimum eigenvalue
    lambda_max: float
        represent maximum eigenvalue    
    """
    if not q_fit:
        sigma = 0.5 
        lambda_min = (sigma**2)*((1 - np.sqrt(beta)) ** 2)
        lambda_max = (sigma**2)*((1 + np.sqrt(beta)) ** 2)
        return lambda_min, lambda_max
    beta_best = _find_best_beta(eigenvalues, beta, sigma)
    lambda_min = (sigma**2)*((1 - np.sqrt(beta_best)) ** 2)
    lambda_max = (sigma**2)*((1 + np.sqrt(beta_best)) ** 2)
    return lambda_min, lambda_max

def _find_best_beta(eigenvalues:np.ndarray, beta:float, sigma:float=0.5, steps:int=1000) -> float:
    """
    Function to derive best beta value by finding minumum of the mean distribution
    Parameters
    ----------
    eigenvalues: numpy.ndarray 
        The eigenvalues of the covariance to fit.
    beta: float 
        The ratio of the number of variables to the number of observations. (e.g. N_stocks/N_days)
    sigma: float 
        The variance of the noise. 
    steps: int
        Number of steps to perform better fit  

    Return
    ------
    beta_best:float
        best bit beta value based on steps                   
    """
    if beta < 1:
        beta_values = np.linspace(beta, 1, steps)
    else:
        beta_values = np.linspace(1, beta, steps)

    min = math.inf
    for beta_ in beta_values[1:]:
        # Compute the theoretical eigenvalue density
        theoretical_pdf = marchenko_pastur_pdf(eigenvalues, beta_, sigma)
        # Compute the mean squared error between the theoretical and sample eigenvalue density
        mse = np.mean((theoretical_pdf - eigenvalues) ** 2)
        if mse < min:
            min = mse
            beta_best=beta_
    return beta_best        

def split_covariance_matrices(C:np.ndarray, beta:float=0.5, q_fit:bool=True):
    """
    This function split the covariance matrices to C_Noise, C_Star, C_Global
    
    Parameters
    ----------
    C:np.ndarray 
        Input correlation matrix
    beta:float 
        ratio of number of variables to number of observations. e.g. N_stocks/N_days
    q_fit:bool
         True or False whether fit is required for best beta or not
    Return
    C_Noise:np.ndarray
        a split of correlation matrices whose eignevalues are at most lambda_max
    C_Star:np.ndarray
        a split of correlation matrices whose eigenvalues between lambda_max and largest lambda
    C_Global:np.ndarray
        a split of correlation matrices , Lambda1 which is typically much greater than lambda+    
    """
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    eigenvalues = eigenvalues[::-1]  # Reverse order sorting
    eigenvectors = eigenvectors[:, ::-1]  # Reverse order sorting

    lambda_min, lambda_max = fit_marchenko_pastur(eigenvalues[1:],beta, q_fit=True)
    # Split the eigenvalues and eigenvectors into three regimes.
    # First regime has eigenvalues less than lambda_max.
    eigenvalues_1 = eigenvalues[eigenvalues < lambda_max]
    eigenvectors_1 = eigenvectors[:, eigenvalues < lambda_max]

    # Second regime has eigenvalues between lambda_max and the largest lambda.
    eigenvalues_2 = eigenvalues[
        (eigenvalues >= lambda_max) & (eigenvalues < eigenvalues[0])
    ]
    eigenvectors_2 = eigenvectors[
        :, (eigenvalues >= lambda_max) & (eigenvalues < eigenvalues[0])
    ]

    # Third regime is the eigenvalue greater than lambda_max.
    eigenvalues_3 = eigenvalues[0]
    eigenvectors_3 = eigenvectors[:, 0]

    # Compute the correlation matrices for each regime
    C_Noise = eigenvectors_1 @ np.diag(eigenvalues_1) @ eigenvectors_1.T # CNoise
    C_Star = eigenvectors_2 @ np.diag(eigenvalues_2) @ eigenvectors_2.T # C*
    C_Global = eigenvectors_3 * eigenvalues_3 * eigenvectors_3.T #CGlobal

    return C_Noise, C_Star, C_Global
