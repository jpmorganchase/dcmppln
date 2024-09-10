###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2024: Amazon Web Services, Inc. - Contributions from JPMC
#
###############################################################################
from typing import Optional, Union
import numpy as np
import time


def get_instance_non_private_attributes(instance):
    """
    instance: instance of a class
    returns: all non private attributes and values in a dictionnary
    """
    return { k:v for k,v in vars(instance).items() if k[0]!="_" }



def compute_objective_value(returns, covariance, weights, risk_factor=0.5):
    weights = np.array(weights).reshape(-1, 1)
    term1 = risk_factor * weights.T @ covariance @ weights
    term2 = returns @ weights
    obj_value = term1 - term2
    return obj_value[0,0]  # Convert single value array to scalar


def threshold_matrix(correlation_matrix, threshold=0.5):
    binary_matrix = (correlation_matrix > threshold).astype(float)
    return binary_matrix


def thresholding_not_touching_diag(correlation_matrix, threshold=0.5):
    binary_matrix = threshold_matrix(correlation_matrix, threshold)
    np.fill_diagonal(binary_matrix, np.diagonal(correlation_matrix))
    return binary_matrix

def sparsification(correlation_matrix, threshold=0.5, allow_to_modify_diag=False):
    sparse_matrix = np.where(correlation_matrix > threshold, correlation_matrix, 0)
    if not allow_to_modify_diag:
        np.fill_diagonal(sparse_matrix, np.diagonal(correlation_matrix))
    return sparse_matrix

def sparsification_covariance_no_diag_mean_ratio(correlation_matrix, threshold_mratio=0.2):
    correlation_matrix_without_diagonal = np.array(correlation_matrix)
    np.fill_diagonal(correlation_matrix_without_diagonal, 0)
    avg_weight_without_diagonal = np.mean(correlation_matrix_without_diagonal)
    threshold = threshold_mratio * avg_weight_without_diagonal
    sparse_matrix = sparsification(correlation_matrix, threshold)
    return sparse_matrix


def timeit(func):
    def wrapper(*args, **kwargs)-> Union[Optional[np.array], Optional[float], Optional[float]]: 
        start_process_time = time.perf_counter()
        start_time = time.process_time()
        result = func(*args, **kwargs)
        end_time = time.perf_counter() - start_process_time
        process_time = time.process_time() - start_time
        return result, process_time, end_time
    return wrapper