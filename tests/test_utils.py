###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2024: Amazon Web Services, Inc. - Contributions from JPMC
#
###############################################################################
from dcmppln.utils.utils import (
    sparsification,
    threshold_matrix,
    thresholding_not_touching_diag,
    sparsification_covariance_no_diag_mean_ratio,
)
import numpy as np

def test_threshold_matrix():
    np.random.seed(42)
    correlation_matrix = np.random.laplace(0.0, 1.0, (3, 3))
    binary_matrix = threshold_matrix(correlation_matrix)
    assert np.array_equal(binary_matrix, binary_matrix.astype(bool))


def test_thresholding_not_touching_diag():
    np.random.seed(42)
    correlation_matrix = np.random.laplace(0.0, 1.0, (3, 3))
    binary_matrix_diagonal_intact = thresholding_not_touching_diag(correlation_matrix)
    di_1 = np.diagonal(correlation_matrix)
    di_2 = np.diagonal(binary_matrix_diagonal_intact)
    assert (di_1 == di_2).all()

def test_sparsification_for_diagonal():
    np.random.seed(42)
    correlation_matrix = np.random.laplace(0.0, 1.0, (3, 3))
    sparse_matrix = sparsification(correlation_matrix)
    di_1 = np.diagonal(correlation_matrix)
    di_2 = np.diagonal(sparse_matrix)
    assert (di_1 == di_2).all()


def test_sparsification_for_threshold():
    np.random.seed(42)
    correlation_matrix = np.random.laplace(0.0, 1.0, (3, 3))
    threshold = 0.5
    sparse_matrix = sparsification(correlation_matrix, allow_to_modify_diag=True)
    sparse_matrix_indices = np.argwhere(sparse_matrix == 0)
    correlation_matrix_indices = np.argwhere(correlation_matrix <= threshold)
    assert (sparse_matrix_indices == correlation_matrix_indices).all()


def test_sparsification_covariance_no_diag_mean_ratio_for_diagonal():
    np.random.seed(42)
    correlation_matrix = np.random.laplace(0.0, 1.0, (3, 3))
    di_1 = np.diag(correlation_matrix)
    sparse_matrix = sparsification_covariance_no_diag_mean_ratio(correlation_matrix)
    di_2 = np.diag(sparse_matrix)
    assert (di_1 == di_2).all()

def test_sparsification_covariance_no_diag_mean_ratio_for_threshold():
    np.random.seed(42)
    correlation_matrix = np.random.laplace(0.0, 1.0, (3, 3))
    sparse_matrix = sparsification_covariance_no_diag_mean_ratio(correlation_matrix)
    sparse_matrix_indices = np.argwhere(sparse_matrix == 0)
    correlation_matrix_without_diagonal = np.array(correlation_matrix)
    np.fill_diagonal(correlation_matrix_without_diagonal, 0)
    avg_weight_without_diagonal = np.mean(correlation_matrix_without_diagonal)
    np.fill_diagonal(
        correlation_matrix_without_diagonal, avg_weight_without_diagonal + 1
    )
    correlation_matrix_indices = np.argwhere(
        correlation_matrix_without_diagonal <= avg_weight_without_diagonal
    )
    assert (sparse_matrix_indices == correlation_matrix_indices).all()
