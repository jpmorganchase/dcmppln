###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2024: Amazon Web Services, Inc. - Contributions from JPMC
#
###############################################################################
import numpy as np

from dcmppln.utils.correlation_rmt import (
    marchenko_pastur_pdf,
    fit_marchenko_pastur,
    _find_best_beta,
    split_covariance_matrices,
)


def test_marchenko_pastur_pdf():
    rng = np.random.default_rng(100)
    a = rng.normal(size=(5, 30))
    covariance_matrix = np.matmul(a, a.transpose()) / a.shape[1]
    eigenvalue, eigenvector = np.linalg.eigh(covariance_matrix)
    q = 5.0 / 30.0
    pdf_1 = marchenko_pastur_pdf(eigenvalue, q)
    rng = np.random.default_rng(100)
    a = rng.normal(size=(5, 30))
    covariance_matrix = np.matmul(a, a.transpose()) / a.shape[1]
    eigenvalue, eigenvector = np.linalg.eigh(covariance_matrix)
    pdf_2 = marchenko_pastur_pdf(eigenvalue, q)
    assert (pdf_1 == pdf_2).all()


def test_marchenko_pastur_pdf_for_zero():
    rng = np.random.default_rng(100)
    a = rng.normal(size=(5, 30))
    covariance_matrix = np.matmul(a, a.transpose()) / a.shape[1]
    eigenvalue, eigenvector = np.linalg.eigh(covariance_matrix)
    """
    Hardcoded to test else condition that should return zero if lamda is not
    within the range of lambda_plus and lambda_minus
    lambda_minus <= lambda <= lambda_plus
    """
    q = 0.05

    pdf = marchenko_pastur_pdf(eigenvalue, q)
    assert 0 in pdf


def test_fit_marchenko_pastur(sample_input_data):
    correlation_matrix, covariance_matrix, full_returns = sample_input_data
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    eigenvalues = eigenvalues[::-1]  # Reverse order sorting
    eigenvectors = eigenvectors[:, ::-1]  # Reverse order sorting
    rows, columns = 90.0, 1000.0
    beta = float(rows) / float(columns)
    lambda_min_1, lambda_max_1 = fit_marchenko_pastur(eigenvalues[1:], beta)
    lambda_min_2, lambda_max_2 = fit_marchenko_pastur(eigenvalues[1:], beta)
    assert (lambda_min_1 == lambda_min_2) and (lambda_max_1 == lambda_max_2)


def test_fit_marchenko_paastur_q_fit_false(sample_input_data):
    correlation_matrix, covariance_matrix, full_returns = sample_input_data
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    eigenvalues = eigenvalues[::-1]  # Reverse order sorting
    eigenvectors = eigenvectors[:, ::-1]  # Reverse order sorting
    rows, columns = 90.0, 1000.0
    beta = float(rows) / float(columns)
    lambda_min_false, lambda_max_false = fit_marchenko_pastur(
        eigenvalues[1:], beta, q_fit=False
    )
    lambda_min_true, lambda_max_true = fit_marchenko_pastur(eigenvalues[1:], beta)
    assert lambda_max_false != lambda_max_true


def test_find_best_beta_sigma_1(sample_input_data):
    correlation_matrix, covariance_matrix, full_returns = sample_input_data
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    eigenvalues = eigenvalues[::-1]  # Reverse order sorting
    eigenvectors = eigenvectors[:, ::-1]  # Reverse order sorting
    rows, columns = 90.0, 1000.0
    beta = float(rows) / float(columns)
    best_beta_1 = _find_best_beta(eigenvalues, beta, sigma=1.0)
    assert beta != best_beta_1


def test_split_covariance_matrices(sample_input_data):
    correlation_matrix, covariance_matrix, full_returns = sample_input_data
    C_Noise_1, C_Star_1, C_Global_1 = split_covariance_matrices(covariance_matrix)
    C_Noise_2, C_Star_2, C_Global_2 = split_covariance_matrices(covariance_matrix)
    assert (
        (C_Noise_1 == C_Noise_2).all()
        and (C_Star_1 == C_Star_2).all()
        and (C_Global_1 == C_Global_2).all()
    )
