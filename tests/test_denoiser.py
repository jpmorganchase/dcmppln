###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2024: Amazon Web Services, Inc. - Contributions from JPMC
#
###############################################################################
from dcmppln.denoiser import Denoiser


def test_denoiser_function(sample_input_data):
    denoiser = Denoiser(active=False)
    correlation_matrix, covariance_matrix, full_returns = sample_input_data
    correlation_matrix_1, covariance_matrix_1, full_returns_1 = sample_input_data
    C_2_1, _, _ = denoiser(correlation_matrix)
    C_2_2, _, _ = denoiser(correlation_matrix_1)
    assert (C_2_1 == C_2_2).all()
