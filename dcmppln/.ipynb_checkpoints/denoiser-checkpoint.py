###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2024: Amazon Web Services, Inc. - Contributions from JPMC
#
###############################################################################
import numpy as np
from dcmppln.utils.utils import get_instance_non_private_attributes
from dcmppln.utils.correlation_rmt import split_covariance_matrices
from dcmppln.utils.utils import timeit


class Denoiser:
    """
    Class to apply noise reduction before clustering
    """

    def __init__(self, active: bool = True, q: float = 0.5, q_fit: bool = True):
        """
        Parameters
        ----------
        active: bool to activate or deactivate the component, if deactivated other parameters are meaningless
        q: ratio of number of variables to number of observations. e.g. N_stocks/N_days
        q_fit: finds the best q
        """

        self.active = active
        self.q = q
        self.q_fit = q_fit

        self.params = get_instance_non_private_attributes(self)

    # use this structure to ease profile readability
    def __call__(self, C):
        return self.denoise(C)

    # Check with q=0.5, q_fit = False,  q=1.5
    @timeit
    def denoise(self, C: np.array) -> np.array:
        """
        function to calculate denoise, if the component is deactivated return the
        C else return

        Parameters
        ----------
        C: np.array input correlation matrix
        """

        if not self.active:
            return C

        C_1, C_2, C_3 = split_covariance_matrices(C, q=self.q, q_fit=self.q_fit)
        return C_2
