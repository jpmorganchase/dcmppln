###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2024: Amazon Web Services, Inc. - Contributions from JPMC
#
###############################################################################
import os, sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from dcmppln.utils.utils import get_instance_non_private_attributes
import numpy as np


class InputData:
    """
    Component to input data into the pipeline
    """

    def __init__(self, path="", abs_cov=True, abs_corr=True, subprob_idx=None):
        default_paths = ["../tests/data/", "tests/data/"]

        # Use the provided path or the first existing default path
        if path:
            self.path = path
        else:
            self.path = None
            for default_path in default_paths:
                if self._path_exists(default_path):
                    self.path = default_path
                    self.path=self.path+"seed_1000_size_90"
                    break
            if not self.path:
                raise FileNotFoundError("None of the default paths exist.")

        self.abs_corr = abs_corr
        self.abs_cov = abs_cov
        self.params = get_instance_non_private_attributes(self)

        self._subprob_idx = subprob_idx

    def __call__(self):
        return self.data_for_pipeline()

    def _path_exists(self, path):
        return os.path.exists(path)
    
    def load_data(self, path):
        try:
            return np.load(path)
        except IOError:
            ## output operation error
            raise IOError(f"File not found: {path}")

    def data_for_pipeline(self):
        corr_path = self.path + "_correlation.npy"
        cov_path = self.path + "_covariance.npy"
        returns_path = self.path + "_returns.npy"

        correlation_matrix = self.load_data(corr_path)
        covariance_matrix = self.load_data(cov_path)
        full_returns = self.load_data(returns_path)

        if self.abs_corr:
            correlation_matrix = np.abs(correlation_matrix)

        if self.abs_cov:
            covaraiance_matrix = np.abs(covariance_matrix)

        if self._subprob_idx is not None:
            return (
                correlation_matrix[self._subprob_idx][:, self._subprob_idx],
                covaraiance_matrix[self._subprob_idx][:, self._subprob_idx],
                full_returns[self._subprob_idx],
            )

        return correlation_matrix, covariance_matrix, full_returns
