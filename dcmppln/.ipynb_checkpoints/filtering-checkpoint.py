###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2024: Amazon Web Services, Inc. - Contributions from JPMC
#
###############################################################################
import numpy as np
from dcmppln.utils.utils import get_instance_non_private_attributes
from dcmppln.utils.utils import threshold_matrix, thresholding_not_touching_diag, sparsification


class Filtering:
    def __init__(
        self,
        threshold_flag=False,
        take_average_threshold=True,
        allow_to_modify_diag=False,
        remove_diag_from_average=True,
        threshold_value=0.0,
        active=True,
    ):
        """
        Be careful, threshold_flag set to true will change the matrix to a binary matrix what ever the threshold value is
        :param threshold_flag:
        :param take_average_threshold: bool to use the threshold value against the average
        :param allow_to_modify_diag: bool to allow to modify the diagonal
        :param remove_diag_from_average: bool to remove the diagonal value when estimating the average
        :param threshold_value: float, value to use as a threshold
        :param active: bool to activate or deactivate the component, if deactivated other parameters are meaningless
        """
        self.active = active
        self.take_average_threshold = take_average_threshold
        self.allow_to_modify_diag = allow_to_modify_diag
        self.remove_diag_from_mean = remove_diag_from_average
        self.threshold_flag = threshold_flag
        self.threshold_value = threshold_value

        self.params = get_instance_non_private_attributes(self)

    def filtering(self, matrix):
        if not self.active:
            return matrix
        if self.take_average_threshold:
            if self.remove_diag_from_mean:
                diag_matrix = np.diag(np.diag(matrix))  # diagonal matrix only
                mean_value = np.mean(matrix - diag_matrix)
            else:
                mean_value = np.mean(matrix)
            self.threshold_value *= mean_value

        if not self.threshold_flag:
            return_matrix = sparsification(
                matrix,
                threshold=self.threshold_value,
                allow_to_modify_diag=self.allow_to_modify_diag,
            )
        else:
            if not self.allow_to_modify_diag:
                return_matrix = thresholding_not_touching_diag(
                    matrix, threshold=self.threshold_value
                )
            else:
                return_matrix = threshold_matrix(matrix, threshold=self.threshold_value)
        return return_matrix

    # use this structure to ease profile readability
    def __call__(self, matrix):
        return self.filtering(matrix)
