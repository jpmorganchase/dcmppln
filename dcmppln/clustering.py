###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2024: Amazon Web Services, Inc. - Contributions from JPMC
#
###############################################################################
from dcmppln.utils.utils import get_instance_non_private_attributes

from dcmppln.modularity_maximization.modularity_spectral_optimization import (
    modularity_spectral_optimization,
    modularity_spectral_threshold,
)
from dcmppln.utils.k_means import k_means_laplacian_threshold
from dcmppln.utils.utils import timeit
import numpy as np

clustering_methods = ("louvain", "kmeans")
default_clustering_method = "louvain"


class Clustering:

    def __init__(
        self,
        max_community_size=0,
        active=True,
        clustering_method="louvain",
        take_absolute_value=True,
        cluster_on_correlation=True,
    ):
        """
        Supported clustering methods: louvain, kmeans,
        take_absolute_value: flag to take absolute value of the matrix before clustering
        cluster_on_correlation: if True, cluster on (denoised) correlation else cluster on covariance

        :param max_community_size: threshold for the biggest community size, 0 drops any constraint
        :param active: bool to activate or deactivate the component, if deactivated other parameters are meaningless
        :param clustering_method: select method from ["louvain", "kmeans"], default to louvain
        :param take_absolute_value: bool to apply abs to matrix before sending it to the clustering method
        :param cluster_on_correlation: bool to cluster on correlation otherwise cluster on covariance
        """
        if clustering_method not in clustering_methods:
            clustering_method = "louvain"

        self.method = clustering_method
        self.active = active
        self.take_absolute_value = take_absolute_value
        self.max_community_size = max_community_size
        self.cluster_on_correlation = cluster_on_correlation

        self.params = get_instance_non_private_attributes(self)

    def _call_louvain(self, matrix):
        # Requires matrix with only non-negative terms
        if self.max_community_size == 0:
            partitions = modularity_spectral_optimization(matrix)
        else:
            partitions = modularity_spectral_threshold(
                matrix,
                threshold=self.max_community_size,
                return_communities=False,
            )
        return partitions

    def _call_kmeans(self, matrix):
        # Unrestricted cluster size default, dirty trick by using huge upper bound
        threshold_cluster_size = 1e9
        if self.max_community_size > 0:
            threshold_cluster_size = self.max_community_size
        return k_means_laplacian_threshold(
            matrix,
            k_values=None,
            low_threshold=2,
            up_threshold=threshold_cluster_size,
            num_eigenvectors=80,
        )[1]

    @timeit
    def cluster(self, matrix):
        if not self.active:
            return np.zeros(len(matrix))
        if self.take_absolute_value:
            matrix = np.abs(matrix)
        if self.method == "louvain":
            partitions = self._call_louvain(matrix)
        elif self.method == "kmeans":
            partitions = self._call_kmeans(matrix)
        else:
            raise NotImplemented
        print(
            "Max commmunity size:", np.max(np.unique(partitions, return_counts=True)[1])
        )
        return partitions

    # use this structure to ease profile readability
    def __call__(self, matrix):
        return self.cluster(matrix)
