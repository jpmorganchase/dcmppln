###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2024: Amazon Web Services, Inc. - Contributions from JPMC
#
###############################################################################
from dcmppln.denoiser import Denoiser
from dcmppln.clustering import Clustering


def test_clustering_with_louvain_false(sample_input_data):
    denoiser = Denoiser(active=False)
    correlation_matrix, covariance_matrix, full_returns = sample_input_data
    C_2, _, _ = denoiser(correlation_matrix)
    clustering_1 = Clustering(active=False)
    partitions_1, _, _ = clustering_1(C_2)
    clustering_2 = Clustering(active=False)
    partitions_2, _, _ = clustering_2(C_2)
    print(partitions_1)
    assert (partitions_1 == partitions_2).all()


def test_clustering_with_louvain_true(sample_input_data):
    denoiser = Denoiser(active=False)
    correlation_matrix, covariance_matrix, full_returns = sample_input_data
    C_2, _, _ = denoiser(correlation_matrix)
    clustering_1 = Clustering(active=True)
    partitions_1, _, _ = clustering_1(C_2)
    clustering_2 = Clustering(active=True)
    partitions_2, _, _ = clustering_2(C_2)
    assert (partitions_1 == partitions_2).all()


def test_clustering_with_kmeans_false(sample_input_data):
    denoiser = Denoiser(active=False)
    correlation_matrix, covariance_matrix, full_returns = sample_input_data
    C_2, _, _ = denoiser(correlation_matrix)
    clustering_1 = Clustering(active=False, clustering_method="kmeans")
    partitions_1, _, _ = clustering_1(C_2)
    clustering_2 = Clustering(active=False, clustering_method="kmeans")
    partitions_2, _, _ = clustering_2(C_2)
    assert (partitions_1 == partitions_2).all()


def test_clustering_with_kmeans_true(sample_input_data):
    denoiser = Denoiser(active=False)
    correlation_matrix, covariance_matrix, full_returns = sample_input_data
    C_2, _, _ = denoiser(correlation_matrix)
    clustering_1 = Clustering(active=True, clustering_method="kmeans")
    partitions_1, _, _ = clustering_1(C_2)
    clustering_2 = Clustering(active=True, clustering_method="kmeans")
    partitions_2, _, _ = clustering_2(C_2)
    assert (partitions_1 == partitions_2).all()


def test_default_clustering_method():
    clustering = Clustering(active=True, clustering_method="xyz")
    assert clustering.method == "louvain"
