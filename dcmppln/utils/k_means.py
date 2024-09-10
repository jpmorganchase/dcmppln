###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2024: Amazon Web Services, Inc. - Contributions from JPMC
#
###############################################################################
from scipy.linalg import eigh
from sklearn.cluster import KMeans
import numpy as np

def silhouette_score(X, labels):
    """
    X: data matrix
    """
    score = 0
    for i in range(len(X)):
        a = 0
        b = 0
        for j in range(len(X)):
            if labels[i] == labels[j]:
                a += np.linalg.norm(X[i] - X[j])
            else:
                b += np.linalg.norm(X[i] - X[j])
        if a < b:
            score += 1 - a / b
        else:
            score += b / a - 1

    return score / len(X)


#### Implementation for Calinski-Harabaz Score.
def calinski_harabaz_score(X, labels):
    """
    Variance ratio criterion.
    Between cluster dispersion divided by within cluster dispersion.
    """
    n_samples = len(X)
    n_labels = len(np.unique(labels))
    if n_labels == 1:
        return 1.0
    mean = np.mean(X, axis=0)
    B, W = 0, 0
    for k in range(n_labels):
        cluster_k = X[labels == k]
        mean_k = np.mean(cluster_k, axis=0)
        B += len(cluster_k) * np.sum((mean_k - mean) ** 2)
        W += np.sum((cluster_k - mean_k) ** 2)
    return (B / W) * (n_samples - n_labels) / (n_labels - 1)


def correlation_to_distance_linear(correlation_matrix):
    num_rows, num_cols = correlation_matrix.shape
    distance_matrix = np.empty((num_rows, num_cols))

    for i in range(num_rows):
        for j in range(num_cols):
            if i == j:
                distance_matrix[i, j] = 0
            else:
                distance_matrix[i, j] = (3 - correlation_matrix[i, j]) / 2

    return distance_matrix


def correlation_to_distance(correlation_matrix):
    num_rows, num_cols = correlation_matrix.shape
    distance_matrix = np.empty((num_rows, num_cols))

    for i in range(num_rows):
        for j in range(num_cols):
            distance_matrix[i, j] = np.sqrt(2 * (1 - correlation_matrix[i, j]))

    return distance_matrix


def k_means_sweep(C, k_values=None, corr=True, distance_type="non_linear"):
    if corr and distance_type == "non_linear":
        D = np.sqrt(2 * (1 - C))
    elif corr and distance_type == "linear":
        D = correlation_to_distance_linear(C)

    else:
        D = C

    if k_values is None:  # If no k values are specified, sweep over a range of k values
        k_values = range(1, len(D))

    best_score = -1
    best_k = -1
    best_clustering = None

    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=5) 
        clustering = kmeans.fit_predict(D)
        score = calinski_harabaz_score(D, clustering)
        if score > best_score:
            best_score = score
            best_k = k
            best_clustering = clustering
    return best_k, best_clustering  ##


######### This is with threshold or contraints on cluster sizes ################
def k_means_sweep_threshold(
    C,
    k_values=None,
    corr=True,
    low_threshold=2,
    up_threshold=30,
    distance_type="non_linear",
):
    if corr and distance_type == "linear":
        D = correlation_to_distance_linear(C)

    if corr and distance_type == "non_linear":
        D = correlation_to_distance_linear(C)

    else:
        D = C

    best_score = -1
    best_k = -1
    best_clustering = None

    if k_values is None:  
        k_values = range(1, len(D))

    for k in k_values:
        kmeans = KMeans(
            n_clusters=k, n_init=5
        ) 
        clustering = kmeans.fit_predict(D)

        # Calculate the size of each cluster
        cluster_sizes = np.bincount(clustering)

        # Check if all clusters satisfy the constraints
        if np.all(cluster_sizes >= low_threshold) and np.all(
            cluster_sizes <= up_threshold
        ):
            score = calinski_harabaz_score(D, clustering)
            if score > best_score:
                best_score = score
                best_k = k
                best_clustering = clustering

    return best_k, best_clustering


######### This is with threshold or contraints on cluster sizes ################
def k_means_laplacian_threshold(
    C, k_values=None, low_threshold=2, up_threshold=30, num_eigenvectors=40
):
    cov_matrix = C.copy()
    # Step 2: Compute Laplacian Matrix
    n = cov_matrix.shape[0]
    D = np.diag(np.sum(cov_matrix, axis=1))
    laplacian_matrix = D - cov_matrix

    # Step 3: Compute Eigenvectors
    # Number of smallest eigenvectors to consider
    eigenvalues, eigenvectors = eigh(laplacian_matrix, D, type=1)

    sorted_indices = np.argsort(eigenvalues)
    eigenvectors[:, sorted_indices]
    eigenvectors = eigenvectors.copy()

    # Step 4: Perform Clustering

    best_score = -1
    best_k = -1
    best_clustering = None

    if k_values is None:  # If no k values are specified, sweep over a range of k values
        k_values = range(1, n)

    found_soln = False

    for k in k_values:
        for num_eigenvectors in np.arange(2, n):
            # print(num_eigenvectors)
            D = eigenvectors[:, 1:num_eigenvectors]

            kmeans = KMeans(
                n_clusters=k, n_init=5
            )
            clustering = kmeans.fit_predict(D)

            # Calculate the size of each cluster
            cluster_sizes = np.bincount(clustering)

        
            # Check if all clusters satisfy the constraints
            if np.all(cluster_sizes <= up_threshold):
                found_soln = True
                best_clustering = clustering
                break

            if found_soln:
                break

        if found_soln:
            break


    return best_k, best_clustering

