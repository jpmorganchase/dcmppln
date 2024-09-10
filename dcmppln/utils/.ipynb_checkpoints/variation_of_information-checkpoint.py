###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2024: Amazon Web Services, Inc. - Contributions from JPMC
#
###############################################################################
""" 
We quantitatively measured the performance of our
method in terms of a metric known as Variation of Information (V I) [52, 53], which measures the entropy
difference between two partitions of the same network,
providing a rigorous way for us to quantify the similarity between the ‘true’ partition and the one identified
by our method.
"""

import numpy as np


def variation_of_information(true_partition, partition):
    """
    partition is a list of integers, where each integer represents a community for the corresponding node
    """
    unique_labels = set(true_partition)
    true_partition_dict = dict(zip(unique_labels, range(len(unique_labels))))
    true_partition = [
        true_partition_dict[true_partition[i]] for i in range(len(true_partition))
    ]

    unique_labels = set(partition)
    partition_dict = dict(zip(unique_labels, range(len(unique_labels))))
    partition = [partition_dict[partition[i]] for i in range(len(partition))]

    true_partition = list(true_partition)
    partition = list(partition)

    n_c1 = len(set(true_partition))
   
    true_probability = [0.0 for i in range(n_c1)]
    for i, cluster_label in enumerate(set(true_partition)):
        # count number of nodes in each cluster
        true_probability[i] = true_partition.count(cluster_label) / len(true_partition)

    entropy_true = 0.0
    for i in range(len(true_probability)):
        entropy_true += -1.0 * true_probability[i] * np.log2(true_probability[i])

    n_c2 = len(set(partition))
    partition_probability = [0.0 for i in range(n_c2)]

    for i, cluster_label in enumerate(set(partition)):
        partition_probability[i] = partition.count(cluster_label) / len(partition)

    # Entropy associated with partition
    entropy_partition = 0.0
    for i in range(len(partition_probability)):
        entropy_partition += (
            -1.0 * partition_probability[i] * np.log2(partition_probability[i])
        )

    # Compute mutual information between two clusterings
    # mutual information is a symmetric measure
    mutual_information = 0.0
    for i in range(n_c1):
        for j in range(n_c2):
            # count number of nodes in each cluster
            n_ij = 0
            for k in range(len(true_partition)):
                if true_partition[k] == i and partition[k] == j:
                    n_ij += 1
            if n_ij > 0:
                mutual_information += (
                    n_ij
                    / len(true_partition)
                    * np.log2(
                        (n_ij / len(true_partition))
                        / (true_probability[i] * partition_probability[j])
                    )
                )

    # Compute variation of information
    variation_of_information = (
        entropy_true + entropy_partition - 2.0 * mutual_information
    )

    return variation_of_information

