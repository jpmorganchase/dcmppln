###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2024: Amazon Web Services, Inc. - Contributions from JPMC
#
###############################################################################
"""
Note: This has to be run outside of py39 since CPLEX installation needs it
"""

from docplex.mp.model import Model

from docplex.cp.model import CpoModel

import time
import numpy as np

import networkx as nx


def maximize_modularity(adjacency_matrix):
    """Doctest
    >>> G = nx.barbell_graph(3, 0)
    >>> adjacency_matrix = nx.to_numpy_array(G)
    >>> expected_modularity = 0.357142
    >>> best_partition, best_modularity = maximize_modularity(adjacency_matrix)
    >>> assert np.isclose(best_modularity, expected_modularity), f"Modularity test failed. Expected: {expected_modularity}, Actual: {best_modularity}"
    >>> assert (np.allclose(best_partition, np.array([0, 0,0, 1,1, 1])) or np.allclose(best_partition, np.array([1,1,1, 0,0,0]) )  )
    """

    """
    Find the best binary string that maximizes the modularity score.

    Parameters:
    adjacency_matrix (numpy.ndarray): The adjacency matrix representing the network.

    Returns:
    best_partition (list): The best partition that maximizes the modularity score.
    best_modularity (float): The modularity score corresponding to the best partition.
    """

    n = len(adjacency_matrix)
    m = np.sum(adjacency_matrix) / 2.0

    # Create the docplex model
    model = Model(name="ModularityMaximization")

    # Create binary variables for each node in the partition
    partition_vars = model.binary_var_list(n, name="partition")

    # Calculate the modularity score
    modularity_score = model.sum(
        (
            adjacency_matrix[i][j]
            - (np.sum(adjacency_matrix[i]) * np.sum(adjacency_matrix[j])) / (2.0 * m)
        )
        * (partition_vars[i] * partition_vars[j])
        for i in range(n)
        for j in range(n)
    ) / (m)

    # Maximize the modularity score
    model.maximize(modularity_score)

    # Solve the model
    if model.solve():
        # Retrieve the best partition and modularity score
        best_partition = np.array(
            [int(partition.solution_value) for partition in partition_vars]
        )
        best_modularity = model.objective_value
        return best_partition, best_modularity
    else:
        print("No feasible solution found.")
        return None


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    print("Passed the test")


