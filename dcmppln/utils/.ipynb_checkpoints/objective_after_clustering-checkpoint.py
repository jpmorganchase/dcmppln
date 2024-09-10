###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2024: Amazon Web Services, Inc. - Contributions from JPMC
#
###############################################################################
from dcmppln.optimization.portfolio_optimization_runtime import *
from dcmppln.optimizer import Optimizer, CplexOptimizer
from dcmppln.utils.utils import (
    compute_objective_value,
    timeit,
)
from dcmppln.utils.portfolio_model import construct_markovitz_model


def get_model(returns,
              covariance_matrix,
              target_return=None,
              risk_factor=None,
              budget=None,
              model_name="Markowitz Portfolio Optimization",
              continuous_variables_flag=False):
    """
    Generator docplex model from input data
    target_return is not used in this version
    :returns docplex model and list of variables for each asset
    """
    n = len(returns)  # Number of assets

    if not budget:
        budget = n // 2

    ## q: Risk factor
    if not risk_factor:
        risk_factor = 0.5

    model, weights = construct_markovitz_model(
        returns,
        covariance_matrix,
        name=model_name,
        budget=budget,
        risk_factor=risk_factor,
        continuous_variables=continuous_variables_flag,
    )
    return model, weights


def split_cardinality_constraint(list_of_int, cardinality):
    """
    list_of_int: list of int of type [n1,...,nk]
    cardinality: int to match
    return list of int such that its sum = cardinality and relative weight of each int is somewhat preserved 
    """
    continuous_solution = np.array(list_of_int)*cardinality/np.sum(list_of_int)
    integer_solution = np.floor(continuous_solution).astype(int)
    increment = cardinality - np.sum(integer_solution)
   
    index_order_for_increase = np.argsort(list_of_int)[::-1]
    
    i = 0
    while increment > 0:
        integer_solution[index_order_for_increase[i]] += 1
        increment -= 1
        i += 1
    assert sum(integer_solution) == cardinality
    return integer_solution


def optimize_and_score(
    full_returns_,
    correlation_,
    partitions_: Union[np.ndarray, list],
    card_: int = 0,
    input_risk_factor=1,
    return_time=False,
    filtering_method=None,
    optimizer: Optimizer=None,
    rebalanced_risk_factor = None,
    unique_identifier_base = None,
    log_best_feasible = False
):
    """
    Each partition will be separated, filtered, solved separately, recombined, and score the combined solution
    Poor copy paste of optimize_base during transition
    :param full_returns_: returns vector
    :param correlation_: correlation matrix
    :param partitions_: partitions detected
    :param card_: cardinality constraint value
    :param input_risk_factor: base risk factor
    :param return_time: bool to return measured runtimes
    :param filtering_method: object of class Filtering
    :param unique_identifier_base: unique identifier uuid4 in hex
    :return:
    """
    if optimizer is None:
        optimizer = CplexOptimizer()
    if filtering_method is None:
        filtering_method = lambda x: x
    if rebalanced_risk_factor is None:
        rebalanced_risk_factor = input_risk_factor
        
    num_communities = len(np.unique(partitions_))
    unique_labels = np.unique(partitions_)
    
    if unique_identifier_base is None:
        unique_identifiers = [None] * len(unique_labels)
    else:
        unique_identifiers = [unique_identifier_base+f"_{int(i)}" for i in unique_labels]

    communities_ = []
    for i, label in enumerate(unique_labels):
        communities_.append(
            np.argwhere(partitions_ == label).reshape(
                -1,
            )
        )
        partitions_[partitions_ == label] = i

    sorted_communities = sorted(communities_, key=lambda x: x.shape[0], reverse=True)

    full_dim = len(partitions_)
    if not card_:
        card_ = full_dim // 2

    card_subproblems = split_cardinality_constraint(list(map(len, sorted_communities)), card_)

    assert card_subproblems.sum() == card_, "The sum of card doesn't match"

    weights_dict = {}
    process_times_dict = {}
    clock_time_dict = {}

    for i, card in enumerate(card_subproblems):
        comm_indices = sorted_communities[i]
        returns = full_returns_[comm_indices]

        community_correlation_matrix = filtering_method(
            correlation_[comm_indices][:, comm_indices]
        )

        model, weights = get_model(returns,
                                   community_correlation_matrix,
                                   budget=card,
                                   risk_factor=rebalanced_risk_factor,)
        (
            optimal_weights,
            clock_time,
            processing_time,
        ) = optimizer(
            model,
            weights,
            unique_identifier = unique_identifiers[i],
            log_best_feasible = log_best_feasible,
        )

        weights_dict[i] = optimal_weights
        process_times_dict[i] = processing_time
        clock_time_dict[i] = clock_time

    # Recombining solution to compare the objective later
    full_recombined_soln = np.zeros(full_dim, dtype=type(weights_dict[0]))

    for i, comm in enumerate(sorted_communities):
        full_recombined_soln[comm] = weights_dict[i]
    
    obj_after_clustering = compute_objective_value(
        full_returns_, correlation_, full_recombined_soln, risk_factor=input_risk_factor
    )

    print("Objective value with this comm detection algorithm", obj_after_clustering)
    print("Communities found at level 1: ", num_communities)

    if return_time:
        return obj_after_clustering, full_recombined_soln, process_times_dict, clock_time_dict
    else:
        return obj_after_clustering, full_recombined_soln


def objective_after_clustering(
    full_returns_,
    correlation_,
    partitions_: Union[np.ndarray, list],
    community_detection_alg: str = "",
    card_: int = 0,
    k_max: int = 10,
    input_risk_factor=0.5,
    return_time=False,
    filtering_method=None,
    threshold_mratio=0,
):
    """
    We do no q (risk apetite) rescaling here

    Parameters:
    partitions: List or array indicating labels for each node
    community_detection_alg: Type of community detection algorthm
    card_= Cardinality constraint for the full problem
    k_max: This argument is passed only for the spectral clustering algo

    Returns:
    objective value
    """

    assert community_detection_alg in (
        "mod_spectral_full",
        "mod_spectral_denoised",
        "louvain_full",
        "louvain_denoised",
        "q_louvain_full",
        "q_louvain_denoised",
        "nx_louvain_full",
        "nx_louvain_denoised",
        "spectral_clustering",
        "",
    ), "The algorithm type has not been found"
    if filtering_method is None:
        filtering_method = sparsification_covariance_no_diag_mean_ratio
    num_communities = len(np.unique(partitions_))
    unique_labels = np.unique(partitions_)

    communities_ = []
    for i, label in enumerate(unique_labels):
        communities_.append(
            np.argwhere(partitions_ == label).reshape(
                -1,
            )
        )
        partitions_[partitions_ == label] = i

    sorted_communities = sorted(communities_, key=lambda x: x.shape[0], reverse=True)
    weights_by_count = [
        sorted_communities[i].shape[0] for i in range(len(sorted_communities))
    ]
    weights_by_count = np.array(weights_by_count) / sum(weights_by_count)

    total_assets = len(partitions_)
    full_dim = total_assets
    if not card_:
        card_ = total_assets // 2

    card_subproblems = np.array([int(el * card_) for el in weights_by_count])
    ## Adding the extra cardinality constraint (if any) to the largest community
    card_subproblems[0] = card_ - card_subproblems[1:].sum()

    assert card_subproblems.sum() == card_, "The sum of card doesn't match"

    ### Running the cplex routine
    weights_dict = {}
    process_times_dict = {}
    for i, card in enumerate(card_subproblems):
        comm_indices = sorted_communities[i]
        returns = full_returns_[comm_indices]

        community_correlation_matrix = correlation_[comm_indices][:, comm_indices]

        #### THRESHOLD
        if threshold_mratio:
            community_correlation_matrix = filtering_method(
                community_correlation_matrix, threshold_mratio=threshold_mratio
            )
        (
            optimal_weights,
            execution_time,
            processing_time,
        ) = Boolean_markowitz_portfolio_optimization_time(
            returns,
            community_correlation_matrix,
            budget=card,
            risk_factor=input_risk_factor,
            return_sol=True,
        )

        weights_dict[i] = optimal_weights
        process_times_dict[i] = processing_time

    total_processing_time = sum(process_times_dict.values())

    ### Recombining solution to compare the objective later
    full_recombined_soln = np.zeros(full_dim, dtype=type(weights_dict[0]))

    for i, comm in enumerate(sorted_communities):
        full_recombined_soln[comm] = weights_dict[i]

    obj_after_clustering = compute_objective_value(
        full_returns_, correlation_, full_recombined_soln, risk_factor=input_risk_factor
    )

    print("Objective value with this comm detection algorithm", obj_after_clustering[0])
    print("Communities found at level 1: ", num_communities)

    if return_time == False:
        return obj_after_clustering, full_recombined_soln
    else:
        return obj_after_clustering, full_recombined_soln, total_processing_time
    





