###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2024: Amazon Web Services, Inc. - Contributions from JPMC
#
###############################################################################
import time
from typing import Union
import os
import numpy as np
from docplex.mp.model import Model

from dcmppln.utils.portfolio_model import construct_markovitz_model
from dcmppln.optimization.cplex_utils import BestFeasibleLogger, append_to_log

from dcmppln.CONSTANT import TIMEOUT,LOG_FOLDER, SAVE_LP_FILE


def optimize_and_measure_time(model):
    start_time = time.perf_counter()
    tp_start = time.process_time()

    # Solve the model
    solution = model.solve()

    ## End time
    tp_stop = time.process_time()
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    processing_time = tp_stop - tp_start
    return solution, execution_time, processing_time


def add_logger(model, unique_identifier=None):
    log_path = None
    if unique_identifier is not None:
        unique_identifier_base, partition_id = unique_identifier.split("_")
        log_path = os.path.join(LOG_FOLDER, unique_identifier_base, f"best_feasible_{unique_identifier}.csv")
        model.add_progress_listener(BestFeasibleLogger(log_path=log_path))
        if SAVE_LP_FILE:
            lp_path = os.path.join(LOG_FOLDER, unique_identifier_base, f"{unique_identifier}.lp")
            model.export_as_lp(lp_path)
    return log_path


def base_markovitz_portfolio_optimization_time(
    returns,
    covariance_matrix,
    target_return=None,
    risk_factor=None,
    budget=None,
    return_sol=False,
    mip_margin=None,
    model_name="Markowitz Portfolio Optimization",
    continuous_variables_flag=False,
    timeout=TIMEOUT,
    unique_identifier = None,
    log_best_feasible = False,
    additional_listerners = []
):
    """
    Trick if the budget is <0 we drop the constraint
    unique_identifier: hex+"_{partition_number}"
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

    if mip_margin is not None:
        model.parameters.mip.tolerances.mipgap.set(mip_margin)
        
    model.set_time_limit(timeout)
    if log_best_feasible:
        log_path = add_logger(model, unique_identifier=unique_identifier)
        
    for listerner in additional_listerners:
        model.add_progress_listener(listerner)
    
    # Solve the model
    solution, execution_time, processing_time = optimize_and_measure_time(model)
    
    if (unique_identifier is not None) and log_best_feasible:
        msg = f"{solution.solve_details.time},{solution.objective_value},{solution.solve_details.best_bound}\n"
        append_to_log(log_path, msg)
    
    # Retrieve the optimal weights
    optimal_weights = [solution.get_value(w) for w in weights]
   
    if return_sol:
        return optimal_weights, execution_time, processing_time
    else:
        return execution_time, processing_time


def continuous_markowitz_portfolio_optimization_time(
    returns,
    covariance_matrix,
    target_return=None,
    risk_factor=None,
    budget=None,
    return_sol=False,
    mip_margin=None,
):
    return base_markovitz_portfolio_optimization_time(
        returns,
        covariance_matrix,
        target_return=target_return,
        risk_factor=risk_factor,
        budget=budget,
        return_sol=return_sol,
        mip_margin=mip_margin,
        model_name="continuous Markowitz Portfolio Optimization",
        continuous_variables_flag=True,
    )


###############################################################################################


def Boolean_markowitz_portfolio_optimization_time(
    returns,
    covariance_matrix,
    target_return=None,
    risk_factor=None,
    budget=None,
    return_sol=False,
    mip_margin=None,
    model_name="",
):
    return base_markovitz_portfolio_optimization_time(
        returns,
        covariance_matrix,
        target_return=target_return,
        risk_factor=risk_factor,
        budget=budget,
        return_sol=return_sol,
        mip_margin=mip_margin,
        model_name=model_name,
        continuous_variables_flag=False,
    )
