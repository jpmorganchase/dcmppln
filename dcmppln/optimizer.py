###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2024: Amazon Web Services, Inc. - Contributions from JPMC
#
###############################################################################
from abc import ABC, abstractmethod

from dcmppln.optimization.cplex_utils import append_to_log
from dcmppln.optimization.portfolio_optimization_runtime import (
    add_logger,
    optimize_and_measure_time,
)
from dcmppln.utils.utils import get_instance_non_private_attributes

from dcmppln.CONSTANT import LOG_FOLDER, TIMEOUT, DEFAULT_MIP_GAP

from memory_profiler import profile
import numpy as np

#from dwave.samplers import SimulatedAnnealingSampler

# Gurobi imports
from gurobipy import GRB, Env, read
import os
import time


class Optimizer(ABC):
    """
    Abstract class of optimizer component
    Must overwrite and match the call function signature
    """

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def __call__(
        self,
        model,
        weights,
        unique_identifier=None,
        log_best_feasible=False,
    ):
        """
        Example signature and return overwriting methods needs to follow
        :param model: docplex model
        :param weights: docplex variable
        :param unique_identifier: hex id of the optimization
        :param log_best_feasible: log best feasible solutions
        :return:
        """
        optimal_weights = np.zeros(10)
        execution_time = 1
        processing_time = 1
        return optimal_weights, execution_time, processing_time


class CplexOptimizer(Optimizer):
    """
    Warning mip_margin=0 actually means default = 1-e4
    """

    def __init__(
        self,
        mip_margin=DEFAULT_MIP_GAP,
        continuous_variables_flag=False,
        timeout=TIMEOUT,
        additional_listeners=None,
        qtolin=-1,
    ):
        super().__init__()
        if additional_listeners is None:
            additional_listeners = list()
        self.mip_margin = mip_margin
        self.continuous_variables_flag = continuous_variables_flag
        self.timeout = timeout
        self.qtolin = qtolin
        self._additional_listeners = additional_listeners
        self.additional_listeners_name = [
            callback.__class__.__name__ for callback in self._additional_listeners
        ]
        self._cplex_parameters = {
            "preprocessing.qtolin": qtolin,
            "mip.tolerances.mipgap": mip_margin,
            "timelimit": TIMEOUT,
        }
        self.params = get_instance_non_private_attributes(self)

    def __call__(self, model, weights, unique_identifier=None, log_best_feasible=False):
        return self.optimize(
            model,
            weights,
            unique_identifier=unique_identifier,
            log_best_feasible=log_best_feasible,
        )

    def optimize(
        self,
        model,
        weights,
        unique_identifier=None,
        log_best_feasible=False,
    ):
        for key, val in self._cplex_parameters.items():
            # set timeout, mipgap, qtolin
            model.parameters.set_from_qualified_name(key, val)

        if log_best_feasible:
            log_path = add_logger(model, unique_identifier=unique_identifier)

        for listerner in self._additional_listeners:
            model.add_progress_listener(listerner)

        # Solve the model
        solution, execution_time, processing_time = optimize_and_measure_time(model)

        if (unique_identifier is not None) and log_best_feasible:
            msg = f"{solution.solve_details.time},{solution.objective_value},{solution.solve_details.best_bound}\n"
            append_to_log(log_path, msg)

        # Retrieve the optimal weights
        optimal_weights = [solution.get_value(w) for w in weights]
        # Retrieve the optimal weights

        return optimal_weights, execution_time, processing_time


class SimulatedAnnealingDwave(Optimizer):
    def __init__(self, **kwargs):
        """
        :param kwargs: hyperparamters to pass to SimulatedAnnealingSampler at sample time
        """
        super().__init__()
        self.optimizer_parameters = kwargs
        self.params = get_instance_non_private_attributes(self)

    def __call__(
        self, model, weights=None, unique_identifier=None, log_best_feasible=False
    ):
        """
        :param model: docplex model
        :param weights: docplex variable
        :param unique_identifier: hex id of the optimization
        :param log_best_feasible: bool log best feasible solutions
        :return:
        """
        # convert to QUBO with penalty
        cqm = dimod.lp.loads(model.lp_string)
        bqm, invert = dimod.cqm_to_bqm(cqm)
        sampler = SimulatedAnnealingSampler()

        start_time = time.perf_counter()
        tp_start = time.process_time()

        # Solve the model
        sampleset = sampler.sample(bqm, **self.optimizer_parameters)

        ## End time
        tp_stop = time.process_time()
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        processing_time = tp_stop - tp_start

        values_dict = sampleset.first.sample
        if cqm.check_feasible(sampleset.first.sample):
            best_sample_weights = [
                values_dict[f"weights_{i}"] for i in range(len(values_dict))
            ]
        else:
            # Attempted default value
            best_sample_weights = [0 for i in range(len(values_dict))]
        return best_sample_weights, execution_time, processing_time


class GurobiOptimizer(Optimizer):
    def __init__(self, **kwargs):
        """
        :param kwargs: hyperparamters to pass to model see https://www.gurobi.com/documentation/10.0/refman/parameters.html#sec:Parameters
        """
        super().__init__()
        self.optimizer_parameters = {
            "TimeLimit": TIMEOUT,
            "MIPGap": DEFAULT_MIP_GAP,
            "Threads": 1,
        }
        self.name = self.__class__.__name__
        self.params = get_instance_non_private_attributes(self)

    def __call__(
        self, model, weights=None, unique_identifier=None, log_best_feasible=False
    ):
        """
        :param model: docplex model
        :param weights: docplex variable
        :param unique_identifier: hex id of the optimization
        :param log_best_feasible: bool log best feasible solutions
        :return:
        """
        with open("temp.lp", "w") as f:
            f.write(model.lp_string.replace("*", " * "))

        m = read("temp.lp")
        for param, values in self.optimizer_parameters.items():
            m.setParam(param, values)

        start_time = time.perf_counter()
        tp_start = time.process_time()

        # Solve the model
        m.optimize()

        ## End time
        tp_stop = time.process_time()
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        processing_time = tp_stop - tp_start

        solution = np.array(
            [m.getVarByName(f"weights_{i}").X for i in range(m.NumVars)]
        )
        solution = solution + 0.0  # trick to avoid -0.0 values
        return solution, execution_time, processing_time


class DummyOptimizer(Optimizer):
    """
    Dummy optimizer to test other component, return only zeros
    """

    def __init__(self):
        super().__init__()
        self.params = get_instance_non_private_attributes(self)

    def __call__(
        self,
        model,
        weights,
        unique_identifier=None,
        log_best_feasible=False,
    ):
        """
        Example signature and return overwriting methods needs to follow
        :param model: docplex model
        :param weights: docplex variable
        :param unique_identifier: hex id of the optimization
        :param log_best_feasible: log best feasible solutions
        :return:
        """
        optimal_weights = np.zeros(len(weights))
        execution_time = 0
        processing_time = 0
        return optimal_weights, execution_time, processing_time
