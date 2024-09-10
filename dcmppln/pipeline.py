###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2024: Amazon Web Services, Inc. - Contributions from JPMC
#
###############################################################################
import numpy as np
import json
import time
import os
from uuid import uuid4

from dcmppln.optimizer import Optimizer, DummyOptimizer
from dcmppln.utils.risk_rebalance import rebalancing_risk_factor
from dcmppln.utils.objective_after_clustering import optimize_and_score

from dcmppln.denoiser import Denoiser
from dcmppln.clustering import Clustering
from dcmppln.filtering import Filtering

from dcmppln.CONSTANT import LOG_FOLDER, TIMEOUT


class Pipeline:
    """
    Structure to efficiently run a grid search on a list of Denoiser, Clustering, Filtering, Optimizer
    main method is run which is 4 for loops
    """

    def __init__(
        self,
        correlation_martix: np.array,
        covariance_matrix: np.array,
        returns: float,
        denoiser: Denoiser = Denoiser(active=False),
        cluster: Clustering = Clustering(active=False),
        filter_func: Filtering = Filtering(active=False),
        optimize_func: Optimizer = DummyOptimizer(),
    ):
        self.correlation_matrix = correlation_martix
        self.covariance_matrix = covariance_matrix
        self.returns = returns
        self.denoiser_func = denoiser
        self.cluster_func = cluster
        self.filter_func = filter_func
        self.optimize_func = optimize_func
        self.__state__ = dict()

    def run(
        self,
        risk_rebalancing=False,
        cluster_on_correlation=True,
        optimize_on_correlation=False,
        log_best_feasible=False,
        input_risk_factor=1,
        run_optimizer=False,
    ):
        """
        Method to run the grid search over the components (one inner for loop per component) and get score metrics

        :param risk_rebalancing: bool to use risk rebalancing technique
        :param cluster_on_correlation: bool to cluster on correlation else cluster on covariance
        :param optimize_on_correlation: bool to optimize on correlation else cluster on covariance
        :param log_best_feasible: bool to log best feasible solution (only available on CPLEX optimizer for now)
        :param cardinality_divider: set cardinality constraint to len(returns)//cardinality_divider
        :param input_risk_factor: risk factor in the markovitz optimization
        :return: dict with key being input and component parameters and values the score, run times etc...
        """

        matrix_to_cluster = (
            self.correlation_matrix
            if cluster_on_correlation
            else self.covariance_matrix
        )
        matrix_to_optimize = (
            self.correlation_matrix
            if optimize_on_correlation
            else self.covariance_matrix
        )

        C_2, denoise_process_time, denoise_time = self.denoiser_func(matrix_to_cluster)
        partitions, clustering_process_time, clustering_time = self.cluster_func(C_2)


        # -----------------------------------------------------------------------------------------

        unique_identifier_base = uuid4().hex
        if log_best_feasible:
            log_sub_dir = os.path.join(LOG_FOLDER, unique_identifier_base)
            os.mkdir(log_sub_dir)
        risk_factor = input_risk_factor
        if run_optimizer:
            return self.optimize_and_score(
                matrix_to_cluster,
                matrix_to_optimize,
                partitions,
                risk_factor,
                unique_identifier_base,
                log_best_feasible,
                cluster_on_correlation,
                optimize_on_correlation,
                denoise_process_time,
                clustering_process_time,
                denoise_time,
                clustering_time,
                risk_rebalancing=False,
                cardinality_divider=2,
            )
        return C_2, partitions

    def optimize_and_score(
        self,
        matrix_to_cluster,
        matrix_to_optimize,
        partitions,
        risk_factor,
        unique_identifier_base,
        log_best_feasible,
        cluster_on_correlation,
        optimize_on_correlation,
        denoise_process_time,
        clustering_process_time,
        denoise_time,
        clustering_time,
        risk_rebalancing=False,
        cardinality_divider=2,
    ):
        (
            obj_after_clustering,
            full_recombined_soln,
            opt_and_filter_process_time_dict,
            opt_and_filter_time_dict,
        ) = optimize_and_score(
            self.returns,
            matrix_to_optimize,
            partitions,
            card_=len(self.returns) // cardinality_divider,
            input_risk_factor=risk_factor,
            return_time=True,
            filtering_method=self.filter_func,
            optimizer=self.optimize_func,
            rebalanced_risk_factor=(
                rebalancing_risk_factor(
                    risk_factor, matrix_to_optimize, self.returns, partitions
                )
                if risk_rebalancing
                else None
            ),
            unique_identifier_base=unique_identifier_base,
            log_best_feasible=log_best_feasible,
        )
        opt_and_filter_process_time = sum(opt_and_filter_process_time_dict.values())
        opt_and_filter_time = sum(opt_and_filter_time_dict.values())

        key_parameters = {
            "denoiser": self.denoiser_func.params,
            "clustering": self.cluster_func.params,
            "filtering": self.filter_func.params,
            "optimizer": self.optimize_func.params,
            "problem_parameters": {
                "problem_size": len(self.returns),
                "cardinality_divider": cardinality_divider,
                "input_risk_factor": risk_factor,
                "cluster_on_correlation": cluster_on_correlation,
                "optimize_on_correlation": optimize_on_correlation,
                "risk_rebalancing": risk_rebalancing,
            },
        }
        self.__state__["key_parameters"] = key_parameters
        run_time = {
            "process_time": denoise_process_time
            + clustering_process_time
            + opt_and_filter_process_time,
            "clock_time": denoise_time + clustering_time + opt_and_filter_time,
            "process_times": [
                denoise_process_time,
                clustering_process_time,
                opt_and_filter_process_time,
            ],
            "clock_times": [
                denoise_time,
                clustering_time,
                opt_and_filter_time,
            ],
            "opt_process_times": opt_and_filter_process_time_dict,
            "opt_clock_times": opt_and_filter_time_dict,
        }
        self.__state__["run_time"] = run_time
        result = {
            "score": obj_after_clustering,
            "recombined_solution": full_recombined_soln,
            "run_time": run_time,
            "partitions": partitions,
            "max_community_size_found": np.max(
                np.unique(partitions, return_counts=True)[1]
            ),
            "size_of_partitions": np.unique(partitions, return_counts=True)[1],
            "unique_identifier_base": unique_identifier_base,
            "total_input_size": len(self.returns),
        }
        self.__state__["result"] = result

        return {
            "score": obj_after_clustering,
            "recombined_solution": full_recombined_soln,
        }
