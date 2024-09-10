###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2024: Amazon Web Services, Inc. - Contributions from JPMC
#
###############################################################################
import os
from dcmppln.pipeline import Pipeline
from dcmppln.optimizer import GurobiOptimizer
import gurobipy as gp
from gurobipy import GRB

def test_pipeline_with_gurobi(sample_input_data):
    correlation_matrix, covariance_matrix, full_returns = sample_input_data
    p = Pipeline(
        correlation_matrix,
        covariance_matrix,
        full_returns,
        optimize_func=GurobiOptimizer(),
    )
    result = p.run(run_optimizer=True)
    print(result)
    assert result["score"] == 0.13237755898088116
