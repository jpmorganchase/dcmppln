###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2024: Amazon Web Services, Inc. - Contributions from JPMC
#
###############################################################################
from docplex.mp.model import Model

def construct_markovitz_model(
    returns,
    covariance_matrix,
    budget=None,
    risk_factor=None,
    name="Markowitz Portfolio Optimization",
    continuous_variables=False,
):
    """
    Trick if the budget is <0 we drop the constraint
    """
    n = len(returns)  # Number of assets
    if budget is None:
        budget = n // 2
    # q: Risk factor
    if risk_factor is None:
        risk_factor = 0.5

    model = Model(name)
    if continuous_variables:
        weights = model.continuous_var_list(n, lb=0.0, ub=1.0, name="weights")
    else:
        weights = model.binary_var_list(n, name="weights")
    portfolio_risk = model.sum(
        risk_factor * weights[i] * covariance_matrix[i, j] * weights[j]
        for i in range(n)
        for j in range(n)
    ) - model.sum(returns[i] * weights[i] for i in range(n))
    
    if budget >= 0:
        model.add_constraint(model.sum(weights[i] for i in range(n)) == budget)
    model.minimize(portfolio_risk)
    return model, weights
