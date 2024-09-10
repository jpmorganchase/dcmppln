###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2024: Amazon Web Services, Inc. - Contributions from JPMC
#
###############################################################################
import numpy as np

def rebalancing_risk_factor(risk_factor, full_covariance, full_avg_return, communities):

    #### we have the full covariance matrix and the full average return vector, and the communities

    balanced_risk_factor = risk_factor

    ### calculate norm of the covariance matrix, and the norm of the average return vector
    cov_norm = np.linalg.norm(full_covariance)
    avg_return_norm = np.linalg.norm(full_avg_return)

    ### calculate the number of the masked covariance that only takes blocks corresponding to the communities
    num_communities = len(communities)

    if type(communities) == dict:
        pass
    else:
        communities = {k: v for k, v in enumerate(communities)}

    community_cov_norm = 0
    community_avg_return_norm = 0
    
    for community_idx, community in communities.items():
        community = np.array([community]).astype(int)
        community_covariance = full_covariance[np.ix_(community, community)]
        community_avg_return = full_avg_return[community]

        ### calculate the norm of the covariance matrix and the norm of the average return vector
        community_cov_norm += np.linalg.norm(community_covariance)
        community_avg_return_norm += np.linalg.norm(community_avg_return)
        

    ### add the ratio of the norms to the balanced risk factor
    balanced_risk_factor *= (cov_norm/community_cov_norm)*(community_avg_return_norm/avg_return_norm)   

    return balanced_risk_factor
