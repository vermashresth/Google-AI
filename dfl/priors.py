import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS, KMeans, SpectralClustering

from dfl.config import dim_dict

def kmeans_missing(X, n_clusters, algo, max_iter=10):
    missing = ~np.isfinite(X)
    mu = np.nanmean(X, 0, keepdims=1)
    X_hat = np.where(missing, mu, X)

    prev_labels = None
    for i in range(max_iter):
        if algo == 'optics':
            cls = OPTICS(min_samples=4)
        elif algo == 'kmeans':
            cls = KMeans(n_clusters, random_state=0)
        elif algo == 'spectral':
            cls = SpectralClustering(n_clusters, random_state=0)

        labels = cls.fit_predict(X_hat)

        if algo == 'kmeans':
            centroids = cls.cluster_centers_
        else:
            if algo == 'optics':
                labels = labels + 1
            unique_labels = len(set(labels))
            centroids = []
            for i in range(unique_labels):
                idxes = np.where(labels == i)[0]
                centroids.append(np.mean(X_hat[idxes], axis=0))
            centroids = np.array(centroids)

        X_hat[missing] = centroids[labels][missing]

        if i > 0 and np.all(labels == prev_labels):
            break

        prev_labels = labels

    return labels, centroids, X_hat, cls, len(set(labels)), i

def clustered_prior(traj, n_clusters, policy_id):
    n_states, n_actions = 2, 2
    n_benefs = traj.shape[-1]

    indiv_emp_T = indiv_emp_T_calculator(traj, range(n_benefs), policy_id)
    transitions_df = pd.DataFrame(indiv_emp_T.reshape(n_benefs, n_states*n_actions*n_states))
    prob_cols = transitions_df.columns
    pass_to_clustering_cols = [0,4] # [[0, 0, 0], [1, 0, 0]]
    labels, centroids, _, cls, num_clusters, max_iters = kmeans_missing(transitions_df[pass_to_clustering_cols],
                                                                        n_clusters, 'kmeans', max_iter=100)
    transitions_df['cluster'] = labels

    all_probs = []
    for i in range(n_clusters):
        cluster_beneficiaries = transitions_df[transitions_df['cluster'] == i]
        cluster_beneficiaries_ids = np.arange(n_benefs)[transitions_df['cluster'] == i]
        probs = grp_emp_T_calculator(traj, cluster_beneficiaries_ids, policy_id)
        all_probs.append(probs)
    all_probs = np.array(all_probs)
    for s in range(n_states):
        for a in range(n_actions):
            nans = np.isnan(all_probs[:, s, a, 0])
            all_probs[nans, s, a, 0] = all_probs[~nans, s, a, 0].mean()
            all_probs[nans, s, a, 1] = 1 - all_probs[nans, s, a, 0]
    
    return all_probs[transitions_df['cluster'].values]

def indiv_emp_T_calculator(traj, benef_ids, policy_id):
    n_states, n_actions = 2, 2

    transition_prob_list = []
    default_prob = np.full((n_states, n_actions, n_states), np.nan)

    for benef_id in benef_ids:
        transitions_df = trajToGrpFreq(traj, [benef_id], policy_id)
        transition_prob = default_prob.copy()

        transition_prob = freqToT(transitions_df, transition_prob, n_states, n_actions)
        transition_prob_list.append(transition_prob.reshape(1, n_states, 2, n_states))

    emp_T_data = np.concatenate(transition_prob_list, axis=0)
    return emp_T_data

def grp_emp_T_calculator(traj, benef_ids, policy_id):
    n_states, n_actions = 2, 2

    default_prob = np.full((n_states, n_actions, n_states), np.nan)

    transitions_df = trajToGrpFreq(traj, benef_ids, policy_id)
    transition_prob = default_prob.copy()

    emp_T_data = freqToT(transitions_df, transition_prob, n_states, n_actions)
    return emp_T_data

def freqToT(transitions_df, default_prob, n_states, n_actions):
    transition_prob = default_prob.copy()
    for s in range(n_states):

            for a in range(n_actions):
                s_a = transitions_df[(transitions_df['s']==s) &
                                        (transitions_df['a']==a)
                                    ]
                s_a_count = s_a.shape[0]
                for s_prime in range(n_states):
                    s_a_s_prime = s_a[(s_a['s_prime']==s_prime)
                                                ]
                    s_a_s_prime_count = s_a_s_prime.shape[0]
                    if s_a_count >= 1:
                        transition_prob[s,a,s_prime] = s_a_s_prime_count / s_a_count
    return transition_prob

def trajToGrpFreq(traj, benef_ids, policy_id):
    benef_ci_traj = traj[:, # trial index
            policy_id, # policy index
            :, # time index
            :, # tuple dimension
            benef_ids # benef index
        ]
    benef_ci_traj = benef_ci_traj.reshape(-1, benef_ci_traj.shape[2], benef_ci_traj.shape[3])
    s_traj_c =  benef_ci_traj[:, :-1, dim_dict['state']]
    a_traj_c =  benef_ci_traj[:, :-1, dim_dict['action']]
    r_traj_c =  benef_ci_traj[:, :-1, dim_dict['reward']]
    s_prime_traj_c =  benef_ci_traj[:, :-1, dim_dict['next_state']]
    a_prime_traj_c = benef_ci_traj[:, 1:, dim_dict['action']]

    transitions_df = pd.DataFrame(columns = ['s', 's_prime', 'r', 'a', 'a_prime'])

    for s_traj, a_traj, r_traj, s_prime_traj, a_prime_traj in \
                        zip(s_traj_c, a_traj_c, r_traj_c, s_prime_traj_c, a_prime_traj_c):
        transitions_df = transitions_df.append(pd.DataFrame({'s':s_traj,
                                's_prime': s_prime_traj,
                                'r': r_traj,
                                'a': a_traj,
                                'a_prime': a_prime_traj}), ignore_index=True)

    return transitions_df