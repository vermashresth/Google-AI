import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

def kmeans_missing(X, n_clusters, max_iter=10,random_state=0):
    missing = ~np.isfinite(X)
    mu = np.nanmean(X, 0, keepdims=1)
    X_hat = np.where(missing, mu, X)

    prev_labels = None
    for i in range(max_iter):
        cls = KMeans(n_clusters, random_state=random_state)


        labels = cls.fit_predict(X_hat)

        centroids = cls.cluster_centers_


        X_hat[missing] = centroids[labels][missing]

        if i > 0 and np.all(labels == prev_labels):
            break

        prev_labels = labels

    return labels, centroids, X_hat, cls, len(set(labels)), i

def get_transition_probabilities(beneficiaries, transitions, min_support=2):
    transitions = transitions[transitions['user_id'].isin(beneficiaries)]

    i_transitions = transitions[transitions['action']=='Intervention']
    n_i_transitions = transitions[transitions['action']=='No Intervention']

    i_L = i_transitions[i_transitions['pre-action state']=="L"]
    i_H = i_transitions[i_transitions['pre-action state']=="H"]

    i_L_L = i_L[i_L['post-action state']=="L"]
    i_L_H = i_L[i_L['post-action state']=="H"]

    i_H_L = i_H[i_H['post-action state']=="L"]
    i_H_H = i_H[i_H['post-action state']=="H"]

    n_i_L = n_i_transitions[n_i_transitions['pre-action state']=="L"]
    n_i_H = n_i_transitions[n_i_transitions['pre-action state']=="H"]

    n_i_L_L = n_i_L[n_i_L['post-action state']=="L"]
    n_i_L_H = n_i_L[n_i_L['post-action state']=="H"]

    n_i_H_L = n_i_H[n_i_H['post-action state']=="L"]
    n_i_H_H = n_i_H[n_i_H['post-action state']=="H"]

    transition_probabilities = dict()
    if i_L.shape[0] >= min_support:
        transition_probabilities['P(L, I, L)'] = i_L_L.shape[0] / i_L.shape[0]
        transition_probabilities['P(L, I, H)'] = i_L_H.shape[0] / i_L.shape[0]
    else:
        transition_probabilities['P(L, I, L)'] = np.nan
        transition_probabilities['P(L, I, H)'] = np.nan

    if i_H.shape[0] >= min_support:
        transition_probabilities['P(H, I, L)'] = i_H_L.shape[0] / i_H.shape[0]
        transition_probabilities['P(H, I, H)'] = i_H_H.shape[0] / i_H.shape[0]
    else:
        transition_probabilities['P(H, I, L)'] = np.nan
        transition_probabilities['P(H, I, H)'] = np.nan
    
    if n_i_L.shape[0] >= min_support:
        transition_probabilities['P(L, N, L)'] = n_i_L_L.shape[0] / n_i_L.shape[0]
        transition_probabilities['P(L, N, H)'] = n_i_L_H.shape[0] / n_i_L.shape[0]
    else:
        transition_probabilities['P(L, N, L)'] = np.nan
        transition_probabilities['P(L, N, H)'] = np.nan

    if n_i_H.shape[0] >= min_support:
        transition_probabilities['P(H, N, L)'] = n_i_H_L.shape[0] / n_i_H.shape[0]
        transition_probabilities['P(H, N, H)'] = n_i_H_H.shape[0] / n_i_H.shape[0]
    else:
        transition_probabilities['P(H, N, L)'] = np.nan
        transition_probabilities['P(H, N, H)'] = np.nan

    return transition_probabilities, {'P(L, I, L)': i_L_L.shape[0], 'P(L, I, H)': i_L_H.shape[0], 'P(H, I, L)': i_H_L.shape[0], 'P(H, I, H)': i_H_H.shape[0], 'P(L, N, L)': n_i_L_L.shape[0], 'P(L, N, H)': n_i_L_H.shape[0], 'P(H, N, L)': n_i_H_L.shape[0], 'P(H, N, H)': n_i_H_H.shape[0]}

def prepare_clustering_X(CLUSTERING_INPUT, feat_cols, pilot_feat_df, warmup_all_probs, after_warmup_all_probs, all_probs):
    if CLUSTERING_INPUT=='WARMUP_ONLY':
        X = pd.concat([pilot_feat_df.reset_index()[feat_cols],
                    warmup_all_probs[['P(L, N, L)', 'P(H, N, L)', 'P(L, I, L)', 'P(H, I, L)']]], axis=1).fillna(-1)
        warmup_cols = ['P(L, N, L)', 'P(H, N, L)']
        scaler = preprocessing.StandardScaler().fit(X[warmup_cols])
        X[warmup_cols] = scaler.transform(X[warmup_cols])
        X = X[warmup_cols]
        cluster_feat_cols = warmup_cols
    elif CLUSTERING_INPUT=='FEATURES_WARMUP':
        X = pd.concat([pilot_feat_df.reset_index()[feat_cols],
                    warmup_all_probs[['P(L, N, L)', 'P(H, N, L)', 'P(L, I, L)', 'P(H, I, L)']]], axis=1).fillna(-1)
        warmup_cols = ['P(L, N, L)', 'P(H, N, L)', 'P(L, I, L)', 'P(H, I, L)']
        scaler = preprocessing.StandardScaler().fit(X[warmup_cols])
        X[warmup_cols] = scaler.transform(X[warmup_cols])
        cluster_feat_cols = feat_cols + warmup_cols
    elif CLUSTERING_INPUT=='FEATURES_ONLY':
        X = pilot_feat_df.reset_index()[feat_cols]
        cluster_feat_cols = feat_cols
    elif CLUSTERING_INPUT=='FEATURES_PASSIVE':
        X = pd.concat([pilot_feat_df.reset_index()[feat_cols],
                    after_warmup_all_probs[['P(L, N, L)', 'P(H, N, L)']]], axis=1)
        warmup_cols = ['P(L, N, L)', 'P(H, N, L)', 'P(L, I, L)', 'P(H, I, L)']
        scaler = preprocessing.StandardScaler().fit(X[warmup_cols])
        X[warmup_cols] = scaler.transform(X[warmup_cols])
        cluster_feat_cols = feat_cols + ['P(L, N, L)', 'P(H, N, L)']
    elif CLUSTERING_INPUT=='PASSIVE_PROB':
        X = after_warmup_all_probs[['P(L, N, L)', 'P(H, N, L)']]
        cluster_feat_cols = ['P(L, N, L)', 'P(H, N, L)']
    elif CLUSTERING_INPUT=='PASSIVE_PROB_ALL':
        X = all_probs[['P(L, N, L)', 'P(H, N, L)']]
        cluster_feat_cols = ['P(L, N, L)', 'P(H, N, L)']
    else:
        raise NotImplementedError
    X['user_id'] = pilot_feat_df.index
    
    return X, cluster_feat_cols

def get_imputed_cluster_TP_estimate(X, pilot_trans_df, N_CLUSTERS):
    '''
    X: Dataframe with user_id and correponding fitted_cluster
    '''

    # Recalculate grouped TP using assigned clusters
    cols = [
            "P(L, I, L)", "P(L, I, H)", "P(H, I, L)", "P(H, I, H)", "P(L, N, L)", "P(L, N, H)", "P(H, N, L)", "P(H, N, H)", 
        ]
    cluster_transition_probabilities = pd.DataFrame(columns=['count', 'cluster'] + cols)

    for i in range(N_CLUSTERS):
        cluster_beneficiaries = X[X['fitted_cluster'] == i]
        cluster_b_user_ids = cluster_beneficiaries['user_id']
        probs, _ = get_transition_probabilities(cluster_b_user_ids, pilot_trans_df, min_support=3)
        probs['cluster'] = i
        probs['count'] = len(cluster_b_user_ids)
        cluster_transition_probabilities = cluster_transition_probabilities.append(probs, ignore_index=True)

    # Imputation to Calculate Missing Active and Passive transition probabilities
    for i in range(4, 8):
        if i % 2 == 0:
            curr_col = cols[i]
            cluster_transition_probabilities[curr_col] = cluster_transition_probabilities[curr_col].fillna(cluster_transition_probabilities[curr_col].mean())
        else:
            cluster_transition_probabilities[cols[i]] = 1 - cluster_transition_probabilities[cols[i - 1]]
    for i in range(4):
        if i % 2 == 0:
            curr_col = cols[i]
            est_col = cols[i + 4]
            diff_col = cluster_transition_probabilities[curr_col] - cluster_transition_probabilities[est_col]
            diff_col = diff_col.fillna(diff_col.mean())
            cluster_transition_probabilities[curr_col] = diff_col + cluster_transition_probabilities[est_col]
            cluster_transition_probabilities.loc[cluster_transition_probabilities[curr_col] >= 1, curr_col] = 1
        else:
            cluster_transition_probabilities[cols[i]] = 1 - cluster_transition_probabilities[cols[i - 1]]
    return cluster_transition_probabilities
