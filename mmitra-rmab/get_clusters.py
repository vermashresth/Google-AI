import os
import sys
import numpy as np
from numpy.lib.arraysetops import unique
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from copy import deepcopy

from datetime import datetime
from pprint import pprint
from tqdm import tqdm
import ipdb
import pickle

plt.style.use("seaborn")
np.random.seed(1)

from training.utils import load_obj, save_obj
from training.data import load_data
from training.dataset import _preprocess_call_data, preprocess_and_make_dataset

from sklearn.cluster import KMeans, OPTICS, SpectralClustering
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

from training.modelling.metrics import F1, Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model
from training.modelling.dataloader import get_train_val_test

from scipy.spatial.distance import cdist

stats = pd.read_csv("new_feb-mar_data/train_feb-mar.csv")#"beneficiary_stats.csv")#"may_data/beneficiary_stats_v5.csv")
beneficiary_data = pd.read_csv("new_feb-mar_data/beneficiary/ai_registration-20210216-20210315.csv")#"feb16-mar15_data/beneficiary/ai_registration-20210216-20210315.csv")#"may_data/beneficiary/AIRegistration-20200501-20200731.csv")
b_data, call_data = load_data("new_feb-mar_data")#"may_data")
call_data = _preprocess_call_data(call_data)
all_beneficiaries = stats#stats[stats['Group'].isin(["Google-AI-Control", "Google-AI-Calls"])]
features_dataset = preprocess_and_make_dataset(b_data, call_data)
user_ids, dynamic_xs, gest_ages, static_xs, ngo_hosp_ids, labels = features_dataset
enroll_gest_age_mean = np.mean(features_dataset[3][:, 0])
days_to_first_call_mean = np.mean(features_dataset[3][:, 7])

# dynamic features preprocessing
dynamic_xs = dynamic_xs.astype(np.float32)
dynamic_xs[:, :, 2] = dynamic_xs[:, :, 2] / 60
dynamic_xs[:, :, 3] = dynamic_xs[:, :, 3] / 60
dynamic_xs[:, :, 4] = dynamic_xs[:, :, 4] / 12

static_xs = static_xs.astype(np.float32)
static_xs[:, 0] = (static_xs[:, 0] - enroll_gest_age_mean)
static_xs[:, 7] = (static_xs[:, 7] - days_to_first_call_mean)

features_dataset = user_ids, dynamic_xs, gest_ages, static_xs, ngo_hosp_ids, labels

aug_states = []
for i in range(6):
    if i % 2 == 0:
        aug_states.append('L{}'.format(i // 2))
    else:
        aug_states.append('H{}'.format(i // 2))

CONFIG = {
    "problem": {
        "orig_states": ['L', 'H'],
        "states": aug_states + ['L', 'H'],
        "actions": ["N", "I"],
    },
    "time_step": 7,
    "gamma": 0.99,
    "clusters": 0,
    "transitions": "weekly",
    "clustering": "kmeans"
}

if CONFIG['transitions'] == 'weekly':
    transitions = pd.read_csv("outputs/pilot_transitions_5months.csv")#"may_data/RMAB_one_month/weekly_transitions_SI_single_group.csv")#"outputs/pilot_transitions_5months.csv")

def kmeans_missing(X, n_clusters, max_iter=10):
    n_clusters = CONFIG['clusters']
    missing = ~np.isfinite(X)
    mu = np.nanmean(X, 0, keepdims=1)
    X_hat = np.where(missing, mu, X)

    prev_labels = None
    for i in range(max_iter):
        if CONFIG['clustering'] == 'optics':
            cls = OPTICS(min_samples=4, n_jobs=-1)
        elif CONFIG['clustering'] == 'kmeans':
            cls = KMeans(n_clusters, n_jobs=-1, random_state=0)
        elif CONFIG['clustering'] == 'spectral':
            cls = SpectralClustering(n_clusters, n_jobs=-1, random_state=0)

        labels = cls.fit_predict(X_hat)

        if CONFIG['clustering'] == 'kmeans':
            centroids = cls.cluster_centers_
        else:
            if CONFIG['clustering'] == 'optics':
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
    distortion = sum(np.min(cdist(X_hat, cls.cluster_centers_,'euclidean'), axis=1)) / X_hat.shape[0]
    inertia = cls.inertia_
    return labels, centroids, X_hat, cls, len(set(labels)), i, distortion, inertia

def get_static_feature_clusters(train_beneficiaries, train_transitions, features_dataset, n_clusters):
    cols = [
        "P(L, I, L)", "P(L, I, H)", "P(H, I, L)", "P(H, I, H)", "P(L, N, L)", "P(L, N, H)", "P(H, N, L)", "P(H, N, H)", 
    ]

    user_ids, dynamic_xs, gest_ages, static_xs, ngo_hosp_ids, labels = features_dataset
    
    train_ids = train_beneficiaries['user_id']
    idxes = [np.where(user_ids == x)[0][0] for x in train_ids]
    train_static_features = static_xs[idxes]
    train_static_features = train_static_features[:, : -8]

    # test_ids = test_beneficiaries['user_id']
    # idxes = [np.where(user_ids == x)[0][0] for x in test_ids]
    # test_static_features = static_xs[idxes]
    # test_static_features = test_static_features[:, : -8]

    train_labels, centroids, _, cls, num_clusters, max_iters = kmeans_missing(train_static_features, n_clusters, max_iter=100)
    train_beneficiaries['cluster'] = train_labels
    # test_beneficiaries['cluster'] = cls.predict(test_static_features)

    cluster_transition_probabilities = pd.DataFrame(columns=['cluster', 'count'] + cols)

    for i in range(n_clusters):
        cluster_beneficiaries = train_beneficiaries[train_beneficiaries['cluster'] == i]
        cluster_b_user_ids = cluster_beneficiaries['user_id']
        probs, _ = get_transition_probabilities(cluster_b_user_ids, train_transitions, min_support=3)
        probs['cluster'] = i
        probs['count'] = len(cluster_b_user_ids)
        cluster_transition_probabilities = cluster_transition_probabilities.append(probs, ignore_index=True)

    # ipdb.set_trace()

    return cluster_transition_probabilities, cls

def get_individual_transition_clusters(train_beneficiaries, train_transitions, features_dataset, n_clusters):
    cols = [
        "P(L, I, L)", "P(L, I, H)", "P(H, I, L)", "P(H, I, H)", "P(L, N, L)", "P(L, N, H)", "P(H, N, L)", "P(H, N, H)", 
    ]

    user_ids, dynamic_xs, gest_ages, static_xs, ngo_hosp_ids, labels = features_dataset
    
    train_ids = train_beneficiaries['user_id']
    idxes = [np.where(user_ids == x)[0][0] for x in train_ids]
    train_static_features = static_xs[idxes]
    train_static_features = train_static_features[:, : -8]

    # test_ids = test_beneficiaries['user_id']
    # idxes = [np.where(user_ids == x)[0][0] for x in test_ids]
    # test_static_features = static_xs[idxes]
    # test_static_features = test_static_features[:, : -8]
    all_transition_probabilities = get_all_transition_probabilities(train_beneficiaries, train_transitions)
    pass_to_kmeans_cols = ['P(L, N, L)', 'P(H, N, L)']

    train_labels, centroids, _, cls, num_clusters, max_iters, distortion, inertia = kmeans_missing(all_transition_probabilities[pass_to_kmeans_cols], n_clusters, max_iter=100)
    
    # ipdb.set_trace()
    train_beneficiaries['cluster'] = train_labels
    # test_beneficiaries['cluster'] = cls.predict(test_static_features)

    dt_clf = RandomForestClassifier(n_estimators=200, criterion="entropy", max_depth=30, n_jobs=-1, random_state=124)
    dt_clf.fit(train_static_features, train_labels)

    cluster_transition_probabilities = pd.DataFrame(columns=['cluster', 'count'] + cols)

    for i in range(n_clusters):
        cluster_beneficiaries = train_beneficiaries[train_beneficiaries['cluster'] == i]
        cluster_b_user_ids = cluster_beneficiaries['user_id']
        probs, _ = get_transition_probabilities(cluster_b_user_ids, train_transitions, min_support=3)
        probs['cluster'] = i
        probs['count'] = len(cluster_b_user_ids)
        cluster_transition_probabilities = cluster_transition_probabilities.append(probs, ignore_index=True)

    # ipdb.set_trace()

    return cluster_transition_probabilities, dt_clf, train_beneficiaries, distortion, inertia

def get_transition_probabilities(beneficiaries, transitions, min_support=3):
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

def get_all_transition_probabilities(train_beneficiaries, transitions):
    cols = [
        "P(L, I, L)", "P(L, I, H)", "P(H, I, L)", "P(H, I, H)", "P(L, N, L)", "P(L, N, H)", "P(H, N, L)", "P(H, N, H)", 
    ]
    transition_probabilities = pd.DataFrame(columns = ['user_id'] + cols)
    user_ids = train_beneficiaries['user_id']

    for user_id in user_ids:
        probs, _ = get_transition_probabilities([user_id], transitions, min_support=1)
        probs['user_id'] = user_id

        transition_probabilities = transition_probabilities.append(probs, ignore_index=True)

    return transition_probabilities

def get_reward(state, action, m):
    if state[0] == "L":
        reward = 1.0
    else:
        reward = -1.0
    if action == 'N':
        reward += m

    return reward

def plan(transition_probabilities, CONFIG):

    v_values = np.zeros((CONFIG['clusters'], len(CONFIG['problem']['states'])))
    q_values = np.zeros((CONFIG['clusters'], len(CONFIG['problem']['states']), len(CONFIG['problem']['actions'])))
    high_m_values = 1 * np.ones((CONFIG['clusters'], len(CONFIG['problem']['states'])))
    low_m_values = -1 * np.ones((CONFIG['clusters'], len(CONFIG['problem']['states'])))

    for cluster in range(CONFIG['clusters']):
        print('Planning for Cluster {}'.format(cluster))
        t_probs = np.zeros((len(CONFIG['problem']['states']), len(CONFIG['problem']['states']), len(CONFIG['problem']['actions'])))
        two_state_probs = np.zeros((2, 2, 2))
        for i in range(two_state_probs.shape[0]):
            for j in range(two_state_probs.shape[1]):
                for k in range(two_state_probs.shape[2]):
                    s = CONFIG['problem']['orig_states'][i]
                    s_prime = CONFIG['problem']['orig_states'][j]
                    a = CONFIG['problem']['actions'][k]
                    two_state_probs[i, j, k] = transition_probabilities.loc[transition_probabilities['cluster']==cluster, "P(" + s + ", " + a + ", " + s_prime + ")"]
        
        t_probs[0 : 2, 2 : 4, 0] = two_state_probs[:, :, 0]
        t_probs[2 : 4, 4 : 6, 0] = two_state_probs[:, :, 0]
        t_probs[4 : 6, 6 : 8, 0] = two_state_probs[:, :, 0]
        t_probs[6 : 8, 6 : 8, 0] = two_state_probs[:, :, 0]

        t_probs[0 : 2, 2 : 4, 1] = two_state_probs[:, :, 0]
        t_probs[2 : 4, 4 : 6, 1] = two_state_probs[:, :, 0]
        t_probs[4 : 6, 6 : 8, 1] = two_state_probs[:, :, 0]
        t_probs[6 : 8, 0 : 2, 1] = two_state_probs[:, :, 1]

        #ipdb.set_trace()

        max_q_diff = np.inf
        prev_m_values, m_values = None, None
        while max_q_diff > 1e-5:
            prev_m_values = m_values
            m_values = (low_m_values + high_m_values) / 2
            if type(prev_m_values) != type(None) and abs(prev_m_values - m_values).max() < 1e-20:
                break
            max_q_diff = 0
            v_values[cluster, :] = np.zeros((len(CONFIG['problem']['states'])))
            q_values[cluster, :, :] = np.zeros((len(CONFIG['problem']['states']), len(CONFIG['problem']['actions'])))
            delta = np.inf
            while delta > 0.0001:
                delta = 0
                for i in range(t_probs.shape[0]):
                    v = v_values[cluster, i]
                    v_a = np.zeros((t_probs.shape[2],))
                    for k in range(v_a.shape[0]):
                        for j in range(t_probs.shape[1]):
                            v_a[k] += t_probs[i, j, k] * (get_reward(CONFIG['problem']['states'][i], CONFIG['problem']['actions'][k], m_values[cluster, i]) + CONFIG["gamma"] * v_values[cluster, j])

                    v_values[cluster, i] = np.max(v_a)
                    delta = max([delta, abs(v_values[cluster, i] - v)])

            state_idx = -1
            for state in range(q_values.shape[1]):
                for action in range(q_values.shape[2]):
                    for next_state in range(q_values.shape[1]):
                        q_values[cluster, state, action] += t_probs[state, next_state, action] * (get_reward(CONFIG['problem']['states'][state], CONFIG['problem']['actions'][action], m_values[cluster, state]) + CONFIG["gamma"] * v_values[cluster, next_state])
                # print(state, q_values[cluster, state, 0], q_values[cluster, state, 1])

            for state in range(q_values.shape[1]):
                if abs(q_values[cluster, state, 1] - q_values[cluster, state, 0]) > max_q_diff:
                    state_idx = state
                    max_q_diff = abs(q_values[cluster, state, 1] - q_values[cluster, state, 0])

            # print(q_values)
            # print(low_m_values, high_m_values)
            if max_q_diff > 1e-5 and q_values[cluster, state_idx, 0] < q_values[cluster, state_idx, 1]:
                low_m_values[cluster, state_idx] = m_values[cluster, state_idx]
            elif max_q_diff > 1e-5 and q_values[cluster, state_idx, 0] > q_values[cluster, state_idx, 1]:
                high_m_values[cluster, state_idx] = m_values[cluster, state_idx]

            # print(low_m_values, high_m_values, state_idx)
            # ipdb.set_trace()
    
    m_values = (low_m_values + high_m_values) / 2

    return q_values, m_values

def run_third_pilot(all_beneficiaries, transitions, call_data, CONFIG, features_dataset, beneficiary_data):
    dependencies = {
        'BinaryAccuracy': BinaryAccuracy,
        'F1': F1,
        'Precision': Precision,
        'Recall': Recall
    }

    cluster_transition_probabilities, cls, train_beneficiaries, distortion, inertia = get_individual_transition_clusters(all_beneficiaries, transitions, features_dataset, CONFIG['clusters'])
    cols = [
        "P(L, I, L)", "P(L, I, H)", "P(H, I, L)", "P(H, I, H)", "P(L, N, L)", "P(L, N, H)", "P(H, N, L)", "P(H, N, H)", 
    ]
    # ipdb.set_trace()
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

    #ipdb.set_trace()

    train_b_ids = train_beneficiaries['user_id'].to_list()
    cluster_transition_probabilities['mean_squared_error'] = 0.0
    cluster_counts = {}
    missing_counts = {}
    mse_vals = {}
    total_mse = 0.0
    total_count = 0
    ground_truth = []
    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())
    count2 = 0
    for user_id in train_b_ids:
        flag = False
        user_cluster = train_beneficiaries[train_beneficiaries['user_id'] == user_id]['cluster'].item()
        user_probs, _ = get_transition_probabilities([user_id], transitions)
        user_prob_list = [user_probs['P(L, N, L)'], user_probs['P(H, N, L)']]
        ground_truth.append(user_prob_list)
        cluster_row = cluster_transition_probabilities[cluster_transition_probabilities['cluster'] == user_cluster]
        cluster_prob_list = [cluster_row['P(L, N, L)'].item(), cluster_row['P(H, N, L)'].item()]
        try:
            mse_val = rmse(np.array(cluster_prob_list), np.array(user_prob_list))
            if mse_val < float('inf'):
                total_mse += mse_val
                total_count += 1
            else:
              count2 += 1
        except:
            flag = True
            mse_val = 0
        # ipdb.set_trace()
        if user_cluster not in cluster_counts:
            cluster_counts[user_cluster] = 1
            mse_vals[user_cluster] = mse_val
        else:
            cluster_counts[user_cluster] += 1
            mse_vals[user_cluster] += mse_val
        
        if flag:
            if user_cluster not in missing_counts:
                missing_counts[user_cluster] = 1
            else:
                missing_counts[user_cluster] += 1
        
        cluster_transition_probabilities.loc[cluster_transition_probabilities['cluster'] == user_cluster, 'mean_squared_error'] = cluster_transition_probabilities[cluster_transition_probabilities['cluster'] == user_cluster]['mean_squared_error'].item() + mse_val
    
    print(total_mse, total_count, total_mse / total_count)
    avg_mse = total_mse/total_count
    cluster_transition_probabilities.to_csv('outputs/{}_{}_transition_probabilities_{}.csv'.format(CONFIG['transitions'], CONFIG['clustering'], CONFIG['clusters']))
    cluster_sizes = list(cluster_counts.values())
    std = np.std(cluster_sizes)
    print(std)
    q_values, m_values = plan(cluster_transition_probabilities, CONFIG)
    whittle_classifier = q_values, m_values, cls
    with open('whittle_classifier_'+str(CONFIG["clusters"])+'.pkl', 'wb') as fr:
        pickle.dump(whittle_classifier, fr)
    fr.close()
    ground_truth = np.array(ground_truth)
    with open('gt_beneficiary_probs.pkl', 'wb') as fr:
        pickle.dump(ground_truth, fr)
    fr.close()

    return total_mse, total_count, avg_mse, std, count2, distortion, inertia

df_dict = {"# of clusters":[],"mse":[],"count":[],"avg_mse":[],"std":[],"count2":[], "distortion":[], "inertia":[]}
for k in [10, 20, 30, 40, 50, 75, 100, 200, 300, 400, 500]:
  CONFIG["clusters"] = k
  mse, count, avg_mse, std, count2, dis, ine = run_third_pilot(all_beneficiaries, transitions, call_data, CONFIG, features_dataset, beneficiary_data)
  df_dict["# of clusters"].append(k)
  df_dict["mse"].append(mse)
  df_dict["count"].append(count)
  df_dict["avg_mse"].append(avg_mse)
  df_dict["std"].append(std)
  df_dict["count2"].append(count2)
  df_dict["distortion"].append(dis)
  df_dict["inertia"].append(ine)
  print("Data now")
  print(df_dict)

df = pd.DataFrame(df_dict)
df.to_csv("outputs/output_stats_clustering.csv")

