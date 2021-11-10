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
    transitions = pd.read_csv("new_feb-mar_data/pilot_transitions_5months.csv")#"may_data/RMAB_one_month/weekly_transitions_SI_single_group.csv")#"outputs/pilot_transitions_5months.csv")

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

    return labels, centroids, X_hat, cls, len(set(labels)), i

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

    train_labels, centroids, _, cls, num_clusters, max_iters = kmeans_missing(all_transition_probabilities[pass_to_kmeans_cols], n_clusters, max_iter=100)
    
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

    return cluster_transition_probabilities, dt_clf, train_beneficiaries

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

def get_group_transition_probabilities(train_beneficiaries, transitions, beneficiary_data):
    features = ["education", "phone_owner", "income_bracket", "age"]
    cols = [
        "P(L, I, L)", "P(L, I, H)", "P(H, I, L)", "P(H, I, H)", "P(L, N, L)", "P(L, N, H)", "P(H, N, L)", "P(H, N, H)", 
    ]
    education_s = [[7], [1], [2], [3], [4], [5], [6]]
    phone_owner_s = ["husband", "woman", "family"]
    income_bracket_s = [['5000-10000', '0-5000'], ['10000-15000', '15000-20000'], ['30000 and above'], ['25000-30000', '20000-25000']]
    age_s = [(0, 19), (19, 24), (24, 29), (29, 36), (36, 100)]

    transition_probabilities = pd.DataFrame(columns=features+cols)
    for education_idx in range(len(education_s)):
        for phone_owner_idx in range(len(phone_owner_s)):
            for income_bracket_idx in range(len(income_bracket_s)):
                for age_idx in range(len(age_s)):
                    beneficiaries = beneficiary_data[
                        (beneficiary_data['user_id'].isin(train_beneficiaries['user_id']))&
                        (beneficiary_data['age']>=age_s[age_idx][0])&
                        (beneficiary_data['age']<age_s[age_idx][1])&
                        (beneficiary_data['education'].isin(education_s[education_idx]))&
                        (beneficiary_data['phone_owner']==phone_owner_s[phone_owner_idx])&
                        (beneficiary_data['income_bracket'].isin(income_bracket_s[income_bracket_idx]))
                    ]["user_id"]
                    probs, _ = get_transition_probabilities(beneficiaries, transitions)

                    probs["education"] = education_idx
                    probs["income_bracket"] = income_bracket_idx
                    probs["phone_owner"] = phone_owner_idx
                    probs["age"] = age_idx
                    probs['count'] = beneficiaries.shape[0]

                    transition_probabilities = transition_probabilities.append(probs, ignore_index=True)

    return transition_probabilities

def get_pooled_estimate(group_transition_probabilities, train_beneficiaries, transitions, beneficiary_data):
    education_s = [[7], [1], [2], [3], [4], [5], [6]]
    phone_owner_s = ["husband", "woman", "family"]
    income_bracket_s = [['5000-10000', '0-5000'], ['10000-15000', '15000-20000'], ['30000 and above'], ['25000-30000', '20000-25000']]
    age_s = [(0, 19), (19, 24), (24, 29), (29, 36), (36, 100)]

    beneficiaries = []
    for i in group_transition_probabilities.index:
        education_idx = int(group_transition_probabilities.loc[i, "education"].item())
        income_bracket_idx = int(group_transition_probabilities.loc[i, "income_bracket"].item())
        phone_owner_idx = int(group_transition_probabilities.loc[i, "phone_owner"].item())
        age_idx = int(group_transition_probabilities.loc[i, "age"].item())

        group_beneficiaries = beneficiary_data[
            (beneficiary_data['user_id'].isin(train_beneficiaries['user_id']))&
            (beneficiary_data['age']>=age_s[age_idx][0])&
            (beneficiary_data['age']<age_s[age_idx][1])&
            (beneficiary_data['education'].isin(education_s[education_idx]))&
            (beneficiary_data['phone_owner']==phone_owner_s[phone_owner_idx])&
            (beneficiary_data['income_bracket'].isin(income_bracket_s[income_bracket_idx]))
        ]["user_id"]
        beneficiaries.append(group_beneficiaries)

    beneficiaries = pd.concat(beneficiaries, axis=0)
    probs, num_lst = get_transition_probabilities(beneficiaries, transitions)

    return probs, num_lst

def get_reward(state, action, m):
    if state[0] == "L":
        reward = 1.0
    else:
        reward = -1.0
    if action == 'N':
        reward += m

    return reward

# def plan2(two_state_probs, sleeping_constraint = 'True'):

#     aug_states = []
#     for i in range(6):
#         if i % 2 == 0:
#             aug_states.append('L{}'.format(i // 2))
#         else:
#             aug_states.append('H{}'.format(i // 2))

#     if sleeping_constraint:
#         local_CONFIG = {
#             'problem': {
#                 "orig_states": ['L', 'H'],
#                 "states": aug_states + ['L', 'H'],
#                 "actions": ["N", "I"],
#             },
#             "time_step": 7,
#             "gamma": 0.99,
#         }
#     else:
#         local_CONFIG = {
#             'problem': {
#                 "orig_states": ['L', 'H'],
#                 "states": ['L', 'H'],
#                 "actions": ["N", "I"],
#             },
#             "time_step": 7,
#             "gamma": 0.99,
#         }

#     v_values = np.zeros(len(local_CONFIG['problem']['states']))
#     q_values = np.zeros((len(local_CONFIG['problem']['states']), len(local_CONFIG['problem']['actions'])))
#     high_m_values = 1 * np.ones(len(local_CONFIG['problem']['states']))
#     low_m_values = -1 * np.ones(len(local_CONFIG['problem']['states']))

#     t_probs = np.zeros((len(local_CONFIG['problem']['states']), len(local_CONFIG['problem']['states']), len(local_CONFIG['problem']['actions'])))

#     if sleeping_constraint:    
#         t_probs[0 : 2, 2 : 4, 0] = two_state_probs[:, :, 0]
#         t_probs[2 : 4, 4 : 6, 0] = two_state_probs[:, :, 0]
#         t_probs[4 : 6, 6 : 8, 0] = two_state_probs[:, :, 0]
#         t_probs[6 : 8, 6 : 8, 0] = two_state_probs[:, :, 0]

#         t_probs[0 : 2, 2 : 4, 1] = two_state_probs[:, :, 0]
#         t_probs[2 : 4, 4 : 6, 1] = two_state_probs[:, :, 0]
#         t_probs[4 : 6, 6 : 8, 1] = two_state_probs[:, :, 0]
#         t_probs[6 : 8, 0 : 2, 1] = two_state_probs[:, :, 1]
#     else:
#         t_probs = two_state_probs

#     max_q_diff = np.inf
#     prev_m_values, m_values = None, None
#     while max_q_diff > 1e-5:
#         prev_m_values = m_values
#         m_values = (low_m_values + high_m_values) / 2
#         if type(prev_m_values) != type(None) and abs(prev_m_values - m_values).max() < 1e-20:
#             break
#         max_q_diff = 0
#         v_values = np.zeros((len(local_CONFIG['problem']['states'])))
#         q_values = np.zeros((len(local_CONFIG['problem']['states']), len(local_CONFIG['problem']['actions'])))
#         delta = np.inf
#         while delta > 0.0001:
#             delta = 0
#             for i in range(t_probs.shape[0]):
#                 v = v_values[i]
#                 v_a = np.zeros((t_probs.shape[2],))
#                 for k in range(v_a.shape[0]):
#                     for j in range(t_probs.shape[1]):
#                         v_a[k] += t_probs[i, j, k] * (get_reward(local_CONFIG['problem']['states'][i], local_CONFIG['problem']['actions'][k], m_values[i]) + local_CONFIG["gamma"] * v_values[j])

#                 v_values[i] = np.max(v_a)
#                 delta = max([delta, abs(v_values[i] - v)])

#         state_idx = -1
#         for state in range(q_values.shape[1]):
#             for action in range(q_values.shape[2]):
#                 for next_state in range(q_values.shape[1]):
#                     q_values[state, action] += t_probs[state, next_state, action] * (get_reward(local_CONFIG['problem']['states'][state], local_CONFIG['problem']['actions'][action], m_values[state]) + local_CONFIG["gamma"] * v_values[next_state])
#             # print(state, q_values[cluster, state, 0], q_values[cluster, state, 1])

#         for state in range(q_values.shape[1]):
#             if abs(q_values[state, 1] - q_values[state, 0]) > max_q_diff:
#                 state_idx = state
#                 max_q_diff = abs(q_values[state, 1] - q_values[state, 0])

#         # print(q_values)
#         # print(low_m_values, high_m_values)
#         if max_q_diff > 1e-5 and q_values[state_idx, 0] < q_values[state_idx, 1]:
#             low_m_values[state_idx] = m_values[state_idx]
#         elif max_q_diff > 1e-5 and q_values[state_idx, 0] > q_values[state_idx, 1]:
#             high_m_values[state_idx] = m_values[state_idx]

#         # print(low_m_values, high_m_values, state_idx)
#         # ipdb.set_trace()
    
#     m_values = (low_m_values + high_m_values) / 2

#     return q_values, m_values

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

def count_overlaps(test_beneficiaries):
    call_beneficiaries = test_beneficiaries[test_beneficiaries['Group']=="Google-AI-Calls"]
    call_succ_beneficiaries = test_beneficiaries[
        (test_beneficiaries['Group']=="Google-AI-Calls")&
        (test_beneficiaries['Intervention Status']=="Successful")
    ]
    control_beneficiaries = test_beneficiaries[test_beneficiaries['Group']=="Google-AI-Control"]

    good_call_response = test_beneficiaries[
        (test_beneficiaries['user_id'].isin(call_beneficiaries['user_id']))&
        (test_beneficiaries['Post-intervention Day: E2C Ratio']>=0.5)
    ]['user_id']

    good_succ_call_response = test_beneficiaries[
        (test_beneficiaries['user_id'].isin(call_succ_beneficiaries['user_id']))&
        (test_beneficiaries['Post-intervention Day: E2C Ratio']>=0.5)
    ]['user_id']

    good_control_response = test_beneficiaries[
        (test_beneficiaries['user_id'].isin(control_beneficiaries['user_id']))&
        (test_beneficiaries['Post-intervention Day: E2C Ratio']>=0.5)
    ]['user_id']

    print([call_beneficiaries.shape[0], call_succ_beneficiaries.shape[0], control_beneficiaries.shape[0]])

    notes = np.array([
        [good_call_response.shape[0], 
        good_succ_call_response.shape[0],
        good_control_response.shape[0]]
    ])

    results = []
    for k in [100, 200]:
        top_k_call_beneficiaries = call_beneficiaries.iloc[:k, :]
        top_k_good_call_beneficiaries = top_k_call_beneficiaries[top_k_call_beneficiaries['user_id'].isin(good_call_response)]

        top_k_succ_call_beneficiaries = call_succ_beneficiaries.iloc[:k, :]
        top_k_good_succ_call_beneficiaries = top_k_succ_call_beneficiaries[top_k_succ_call_beneficiaries['user_id'].isin(good_succ_call_response)]

        top_k_control_beneficiaries = control_beneficiaries.iloc[:k, :]
        top_k_good_control_beneficiaries = top_k_control_beneficiaries[top_k_control_beneficiaries['user_id'].isin(good_control_response)]

        results.append([
            top_k_good_call_beneficiaries.shape[0]/k, 
            top_k_good_succ_call_beneficiaries.shape[0]/k,
            top_k_good_control_beneficiaries.shape[0]/k
        ])

    return np.array(results), notes

def run_experiment(train_beneficiaries, train_transitions, test_beneficiaries, call_data, CONFIG, features_dataset):
    cluster_transition_probabilities, test_beneficiaries = get_static_feature_clusters(train_beneficiaries, train_transitions, test_beneficiaries, features_dataset, CONFIG['clusters'])
    cluster_transition_probabilities.to_csv('outputs/weekly_cluster_transition_probabilities_{}.csv'.format(CONFIG['clusters']))
    q_values, m_values = plan(cluster_transition_probabilities, CONFIG)

    print('-'*60)
    print('M Values: {}'.format(m_values))
    print('-'*60)

    print('-'*60)
    print('Q Values: {}'.format(q_values))
    print('-'*60)

    for i in tqdm(test_beneficiaries.index):
        user_id = test_beneficiaries.loc[i, "user_id"]
        state = int(test_beneficiaries.loc[i, "state"])
        cluster = int(test_beneficiaries.loc[i, 'cluster'])

        # test_beneficiaries.loc[i, "Whittle Index"] = q_values[cluster, state, 1] - q_values[cluster, state, 0]
        test_beneficiaries.loc[i, "Whittle Index"] = m_values[cluster, state]

    test_beneficiaries = test_beneficiaries.sort_values(by="Whittle Index", ascending=False)
    test_beneficiaries.to_csv('outputs/weekly_test_set_{}.csv'.format(CONFIG['clusters']))
    results, notes = count_overlaps(test_beneficiaries)

    return results, notes

def run_third_pilot(all_beneficiaries, transitions, call_data, CONFIG, features_dataset, pilot_data, beneficiary_data):
    #GARGI
    pilot_stats = pd.read_csv("new_feb-mar_data/test_feb-mar.csv")#GARGI
    pilot_beneficiary_data, pilot_call_data = load_data(pilot_data)
    inf_dataset = preprocess_and_make_dataset(pilot_beneficiary_data, pilot_call_data)
    pilot_call_data = _preprocess_call_data(pilot_call_data)
    pilot_user_ids, pilot_dynamic_xs, pilot_gest_age, pilot_static_xs, pilot_hosp_id, pilot_labels = inf_dataset
    pilot_test_ids = pilot_stats['user_id']
    pilot_idxes = [np.where(pilot_user_ids == x)[0][0] for x in pilot_test_ids]
    pilot_static_xs = pilot_static_xs[pilot_idxes]
    # pilot_gold_e2c = pilot_labels[:, 3] / pilot_labels[:, 2]
    # pilot_gold_e2c_processed = np.nan_to_num(pilot_gold_e2c)
    # pilot_gold_labels = (pilot_gold_e2c_processed < 0.5) * 1.0
    # pilot_gold_labels = pilot_gold_labels.astype(np.int)

    # ipdb.set_trace()

    enroll_gest_age_mean = np.mean(inf_dataset[3][:, 0])
    days_to_first_call_mean = np.mean(inf_dataset[3][:, 7])

    # dynamic features preprocessing
    pilot_dynamic_xs = pilot_dynamic_xs.astype(np.float32)
    pilot_dynamic_xs[:, :, 2] = pilot_dynamic_xs[:, :, 2] / 60
    pilot_dynamic_xs[:, :, 3] = pilot_dynamic_xs[:, :, 3] / 60
    pilot_dynamic_xs[:, :, 4] = pilot_dynamic_xs[:, :, 4] / 12

    # static features preprocessing
    pilot_static_xs = pilot_static_xs.astype(np.float32)
    pilot_static_xs[:, 0] = (pilot_static_xs[:, 0] - enroll_gest_age_mean)
    pilot_static_xs[:, 7] = (pilot_static_xs[:, 7] - days_to_first_call_mean)

    dependencies = {
        'BinaryAccuracy': BinaryAccuracy,
        'F1': F1,
        'Precision': Precision,
        'Recall': Recall
    }
    #GARGI
    # model = load_model(os.path.join("models", 'lstm_model_final', "model"), custom_objects=dependencies)
    # output_probs = model.predict(x=[pilot_static_xs, pilot_dynamic_xs, pilot_hosp_id, pilot_gest_age])
    # out_lstm_labels = (output_probs >= 0.5).astype(np.int)
    # low_eng_idxes = np.where(out_lstm_labels == 1)[0]
    # low_eng_user_ids = np.array(pilot_user_ids)[low_eng_idxes]
    pilot_static_features = np.array(pilot_static_xs, dtype=np.float)
    pilot_static_features = pilot_static_features[:, : -8]

    cluster_transition_probabilities, cls, train_beneficiaries = get_individual_transition_clusters(all_beneficiaries, transitions, features_dataset, CONFIG['clusters'])
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
    cluster_transition_probabilities.to_csv('outputs/individual_clustering/{}_{}_transition_probabilities_{}.csv'.format(CONFIG['transitions'], CONFIG['clustering'], CONFIG['clusters']))
    #exit()

    ground_truth = np.array(ground_truth)
    with open('gt_beneficiary_probs.pkl', 'wb') as fr:
        pickle.dump(ground_truth, fr)
    fr.close()

    #ipdb.set_trace()
    # cluster_transition_probabilities['P(L] = cluster_transition_probabilities.fillna(cluster_transition_probabilities.mean())
    q_values, m_values = plan(cluster_transition_probabilities, CONFIG)

    print('-'*60)
    print('M Values: {}'.format(m_values))
    print('-'*60)

    print('-'*60)
    print('Q Values: {}'.format(q_values))
    print('-'*60)

    pilot_cluster_predictions = cls.predict(pilot_static_features)

    whittle_indices = {'user_id': [], 'whittle_index': [], 'cluster': [], 'start_state': [], 'registration_date': [], 'current_E2C': []}#, 'lstm_prediction': [], 'gold_e2c': [], 'gold_label': [], 'registration_date': [], 'current_E2C': []} #GARGI
    for idx, puser_id in enumerate(pilot_test_ids):#pilot_user_ids): #GARGI
        pilot_date_num = (pd.to_datetime(CONFIG['pilot_start_date'], format="%Y-%m-%d") - pd.to_datetime("2018-01-01", format="%Y-%m-%d")).days

        if CONFIG['transitions'] == 'weekly':
            past_days_calls = pilot_call_data[
            (pilot_call_data["user_id"]==puser_id)&
            (pilot_call_data["startdate"]<pilot_date_num)&
            (pilot_call_data["startdate"]>=pilot_date_num - 7)
        ]
        else:
            past_days_calls = pilot_call_data[
                (pilot_call_data["user_id"]==puser_id)&
                (pilot_call_data["startdate"]<pilot_date_num)&
                (pilot_call_data["startdate"]>=pilot_date_num - 30)
            ]
        past_days_connections = past_days_calls[past_days_calls['duration']>0].shape[0]
        past_days_engagements = past_days_calls[past_days_calls['duration'] >= 30].shape[0]
        if CONFIG['transitions'] == 'weekly':
            if past_days_engagements == 0:
                curr_state = 7
            else:
                curr_state = 6
        else:
            if past_days_connections == 0:
                curr_state = 7
            else:
                if past_days_engagements/past_days_connections >= 0.5:
                    curr_state = 6
                else:
                    curr_state = 7
        
        curr_cluster = pilot_cluster_predictions[idx]

        whittle_indices['user_id'].append(puser_id)
        whittle_indices['whittle_index'].append(m_values[curr_cluster, curr_state])
        whittle_indices['cluster'].append(curr_cluster)
        if curr_state == 7:
            whittle_indices['start_state'].append('NE')
        elif curr_state == 6:
            whittle_indices['start_state'].append('E')
        #GARGI
        # if out_lstm_labels[idx] == 0:
        #     whittle_indices['lstm_prediction'].append('high_engagement')
        # elif out_lstm_labels[idx] == 1:
        #     whittle_indices['lstm_prediction'].append('low_engagement')
        # whittle_indices['gold_e2c'].append(pilot_gold_e2c[idx])
        # if pilot_gold_labels[idx] == 0:
        #     whittle_indices['gold_label'].append('high_engagement')
        # elif pilot_gold_labels[idx] == 1:
        #     whittle_indices['gold_label'].append('low_engagement')
        
        all_days_calls = pilot_call_data[
            (pilot_call_data['user_id'] == puser_id)
        ]
        all_days_connections = all_days_calls[all_days_calls['duration'] > 0].shape[0]
        all_days_engagements = all_days_calls[all_days_calls['duration'] >= 30].shape[0]
        if all_days_connections == 0:
            whittle_indices['current_E2C'].append('')
        else:
            curr_e2c = all_days_engagements / all_days_connections
            whittle_indices['current_E2C'].append(curr_e2c)
        regis_date = pilot_beneficiary_data[pilot_beneficiary_data['user_id'] == puser_id]['registration_date'].item()

        whittle_indices['registration_date'].append(regis_date)

    with open('policy_dump.pkl', 'wb') as fr:
        policy = pilot_test_ids, pilot_static_features, cls, cluster_transition_probabilities, m_values, q_values
        pickle.dump(policy, fr)
    fr.close()

    df = pd.DataFrame(whittle_indices)
    df = df.sort_values('whittle_index', ascending=False)
    df.to_csv('outputs/individual_clustering/checking_{}_{}_pilot_stats_{}.csv'.format(CONFIG['transitions'], CONFIG['clustering'], CONFIG['clusters']))

    # ipdb.set_trace()

    return

def run_and_repeat(beneficiaries, transitions, call_data, CONFIG, features_dataset):
    all_results, all_notes = [], []

    for train_beneficiaries, test_beneficiaries in beneficiaries:
        train_transitions = transitions[transitions['user_id'].isin(train_beneficiaries['user_id'])]
        results, notes = run_experiment(train_beneficiaries, train_transitions, test_beneficiaries, call_data, CONFIG, features_dataset)

        print(results)
        print(notes)
        
        all_results.append(results)
        all_notes.append(notes)

    return np.mean(np.stack(all_results, axis=0), axis=0), np.mean(np.stack(all_notes, axis=0), axis=0)

run_third_pilot(all_beneficiaries, transitions, call_data, CONFIG, features_dataset, 'feb16-mar15_data', beneficiary_data)

# results, avgs = run_and_repeat(beneficiary_splits, transitions, call_data, CONFIG, features_dataset)
# print("-"*60)
# print(results)
# print(avgs)
# print("-"*60)
