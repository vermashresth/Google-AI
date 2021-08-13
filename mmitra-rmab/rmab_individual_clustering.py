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

stats = pd.read_csv("may_data/beneficiary_stats_v5.csv")
beneficiary_data = pd.read_csv("may_data/beneficiary/AIRegistration-20200501-20200731.csv")
b_data, call_data = load_data("may_data")
call_data = _preprocess_call_data(call_data)
all_beneficiaries = stats[stats['Group'].isin(["Google-AI-Control", "Google-AI-Calls"])]

# features_dataset = preprocess_and_make_dataset(b_data, call_data)
with open('may_data/features_dataset.pkl', 'rb') as fw:
    features_dataset = pickle.load(fw)
fw.close()
# exit()

# beneficiary_splits = load_obj("may_data/RMAB_one_month/weekly_beneficiary_splits_single_group.pkl")

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
    "clusters": int(sys.argv[1]),
    "transitions": "weekly",
    "clustering": sys.argv[2],
    "pilot_start_date": sys.argv[3]
}

if CONFIG['transitions'] == 'weekly':
    transitions = pd.read_csv("may_data/RMAB_one_month/weekly_transitions_SI_single_group.csv")
else:
    transitions = pd.read_csv("may_data/RMAB_one_month/transitions_SI_single_group.csv")

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

def run_third_pilot(all_beneficiaries, transitions, call_data, CONFIG, features_dataset, pilot_data, beneficiary_data):
    pilot_beneficiary_data, pilot_call_data = load_data(pilot_data)
    inf_dataset = preprocess_and_make_dataset(pilot_beneficiary_data, pilot_call_data)
    pilot_call_data = _preprocess_call_data(pilot_call_data)
    pilot_user_ids, pilot_dynamic_xs, pilot_gest_age, pilot_static_xs, pilot_hosp_id, pilot_labels = inf_dataset
    pilot_gold_e2c = pilot_labels[:, 3] / pilot_labels[:, 2]
    pilot_gold_e2c_processed = np.nan_to_num(pilot_gold_e2c)
    pilot_gold_labels = (pilot_gold_e2c_processed < 0.5) * 1.0
    pilot_gold_labels = pilot_gold_labels.astype(np.int)

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

    pilot_static_features = np.array(pilot_static_xs, dtype=np.float)
    pilot_static_features = pilot_static_features[:, : -8]

    #ipdb.set_trace()
    # cluster_transition_probabilities['P(L] = cluster_transition_probabilities.fillna(cluster_transition_probabilities.mean())
    #q_values, m_values = plan(cluster_transition_probabilities, CONFIG)
    #pickle.dump(q_values, open("q_values",'wb'))
    #pickle.dump(m_values, open("m_values",'wb'))
    q_values = pickle.load(open("q_values",'rb'))
    m_values = pickle.load(open("m_values",'rb'))
    print('-'*60)
    print('M Values: {}'.format(m_values))
    print('-'*60)

    print('-'*60)
    print('Q Values: {}'.format(q_values))
    print('-'*60)
    #pickle.dump(cls, open("cls_model",'wb'))
    cls = pickle.load(open("cls_model",'rb'))
    pilot_cluster_predictions = cls.predict(pilot_static_features)

    whittle_indices = {'user_id': [], 'whittle_index': [], 'cluster': [], 'start_state': [], 'registration_date': [], 'current_E2C': []}
    for idx, puser_id in enumerate(pilot_user_ids):
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


    df = pd.DataFrame(whittle_indices)
    df = df.sort_values('whittle_index', ascending=False)
    df.to_csv('outputs/individual_clustering/checking_{}_{}_pilot_stats_{}.csv'.format(CONFIG['transitions'], CONFIG['clustering'], CONFIG['clusters']))

    # ipdb.set_trace()

    return

run_third_pilot(all_beneficiaries, transitions, call_data, CONFIG, features_dataset, 'feb16-mar15_data', beneficiary_data)
