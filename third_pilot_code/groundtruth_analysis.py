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

plt.style.use('seaborn')
np.random.seed(1)

from training.utils import load_obj, save_obj
from training.data import load_data
from training.call_data import load_call_data, load_call_file
from training.dataset import _preprocess_call_data, preprocess_and_make_dataset

from sklearn.cluster import KMeans, OPTICS, SpectralClustering
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from training.modelling.metrics import F1, Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model
from training.modelling.dataloader import get_train_val_test


CONFIG = {
    'clusters': 40,
    'clustering': 'kmeans',
    }

def get_transition_probabilities(beneficiaries, transitions, min_support=3):
    transitions = transitions[transitions['user_id'].isin(beneficiaries)]

    i_transitions = transitions[transitions['action']=='Intervention']
    n_i_transitions = transitions[transitions['action']=='No Intervention']

    i_E = i_transitions[i_transitions['pre-action state']=='L']
    i_NE = i_transitions[i_transitions['pre-action state']=='H']

    i_E_E = i_E[i_E['post-action state']=='L']
    i_E_NE = i_E[i_E['post-action state']=='H']

    i_NE_E = i_NE[i_NE['post-action state']=='L']
    i_NE_NE = i_NE[i_NE['post-action state']=='H']

    n_i_E = n_i_transitions[n_i_transitions['pre-action state']=='L']
    n_i_NE = n_i_transitions[n_i_transitions['pre-action state']=='H']

    n_i_E_E = n_i_E[n_i_E['post-action state']=='L']
    n_i_E_NE = n_i_E[n_i_E['post-action state']=='H']

    n_i_NE_E = n_i_NE[n_i_NE['post-action state']=='L']
    n_i_NE_NE = n_i_NE[n_i_NE['post-action state']=='H']

    transition_probabilities = dict()
    if i_E.shape[0] >= min_support:
        transition_probabilities['P(E, I, E)'] = i_E_E.shape[0] / i_E.shape[0]
        transition_probabilities['P(E, I, NE)'] = i_E_NE.shape[0] / i_E.shape[0]
    else:
        transition_probabilities['P(E, I, E)'] = np.nan
        transition_probabilities['P(E, I, NE)'] = np.nan

    if i_NE.shape[0] >= min_support:
        transition_probabilities['P(NE, I, E)'] = i_NE_E.shape[0] / i_NE.shape[0]
        transition_probabilities['P(NE, I, NE)'] = i_NE_NE.shape[0] / i_NE.shape[0]
    else:
        transition_probabilities['P(NE, I, E)'] = np.nan
        transition_probabilities['P(NE, I, NE)'] = np.nan
    
    if n_i_E.shape[0] >= min_support:
        transition_probabilities['P(E, A, E)'] = n_i_E_E.shape[0] / n_i_E.shape[0]
        transition_probabilities['P(E, A, NE)'] = n_i_E_NE.shape[0] / n_i_E.shape[0]
    else:
        transition_probabilities['P(E, A, E)'] = np.nan
        transition_probabilities['P(E, A, NE)'] = np.nan

    if n_i_NE.shape[0] >= min_support:
        transition_probabilities['P(NE, A, E)'] = n_i_NE_E.shape[0] / n_i_NE.shape[0]
        transition_probabilities['P(NE, A, NE)'] = n_i_NE_NE.shape[0] / n_i_NE.shape[0]
    else:
        transition_probabilities['P(NE, A, E)'] = np.nan
        transition_probabilities['P(NE, A, NE)'] = np.nan

    transition_probabilities['C(E, I, E)'] = i_E_E.shape[0]
    transition_probabilities['C(E, I, NE)'] = i_E_NE.shape[0]
    transition_probabilities['C(NE, I, E)'] = i_NE_E.shape[0]
    transition_probabilities['C(NE, I, NE)'] = i_NE_NE.shape[0]
    transition_probabilities['C(E, A, E)'] = n_i_E_E.shape[0]
    transition_probabilities['C(E, A, NE)'] = n_i_E_NE.shape[0]
    transition_probabilities['C(NE, A, E)'] = n_i_NE_E.shape[0]
    transition_probabilities['C(NE, A, NE)'] = n_i_NE_NE.shape[0]
    
    return transition_probabilities


def get_static_feature_clusters(train_beneficiaries, train_transitions, features_dataset, n_clusters):
    cols = [
        "P(E, I, E)", "P(E, I, NE)", "P(NE, I, E)", "P(NE, I, NE)", "P(E, A, E)", "P(E, A, NE)", "P(NE, A, E)", "P(NE, A, NE)", 
    ]
    cols += [
        'C(E, I, E)', 'C(E, I, NE)', 'C(NE, I, E)', 'C(NE, I, NE)', 'C(E, A, E)', 'C(E, A, NE)', 'C(NE, A, E)', 'C(NE, A, NE)', 
    ]
    
    user_ids, dynamic_xs, gest_ages, static_xs, ngo_hosp_ids, labels = features_dataset
    
    train_ids = train_beneficiaries['user_id']
    idxes = [np.where(user_ids == x)[0][0] for x in train_ids]
    train_static_features = static_xs[idxes]
    train_static_features = train_static_features[:, : -8]

    train_labels, centroids, _, cls, num_clusters, max_iters = kmeans_missing(train_static_features, n_clusters, max_iter=100)
    train_beneficiaries['cluster'] = train_labels

    cluster_transition_probabilities = pd.DataFrame(columns=['cluster', 'count'] + cols)

    for i in range(n_clusters):
        cluster_beneficiaries = train_beneficiaries[train_beneficiaries['cluster'] == i]
        cluster_b_user_ids = cluster_beneficiaries['user_id']
        probs, _ = get_transition_probabilities(cluster_b_user_ids, train_transitions, min_support=3)
        probs['cluster'] = i
        probs['count'] = len(cluster_b_user_ids)
        # tuple_count = get_tuple_count(cluster_b_user_ids, train_transitions)
        # for key in tuple_count:
        #     probs[key] = tuple_count[key]
        cluster_transition_probabilities = cluster_transition_probabilities.append(probs, ignore_index=True)

    cluster_transition_probabilities.to_csv("groundtruth_analysis/static_feature_cluster.csv")

    return cluster_transition_probabilities, cls

def get_all_transition_probabilities(train_beneficiaries, transitions):
    cols = [
        'P(E, I, E)', 'P(E, I, NE)', 'P(NE, I, E)', 'P(NE, I, NE)', 'P(E, A, E)', 'P(E, A, NE)', 'P(NE, A, E)', 'P(NE, A, NE)', 
    ]

    transition_probabilities = pd.DataFrame(columns = ['user_id'] + cols)
    user_ids = train_beneficiaries['user_id']

    for user_id in user_ids:
        probs = get_transition_probabilities([user_id], transitions, min_support=1)
        probs['user_id'] = user_id

        transition_probabilities = transition_probabilities.append(probs, ignore_index=True)

    return transition_probabilities

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

def get_transitions():
    # b_data, _ = load_data("feb16-mar15_data")
    b_data = pd.read_csv("outputs/individual_clustering/weekly_kmeans_pilot_stats_40.csv")
    call_data_week_2 = load_call_file(f'feb16-mar15_data/call/older/call_data_week_1.csv', b_data['user_id'])
    call_data_week_2 = _preprocess_call_data(call_data_week_2)
    save_obj(call_data_week_2, "temp/call_data_week_1.pkl")
    transitions = pd.DataFrame(columns=['user_id', 'pre-action state', 'action', 'post-action state'])
    beneficiaries = b_data['user_id']
    for i in range(1, 7):

        previous_call_data = call_data_week_2
        call_data_week_2 = load_call_file(f'feb16-mar15_data/call/older/call_data_week_{i+1}.csv', b_data['user_id'])
        call_data_week_2 = _preprocess_call_data(call_data_week_2)
        save_obj(call_data_week_2, f'temp/call/older/call_data_week_{i+1}.pkl' )
        interventions = pd.read_csv(f'intervention_lists/calling_list_week{i}.csv')
        
        for beneficiary in beneficiaries:
            sequence = []
            calls = previous_call_data[previous_call_data['user_id']==beneficiary]
            period_connections = calls[calls['duration'] > 0].shape[0]
            period_engagements = calls[calls['duration'] >= 30].shape[0]

            if period_engagements == 0:
                sequence.append('H')
            else:
                sequence.append('L')
            

            intervention = interventions[interventions['user_id']==beneficiary]
            if intervention.shape[0] == 0:
                sequence.append('N')
            else:
                sequence.append('I')

            calls = call_data_week_2[call_data_week_2['user_id']==beneficiary]

            period_connections = calls[calls['duration'] > 0].shape[0]
            period_engagements = calls[calls['duration'] >= 30].shape[0]

            if period_engagements == 0:
                sequence.append('H')
            else:
                sequence.append('L')
            

            if sequence[1] == 'I':
                transitions = transitions.append({
                            'user_id': beneficiary, 
                            'pre-action state': sequence[0], 
                            'action': 'Intervention', 
                            'post-action state': sequence[2]
                        }, ignore_index=True)
            else:
                transitions = transitions.append({
                        'user_id': beneficiary, 
                        'pre-action state': sequence[0], 
                        'action': 'No Intervention', 
                        'post-action state': sequence[2]
                    }, ignore_index=True)

    print(transitions)
    transitions.to_csv(path_or_buf = 'feb16-mar15_data/transitions.csv')

    return transitions

def get_individual_transition_clusters(train_beneficiaries, train_transitions, features_dataset, n_clusters):
    cols = [
        'P(E, I, E)', 'P(E, I, NE)', 'P(NE, I, E)', 'P(NE, I, NE)', 'P(E, A, E)', 'P(E, A, NE)', 'P(NE, A, E)', 'P(NE, A, NE)', 
    ]
    cols += [
        'C(E, I, E)', 'C(E, I, NE)', 'C(NE, I, E)', 'C(NE, I, NE)', 'C(E, A, E)', 'C(E, A, NE)', 'C(NE, A, E)', 'C(NE, A, NE)', 
    ]
    user_ids, dynamic_xs, gest_ages, static_xs, ngo_hosp_ids, labels = features_dataset
    
    train_ids = train_beneficiaries['user_id']
    idxes = [np.where(user_ids == x)[0][0] for x in train_ids]
    train_static_features = static_xs[idxes]
    train_static_features = train_static_features[:, : -8]

    all_transition_probabilities = get_all_transition_probabilities(train_beneficiaries, train_transitions)
    pass_to_kmeans_cols = ['P(E, A, E)', 'P(NE, A, E)']

    train_labels, centroids, _, cls, num_clusters, max_iters = kmeans_missing(all_transition_probabilities[pass_to_kmeans_cols], n_clusters, max_iter=100)
    
    train_beneficiaries['cluster'] = train_labels

    # dt_clf = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=30, n_jobs=-1, random_state=124)
    # dt_clf.fit(train_static_features, train_labels)

    # cluster_transition_probabilities = pd.DataFrame(columns=['cluster', 'count'] + cols)

    # for i in range(n_clusters):
    #     cluster_beneficiaries = train_beneficiaries[train_beneficiaries['cluster'] == i]
    #     cluster_b_user_ids = cluster_beneficiaries['user_id']
    #     probs = get_transition_probabilities(cluster_b_user_ids, train_transitions, min_support=3)
    #     # tuple_count = get_tuple_count(cluster_b_user_ids, train_transitions)
    #     probs['cluster'] = i
    #     probs['count'] = len(cluster_b_user_ids)
    #     # for key in tuple_count:
    #     #     probs[key] = tuple_count[key]
    #     cluster_transition_probabilities = cluster_transition_probabilities.append(probs, ignore_index=True)

    # cluster_transition_probabilities.to_csv("groundtruth_analysis/cluster_transition_probabilities_training_data_kmeans_40.csv")
    return train_beneficiaries
    # return cluster_transition_probabilities, dt_clf

def get_clustered_beneficiaries():

    stats = pd.read_csv('may_data/beneficiary_stats_v5.csv')
    train_beneficiary_data = pd.read_csv('may_data/beneficiary/AIRegistration-20200501-20200731.csv')
    train_beneficiaries = stats[stats['Group'].isin(['Google-AI-Control', 'Google-AI-Calls'])]
    with open('may_data/features_dataset.pkl', 'rb') as fw:
        features_dataset = pickle.load(fw)
    fw.close()
    train_transitions =  pd.read_csv('may_data/RMAB_one_month/weekly_transitions_SI_single_group.csv')

    # pilot_beneficiary_data, pilot_call_data = load_data('feb16-mar15_data')
    # inf_dataset = preprocess_and_make_dataset(pilot_beneficiary_data, pilot_call_data)
    inf_dataset = load_obj('feb16-mar15_data/inf_dataset.pkl')
    
    pilot_user_ids = inf_dataset[0]
    pilot_static_xs = inf_dataset[3]
    
    enroll_gest_age_mean = np.mean(inf_dataset[3][:, 0])
    days_to_first_call_mean = np.mean(inf_dataset[3][:, 7])

    # static features preprocessing
    pilot_static_xs = pilot_static_xs.astype(np.float32)
    pilot_static_xs[:, 0] = (pilot_static_xs[:, 0] - enroll_gest_age_mean)
    pilot_static_xs[:, 7] = (pilot_static_xs[:, 7] - days_to_first_call_mean)
    pilot_static_features = np.array(pilot_static_xs, dtype=np.float)
    pilot_static_features = pilot_static_features[:, : -8]

    cluster_transition_probabilities, cls = get_individual_transition_clusters(train_beneficiaries, train_transitions, features_dataset, CONFIG['clusters'])
    pilot_cluster_predictions = cls.predict(pilot_static_features)
    # save_obj(pilot_cluster_predictions, 'feb16-mar15_data/pilot_cluster_predictions.pkl')
    # pilot_cluster_predictions = load_obj('feb16-mar15_data/pilot_cluster_predictions.pkl')

    clustered_beneficiaries = pd.DataFrame(columns=['user_id', 'cluster'])
    for idx, puser_id in enumerate(pilot_user_ids):
        clustered_beneficiaries = clustered_beneficiaries.append(
            {'user_id': puser_id,
            'cluster': pilot_cluster_predictions[idx]},
            ignore_index=True)
    return clustered_beneficiaries

# print(get_clustered_beneficiaries())

def get_cluster_transition_probabilities(clustered_beneficiaries, transitions):

    cols = [
        'P(E, I, E)', 'P(E, I, NE)', 'P(NE, I, E)', 'P(NE, I, NE)', 'P(E, A, E)', 'P(E, A, NE)', 'P(NE, A, E)', 'P(NE, A, NE)', 
    ]
    cols += [
        'C(E, I, E)', 'C(E, I, NE)', 'C(NE, I, E)', 'C(NE, I, NE)', 'C(E, A, E)', 'C(E, A, NE)', 'C(NE, A, E)', 'C(NE, A, NE)', 
    ]
    
    cluster_transition_probabilities = pd.DataFrame(columns=['cluster', 'count'] + cols)

    for i in range(CONFIG['clusters']):
        cluster_beneficiaries = clustered_beneficiaries[clustered_beneficiaries['cluster'] == i]
        cluster_b_user_ids = cluster_beneficiaries['user_id']
        probs= get_transition_probabilities(cluster_b_user_ids, transitions, min_support=3)
        probs['cluster'] = i
        probs['count'] = len(cluster_b_user_ids)
        # tuple_count = get_tuple_count(cluster_b_user_ids, transitions)
        # for key in tuple_count:
        #     probs[key] = tuple_count[key] 
        cluster_transition_probabilities = cluster_transition_probabilities.append(probs, ignore_index=True)
    
    cluster_transition_probabilities.to_csv('groundtruth_analysis/cluster_transition_probabilities_kmeans_40.csv')

    return cluster_transition_probabilities

def get_tuple_count(beneficiaries, transitions):
    transitions = transitions[transitions['user_id'].isin(beneficiaries)]

    i_transitions = transitions[transitions['action']=='Intervention']
    n_i_transitions = transitions[transitions['action']=='No Intervention']

    i_E = i_transitions[i_transitions['pre-action state']=='L']
    i_NE = i_transitions[i_transitions['pre-action state']=='H']

    i_E_E = i_E[i_E['post-action state']=='L']
    i_E_NE = i_E[i_E['post-action state']=='H']

    i_NE_E = i_NE[i_NE['post-action state']=='L']
    i_NE_NE = i_NE[i_NE['post-action state']=='H']

    n_i_E = n_i_transitions[n_i_transitions['pre-action state']=='L']
    n_i_NE = n_i_transitions[n_i_transitions['pre-action state']=='H']

    n_i_E_E = n_i_E[n_i_E['post-action state']=='L']
    n_i_E_NE = n_i_E[n_i_E['post-action state']=='H']

    n_i_NE_E = n_i_NE[n_i_NE['post-action state']=='L']
    n_i_NE_NE = n_i_NE[n_i_NE['post-action state']=='H']

    tuple_counts = dict()
    tuple_counts['C(L, I, L)'] = i_E_E.shape[0]
    tuple_counts['C(L, I, H)'] = i_E_NE.shape[0]
    tuple_counts['C(H, I, L)'] = i_NE_E.shape[0]
    tuple_counts['C(H, I, H)'] = i_NE_NE.shape[0]
    tuple_counts['C(L, A, L)'] = n_i_E_E.shape[0]
    tuple_counts['C(L, A, H)'] = n_i_E_NE.shape[0]
    tuple_counts['C(H, A, L)'] = n_i_NE_E.shape[0]
    tuple_counts['C(H, A, H)'] = n_i_NE_NE.shape[0]
    return tuple_counts


def get_tuple_count_per_cluster(clustered_beneficiaries, transitions):
    cols = [
            'C(L, I, L)', 'C(L, I, H)', 'C(H, I, L)', 'C(H, I, H)', 'C(L, A, L)', 'C(L, A, H)', 'C(H, A, L)', 'C(H, A, H)', 
        ]
    
    cluster_tuple_counts = pd.DataFrame(columns=['cluster', 'count'] + cols)

    for i in range(CONFIG['clusters']):
        cluster_beneficiaries = clustered_beneficiaries[clustered_beneficiaries['cluster'] == i]
        cluster_b_user_ids = cluster_beneficiaries['user_id']
        tuple_counts = get_tuple_count(cluster_b_user_ids, transitions)
        tuple_counts['cluster'] = i
        tuple_counts['count'] = len(cluster_b_user_ids)
        cluster_tuple_counts = cluster_tuple_counts.append(tuple_counts, ignore_index=True)

    cluster_tuple_counts.to_csv("feb16-mar15_data/cluster_tuple_count_kmeans_40")
    return cluster_tuple_counts

# clustered_beneficiaries = get_clustered_beneficiaries()
# transitions = pd.read_csv('feb16-mar15_data/transitions.csv')

# cluster_transition_probabilities = get_cluster_transition_probabilities(clustered_beneficiaries, transitions)

# cluster_tuple_counts = get_tuple_count_per_cluster(clustered_beneficiaries, transitions)
# get_transitions()
transitions = pd.read_csv("groundtruth_analysis/transitions_week_9.csv")
beneficiaries = pd.read_csv("outputs/individual_clustering/weekly_kmeans_pilot_stats_40.csv")

cols = [
        'P(E, I, E)', 'P(E, I, NE)', 'P(NE, I, E)', 'P(NE, I, NE)', 'P(E, A, E)', 'P(E, A, NE)', 'P(NE, A, E)', 'P(NE, A, NE)', 
    ]
cols += [
        'C(E, I, E)', 'C(E, I, NE)', 'C(NE, I, E)', 'C(NE, I, NE)', 'C(E, A, E)', 'C(E, A, NE)', 'C(NE, A, E)', 'C(NE, A, NE)', 
    ]

# stats = pd.read_csv('may_data/beneficiary_stats_v5.csv')
# train_beneficiary_data = pd.read_csv('may_data/beneficiary/AIRegistration-20200501-20200731.csv')
# train_beneficiaries = stats[stats['Group'].isin(['Google-AI-Control', 'Google-AI-Calls'])]
# with open('may_data/features_dataset.pkl', 'rb') as fw:
#     features_dataset = pickle.load(fw)
# fw.close()
# train_transitions =  pd.read_csv('may_data/RMAB_one_month/weekly_transitions_SI_single_group.csv')
# train_beneficiaries = get_individual_transition_clusters(train_beneficiaries, train_transitions, features_dataset, CONFIG['clusters'])
    
transition_probabilities = pd.DataFrame(columns=['TEST/TRAIN', 'cluster', 'count'] + cols)

with open('policy_dump.pkl', 'rb') as fr:
  pilot_user_ids, pilot_static_features, cls, cluster_transition_probabilities, m_values, q_values = pickle.load(fr)
fr.close()

for i in range(CONFIG['clusters']):
    cluster_beneficiaries = beneficiaries[beneficiaries['cluster']==i]
    cluster_b_user_ids = cluster_beneficiaries['user_id']
    probs = get_transition_probabilities(cluster_b_user_ids, transitions, min_support=3)
    probs['cluster'] = i
    probs['count'] = cluster_b_user_ids.shape[0]
    probs['TEST/TRAIN'] = 'Test - pilot data'
    transition_probabilities = transition_probabilities.append(probs, ignore_index=True)
    # cluster_beneficiaries = train_beneficiaries[train_beneficiaries['cluster'] == i]
    # cluster_b_user_ids = cluster_beneficiaries['user_id']
    # probs = get_transition_probabilities(cluster_b_user_ids, train_transitions, min_support=3)
    # # tuple_count = get_tuple_count(cluster_b_user_ids, train_transitions)
    # probs['cluster'] = i
    # probs['count'] = len(cluster_b_user_ids)
    probs = dict()
    x = cluster_transition_probabilities[cluster_transition_probabilities['cluster'] == i ]
    probs['P(E, I, E)'] = x['P(E, I, E)'].item()
    probs['P(E, I, NE)'] = x['P(E, I, NE)'].item()
    probs['P(NE, I, E)'] = x['P(NE, I, E)'].item()
    probs['P(NE, I, NE)'] = x['P(NE, I, NE)'].item()
    probs['P(E, A, E)'] = x['P(E, A, E)'].item()
    probs['P(E, A, NE)'] = x['P(E, A, NE)'].item()
    probs['P(NE, A, E)'] = x['P(NE, A, E)'].item()
    probs['P(NE, A, NE)'] = x['P(NE, A, NE)'].item()
    
    probs['C(E, I, E)'] = '-'
    probs['C(E, I, NE)'] = '-'
    probs['C(NE, I, E)'] = '-'
    probs['C(NE, I, NE)'] = '-'
    probs['C(E, A, E)'] = '-'
    probs['C(E, A, NE)'] = '-'
    probs['C(NE, A, E)'] = '-'
    probs['C(NE, A, NE)'] = '-'

    probs['cluster'] = i
    probs['count'] = x['count'].item()   
    probs['TEST/TRAIN'] = 'Train'
    transition_probabilities = transition_probabilities.append(probs, ignore_index=True)

transition_probabilities.to_csv("groundtruth_analysis/transition_probabilities_week_9_comp.csv")
