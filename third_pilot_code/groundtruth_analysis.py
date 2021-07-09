import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import ipdb
import pickle


CONFIG = {
    'clusters': int(sys.argv[1]),
    'clustering': sys.argv[2],
    'file_path': sys.argv[3]    #Path to the pickle file
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

def get_probability_comparison():

    transitions = pd.read_csv("groundtruth_analysis/transitions_week_9.csv")
    beneficiaries = pd.read_csv("outputs/individual_clustering/weekly_{}_pilot_stats_{}.csv".format(CONFIG['clustering'], CONFIG['clusters']))

    cols = [
            'P(E, I, E)', 'P(E, I, NE)', 'P(NE, I, E)', 'P(NE, I, NE)', 'P(E, A, E)', 'P(E, A, NE)', 'P(NE, A, E)', 'P(NE, A, NE)', 
        ]
    cols += [
            'C(E, I, E)', 'C(E, I, NE)', 'C(NE, I, E)', 'C(NE, I, NE)', 'C(E, A, E)', 'C(E, A, NE)', 'C(NE, A, E)', 'C(NE, A, NE)', 
        ]
        
    transition_probabilities = pd.DataFrame(columns=['TEST/TRAIN', 'cluster', 'count'] + cols)
    try:
        with open(CONFIG['file_path'], 'rb') as fr:
            pilot_user_ids = pickle.load(fr) 
            pilot_static_features = pickle.load(fr) 
            cls = pickle.load(fr) 
            cluster_transition_probabilities = pickle.load(fr) 
            m_values = pickle.load(fr) 
            q_values = pickle.load(fr)
        fr.close()
    except Exception as e:
        with open(CONFIG['file_path'], 'rb') as fr:
            pilot_user_ids,pilot_static_features,cls,cluster_transition_probabilities,m_values,q_values = pickle.load(fr)
        fr.close()

    # ipdb.set_trace()

    for i in range(CONFIG['clusters']):
        cluster_beneficiaries = beneficiaries[beneficiaries['cluster']==i]
        cluster_b_user_ids = cluster_beneficiaries['user_id']
        probs = get_transition_probabilities(cluster_b_user_ids, transitions, min_support=3)
        probs['cluster'] = i
        probs['count'] = cluster_b_user_ids.shape[0]
        probs['TEST/TRAIN'] = 'Test - pilot data'
        transition_probabilities = transition_probabilities.append(probs, ignore_index=True)
        probs = dict()
        x = cluster_transition_probabilities[cluster_transition_probabilities['cluster'] == i ]
        try:
            probs['P(E, I, E)'] = x['P(L, I, L)'].item()
            probs['P(E, I, NE)'] = x['P(L, I, H)'].item()
            probs['P(NE, I, E)'] = x['P(H, I, L)'].item()
            probs['P(NE, I, NE)'] = x['P(H, I, H)'].item()
            probs['P(E, A, E)'] = x['P(L, N, L)'].item()
            probs['P(E, A, NE)'] = x['P(L, N, H)'].item()
            probs['P(NE, A, E)'] = x['P(H, N, L)'].item()
            probs['P(NE, A, NE)'] = x['P(H, N, H)'].item()
        except Exception as e:
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

    transition_probabilities.to_csv("groundtruth_analysis/{}_{}_week_9_comp.csv".format(CONFIG['clustering'], CONFIG['clusters']))

def get_rmse():

    transition_probabilities = pd.read_csv("groundtruth_analysis/{}_{}_week_9_comp.csv".format(CONFIG['clustering'], CONFIG['clusters']))
    test_transition_probabilities = transition_probabilities[transition_probabilities['TEST/TRAIN'] == 'Test - pilot data']
    train_transition_probabilities = transition_probabilities[transition_probabilities['TEST/TRAIN'] == 'Train']
    # ipdb.set_trace()
    active_transition_probs_test = []
    active_transition_probs_test += test_transition_probabilities['P(E, I, E)'].values.tolist()
    active_transition_probs_test += test_transition_probabilities['P(E, I, NE)'].values.tolist()
    active_transition_probs_test += test_transition_probabilities['P(NE, I, E)'].values.tolist()
    active_transition_probs_test += test_transition_probabilities['P(NE, I, NE)'].values.tolist()

    active_transition_probs_train = []
    active_transition_probs_train += train_transition_probabilities['P(E, I, E)'].values.tolist()
    active_transition_probs_train += train_transition_probabilities['P(E, I, NE)'].values.tolist()
    active_transition_probs_train += train_transition_probabilities['P(NE, I, E)'].values.tolist()
    active_transition_probs_train += train_transition_probabilities['P(NE, I, NE)'].values.tolist()

    passive_transition_probs_test = []
    passive_transition_probs_test += test_transition_probabilities['P(E, A, E)'].values.tolist()
    passive_transition_probs_test += test_transition_probabilities['P(E, A, NE)'].values.tolist()
    passive_transition_probs_test += test_transition_probabilities['P(NE, A, E)'].values.tolist()
    passive_transition_probs_test += test_transition_probabilities['P(NE, A, NE)'].values.tolist()

    passive_transition_probs_train = []
    passive_transition_probs_train += train_transition_probabilities['P(E, A, E)'].values.tolist()
    passive_transition_probs_train += train_transition_probabilities['P(E, A, NE)'].values.tolist()
    passive_transition_probs_train += train_transition_probabilities['P(NE, A, E)'].values.tolist()
    passive_transition_probs_train += train_transition_probabilities['P(NE, A, NE)'].values.tolist()
    # ipdb.set_trace()
    p_train = []
    p_test = []
    for i in range(len(passive_transition_probs_train)):
        if pd.isna(passive_transition_probs_train[i]) or pd.isna(passive_transition_probs_test[i]):
            continue
        p_train.append(passive_transition_probs_train[i])
        p_test.append(passive_transition_probs_test[i])

    a_test = []
    a_train = []
    for i in range(len(active_transition_probs_train)):
        if pd.isna(active_transition_probs_train[i]) or pd.isna(active_transition_probs_test[i]):
            continue
        a_train.append(active_transition_probs_train[i])
        a_test.append(active_transition_probs_test[i])

    # ipdb.set_trace()
    passive_rmse = mean_squared_error(p_test, p_train, squared=False)
    print(passive_rmse)
    active_rmse = mean_squared_error(a_test, a_train, squared=False)
    print(active_rmse)
    overall_rmse = mean_squared_error(p_test+a_test, p_train+a_train, squared=False)
    print(overall_rmse)

get_probability_comparison()
get_rmse()