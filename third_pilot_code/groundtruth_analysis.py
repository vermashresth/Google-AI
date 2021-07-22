import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import ipdb
import pickle


CONFIG = {
    'clusters': int(sys.argv[1]),
    # 'm_n': sys.argv[1],
    'clustering': sys.argv[2],
    'file_path': sys.argv[3],    #Path to the pickle file
    'linkage': 'average'
    }

def get_transition_probabilities(beneficiaries, transitions, min_support=3):
    """
    This method is used to get the transition probabilities and the count corresponding 
    to each transition tuple.
    """

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
    """
    This methods returns a df that has a comparison of cluster transition probabilities 
    based on the train data and test data respectively.
    """

    transitions = pd.read_csv("groundtruth_analysis/transitions_week_9.csv")
    if CONFIG['clustering'] == 'agglomerative':
        beneficiaries = pd.read_csv("outputs/individual_clustering/weekly_{}_pilot_stats_{}_{}.csv".format(CONFIG['clustering'], CONFIG['clusters'], CONFIG['linkage']))
    elif CONFIG['clustering'] == 'som':
        beneficiaries = pd.read_csv("outputs/individual_clustering/weekly_{}_pilot_stats_{}.csv".format(CONFIG['clustering'], CONFIG['m_n']))
    else:
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
        # ipdb.set_trace()
        try:
            try:
                probs['P(E, I, E)'] = x['P(L, I, L)'].item()
                probs['P(E, I, NE)'] = x['P(L, I, H)'].item()
                probs['P(NE, I, E)'] = x['P(H, I, L)'].item()
                probs['P(NE, I, NE)'] = x['P(H, I, H)'].item()
                probs['P(E, A, E)'] = x['P(L, N, L)'].item()
                probs['P(E, A, NE)'] = x['P(L, N, H)'].item()
                probs['P(NE, A, E)'] = x['P(H, N, L)'].item()
                probs['P(NE, A, NE)'] = x['P(H, N, H)'].item()
            except IndexError as e:
                # ipdb.set_trace()
                probs['P(E, I, E)'] = x['P(E, I, E)'].item()
                probs['P(E, I, NE)'] = x['P(E, I, NE)'].item()
                probs['P(NE, I, E)'] = x['P(NE, I, E)'].item()
                probs['P(NE, I, NE)'] = x['P(NE, I, NE)'].item()
                probs['P(E, A, E)'] = x['P(E, A, E)'].item()
                probs['P(E, A, NE)'] = x['P(E, A, NE)'].item()
                probs['P(NE, A, E)'] = x['P(NE, A, E)'].item()
                probs['P(NE, A, NE)'] = x['P(NE, A, NE)'].item()
        except ValueError as e:
            continue
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

def get_all_transition_probabilities(train_beneficiaries, transitions):
    """
    This method returns the transition probabilities corresponding to each 
    beneficiary.
    """
    cols = [
        "P(E, I, E)", "P(E, I, NE)", "P(NE, I, E)", "P(NE, I, NE)", "P(E, A, E)", "P(E, A, NE)", "P(NE, A, E)", "P(NE, A, NE)",
    ]
    transition_probabilities = pd.DataFrame(columns = ['user_id'] + cols)
    user_ids = train_beneficiaries['user_id']
 
    for user_id in user_ids:
        probs = get_transition_probabilities([user_id], transitions, min_support=1)
        probs['user_id'] = user_id
 
        transition_probabilities = transition_probabilities.append(probs, ignore_index=True)
 
    return transition_probabilities


def get_rmse():
    """
    This method returns the overall RMSE, RMSE based on active transition statistics,
    RMSE based on passive transition statistics of the ground truth.
    """

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

    # Here, we remove the nan values and corresponding comparison probabilities

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


def get_average_rmse(beneficiaries, transitions):
    """
    This method returns the average RMSE based on the transitions
    and cluster transition probabilities.
    """
    
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

    tp = get_all_transition_probabilities(beneficiaries, transitions)

    rmse_sum = 0

    for user_id in list(tp['user_id'].values):
        curr_row = tp[tp['user_id'] ==  user_id]
        probs_test = curr_row.values.tolist()[0][1:9]
        cluster = beneficiaries[beneficiaries['user_id'] == user_id ]['cluster'].item()
        cluster_row = cluster_transition_probabilities[cluster_transition_probabilities['cluster'] == cluster]
        cluster_probs = cluster_row.values.tolist()[0][2:]
        
        # a and b are the transition statistics corresponding to each beneficiary during test and train respectively
        a = list()
        b = list()
        for i in range(8):
            if pd.isna(probs_test[i]) or pd.isna(cluster_probs[i]):
                continue
            a.append(probs_test[i])
            b.append(cluster_probs[i])
        if len(a) == 0:
            continue
        # This method return rmse for sklearn > 0.22.0
        rmse = mean_squared_error(a, b, squared=False)
        rmse_sum += rmse

    average_rmse = rmse_sum/len(tp)
    return average_rmse

get_probability_comparison()
get_rmse()
transitions = pd.read_csv("groundtruth_analysis/transitions_week_9.csv")
if CONFIG['clustering'] == 'agglomerative':
    beneficiaries = pd.read_csv("outputs/individual_clustering/weekly_{}_pilot_stats_{}_{}.csv".format(CONFIG['clustering'], CONFIG['clusters'], CONFIG['linkage']))
elif CONFIG['clustering'] == 'som':
    beneficiaries = pd.read_csv("outputs/individual_clustering/weekly_{}_pilot_stats_{}.csv".format(CONFIG['clustering'], CONFIG['m_n']))
else:
    beneficiaries = pd.read_csv("outputs/individual_clustering/weekly_{}_pilot_stats_{}.csv".format(CONFIG['clustering'], CONFIG['clusters']))

print(get_average_rmse(beneficiaries, transitions))