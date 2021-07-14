from matplotlib.pyplot import get
import numpy as np
import pandas as pd
import sys
import os
import pickle
import csv
# import ipdb
from tqdm import tqdm
from collections import OrderedDict
# import random

from training.utils import load_obj, save_obj
from training.data import load_data
from training.dataset import _preprocess_call_data, preprocess_and_make_dataset

from training.modelling.metrics import F1, Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model

np.random.seed(1)

CONFIG = {
    'pilot_data': 'feb16-mar15_data',
    'current_week': 9,
    'week': 'week9'
}

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


pilot_beneficiary_data, pilot_call_data = load_data(CONFIG['pilot_data'])
pilot_call_data = _preprocess_call_data(pilot_call_data)

complete_group = pd.read_csv('outputs/individual_clustering/weekly_kmeans_pilot_stats_40.csv')
all_user_ids = complete_group['user_id'].to_list()
rmab_group = pd.read_csv('outputs/pilot_outputs/rmab_pilot.csv')
round_robin_group = pd.read_csv('outputs/pilot_outputs/round_robin_pilot.csv')

rmab_user_ids = rmab_group['user_id'].to_list()
round_robin_user_ids = round_robin_group['user_id'].to_list()

arm_dict = {}
for user_id in rmab_user_ids:
    arm_dict[user_id] = 'rmab'
for user_id in round_robin_user_ids:
    arm_dict[user_id] = 'round_robin'

with open('policy_dump.pkl', 'rb') as fr:
  pilot_user_ids, pilot_static_features, cls, cluster_transition_probabilities, m_values, q_values = pickle.load(fr)
fr.close()

intervention_dict = {}
for file in ['250_week1', '400_week2', '400_week3', '400_week4', '435_week5', '600_week6', '700_week7' ,'1000_week8']:
    with open('intervention_lists/calling_list_{}.txt'.format(file), 'r') as fr:
        for line in fr:
            user_id = int(line.strip())
            if user_id not in intervention_dict:
                intervention_dict[user_id] = [int(file.split('_')[1][4:])]
            else:
                intervention_dict[user_id].append(int(file.split('_')[1][4:]))

df = pd.read_csv(sys.argv[1])
df = df.sort_values('{}_whittle'.format(CONFIG['week']), ascending=False)

out_dict = {'user_id': [], 'pre-action state': [], 'action': [], 'post-action state': []}

for user_id in tqdm(all_user_ids):
    curr_row = df[df['user_id'] == user_id]

    engagements = []
    for i in range(9):
        counte = int(curr_row['week{}_E/C'.format(i)].item().split('/')[0])
        engagements.append(counte)

    if user_id in intervention_dict:
        user_intervention_list = intervention_dict[user_id]
    else:
        user_intervention_list = []

    assert len(engagements) == 9
    for i in range(8):
        start_state = 'L' if engagements[i] > 0 else 'H'
        next_state = 'L' if engagements[i + 1] > 0 else 'H'
        action = 'Intervention' if i in user_intervention_list else 'No Intervention'
        out_dict['user_id'].append(user_id)
        out_dict['pre-action state'].append(start_state)
        out_dict['action'].append(action)
        out_dict['post-action state'].append(next_state)

transitions_df = pd.DataFrame(out_dict)
transitions_df.to_csv("groundtruth_analysis/transitions_week_9.csv")

cluster_27_beneficiaries = df[(df['cluster'] == 27)]['user_id'].to_list()
cluster_29_beneficiaries = df[(df['cluster'] == 29)]['user_id'].to_list()

cluster_27_probs = get_transition_probabilities(cluster_27_beneficiaries, transitions_df)
cluster_29_probs = get_transition_probabilities(cluster_29_beneficiaries, transitions_df)
cluster_27_df = pd.DataFrame(cluster_27_probs)
cluster_29_df = pd.DataFrame(cluster_29_probs)

cluster_27_df.to_csv('27probs.csv')
cluster_29_df.to_csv('29probs.csv')

# ipdb.set_trace()