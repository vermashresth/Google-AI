from matplotlib.pyplot import get
import numpy as np
import pandas as pd
import sys
import os
import pickle
import csv
import ipdb
from tqdm import tqdm
from collections import OrderedDict, defaultdict
# import random

from training.utils import load_obj, save_obj
from training.data import load_data
from training.dataset import _preprocess_call_data, preprocess_and_make_dataset

from training.modelling.metrics import F1, Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model

np.random.seed(1)

CONFIG = {
    'pilot_data': 'feb16-mar15_data',
    'current_week': 11,
    'week': 'week11'
}
T = 12


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
group_cluster_size = defaultdict(lambda: defaultdict(lambda: 0))
for file in ['250_week1_290421', '400_week2_060521', '400_week3_120521', '400_week4_180521', '435_week5_240521', '600_week6_310521', '700_week7_070621', '1000_week8_140621']:
    with open('outputs/pilot_generations/calling_list_{}.txt'.format(file), 'r') as fr:
        for line in fr:
            user_id = int(line.strip())
            curr_cluster = complete_group[complete_group['user_id'] == user_id]['cluster'].item()
            if user_id not in intervention_dict:
                intervention_dict[user_id] = [int(file.split('_')[1][4:])]
            else:
                intervention_dict[user_id].append(int(file.split('_')[1][4:]))
            if user_id in rmab_user_ids:
                group_cluster_size['rmab'][curr_cluster] += 1
            elif user_id in round_robin_user_ids:
                group_cluster_size['round_robin'][curr_cluster] += 1

rmab_cluster_list = [(group_cluster_size['rmab'][cluster], cluster) for cluster in range(40)]
round_robin_cluster_list = [(group_cluster_size['round_robin'][cluster], cluster) for cluster in range(40)]

rmab_cluster_list = sorted(rmab_cluster_list, reverse=True)[:5]
round_robin_cluster_list = sorted(round_robin_cluster_list, reverse=True)[:5]

top_cluster_list = {'rmab': rmab_cluster_list, 'round_robin': round_robin_cluster_list}


df = pd.read_csv('outputs/analysis_lists/all_analysis_week_12.csv')

out_dict = {'user_id': [], 'pre-action state': [], 'action': [], 'post-action state': []}

for user_id in tqdm(all_user_ids):
    curr_row = df[df['user_id'] == user_id]

    engagements = []
    for i in range(T):
        counte = int(curr_row['week{}_E/C'.format(i)].item().split('/')[0])
        engagements.append(counte)

    if user_id in intervention_dict:
        user_intervention_list = intervention_dict[user_id]
    else:
        user_intervention_list = []

    assert len(engagements) == T
    for i in range(T-1):
        start_state = 'L' if engagements[i] > 0 else 'H'
        next_state = 'L' if engagements[i + 1] > 0 else 'H'
        action = 'Intervention' if i in user_intervention_list else 'No Intervention'
        out_dict['user_id'].append(user_id)
        out_dict['pre-action state'].append(start_state)
        out_dict['action'].append(action)
        out_dict['post-action state'].append(next_state)

transitions_df = pd.DataFrame(out_dict)

transitions_df.to_csv('outputs/pilot_transitions.csv')
