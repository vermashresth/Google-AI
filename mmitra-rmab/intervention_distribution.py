import numpy as np
from numpy.core.fromnumeric import reshape
import pandas as pd
import sys
import os
import csv
import pickle
import ipdb
from collections import OrderedDict, defaultdict

np.random.seed(1)

CONFIG = {
    'calling_files': ['250_week1_290421', '400_week2_060521', '400_week3_120521', '400_week4_180521', '435_week5_240521', '600_week6_310521', '700_week7_070621', '1000_week8_140621'],
    'neha_list_weeks': ['week1', 'week2', 'week3', 'week4', 'week5', 'week6', 'week7', 'week8']
}

intervention_dict = {}
for file in CONFIG['calling_files']:
    with open('outputs/pilot_generations/calling_list_{}.txt'.format(file), 'r') as fr:
        for line in fr:
            user_id = int(line.strip())
            if user_id not in intervention_dict:
                intervention_dict[user_id] = [file.split('_')[1]]
            else:
                intervention_dict[user_id].append(file.split('_')[1])

opt_out_excel = pd.read_excel('neha_lists/optout_requests_week2.xlsx', header=None)
opt_out_list = set(opt_out_excel[0].to_list())

intervention_calling_status = []
intervention_calling_status_list = []
for idx, neha_list_week in enumerate(CONFIG['neha_list_weeks']):
    df_excel = pd.read_excel('neha_lists/neha_calling_status_{}.xlsx'.format(neha_list_week))
    intervention_calling_status.append(df_excel)
    intervention_calling_status_list.append(intervention_calling_status[-1]['UserID'].to_list())
    if idx > 1:
        curr_user_list = df_excel['UserID'].to_list()
        for user in curr_user_list:
            row_excel = df_excel[df_excel['UserID'] == user]
            opt_out = row_excel['Opt Out'].item()
            if idx <= 3:
                if type(opt_out) == type('str') and len(opt_out.strip()) > 0:
                    opt_out_list.add(user)
            else:
                if type(opt_out) == type('str') and opt_out.strip().lower() == 'yes':
                    opt_out_list.add(user)

complete_group = pd.read_csv('outputs/individual_clustering/weekly_kmeans_pilot_stats_40.csv')
all_user_ids = complete_group['user_id'].to_list()
rmab_group = pd.read_csv('outputs/pilot_outputs/rmab_pilot.csv')
round_robin_group = pd.read_csv('outputs/pilot_outputs/round_robin_pilot.csv')

rmab_user_ids = rmab_group['user_id'].to_list()
round_robin_user_ids = round_robin_group['user_id'].to_list()

groups = {'rmab_success': [], 'rmab_intervention': [], 'round_robin_success': [], 'round_robin_intervention': [], 'no_intervention': []}

for user_id in all_user_ids:
    if user_id in intervention_dict and user_id in rmab_user_ids:
        for i in range(len(intervention_calling_status_list)):
            if user_id in intervention_calling_status_list[i]:
                int_call_status = intervention_calling_status[i][intervention_calling_status[i]['UserID'] == user_id]['Call Outcome'].item()
                if type(int_call_status) == type('str') and int_call_status.lower() == 'successful call':
                    groups['rmab_success'].append(user_id)
                    break
        groups['rmab_intervention'].append(user_id)
    elif user_id in intervention_dict and user_id in round_robin_user_ids:
        for i in range(len(intervention_calling_status_list)):
            if user_id in intervention_calling_status_list[i]:
                int_call_status = intervention_calling_status[i][intervention_calling_status[i]['UserID'] == user_id]['Call Outcome'].item()
                if type(int_call_status) == type('str') and int_call_status.lower() == 'successful call':
                    groups['round_robin_success'].append(user_id)
                    break
        groups['round_robin_intervention'].append(user_id)
    elif user_id not in intervention_dict:
        groups['no_intervention'].append(user_id)
ipdb.set_trace()
pilot_pd_data = pd.read_csv("feb16-mar15_data/beneficiary/ai_registration-20210216-20210315.csv", sep='\t')

# g1 = pilot_pd_data[pilot_pd_data['user_id'].isin(groups['rmab_intervention'])]
# g2 = pilot_pd_data[pilot_pd_data['user_id'].isin(groups['all_intervention'])]
# g3 = pilot_pd_data[pilot_pd_data['user_id'].isin(groups['no_intervention'])]

g = {}
for idx, key in enumerate(groups):
    g[idx + 1] = pilot_pd_data[pilot_pd_data['user_id'].isin(groups[key])]

numeric_features = ['enroll_gest_age', 'stage', 'age', 'g', 'p', 's', 'l', 'a']
date_based_features = ['lmp', 'registration_date']
class_based_features = ['ChannelType', 'education', 'phone_owner', 'income_bracket', 'ngo_hosp_id']

print('NUMERIC FEATURES STATS')
for f in numeric_features:
    print('-'*60)
    print('Feature {}'.format(f))
    for i in range(len(g)):
        print('G{}: Mean - {}, Std - {}'.format(i + 1, g[i + 1][f].mean(), g[i + 1][f].std()))
        # print('G2: Mean - {}, Std - {}'.format(g2[f].mean(), g2[f].std()))
        # print('G3: Mean - {}, Std - {}'.format(g3[f].mean(), g3[f].std()))

print('\n\nDATE-BASED FEATURES STATS')
for f in date_based_features:
    print('-'*60)
    print('Feature {}'.format(f))
    for i in range(len(g)):
        g_date = pd.to_datetime(g[i + 1][f], format='%Y-%m-%d') - pd.to_datetime('2018-01-01', format='%Y-%m-%d')
        print('G{}: Mean - {}, Std - {}'.format(i + 1, g_date.mean(), g_date.std()))
    # print('G2: Mean - {}, Std - {}'.format(g2_date.mean(), g2_date.std()))
    # print('G3: Mean - {}, Std - {}'.format(g3_date.mean(), g3_date.std()))

print('\n\nCLASS-BASED FEATURES STATS')
for f in class_based_features:
    print('-'*60)
    print('Feature {}'.format(f))
    for i in range(len(g)):
        g_dict = {}
        g_vals = g[i + 1][f].to_list()
        for val in g_vals:
            if val in g_dict:
                g_dict[val] += 1
            else:
                g_dict[val] = 1
        for key in g_dict:
            g_dict[key] = round((100.0 * g_dict[key]) / len(g_vals), 2)
    # g2_dict = {}
    # g2_vals = g2[f].to_list()
    # for val in g2_vals:
    #     if val in g2_dict:
    #         g2_dict[val] += 1
    #     else:
    #         g2_dict[val] = 1
    # g3_dict = {}
    # g3_vals = g3[f].to_list()
    # for val in g3_vals:
    #     if val in g3_dict:
    #         g3_dict[val] += 1
    #     else:
    #         g3_dict[val] = 1
        print('G{}: {}'.format(i + 1, OrderedDict(sorted(g_dict.items()))))
    # print('G2: {}'.format(OrderedDict(sorted(g2_dict.items()))))
    # print('G3: {}'.format(OrderedDict(sorted(g3_dict.items()))))

# ipdb.set_trace()
columns = [
    "enroll_gest_age",
    "enroll_delivery_status",
    "g",
    "p",
    "s",
    "l",
    "a",
    "days_to_first_call",
    "age_0",
    "age_1",
    "age_2",
    "age_3",
    "age_4",
    "language_2",
    "language_3",
    "language_4",
    "language_5",
    "education_1",
    "education_2",
    "education_3",
    "education_4",
    "education_5",
    "education_6",
    "education_7",
    "phone_owner_0",
    "phone_owner_1",
    "phone_owner_2",
    "call_slots_1",
    "call_slots_2",
    "call_slots_3",
    "call_slots_4",
    "call_slots_5",
    "call_slots_6",
    "ChannelType_0",
    "ChannelType_1",
    "ChannelType_2",
    "income_bracket_-1",
    "income_bracket_0",
    "income_bracket_1",
    "income_bracket_2",
    "income_bracket_3",
    "income_bracket_4",
    "income_bracket_5",
    "income_bracket_6",
]

with open('policy_dump.pkl', 'rb') as fr:
  pilot_user_ids, pilot_static_features, cls, cluster_transition_probabilities, m_values, q_values = pickle.load(fr)
fr.close()

fimp = cls.feature_importances_
fimp = fimp.reshape((fimp.shape[0], 1))
fimp_idx = np.concatenate((fimp, np.arange(fimp.shape[0]).reshape((fimp.shape[0], 1))), axis=1)

sorted_arr = fimp_idx[(-fimp_idx)[:, 0].argsort()]

class_scores = defaultdict(lambda: 0)
for i in range(fimp_idx.shape[0]):
    col_name = columns[int(sorted_arr[i, 1])]
    curr_score = round(sorted_arr[i, 0], 3)
    print('{}\t{}'.format(col_name, curr_score))
    if 'age_' in col_name:
        class_scores['age'] += curr_score
    if 'education_' in col_name:
        class_scores['education'] += curr_score
    if 'income_bracket_' in col_name:
        class_scores['income_bracket'] += curr_score
    if 'language_' in col_name:
        class_scores['language'] += curr_score

print(class_scores)