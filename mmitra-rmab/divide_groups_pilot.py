import numpy as np
import pandas as pd
import sys
import os
import csv
import ipdb
from collections import OrderedDict
# import random

# UPDATE THIS CONFIG WEEKLY TO GENERATE THE PREDICTIONS
CONFIG = {
    'top-k': 200,
    'date': '060521',
    'week': 'week2',
    'calling_list': sys.argv[2]
}

np.random.seed(1)

df = pd.read_csv(sys.argv[1])
rmab_list = pd.read_csv('outputs/pilot_outputs/rmab_pilot.csv')['user_id'].to_list()
round_robin_list = pd.read_csv('outputs/pilot_outputs/round_robin_pilot.csv')['user_id'].to_list()
control_list = pd.read_csv('outputs/pilot_outputs/control_pilot.csv')['user_id'].to_list()

rmab_group = df[df['user_id'].isin(rmab_list)]
rmab_group = rmab_group.sort_values('whittle_index', ascending=False)
rmab_group.to_csv('outputs/pilot_outputs/rmab_pilot_{}.csv'.format(CONFIG['week']))

round_robin_group = df[df['user_id'].isin(round_robin_list)]
round_robin_group = round_robin_group.sort_values('registration_date', ascending=True)
round_robin_group.to_csv('outputs/pilot_outputs/round_robin_pilot_{}.csv'.format(CONFIG['week']))

control_group = df[df['user_id'].isin(control_list)]
control_group.to_csv('outputs/pilot_outputs/control_pilot_{}.csv'.format(CONFIG['week']))

rmab_user_ids = rmab_group['user_id'].to_list()
round_robin_user_ids = round_robin_group['user_id'].to_list()

def write_file(lst, fname):
    with open(fname, 'w') as fw:
        for user in lst:
            fw.write('{}\n'.format(user))
    fw.close()
    return

previous_calling_list = pd.read_csv(CONFIG['calling_list'], header=None, names=['user_id'])
previous_calling_list = previous_calling_list['user_id'].to_list()

round_robin_filtered_ids = []
for user_id in round_robin_user_ids:
    if user_id not in previous_calling_list:
        round_robin_filtered_ids.append(user_id)

write_file(rmab_user_ids[:CONFIG['top-k']], 'outputs/pilot_generations/rmab_list_{}_{}_{}.txt'.format(CONFIG['top-k'], CONFIG['week'], CONFIG['date']))
write_file(round_robin_filtered_ids[:CONFIG['top-k']], 'outputs/pilot_generations/round_robin_list_{}_{}_{}.txt'.format(CONFIG['top-k'], CONFIG['week'], CONFIG['date']))

calling_list = rmab_user_ids[:CONFIG['top-k']] + round_robin_filtered_ids[:CONFIG['top-k']]
np.random.shuffle(calling_list)
write_file(calling_list, 'outputs/pilot_generations/calling_list_{}_{}_{}.txt'.format(2*CONFIG['top-k'], CONFIG['week'], CONFIG['date']))
control_list = control_group['user_id'].to_list()
np.random.shuffle(control_list)
write_file(control_list, 'outputs/pilot_generations/control_group_list_{}_{}.txt'.format(CONFIG['week'], CONFIG['date']))

## GET FEATURE DISTRIBUTIONS (UNCOMMENT TO GENERATE THE DISTRIBUTIONS: ONLY FOR VALIDATION)

# def partition(lst, n):
#     np.random.shuffle(lst)
#     return [lst[i::n] for i in range(n)]
# user_ids = df['user_id'].to_list()

# groups = partition(user_ids, 3)

# pilot_pd_data = pd.read_csv("feb16-mar15_data/beneficiary/ai_registration-20210216-20210315.csv", sep='\t')

# g1 = pilot_pd_data[pilot_pd_data['user_id'].isin(groups[0])]
# g2 = pilot_pd_data[pilot_pd_data['user_id'].isin(groups[1])]
# g3 = pilot_pd_data[pilot_pd_data['user_id'].isin(groups[2])]

# numeric_features = ['enroll_gest_age', 'stage', 'age', 'g', 'p', 's', 'l', 'a', 'ngo_hosp_id']
# date_based_features = ['lmp', 'registration_date']
# class_based_features = ['ChannelType', 'education', 'phone_owner', 'income_bracket']

# print('NUMERIC FEATURES STATS')
# for f in numeric_features:
#     print('-'*60)
#     print('Feature {}'.format(f))
#     print('G1: Mean - {}, Std - {}'.format(g1[f].mean(), g1[f].std()))
#     print('G2: Mean - {}, Std - {}'.format(g2[f].mean(), g2[f].std()))
#     print('G3: Mean - {}, Std - {}'.format(g3[f].mean(), g3[f].std()))

# print('\n\nDATE-BASED FEATURES STATS')
# for f in date_based_features:
#     print('-'*60)
#     print('Feature {}'.format(f))
#     g1_date = pd.to_datetime(g1[f], format='%Y-%m-%d') - pd.to_datetime('2018-01-01', format='%Y-%m-%d')
#     g2_date = pd.to_datetime(g2[f], format='%Y-%m-%d') - pd.to_datetime('2018-01-01', format='%Y-%m-%d')
#     g3_date = pd.to_datetime(g3[f], format='%Y-%m-%d') - pd.to_datetime('2018-01-01', format='%Y-%m-%d')
#     print('G1: Mean - {}, Std - {}'.format(g1_date.mean(), g1_date.std()))
#     print('G2: Mean - {}, Std - {}'.format(g2_date.mean(), g2_date.std()))
#     print('G3: Mean - {}, Std - {}'.format(g3_date.mean(), g3_date.std()))

# print('\n\nCLASS-BASED FEATURES STATS')
# for f in class_based_features:
#     print('-'*60)
#     print('Feature {}'.format(f))
#     g1_dict = {}
#     g1_vals = g1[f].to_list()
#     for val in g1_vals:
#         if val in g1_dict:
#             g1_dict[val] += 1
#         else:
#             g1_dict[val] = 1
#     g2_dict = {}
#     g2_vals = g2[f].to_list()
#     for val in g2_vals:
#         if val in g2_dict:
#             g2_dict[val] += 1
#         else:
#             g2_dict[val] = 1
#     g3_dict = {}
#     g3_vals = g3[f].to_list()
#     for val in g3_vals:
#         if val in g3_dict:
#             g3_dict[val] += 1
#         else:
#             g3_dict[val] = 1
#     print('G1: {}'.format(OrderedDict(sorted(g1_dict.items()))))
#     print('G2: {}'.format(OrderedDict(sorted(g2_dict.items()))))
#     print('G3: {}'.format(OrderedDict(sorted(g3_dict.items()))))
