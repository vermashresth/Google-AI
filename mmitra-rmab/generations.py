import numpy as np
import pandas as pd
import sys
import os
import pickle
import csv
import ipdb
from collections import OrderedDict
# import random

CONFIG = {
    'top-k_rmab': 500,
    'top-k_round_robin': 500,
    'date': '050721',
    'week': 'week11',
    'calling_files': ['250_week1_290421', '400_week2_060521', '400_week3_120521', '400_week4_180521', '435_week5_240521', '600_week6_310521', '700_week7_070621', '1000_week8_140621', '1000_week9_210621', '1000_week10_280621'],
    'neha_list_weeks': ['week1', 'week2', 'week3', 'week4', 'week5', 'week6', 'week7'],
    'calling_list': sys.argv[2],
    'round_robin_list': sys.argv[3]
}

np.random.seed(1)

df = pd.read_csv(sys.argv[1])
df = df.sort_values('{}_whittle'.format(CONFIG['week']), ascending=False)

df2 = pd.read_csv('outputs/individual_clustering/weekly_kmeans_pilot_stats_40.csv')

rmab_list = pd.read_csv('outputs/pilot_outputs/rmab_pilot.csv')['user_id'].to_list()
round_robin_list = pd.read_csv('outputs/pilot_outputs/round_robin_pilot.csv')['user_id'].to_list()
control_list = pd.read_csv('outputs/pilot_outputs/control_pilot.csv')['user_id'].to_list()

rmab_group = df[df['user_id'].isin(rmab_list)]
rmab_group = rmab_group.sort_values('{}_whittle'.format(CONFIG['week']), ascending=False)
rmab_group.to_csv('outputs/pilot_outputs/rmab_pilot_{}.csv'.format(CONFIG['week']))

round_robin_group = df2[df2['user_id'].isin(round_robin_list)]
round_robin_group = round_robin_group.sort_values('registration_date', ascending=True)
round_robin_group.to_csv('outputs/pilot_outputs/round_robin_pilot_{}.csv'.format(CONFIG['week']))

# control_group = df[df['user_id'].isin(control_list)]
# control_group.to_csv('outputs/pilot_outputs/control_pilot_{}.csv'.format(CONFIG['week']))

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

round_robin_all_list = pd.read_csv(CONFIG['round_robin_list'], header=None, names=['user_id'])
round_robin_all_list = round_robin_all_list['user_id'].to_list()

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

# skip_categories = [0,6,7,9,11]
# skip_categories = intervention_calling_status['Call Outcome'].unique()[skip_categories]
skip_categories = ['disconnected', 'out of coverage', 'ringing', 'switched off', 'network busy' , 'successful call']

rmab_filtered_ids = []
for user_id in rmab_user_ids:
    if user_id in intervention_dict and len(intervention_dict[user_id]) >= 3:
        continue
    if user_id not in opt_out_list:
        flag = True
        for i in range(len(intervention_calling_status_list)):
            if user_id in intervention_calling_status_list[i]:
                int_call_status = intervention_calling_status[i][intervention_calling_status[i]['UserID'] == user_id]['Call Outcome'].item()
                if type(int_call_status) == type('str') and int_call_status.lower() not in skip_categories:
                    flag = False
        if flag:
            rmab_filtered_ids.append(user_id)

round_robin_filtered_ids = []
for user_id in round_robin_user_ids:
    # int_calling_status = intervention_calling_status[intervention_calling_status['UserID'] == user_id]['Call Outcome'].item()
    # if user_id in intervention_calling_status_list and int_calling_status.lower() in skip_categories:
    #     round_robin_filtered_ids.append(user_id)
    if user_id not in round_robin_all_list:
        round_robin_filtered_ids.append(user_id)

for i in range(CONFIG['top-k_rmab']):
    assert rmab_filtered_ids[i] not in previous_calling_list

for i in range(CONFIG['top-k_round_robin']):
    assert round_robin_filtered_ids[i] not in previous_calling_list

write_file(rmab_filtered_ids[:CONFIG['top-k_rmab']], 'outputs/pilot_generations/rmab_list_{}_{}_{}.txt'.format(CONFIG['top-k_rmab'], CONFIG['week'], CONFIG['date']))
write_file(round_robin_filtered_ids[:CONFIG['top-k_round_robin']], 'outputs/pilot_generations/round_robin_list_{}_{}_{}.txt'.format(CONFIG['top-k_round_robin'], CONFIG['week'], CONFIG['date']))

calling_list = rmab_filtered_ids[:CONFIG['top-k_rmab']] + round_robin_filtered_ids[:CONFIG['top-k_round_robin']]
np.random.shuffle(calling_list)
write_file(calling_list, 'outputs/pilot_generations/calling_list_{}_{}_{}.txt'.format(CONFIG['top-k_rmab'] + CONFIG['top-k_round_robin'], CONFIG['week'], CONFIG['date']))

# ipdb.set_trace()