import numpy as np
import pandas as pd
import sys
import os
import csv
import ipdb
import pickle
from collections import OrderedDict
from tqdm import tqdm

CONFIG = {
    'week': 'week8'
    # 'calling_list': sys.argv[2],
    # 'round_robin_list': sys.argv[3]
}

np.random.seed(1)

df = pd.read_csv(sys.argv[1])
df = df.sort_values('{}_whittle'.format(CONFIG['week']), ascending=False)

df_check = pd.read_csv(sys.argv[2])
df_check = df_check.sort_values('week7_whittle', ascending=False)

df2 = pd.read_csv('outputs/individual_clustering/weekly_kmeans_pilot_stats_40.csv')

rmab_list = pd.read_csv('outputs/pilot_outputs/rmab_pilot.csv')['user_id'].to_list()
round_robin_list = pd.read_csv('outputs/pilot_outputs/round_robin_pilot.csv')['user_id'].to_list()
control_list = pd.read_csv('outputs/pilot_outputs/control_pilot.csv')['user_id'].to_list()

rmab_group = df[df['user_id'].isin(rmab_list)]
rmab_group = rmab_group.sort_values('{}_whittle'.format(CONFIG['week']), ascending=False)
# rmab_group.to_csv('outputs/pilot_outputs/rmab_pilot_{}.csv'.format(CONFIG['week']))

round_robin_group = df2[df2['user_id'].isin(round_robin_list)]
round_robin_group = round_robin_group.sort_values('registration_date', ascending=True)
# round_robin_group.to_csv('outputs/pilot_outputs/round_robin_pilot_{}.csv'.format(CONFIG['week']))

# control_group = df[df['user_id'].isin(control_list)]
# control_group.to_csv('outputs/pilot_outputs/control_pilot_{}.csv'.format(CONFIG['week']))

rmab_user_ids = rmab_group['user_id'].to_list()
round_robin_user_ids = round_robin_group['user_id'].to_list()

intervention_benefit = {'rmab': [], 'round_robin': [], 'control': []}

all_user_ids = df['user_id'].to_list()

full_mat = {'rmab': [], 'round_robin': [], 'control': []}

for user_id in tqdm(all_user_ids):
    curr_mat = []
    curr_row = df[df['user_id'] == user_id]

    check_row = df_check[df_check['user_id'] == user_id]

    arm = curr_row['arm'].item()
    count = 0

    week0e = int(curr_row['week{}_E/C'.format(0)].item().split('/')[0])
    week0e_check = int(check_row['week{}_E/C'.format(0)].item().split('/')[0])
    assert week0e == week0e_check
    if week0e:
        curr_mat.append(1)
    else:
        curr_mat.append(0)

    for i in range(1,8):
        nume = int(curr_row['week{}_E/C'.format(i)].item().split('/')[0])
        if i < 7:
            nume_check = int(check_row['week{}_E/C'.format(i)].item().split('/')[0])
            try:
                assert nume == nume_check
            except:
                print(user_id)
        if nume > 0:
            count += 1
            curr_mat.append(1)
        else:
            curr_mat.append(0)

    full_mat[arm].append(curr_mat)

    # if not week0e:
        # intervention_benefit[arm].append(count)

for key in full_mat:
    full_mat[key] = np.array(full_mat[key], dtype=np.int)

ipdb.set_trace()