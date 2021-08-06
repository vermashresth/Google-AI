import numpy as np
import pandas as pd
import sys
import os
import csv
import ipdb
from collections import OrderedDict
# import random

CONFIG = {
    'calling_files': ['250_week1', '400_week2', '400_week3', '400_week4', '435_week5', '600_week6', '700_week7', '1000_week8', '1000_week9', '1000_week10', '1000_week11'],
    'current_week': 11
}

complete_group = pd.read_csv('outputs/individual_clustering/weekly_kmeans_pilot_stats_40.csv')
all_user_ids = complete_group['user_id'].to_list()

intervention_dict = {}
week_wise_list = {}
for idx, file in enumerate(CONFIG['calling_files']):
    week_wise_list[idx + 1] = []
    with open('outputs/pilot_generations/share_external/calling_list_{}.txt'.format(file), 'r') as fr:
        for line in fr:
            user_id = int(line.strip())
            if user_id not in intervention_dict:
                intervention_dict[user_id] = [file.split('_')[1]]
            else:
                intervention_dict[user_id].append(file.split('_')[1])
            week_wise_list[idx + 1].append(user_id)
    fr.close()


out_dict = {'user_id': [], 'predictions_latest_week': []}

for user_id in all_user_ids:
    if user_id not in week_wise_list[CONFIG['current_week']] and user_id not in week_wise_list[CONFIG['current_week'] - 1]:
        continue
    out_dict['user_id'].append(user_id)
    if user_id in week_wise_list[CONFIG['current_week']]:
        out_dict['predictions_latest_week'].append(1)
    else:
        out_dict['predictions_latest_week'].append(0)

df = pd.DataFrame(out_dict)
df.to_csv('database_table_{}.csv'.format(CONFIG['current_week']), index=False)