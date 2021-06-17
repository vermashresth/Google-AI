import numpy as np
import pandas as pd
import ipdb
import pickle
from tqdm import tqdm

from training.utils import load_obj, save_obj
from training.data import load_data
from training.dataset import _preprocess_call_data, preprocess_and_make_dataset

from training.modelling.metrics import F1, Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model

CONFIG = {
    'pilot_data': 'feb16-mar15_data',
    'current_week': 8,
    'calling_files': ['250_week1_290421', '400_week2_060521', '400_week3_120521', '400_week4_180521', '435_week5_240521', '600_week6_310521', '700_week7_070621'],
    'pilot_dates' : ['2021-04-26', '2021-05-03', '2021-05-10', '2021-05-17', '2021-05-24', '2021-05-31', '2021-06-07', '2021-06-14']
}

assert len(CONFIG['calling_files']) == CONFIG['current_week'] - 1
assert len(CONFIG['pilot_dates']) == CONFIG['current_week']

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
for file in CONFIG['calling_files']:
    with open('outputs/pilot_generations/calling_list_{}.txt'.format(file), 'r') as fr:
        for line in fr:
            user_id = int(line.strip())
            if user_id not in intervention_dict:
                intervention_dict[user_id] = [file.split('_')[1]]
            else:
                intervention_dict[user_id].append(file.split('_')[1])

# ipdb.set_trace()

out_dict = {'user_id': [], 'cluster': [], 'arm': [], 'registration_date': [], 'intervention_week': [], 'week1_whittle': [], 'after_intervention_E/C' : []}
out_dict['week{}_whittle'.format(CONFIG['current_week'])] = []
for i in range(CONFIG['current_week']):
    out_dict['week{}_E/C'.format(i)] = []

for user_id in tqdm(all_user_ids):
    out_dict['user_id'].append(user_id)
    curr_row = complete_group[complete_group['user_id'] == user_id]
    curr_cluster = curr_row['cluster'].item()
    out_dict['cluster'].append(curr_cluster)
    out_dict['registration_date'].append(curr_row['registration_date'].item())
    if user_id in arm_dict:
        out_dict['arm'].append(arm_dict[user_id])
    else:
        out_dict['arm'].append('control')

    if user_id in intervention_dict:
        out_dict['intervention_week'].append(intervention_dict[user_id][-1])
    else:
        out_dict['intervention_week'].append('')

    curr_state = None
    countc, counte = 0, 0
    for i, date_val in enumerate(CONFIG['pilot_dates']):
        pilot_date_num = (pd.to_datetime(date_val, format="%Y-%m-%d") - pd.to_datetime("2018-01-01", format="%Y-%m-%d")).days
        
        past_days_calls = pilot_call_data[
            (pilot_call_data["user_id"]==user_id)&
            (pilot_call_data["startdate"]<pilot_date_num)&
            (pilot_call_data["startdate"]>=pilot_date_num - 7)
        ]

        past_days_connections = past_days_calls[past_days_calls['duration']>0].shape[0]
        past_days_engagements = past_days_calls[past_days_calls['duration'] >= 30].shape[0]

        if past_days_engagements == 0:
            curr_state = 7
        else:
            curr_state = 6
        
        if len(out_dict['intervention_week'][-1]) > 0 and i > int(out_dict['intervention_week'][-1][-1]):
            countc += past_days_connections
            counte += past_days_engagements
        
        out_dict['week{}_E/C'.format(i)].append('{}/{}'.format(past_days_engagements, past_days_connections))
    
    if user_id in intervention_dict and int(intervention_dict[user_id][-1][-1]) > CONFIG['current_week'] - 4:
        curr_state -= 6
    
    out_dict['after_intervention_E/C'].append('{}/{}'.format(counte, countc))

    out_dict['week{}_whittle'.format(CONFIG['current_week'])].append(m_values[curr_cluster, curr_state])
    out_dict['week1_whittle'].append(curr_row['whittle_index'].item())

df = pd.DataFrame(out_dict)
df = df.sort_values('week1_whittle', ascending=False)
df.to_csv('outputs/analysis_lists/all_analysis_week_{}.csv'.format(CONFIG['current_week']))

exit()

df = pd.read_csv('outputs/analysis_lists/all_analysis_first_int_week_7.csv')
# df2 = pd.read_csv('outputs/analysis_lists/all_analysis_week_7.csv')

user_ids = df['user_id'].to_list()

stats_dict = {'rmab': {'start_engaging': [], 'start_non_engaging': []}, 'round_robin': {'start_engaging': [], 'start_non_engaging': []}, 'control': {'start_engaging': [], 'start_non_engaging': []}}

for user_id in user_ids:
    curr_row = df[df['user_id'] == user_id]

    arm = curr_row['arm'].item()
    week0_ec = curr_row['week0_E/C'].item()
    week0_ec = list(map(int, week0_ec.split('/')))
    week0e, week0c = week0_ec[0], week0_ec[1]

    after_ec = curr_row['after_intervention_E/C'].item()
    after_ec = list(map(int, after_ec.split('/')))
    aftere, afterc = after_ec[0], after_ec[1]
    # if arm != 'control':
    if week0e:
        stats_dict[arm]['start_engaging'].append(user_id)
    else:
        stats_dict[arm]['start_non_engaging'].append(user_id)

for arm in ['rmab', 'round_robin']:
    for start_condition in ['start_engaging', 'start_non_engaging']:
        count_all, count_only_i1, count_only_i2 = 0, 0, 0
        count1 = 0
        print('-'*60)
        print('Arm: {}, Starting State: {}'.format(arm, start_condition))

        curr_group = stats_dict[arm][start_condition]
        print('Beneficiaries in this group: {}'.format(len(curr_group)))

        for user_id in curr_group:
            curr_row = df[df['user_id'] == user_id]
            iweek = curr_row['intervention_week'].item()
            after_ec = curr_row['after_intervention_E/C'].item()
            after_ec = list(map(int, after_ec.split('/')))
            aftere, afterc = after_ec[0], after_ec[1]

            e_count = 0
            for i in range(1, 7):
                i_ec = curr_row['week{}_E/C'.format(i)].item()
                i_ec = list(map(int, i_ec.split('/')))
                ie, ic = i_ec[0], i_ec[1]
                e_count += ie

            # if iweek == 'week2':
            #     count_all += 1
            #     if aftere:
            #         count_only_i1 += 1
            #     else:
            #         count_only_i2 += 1

            if e_count:
                count_all += 1
            else:
                count1 += 1
            
            # if type(iweek) == type('a') and e_count:
            #     count_only_i1 += 1
            # if type(iweek) == type('a') and not e_count:
            #     count_only_i2 += 1

            
            # if e_count:
            #     count_all += 1
            if aftere and iweek == 'week1':
                count_only_i1 += 1
            elif iweek == 'week1':
                count_only_i2 += 1
            # if aftere and iweek == 'week2':
            #     count_only_i2 += 1
        # print(count1, len(stats_dict['rmab']['start_engaging']))
        print('Beneficiaries with final state = E: {}'.format(count_all))
        print('Beneficiaries with final state = NE: {}'.format(count1))
        print('Week 1 Beneficiaries who are finally E: {}'.format(count_only_i1))
        print('Week 1 Beneficiaries who are finally NE: {}'.format(count_only_i2))
        # print('Final state = E and received I: {}'.format(count_only_i1))
        # print('Final state = NE and received I: {}'.format(count_only_i2))
        # print(count_all, count_only_i1, count_only_i2)

# ipdb.set_trace()