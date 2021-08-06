import numpy as np
import pandas as pd
import sys
import os
import csv
import ipdb
import pickle
from collections import OrderedDict, defaultdict
from tqdm import tqdm

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

CONFIG = {
    'week': 'week12',
    'current_week': 12
    # 'calling_list': sys.argv[2],
    # 'round_robin_list': sys.argv[3]
}

np.random.seed(1)

df = pd.read_csv(sys.argv[1])
df = df.sort_values('{}_whittle'.format(CONFIG['week']), ascending=False)

# df_check = pd.read_csv(sys.argv[2])
# df_check = df_check.sort_values('week7_whittle', ascending=False)

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

all_user_ids = set(df['user_id'].to_list())

full_mat = {'rmab': [], 'round_robin': [], 'control': []}

with open('policy_dump.pkl', 'rb') as fr:
  pilot_user_ids, pilot_static_features, cls, cluster_transition_probabilities, m_values, q_values = pickle.load(fr)
fr.close()

arm_to_cat = {'rmab': [1], 'round_robin': [1], 'control': [0]}
X_mat, Y_mat = [], []

# ipdb.set_trace()

for idx, user_id in tqdm(enumerate(pilot_user_ids)):
    if user_id not in all_user_ids:
        continue

    curr_mat = []
    curr_row = df[df['user_id'] == user_id]

    # check_row = df_check[df_check['user_id'] == user_id]

    arm = curr_row['arm'].item()
    # if arm == 'rmab':
        # continue

    count = 0

    week0e = int(curr_row['week{}_E/C'.format(0)].item().split('/')[0])
    # week0e_check = int(check_row['week{}_E/C'.format(0)].item().split('/')[0])
    # assert week0e == week0e_check
    if week0e:
        curr_mat.append(1)
    else:
        curr_mat.append(0)

    for i in range(1,CONFIG['current_week']):
        nume = int(curr_row['week{}_E/C'.format(i)].item().split('/')[0])
        # if i < 7:
        #     nume_check = int(check_row['week{}_E/C'.format(i)].item().split('/')[0])
        #     try:
        #         assert nume == nume_check
        #     except:
        #         print(user_id)
        if nume > 0:
            count += 1
            curr_mat.append(1)
        else:
            curr_mat.append(0)
    
    X_mat.append(list(pilot_static_features[idx, :]) + [curr_mat[0]] + arm_to_cat[arm])
    Y_mat.append(count)

    full_mat[arm].append(curr_mat)

    # if not week0e:
        # intervention_benefit[arm].append(count)

for key in full_mat:
    full_mat[key] = np.array(full_mat[key], dtype=np.int)

# with open('full_matrix_week{}_end.pkl'.format(CONFIG['current_week'] - 1), 'wb') as fw:
#     pickle.dump(full_mat, fw)
# fw.close()
# exit()
reg = LinearRegression().fit(X_mat, Y_mat)

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
    "start_state",
    "exp_arm_rr"
]

fimp = reg.coef_
fimp = fimp.reshape((fimp.shape[0], 1))
fimp_idx = np.concatenate((fimp, np.arange(fimp.shape[0]).reshape((fimp.shape[0], 1))), axis=1)

mod = sm.OLS(Y_mat,X_mat)
fii = mod.fit()
p_values = fii.summary2().tables[1]['P>|t|']
p_values = p_values.to_list()
sorted_arr = fimp_idx[(-fimp_idx)[:, 0].argsort()]

class_scores = defaultdict(lambda: 0)

for i in range(fimp_idx.shape[0]):
    col_name = columns[int(sorted_arr[i, 1])]
    curr_score = round(sorted_arr[i, 0], 3)
    print('{}\t{}\t{}'.format(col_name, curr_score, p_values[i]))
    if 'age_' in col_name:
        class_scores['age'] += curr_score
    if 'education_' in col_name:
        class_scores['education'] += curr_score
    if 'income_bracket_' in col_name:
        class_scores['income_bracket'] += curr_score
    if 'language_' in col_name:
        class_scores['language'] += curr_score
    if 'exp_arm_' in col_name:
        class_scores['experimental_arm'] += curr_score

print(class_scores)

# ipdb.set_trace()