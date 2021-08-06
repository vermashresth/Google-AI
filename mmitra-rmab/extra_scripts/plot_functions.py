import numpy as np
import pandas as pd
import sys
import os
import csv
import ipdb
import pickle
from collections import OrderedDict, defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

CONFIG = {
    'week': 'week10',
    'current_week': 10,
    'calling_files': ['250_week1_290421', '400_week2_060521', '400_week3_120521', '400_week4_180521', '435_week5_240521', '600_week6_310521', '700_week7_070621', '1000_week8_140621', '1000_week9_210621']
}

np.random.seed(1)

df = pd.read_csv(sys.argv[1])
df = df.sort_values('{}_whittle'.format(CONFIG['week']), ascending=False)

intervention_dict = {}
week_wise_intervention_dict = defaultdict(lambda: [])
for file in CONFIG['calling_files']:
    with open('outputs/pilot_generations/calling_list_{}.txt'.format(file), 'r') as fr:
        for line in fr:
            user_id = int(line.strip())
            if user_id not in intervention_dict:
                intervention_dict[user_id] = [file.split('_')[1]]
            else:
                intervention_dict[user_id].append(file.split('_')[1])
            week_wise_intervention_dict[file.split('_')[1]].append(user_id)

rmab_list = pd.read_csv('outputs/pilot_outputs/rmab_pilot.csv')['user_id'].to_list()
round_robin_list = pd.read_csv('outputs/pilot_outputs/round_robin_pilot.csv')['user_id'].to_list()
control_list = pd.read_csv('outputs/pilot_outputs/control_pilot.csv')['user_id'].to_list()

rmab_group = df[df['user_id'].isin(rmab_list)]
rmab_group = rmab_group.sort_values('{}_whittle'.format(CONFIG['week']), ascending=False)
# rmab_group.to_csv('outputs/pilot_outputs/rmab_pilot_{}.csv'.format(CONFIG['week']))

round_robin_group = df[df['user_id'].isin(round_robin_list)]
round_robin_group = round_robin_group.sort_values('registration_date', ascending=True)
# round_robin_group.to_csv('outputs/pilot_outputs/round_robin_pilot_{}.csv'.format(CONFIG['week']))

# control_group = df[df['user_id'].isin(control_list)]
# control_group.to_csv('outputs/pilot_outputs/control_pilot_{}.csv'.format(CONFIG['week']))

rmab_user_ids = rmab_group['user_id'].to_list()
round_robin_user_ids = round_robin_group['user_id'].to_list()

# int_week=sys.argv[2]
# plot_line = {'rmab': [], 'round_robin': []}

# for user_id in week_wise_intervention_dict[int_week]:
#     curr_mat = []
#     curr_row = df[df['user_id'] == user_id]
#     arm = curr_row['arm'].item()

#     week0e = int(curr_row['week{}_E/C'.format(0)].item().split('/')[0])
#     if week0e:
#         curr_mat.append(1)
#     else:
#         curr_mat.append(0)
    
#     for i in range(1,CONFIG['current_week']):
#         nume = int(curr_row['week{}_E/C'.format(i)].item().split('/')[0])
#         if nume > 0:
#             curr_mat.append(1)
#         else:
#             curr_mat.append(0)
    
#     plot_line[arm].append(curr_mat)

# for key in plot_line:
#     plot_line[key] = np.array(plot_line[key])

plt.figure(figsize=(10,9))
all_weeks = ['week{}'.format(i) for i in range(1,5)]
week_plot_line = {}
for week_val in all_weeks:
    plot_line = {'rmab': [], 'round_robin': []}

    for user_id in week_wise_intervention_dict[week_val]:
        curr_mat = []
        curr_row = df[df['user_id'] == user_id]
        arm = curr_row['arm'].item()

        week0e = int(curr_row['week{}_E/C'.format(0)].item().split('/')[0])
        if week0e:
            curr_mat.append(1)
        else:
            curr_mat.append(0)
        
        for i in range(1,CONFIG['current_week']):
            nume = int(curr_row['week{}_E/C'.format(i)].item().split('/')[0])
            if nume > 0:
                curr_mat.append(1)
            else:
                curr_mat.append(0)
        
        plot_line[arm].append(curr_mat)

    for key in plot_line:
        plot_line[key] = np.array(plot_line[key])
    week_plot_line[week_val] = plot_line
    plt.subplot(int('41{}'.format(week_val[-1])))
    axes = plt.gca()
    # axes.set_ylim([0,0.9])
    for key in plot_line:
        print('Group: {}, Shape: {}'.format(key, plot_line[key].shape))
        arr = np.mean(plot_line[key], axis=0)
        # arr = np.sum(plot_line[key], axis=0)
        plt.plot(np.arange(CONFIG['current_week']), arr, '-^', label=key)
    plt.axvline(x=int(week_val[4:]) + 0.5, color='r', linestyle='--')
    plt.legend(loc='lower right')
    plt.xlabel('Weeks')
    plt.ylabel('Engaging (Fraction)')
    plt.title('Engagement patterns of {} interventions (RMAB - {}, Round Robin - {})'.format(week_val, plot_line['rmab'].shape[0], plot_line['round_robin'].shape[0]))
plt.tight_layout()
# plt.savefig('week_wise_int_eng_absolute.png', dpi=500, bbox_inches='tight')

# plt.show()

# ipdb.set_trace()
plt.figure(figsize=(10,4))
combined_relative_plot = {}
for key in ['rmab', 'round_robin']:
    combined_relative_plot[key] = np.concatenate((week_plot_line['week1'][key][:, : 7], week_plot_line['week2'][key][:, 1 : 8], week_plot_line['week3'][key][:, 2 : 9], week_plot_line['week4'][key][:, 3 : 10]), axis=0)
    arr = np.mean(combined_relative_plot[key], axis=0)
    plt.plot(np.arange(CONFIG['current_week'] - 3), arr, '-^', label=key)

plt.axvline(x=1.5, color='r', linestyle='--')
plt.legend()
plt.xlabel('Weeks')
plt.ylabel('Engaging (Fraction)')
plt.title('Engagement patterns of Week 1-4 interventions (RMAB - {}, Round Robin - {})'.format(combined_relative_plot['rmab'].shape[0], combined_relative_plot['round_robin'].shape[0]))
plt.show()