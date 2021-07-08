import pandas as pd
from tqdm import tqdm
import ipdb
from training.call_data import load_call_data, load_call_file
from training.dataset import _preprocess_call_data


def get_transitions():
    transitions = pd.DataFrame(columns=['user_id', 'pre-action state', 'action', 'post-action state'])

    b_data = pd.read_csv("outputs/individual_clustering/weekly_kmeans_pilot_stats_40.csv")
    user_ids = b_data['user_id'].to_list()


    analysis = pd.read_csv("analysis_lists/all_analysis_week_7.csv")
    for i in range(6):
        interventions = pd.read_csv(f'intervention_lists/calling_list_week{i+1}.csv')['user_id'].to_list()
        for user in user_ids:
            start_state = 'H' if int(analysis[analysis['user_id'] == user][f'week{i}_E/C'].item().split('/')[0]) == 0 else 'L'
            end_state = 'H' if int(analysis[analysis['user_id'] == user][f'week{i+1}_E/C'].item().split('/')[0]) else 'L'
            if user in interventions:
                transitions = transitions.append(
                    { 'user_id': user,
                    'pre-action state': start_state,
                    'action': 'Intervention',
                    'post-action state': end_state
                    }, ignore_index=True)
            else:
                transitions = transitions.append(
                    { 'user_id': user,
                    'pre-action state': start_state,
                    'action': 'No Intervention',
                    'post-action state': end_state
                    }, ignore_index=True)
        print(i)
    transitions.to_csv("groundtruth_analysis/transitions.csv")

# get_transitions()

intervention_dict = {}
for i in range(8):
    interventions = pd.read_csv(f"intervention_lists/calling_list_week{i+1}.csv")['user_id'].to_list()
    for user_id in interventions:
        if user_id not in intervention_dict:
            intervention_dict[user_id] = [i]
        else:
            intervention_dict[user_id].append(i)

beneficiaries = pd.read_csv("outputs/individual_clustering/weekly_kmeans_pilot_stats_40.csv")
all_user_ids = beneficiaries['user_id'].to_list()

analysis = pd.read_csv("analysis_lists/all_analysis_week_9.csv")
# analysis = analysis


out_dict = {'user_id': [], 'pre-action state': [], 'action': [], 'post-action state': []}


for user_id in tqdm(all_user_ids):
    curr_row = analysis[analysis['user_id'] == user_id]
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
transitions_df.to_csv("groundtruth_analysis/transitions_week9.csv")

# ipdb.set_trace()