import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
from pprint import pprint
from tqdm import tqdm

plt.style.use("seaborn")
np.random.seed(1)

from training.utils import load_obj, save_obj
from training.data import load_data
from training.dataset import _preprocess_call_data

stats = pd.read_csv("may_data/beneficiary_stats_v5.csv")
beneficiary_data = pd.read_csv("may_data/beneficiary/AIRegistration-20200501-20200731.csv")
_, call_data = load_data("may_data")
call_data = _preprocess_call_data(call_data)

# Dataset Creation
CONFIG = dict()
CONFIG['intervention_period_start'] = 30
CONFIG['intervention_period_end'] = 210
CONFIG['time_step'] = 7

def get_transitions(stats, beneficiaries, call_data, CONFIG):
    transitions = pd.DataFrame(columns=["user_id", "pre-action state", "action", "post-action state"])
    zero_c_lst = []
    for beneficiary in tqdm(beneficiaries):
        start = stats[stats['user_id']==beneficiary]['start'].item()
        intervention = stats[stats['user_id']==beneficiary]['Intervention Date'].item()
        success = stats[stats['user_id']==beneficiary]['Intervention Status'].item()
        calls = call_data[call_data['user_id']==beneficiary]

        group = stats[stats['user_id']==beneficiary]['Group'].item()
        start = start + CONFIG['intervention_period_start']
        end = start + CONFIG['intervention_period_end']
        zero_counts = 0
        sequence = ""
        for day in range(start, end, CONFIG['time_step']):
            period_start = day
            period_end = day + CONFIG['time_step']

            if (group=="Google-AI-Calls") and (success=="Successful"): 
                if (intervention >= period_start) and (intervention < period_end):
                    sequence += "I"
                    continue

            period_calls = calls[
                (calls['startdate']>=period_start)&
                (calls['startdate']<period_end)
            ]

            period_connections = period_calls[period_calls['duration']>0].shape[0]
            period_engagements = period_calls[period_calls['duration']>=30].shape[0]

            # if period_connections == 0:
            #     if period_calls.shape[0] > 0:
            #         sequence += "H"
            #     else:
            #         continue
            # else:
            #     if period_engagements/period_connections < 0.5:
            #         sequence += "H"
            #     else:
            #         sequence += "L"

            if period_connections == 0:
                zero_counts += 1

            if period_engagements == 0:
                sequence += 'H'
            else:
                sequence += 'L'

        for i in range(len(sequence)-1):
            if sequence[i] == "I":
                continue
            elif sequence[i+1] == "I":
                if sequence[-1] == "I":
                    continue
                else:
                    transitions = transitions.append({
                        'user_id': beneficiary, 
                        'pre-action state': sequence[i], 
                        'action': "Intervention", 
                        'post-action state': sequence[i+2]
                    }, ignore_index=True)
            else:
                transitions = transitions.append({
                    'user_id': beneficiary, 
                    'pre-action state': sequence[i], 
                    'action': "No Intervention", 
                    'post-action state': sequence[i+1]
                }, ignore_index=True)
        zero_c_lst.append(zero_counts)
    print('Avg Zero Count: {}'.format(np.mean(np.array(zero_c_lst))))
    return transitions

beneficiaries = stats[stats['Group'].isin(["Google-AI-Control", "Google-AI-Calls"])]["user_id"]
transitions = get_transitions(stats, beneficiaries, call_data, CONFIG)
transitions.to_csv("may_data/RMAB_one_month/weekly_transitions_SI_all.csv")

call_beneficiaries = stats[stats['Group'] == "Google-AI-Calls"][["user_id", "Group", "Intervention Status", "Post-intervention Day: E2C Ratio"]]
control_beneficiaries = stats[stats['Group'] == "Google-AI-Control"][["user_id", "Group", "Post-intervention Day: E2C Ratio"]]

beneficiary_splits = []
for i in range(1):
    train_call = call_beneficiaries.sample(frac=1.0)
    test_call = call_beneficiaries[~call_beneficiaries['user_id'].isin(train_call['user_id'])]

    train_control = control_beneficiaries.sample(frac=1.0)
    test_control = control_beneficiaries[~control_beneficiaries['user_id'].isin(train_control['user_id'])]

    train = pd.concat([train_call, train_control], axis=0)
    test = pd.concat([test_call, test_control], axis=0)

    for i in tqdm(test.index):
        user_id = test.loc[i, "user_id"]


        past_30_days_calls = call_data[
            (call_data["user_id"]==user_id)&
            (call_data["startdate"]<1024)&
            (call_data["startdate"]>=1017)
        ]
        past_30_days_connections = past_30_days_calls[past_30_days_calls['duration']>0].shape[0]
        past_30_days_engagements = past_30_days_calls[past_30_days_calls['duration']>=30].shape[0]

        # if past_30_days_connections == 0:
        #     state = 1
        # else:
        #     if past_30_days_engagements/past_30_days_connections >= 0.5:
        #         state = 0
        #     else:
        #         state = 1
        if past_30_days_engagements == 0:
            state = 1
        else:
            state = 0

        test.loc[i, "state"] = state

    beneficiary_splits.append((train, test))

save_obj(beneficiary_splits, "may_data/RMAB_one_month/weekly_beneficiary_splits_all.pkl")