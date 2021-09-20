import os
import sys
import numpy as np
from numpy.lib.arraysetops import unique
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from copy import deepcopy

from datetime import datetime
from pprint import pprint
from tqdm import tqdm
import ipdb
import pickle

plt.style.use("seaborn")
np.random.seed(1)

from training.utils import load_obj, save_obj
from training.data import load_data
from training.dataset import _preprocess_call_data, preprocess_and_make_dataset

from sklearn.cluster import KMeans, OPTICS, SpectralClustering
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

from training.modelling.metrics import F1, Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model
from training.modelling.dataloader import get_train_val_test

aug_states = []
for i in range(6):
    if i % 2 == 0:
        aug_states.append('L{}'.format(i // 2))
    else:
        aug_states.append('H{}'.format(i // 2))

CONFIG = {
    "problem": {
        "orig_states": ['L', 'H'],
        "states": aug_states + ['L', 'H'],
        "actions": ["N", "I"],
    },
    "time_step": 7,
    "gamma": 0.99,
    "clusters": 40,
    "transitions": "weekly",
    "clustering": "kmeans",
    "pilot_start_date": sys.argv[1],
    "pilot_data": "data",
    "interventions": int(sys.argv[2])
}

def run_third_pilot(CONFIG):
    pilot_data = CONFIG['pilot_data']
    pilot_beneficiary_data, pilot_call_data = load_data(pilot_data)
    inf_dataset = preprocess_and_make_dataset(pilot_beneficiary_data, pilot_call_data)
    pilot_call_data = _preprocess_call_data(pilot_call_data)
    pilot_user_ids, pilot_dynamic_xs, pilot_gest_age, pilot_static_xs, pilot_hosp_id, pilot_labels = inf_dataset

    enroll_gest_age_mean = np.mean(inf_dataset[3][:, 0])
    days_to_first_call_mean = np.mean(inf_dataset[3][:, 7])

    # dynamic features preprocessing
    pilot_dynamic_xs = pilot_dynamic_xs.astype(np.float32)
    pilot_dynamic_xs[:, :, 2] = pilot_dynamic_xs[:, :, 2] / 60
    pilot_dynamic_xs[:, :, 3] = pilot_dynamic_xs[:, :, 3] / 60
    pilot_dynamic_xs[:, :, 4] = pilot_dynamic_xs[:, :, 4] / 12

    # static features preprocessing
    pilot_static_xs = pilot_static_xs.astype(np.float32)
    pilot_static_xs[:, 0] = (pilot_static_xs[:, 0] - enroll_gest_age_mean)
    pilot_static_xs[:, 7] = (pilot_static_xs[:, 7] - days_to_first_call_mean)

    dependencies = {
        'BinaryAccuracy': BinaryAccuracy,
        'F1': F1,
        'Precision': Precision,
        'Recall': Recall
    }

    pilot_static_features = np.array(pilot_static_xs, dtype=np.float)
    pilot_static_features = pilot_static_features[:, : -8]

#     q_values = pickle.load(open("q_values",'rb'))
#     m_values = pickle.load(open("m_values",'rb'))
#     cls = pickle.load(open("cls_model",'rb'))
    with open('policy_dump.pkl', 'rb') as fr:
      pilot_user_ids_, pilot_static_features_, cls, cluster_transition_probabilities_, m_values, q_values = pickle.load(fr)
    fr.close()
    pilot_cluster_predictions = cls.predict(pilot_static_features)
    
    import mysql.connector
    from mysql.connector.constants import ClientFlag
    import pandas as pd
    config = {
        'user': 'googleai',
        'password': '4UY(@{SqH{',
        'host': '34.93.237.61',
        'client_flags': [ClientFlag.SSL]
    }

    # now we establish our connection

    config['database'] = 'mmitrav2'  # add new database to config dict
    cnxn = mysql.connector.connect(**config)
    cursor = cnxn.cursor()
    query = "SELECT beneficiary_id, intervention_date, intervention_success
    FROM intervention_list
    WHERE intervention_date < '${DATE}' AND intervention_date >= date_add('${DATE}', interval -21 day) AND intervention_success = 1;"
    try:
        df_intervention = pd.read_sql(query, cnxn)
        intervention_users = df_intervention['beneficiary_id'].to_list()
    except:
        intervention_users = []
#     try:
#         df_intervention = pd.read_csv("data/intervention_data.csv",sep="\t")
#         intervention_users = df_intervention['beneficiary_id'].to_list()
#     except:
#         intervention_users = []
        
    whittle_indices = {'user_id': [], 'whittle_index': [], 'cluster': [], 'start_state': [], 'registration_date': [], 'current_E2C': []}
    for idx, puser_id in enumerate(pilot_user_ids):
        pilot_date_num = (pd.to_datetime(CONFIG['pilot_start_date'], format="%Y-%m-%d") - pd.to_datetime("2018-01-01", format="%Y-%m-%d")).days
        
        if puser_id in intervention_users:
            continue
        
        if CONFIG['transitions'] == 'weekly':
            past_days_calls = pilot_call_data[
            (pilot_call_data["user_id"]==puser_id)&
            (pilot_call_data["startdate"]<pilot_date_num)&
            (pilot_call_data["startdate"]>=pilot_date_num - 7)
        ]
        else:
            past_days_calls = pilot_call_data[
                (pilot_call_data["user_id"]==puser_id)&
                (pilot_call_data["startdate"]<pilot_date_num)&
                (pilot_call_data["startdate"]>=pilot_date_num - 30)
            ]
        past_days_connections = past_days_calls[past_days_calls['duration']>0].shape[0]
        past_days_engagements = past_days_calls[past_days_calls['duration'] >= 30].shape[0]
        if CONFIG['transitions'] == 'weekly':
            if past_days_engagements == 0:
                curr_state = 7
            else:
                curr_state = 6
        else:
            if past_days_connections == 0:
                curr_state = 7
            else:
                if past_days_engagements/past_days_connections >= 0.5:
                    curr_state = 6
                else:
                    curr_state = 7
        
        curr_cluster = pilot_cluster_predictions[idx]

        whittle_indices['user_id'].append(puser_id)
        whittle_indices['whittle_index'].append(m_values[curr_cluster, curr_state])
        whittle_indices['cluster'].append(curr_cluster)
        if curr_state == 7:
            whittle_indices['start_state'].append('NE')
        elif curr_state == 6:
            whittle_indices['start_state'].append('E')
        
        all_days_calls = pilot_call_data[
            (pilot_call_data['user_id'] == puser_id)
        ]
        all_days_connections = all_days_calls[all_days_calls['duration'] > 0].shape[0]
        all_days_engagements = all_days_calls[all_days_calls['duration'] >= 30].shape[0]
        if all_days_connections == 0:
            whittle_indices['current_E2C'].append('')
        else:
            curr_e2c = all_days_engagements / all_days_connections
            whittle_indices['current_E2C'].append(curr_e2c)
        regis_date = pilot_beneficiary_data[pilot_beneficiary_data['user_id'] == puser_id]['registration_date'].item()

        whittle_indices['registration_date'].append(regis_date)


    df = pd.DataFrame(whittle_indices)
    df = df.sort_values('whittle_index', ascending=False)
    #df.to_csv('checking_{}_{}_pilot_stats_{}.csv'.format(CONFIG['transitions'], CONFIG['clustering'], CONFIG['clusters']))
    df_ = df[df['start_state']=='NE']
    df_ = df_[:CONFIG["interventions"]]
    df_ = df_['user_id']
    df_.to_csv('user_interventions.csv', index=False, header=False)
    return

run_third_pilot(CONFIG)
