import os
import sys
import numpy as np
import seaborn as sns
from numpy.lib.arraysetops import unique
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
import ipdb

import pickle

plt.style.use("seaborn")
np.random.seed(1)

from training.utils import load_obj, save_obj
from training.data import load_data
from training.dataset import _preprocess_call_data, preprocess_and_make_dataset

from sklearn.ensemble import RandomForestClassifier

CONFIG = {
    "clusters": 40,
    "transitions": "weekly",
    "pilot_start_date": sys.argv[1],
    "calling_list": sys.argv[2],
    "pickle_file_path": sys.argv[3] ,
    "file_path": sys.argv[4]   #Path where soft assignment cluster dist need to be saved
}

def get_soft_clusters():
    """
    This method assigns each beneficiary a set of clusters with a certain probability.
    Returns data with the weighted whittle indices, the cluster_set, probability_set, max_probabiltiy
    among all and the cluster corresponding to the max_probability corresponding to each beneficiary.
    """
    pilot_beneficiary_data, pilot_call_data = load_data('feb16-mar15_data')
    pilot_call_data = _preprocess_call_data(pilot_call_data)

    with open("{}.pkl".format(CONFIG['pickle_file_path']), 'rb') as fr:
        pilot_user_ids = pickle.load(fr) 
        pilot_static_features = pickle.load(fr) 
        cls = pickle.load(fr) 
        cluster_transition_probabilities = pickle.load(fr) 
        m_values = pickle.load(fr) 
        q_values = pickle.load(fr)
    fr.close()
    CONFIG['clusters'] = len(list(cluster_transition_probabilities['cluster']))
    previous_calling_list = pd.read_csv(CONFIG['calling_list'], header=None, names=['user_id'])
    previous_calling_list = set(previous_calling_list['user_id'].to_list())

    pilot_cluster_predictions = cls.predict_proba(pilot_static_features)

    whittle_indices = {'user_id': [], 'whittle_index': [], 'start_state': [],'cluster': [], 'max_probability': [], 'cluster_set': [], 'probability_set': []}
    for idx, puser_id in enumerate(pilot_user_ids):

        pilot_date_num = (pd.to_datetime(CONFIG['pilot_start_date'], format="%Y-%m-%d") - pd.to_datetime("2018-01-01", format="%Y-%m-%d")).days

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
        
        if puser_id in previous_calling_list:
            curr_state -= 6

        if curr_state == 7:
            whittle_indices['start_state'].append('NE')
        elif curr_state == 6:
            whittle_indices['start_state'].append('E')
        elif curr_state == 1:
            whittle_indices['start_state'].append('NE0')
        elif curr_state == 0:
            whittle_indices['start_state'].append('E0')
                
        whittle_indices['user_id'].append(puser_id)
        whittle_index = 0
        cur_probs = pilot_cluster_predictions[idx]
        curr_cluster = -1
        max_prob = 0
        cluster_set = []
        probability_set = []
        for i in range(CONFIG['clusters']):
            if cur_probs[i] > max_prob:
                max_prob = cur_probs[i]
                curr_cluster = i
            whittle_index += cur_probs[i] * m_values[i, curr_state]
            if cur_probs[i] != 0:
                probability_set.append(cur_probs[i])
                cluster_set.append(i)

        # It is a weighted whittle index
        whittle_indices['whittle_index'].append(whittle_index)
        whittle_indices['cluster'].append(curr_cluster)
        whittle_indices['max_probability'].append(max_prob)
        whittle_indices['cluster_set'].append(cluster_set)
        whittle_indices['probability_set'].append(probability_set)

    df = pd.DataFrame(whittle_indices)
    df = df.sort_values('whittle_index', ascending=False)
    df.to_csv(CONFIG['file_path'])
    return whittle_indices

def analysis():
    """
    This method compares the soft assignment and hard assignment results.
    """

    # df_soft is the dataframe that has soft clustering assignment information
    # i.e., user_id, cluster_set, probability_set, max_probability, weighted_whittle_index
    df_soft = pd.read_csv(CONFIG['file_path'])
    df_soft_500 = df_soft.iloc[:500]
    df_soft_200 = df_soft.iloc[:200]
    df_hard = pd.read_csv('outputs/individual_clustering/weekly_kmeans_pilot_stats_40.csv')
    df_hard_500 = df_hard.iloc[:500]
    df_hard_200 = df_hard.iloc[:200]

    df = df_soft.groupby(by=['whittle_index','cluster_set', 'probability_set'])['user_id'].count()
    # This is a hack used. Directly trying to sort the values throw an error.
    df.to_csv('outputs/cluster_count_soft_{}.csv'.format(CONFIG['pickle_file_path']))
    df = pd.read_csv('outputs/cluster_count_soft_{}.csv'.format(CONFIG['pickle_file_path']))
    df = df.sort_values('whittle_index', ascending=False)
    df.to_csv('outputs/cluster_count_soft_{}.csv'.format(CONFIG['pickle_file_path']))

    print('For top 200 beneficiaries: ')
    print('Number of different whittle indices when soft assignment {}'.format(len(pd.unique(df_soft_200['whittle_index']))) )
    print('Number of different whittle indices when hard assignment {}'.format(len(pd.unique(df_hard_200['whittle_index']))) )

    df_soft_clusters = df_soft_200['cluster_set'].apply(literal_eval)
    cluster_set = []
    for curr_cluster_set in df_soft_clusters:
        for c in curr_cluster_set:
            if c not in cluster_set:
                cluster_set.append(c)

    print('Number of unique clusters encountered soft assignment {}'.format(len(cluster_set)))
    print('Unique clusters encountered {}'.format(cluster_set))
    
    df_soft_clusters = df_soft_200['cluster_set'].apply(literal_eval)
    cluster_sets = set( tuple(i) for i in df_soft_clusters )
    print('Number of unique cluster combinations encountered {}'.format(len(cluster_sets)))

    print('For top 500 beneficiaries: ')
    print('Number of different whittle indices when soft assignment {}'.format(len(pd.unique(df_soft_500['whittle_index']))) )
    print('Number of different whittle indices when hard assignment {}'.format(len(pd.unique(df_hard_500['whittle_index']))) )

    df_soft_clusters = df_soft_500['cluster_set'].apply(literal_eval)
    cluster_set = []
    for curr_cluster_set in df_soft_clusters:
        for c in curr_cluster_set:
            if c not in cluster_set:
                cluster_set.append(c)

    print('Number of unique clusters encountered soft assignment {}'.format(len(cluster_set)))
    print('Unique clusters encountered {}'.format(cluster_set))
    
    df_soft_clusters = df_soft_500['cluster_set'].apply(literal_eval)
    cluster_sets = set( tuple(i) for i in df_soft_clusters )
    print('Number of unique cluster combinations encountered {}'.format(len(cluster_sets)))


def plot_probs():
    """
    This method analyzes the max_probabilities, it outputs a sheet with 
    max probabilities vs their frequency and plots the frequency of probabilities vs
    probability (probabilities lying in different ranges as [0, 0.1), [0.1, 0.2) and so on)
    """
    
    # df_soft is the dataframe that has soft clustering assignment information
    # i.e., user_id, cluster_set, probability_set, max_probability, weighted_whittle_index
    df_soft = pd.read_csv(CONFIG['file_path'])
    df_soft = df_soft['max_probability']

    # plot_p has the frequency of  beneficiaries with a max_probability in certain range say [0.1, 0.2)
    plot_p = {}
    # probs has the frequency of beneficiaries with a max_prbability
    probs = {}
    for p in df_soft:
        p = float(p)
        plot_p[p//0.1] = plot_p.get(p//0.1, 0) + 1
        probs[p] = probs.get(p, 0) + 1
    for i in range(10):
        plot_p[i] = plot_p.get(i, 0)
    x = probs.keys()
    y = probs.values()
    probs = {'max_probability': [], 'frequency': []}
    probs['max_probability'] = list(x)
    probs['frequency'] = list(y)
    df = pd.DataFrame(probs)
    df = df.sort_values('frequency', ascending=False)
    df.to_csv('outputs/probability_distribution_soft_{}.csv'.format(CONFIG['pickle_file_path']))

    # Change labels of x-axis plot to probability ranges
    x_labels = list(plot_p.keys())
    for i in range( len(x_labels) ) :
        x_labels[i] = f'[0.{x_labels[i]}, 0.{x_labels[i]+1})'
    data = {'Probability': x_labels, 'frequency': list(plot_p.values())}
    df = pd.DataFrame(data, columns=['Probability', 'frequency'])
    
    plt.figure(figsize=(8, 8))
      
    plots = sns.barplot(x="Probability", y="frequency", data=df)

    # Annotate the bars of the plot
    for bar in plots.patches:
        plots.annotate(format(bar.get_height(), '.2f'), 
                       (bar.get_x() + bar.get_width() / 2, 
                        bar.get_height()), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')
      
    plt.xlabel("Probability", size=14)
      
    plt.ylabel("frequency", size=14)
    plt.savefig('som_exps/{}.png'.format(CONFIG['pickle_file_path']))
    # plt.show()

get_soft_clusters()
analysis()
plot_probs()