import os
import sys
import numpy as np
from numpy.lib.arraysetops import unique
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from copy import deepcopy
from functools import partial

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
from sklearn import preprocessing


from training.modelling.metrics import F1, Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model
from training.modelling.dataloader import get_train_val_test

beneficiary_data = pd.read_csv("feb16-mar15_data/beneficiary/ai_registration-20210216-20210315.csv")
b_data, call_data = load_data("feb16-mar15_data")
call_data = _preprocess_call_data(call_data)
all_beneficiaries = list(beneficiary_data.user_id)

features_dataset = preprocess_and_make_dataset(b_data, call_data)
# exit()

# beneficiary_splits = load_obj("may_data/RMAB_one_month/weekly_beneficiary_splits_single_group.pkl")

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
    "clusters": int(sys.argv[1]),
    "transitions": "weekly",
    "clustering": sys.argv[2],
    "pilot_start_date": sys.argv[3],
    #REVIEW: Adding New mapping config params
    "mapping_method": 'FO', # FO: Feature Only, WO: Warmup Only, FW: Feature+Warmup
    "train_warmup_end_date": 9, #TODO: Date format
    "test_warmup_end_date": 9,
    'warmup_feat_cols': ['P(L, N, L)', 'P(H, N, L)']
}

f = partial(pd.to_datetime, dayfirst=True)

if CONFIG['transitions'] == 'weekly':
    transitions = pd.read_csv("may_data/RMAB_one_month/weekly_transitions_SI_single_group.csv")
else:
    transitions = pd.read_csv("may_data/RMAB_one_month/transitions_SI_single_group.csv")

#TODO: Add path to test transitions
test_transitions = pd.read_csv('')

def kmeans_missing(X, n_clusters, max_iter=10):
    n_clusters = CONFIG['clusters']
    missing = ~np.isfinite(X)
    mu = np.nanmean(X, 0, keepdims=1)
    X_hat = np.where(missing, mu, X)

    prev_labels = None
    for i in range(max_iter):
        if CONFIG['clustering'] == 'optics':
            cls = OPTICS(min_samples=4, n_jobs=-1)
        elif CONFIG['clustering'] == 'kmeans':
            cls = KMeans(n_clusters, n_jobs=-1, random_state=0)
        elif CONFIG['clustering'] == 'spectral':
            cls = SpectralClustering(n_clusters, n_jobs=-1, random_state=0)

        labels = cls.fit_predict(X_hat)

        if CONFIG['clustering'] == 'kmeans':
            centroids = cls.cluster_centers_
        else:
            if CONFIG['clustering'] == 'optics':
                labels = labels + 1
            unique_labels = len(set(labels))
            centroids = []
            for i in range(unique_labels):
                idxes = np.where(labels == i)[0]
                centroids.append(np.mean(X_hat[idxes], axis=0))
            centroids = np.array(centroids)

        X_hat[missing] = centroids[labels][missing]

        if i > 0 and np.all(labels == prev_labels):
            break

        prev_labels = labels

    return labels, centroids, X_hat, cls, len(set(labels)), i


def get_individual_transition_clusters(train_beneficiaries, train_transitions, features_dataset, n_clusters):
    cols = [
        "P(L, I, L)", "P(L, I, H)", "P(H, I, L)", "P(H, I, H)", "P(L, N, L)", "P(L, N, H)", "P(H, N, L)", "P(H, N, H)", 
    ]

    user_ids, dynamic_xs, gest_ages, static_xs, ngo_hosp_ids, labels = features_dataset
    
    train_ids = train_beneficiaries['user_id']
    idxes = [np.where(user_ids == x)[0][0] for x in train_ids]
    train_static_features = static_xs[idxes]
    train_static_features = train_static_features[:, : -8]

    # test_ids = test_beneficiaries['user_id']
    # idxes = [np.where(user_ids == x)[0][0] for x in test_ids]
    # test_static_features = static_xs[idxes]
    # test_static_features = test_static_features[:, : -8]

    # REVIEW: Added Code to split all transitions into warmup and post warmup transition
    warmup_train_transitions = train_transitions[pd.to_datetime(train_transitions['start_date'])<f(CONFIG['train_warmup_end_date'])]
    train_transitions = train_transitions[pd.to_datetime(train_transitions['start_date'])>=f(CONFIG['train_warmup_end_date'])]
    
    all_transition_probabilities, _ = get_all_transition_probabilities(train_beneficiaries, train_transitions)
    all_warmup_transition_probabilities, warmup_sup = get_all_transition_probabilities(train_beneficiaries, warmup_train_transitions)
    pass_to_kmeans_cols = ['P(L, N, L)', 'P(H, N, L)']

    train_labels, centroids, _, cls, num_clusters, max_iters = kmeans_missing(all_transition_probabilities[pass_to_kmeans_cols], n_clusters, max_iter=100)
    
    # ipdb.set_trace()
    train_beneficiaries['cluster'] = train_labels
    # test_beneficiaries['cluster'] = cls.predict(test_static_features)

    dt_clf = RandomForestClassifier(n_estimators=200, criterion="entropy", max_depth=30, n_jobs=-1, random_state=124)
    # REVIEW: Added Code for mapping methods
    if CONFIG['mapping_method']=='FO':
        scaler = preprocessing.StandardScaler().fit(train_static_features)
        mapping_X = scaler.transform(train_static_features)
        dt_clf.fit(mapping_X, train_labels)
    elif CONFIG['mapping_method'] == 'WO':
        # TODO: Maybe use cluster predict fn
        warmup_feats = np.concatenate([warmup_sup[CONFIG['warmup_feat_cols']].values,
                                       all_warmup_transition_probabilities[CONFIG['warmup_feat_cols']].values], axis=1)
        scaler = preprocessing.StandardScaler().fit(warmup_feats)
        mapping_X = scaler.transform(warmup_feats)
        dt_clf.fit(mapping_X, train_labels)
    elif CONFIG['mapping_method'] == 'FW':
        all_feats = np.concatenate([train_static_features, warmup_sup[CONFIG['warmup_feat_cols']].values,
                                       all_warmup_transition_probabilities[CONFIG['warmup_feat_cols']].values], axis=1)
        scaler = preprocessing.StandardScaler().fit(all_feats)
        mapping_X = scaler.transform(all_feats)
        dt_clf.fit(mapping_X, train_labels)
    else:
        raise NotImplementedError

    cluster_transition_probabilities = pd.DataFrame(columns=['cluster', 'count'] + cols)

    for i in range(n_clusters):
        cluster_beneficiaries = train_beneficiaries[train_beneficiaries['cluster'] == i]
        cluster_b_user_ids = cluster_beneficiaries['user_id']
        probs, _ = get_transition_probabilities(cluster_b_user_ids, train_transitions, min_support=3)
        probs['cluster'] = i
        probs['count'] = len(cluster_b_user_ids)
        cluster_transition_probabilities = cluster_transition_probabilities.append(probs, ignore_index=True)

    # ipdb.set_trace()
    # TODO: Write unit test to verify alignment of features, transitions, predictions on same user ids

    return cluster_transition_probabilities, dt_clf, scaler, train_beneficiaries

def get_transition_probabilities(beneficiaries, transitions, min_support=3):
    transitions = transitions[transitions['user_id'].isin(beneficiaries)]

    i_transitions = transitions[transitions['action']=='Intervention']
    n_i_transitions = transitions[transitions['action']=='No Intervention']

    i_L = i_transitions[i_transitions['pre-action state']=="L"]
    i_H = i_transitions[i_transitions['pre-action state']=="H"]

    i_L_L = i_L[i_L['post-action state']=="L"]
    i_L_H = i_L[i_L['post-action state']=="H"]

    i_H_L = i_H[i_H['post-action state']=="L"]
    i_H_H = i_H[i_H['post-action state']=="H"]

    n_i_L = n_i_transitions[n_i_transitions['pre-action state']=="L"]
    n_i_H = n_i_transitions[n_i_transitions['pre-action state']=="H"]

    n_i_L_L = n_i_L[n_i_L['post-action state']=="L"]
    n_i_L_H = n_i_L[n_i_L['post-action state']=="H"]

    n_i_H_L = n_i_H[n_i_H['post-action state']=="L"]
    n_i_H_H = n_i_H[n_i_H['post-action state']=="H"]

    transition_probabilities = dict()
    if i_L.shape[0] >= min_support:
        transition_probabilities['P(L, I, L)'] = i_L_L.shape[0] / i_L.shape[0]
        transition_probabilities['P(L, I, H)'] = i_L_H.shape[0] / i_L.shape[0]
    else:
        transition_probabilities['P(L, I, L)'] = np.nan
        transition_probabilities['P(L, I, H)'] = np.nan

    if i_H.shape[0] >= min_support:
        transition_probabilities['P(H, I, L)'] = i_H_L.shape[0] / i_H.shape[0]
        transition_probabilities['P(H, I, H)'] = i_H_H.shape[0] / i_H.shape[0]
    else:
        transition_probabilities['P(H, I, L)'] = np.nan
        transition_probabilities['P(H, I, H)'] = np.nan
    
    if n_i_L.shape[0] >= min_support:
        transition_probabilities['P(L, N, L)'] = n_i_L_L.shape[0] / n_i_L.shape[0]
        transition_probabilities['P(L, N, H)'] = n_i_L_H.shape[0] / n_i_L.shape[0]
    else:
        transition_probabilities['P(L, N, L)'] = np.nan
        transition_probabilities['P(L, N, H)'] = np.nan

    if n_i_H.shape[0] >= min_support:
        transition_probabilities['P(H, N, L)'] = n_i_H_L.shape[0] / n_i_H.shape[0]
        transition_probabilities['P(H, N, H)'] = n_i_H_H.shape[0] / n_i_H.shape[0]
    else:
        transition_probabilities['P(H, N, L)'] = np.nan
        transition_probabilities['P(H, N, H)'] = np.nan

    return transition_probabilities, {'P(L, I, L)': i_L_L.shape[0], 'P(L, I, H)': i_L_H.shape[0], 'P(H, I, L)': i_H_L.shape[0], 'P(H, I, H)': i_H_H.shape[0], 'P(L, N, L)': n_i_L_L.shape[0], 'P(L, N, H)': n_i_L_H.shape[0], 'P(H, N, L)': n_i_H_L.shape[0], 'P(H, N, H)': n_i_H_H.shape[0]}

def get_all_transition_probabilities(train_beneficiaries, transitions):
    cols = [
        "P(L, I, L)", "P(L, I, H)", "P(H, I, L)", "P(H, I, H)", "P(L, N, L)", "P(L, N, H)", "P(H, N, L)", "P(H, N, H)", 
    ]
    transition_probabilities = pd.DataFrame(columns = ['user_id'] + cols)
    transition_sup = pd.DataFrame(columns = ['user_id'] + cols)
    user_ids = train_beneficiaries['user_id']

    for user_id in user_ids:
        probs, sup = get_transition_probabilities([user_id], transitions, min_support=1)
        probs['user_id'] = user_id
        sup['user_id'] = user_id
        transition_probabilities = transition_probabilities.append(probs, ignore_index=True)
        transition_sup = transition_sup.append(sup, ignore_index=True)

    return transition_probabilities, transition_sup


def get_reward(state, action, m):
    if state[0] == "L":
        reward = 1.0
    else:
        reward = -1.0
    if action == 'N':
        reward += m

    return reward


def plan(transition_probabilities, CONFIG):

    v_values = np.zeros((CONFIG['clusters'], len(CONFIG['problem']['states'])))
    q_values = np.zeros((CONFIG['clusters'], len(CONFIG['problem']['states']), len(CONFIG['problem']['actions'])))
    high_m_values = 1 * np.ones((CONFIG['clusters'], len(CONFIG['problem']['states'])))
    low_m_values = -1 * np.ones((CONFIG['clusters'], len(CONFIG['problem']['states'])))

    for cluster in range(CONFIG['clusters']):
        print('Planning for Cluster {}'.format(cluster))
        t_probs = np.zeros((len(CONFIG['problem']['states']), len(CONFIG['problem']['states']), len(CONFIG['problem']['actions'])))
        two_state_probs = np.zeros((2, 2, 2))
        for i in range(two_state_probs.shape[0]):
            for j in range(two_state_probs.shape[1]):
                for k in range(two_state_probs.shape[2]):
                    s = CONFIG['problem']['orig_states'][i]
                    s_prime = CONFIG['problem']['orig_states'][j]
                    a = CONFIG['problem']['actions'][k]
                    two_state_probs[i, j, k] = transition_probabilities.loc[transition_probabilities['cluster']==cluster, "P(" + s + ", " + a + ", " + s_prime + ")"]
        
        t_probs[0 : 2, 2 : 4, 0] = two_state_probs[:, :, 0]
        t_probs[2 : 4, 4 : 6, 0] = two_state_probs[:, :, 0]
        t_probs[4 : 6, 6 : 8, 0] = two_state_probs[:, :, 0]
        t_probs[6 : 8, 6 : 8, 0] = two_state_probs[:, :, 0]

        t_probs[0 : 2, 2 : 4, 1] = two_state_probs[:, :, 0]
        t_probs[2 : 4, 4 : 6, 1] = two_state_probs[:, :, 0]
        t_probs[4 : 6, 6 : 8, 1] = two_state_probs[:, :, 0]
        t_probs[6 : 8, 0 : 2, 1] = two_state_probs[:, :, 1]

        ipdb.set_trace()

        max_q_diff = np.inf
        prev_m_values, m_values = None, None
        while max_q_diff > 1e-5:
            prev_m_values = m_values
            m_values = (low_m_values + high_m_values) / 2
            if type(prev_m_values) != type(None) and abs(prev_m_values - m_values).max() < 1e-20:
                break
            max_q_diff = 0
            v_values[cluster, :] = np.zeros((len(CONFIG['problem']['states'])))
            q_values[cluster, :, :] = np.zeros((len(CONFIG['problem']['states']), len(CONFIG['problem']['actions'])))
            delta = np.inf
            while delta > 0.0001:
                delta = 0
                for i in range(t_probs.shape[0]):
                    v = v_values[cluster, i]
                    v_a = np.zeros((t_probs.shape[2],))
                    for k in range(v_a.shape[0]):
                        for j in range(t_probs.shape[1]):
                            v_a[k] += t_probs[i, j, k] * (get_reward(CONFIG['problem']['states'][i], CONFIG['problem']['actions'][k], m_values[cluster, i]) + CONFIG["gamma"] * v_values[cluster, j])

                    v_values[cluster, i] = np.max(v_a)
                    delta = max([delta, abs(v_values[cluster, i] - v)])

            state_idx = -1
            for state in range(q_values.shape[1]):
                for action in range(q_values.shape[2]):
                    for next_state in range(q_values.shape[1]):
                        q_values[cluster, state, action] += t_probs[state, next_state, action] * (get_reward(CONFIG['problem']['states'][state], CONFIG['problem']['actions'][action], m_values[cluster, state]) + CONFIG["gamma"] * v_values[cluster, next_state])
                # print(state, q_values[cluster, state, 0], q_values[cluster, state, 1])

            for state in range(q_values.shape[1]):
                if abs(q_values[cluster, state, 1] - q_values[cluster, state, 0]) > max_q_diff:
                    state_idx = state
                    max_q_diff = abs(q_values[cluster, state, 1] - q_values[cluster, state, 0])

            # print(q_values)
            # print(low_m_values, high_m_values)
            if max_q_diff > 1e-5 and q_values[cluster, state_idx, 0] < q_values[cluster, state_idx, 1]:
                low_m_values[cluster, state_idx] = m_values[cluster, state_idx]
            elif max_q_diff > 1e-5 and q_values[cluster, state_idx, 0] > q_values[cluster, state_idx, 1]:
                high_m_values[cluster, state_idx] = m_values[cluster, state_idx]

            # print(low_m_values, high_m_values, state_idx)
            # ipdb.set_trace()
    
    m_values = (low_m_values + high_m_values) / 2

    return q_values, m_values



def run_third_pilot(all_beneficiaries, transitions, call_data, CONFIG, features_dataset, pilot_data, beneficiary_data):
    CONFIG["read_sql"]=0
    from training_new.data import load_data as load_data_new
    from training_new.dataset import _preprocess_call_data as _preprocess_call_data_new, preprocess_and_make_dataset as preprocess_and_make_dataset_new
    pilot_beneficiary_data, pilot_call_data = load_data_new(CONFIG)
    inf_dataset = preprocess_and_make_dataset_new(pilot_beneficiary_data, pilot_call_data)
    pilot_call_data = _preprocess_call_data_new(pilot_call_data)
    pilot_user_ids, pilot_dynamic_xs, pilot_static_xs, pilot_hosp_id, pilot_labels = inf_dataset
    pilot_gold_e2c = pilot_labels[:, 3] / pilot_labels[:, 2]
    pilot_gold_e2c_processed = np.nan_to_num(pilot_gold_e2c)
    pilot_gold_labels = (pilot_gold_e2c_processed < 0.5) * 1.0
    pilot_gold_labels = pilot_gold_labels.astype(np.int)

    # ipdb.set_trace()

    enroll_gest_age_mean = np.mean(inf_dataset[2][:, 0])
    days_to_first_call_mean = np.mean(inf_dataset[2][:, 6])

    # dynamic features preprocessing
    pilot_dynamic_xs = pilot_dynamic_xs.astype(np.float32)
    pilot_dynamic_xs[:, :, 2] = pilot_dynamic_xs[:, :, 2] / 60
    pilot_dynamic_xs[:, :, 3] = pilot_dynamic_xs[:, :, 3] / 60
    pilot_dynamic_xs[:, :, 4] = pilot_dynamic_xs[:, :, 4] / 12

    # static features preprocessing
    pilot_static_xs = pilot_static_xs.astype(np.float32)
    pilot_static_xs[:, 0] = (pilot_static_xs[:, 0] - enroll_gest_age_mean)
    pilot_static_xs[:, 6] = (pilot_static_xs[:, 6] - days_to_first_call_mean)

    dependencies = {
        'BinaryAccuracy': BinaryAccuracy,
        'F1': F1,
        'Precision': Precision,
        'Recall': Recall
    }

    # model = load_model(os.path.join("models", 'lstm_model_final', "model"), custom_objects=dependencies)
    # output_probs = model.predict(x=[pilot_static_xs, pilot_dynamic_xs, pilot_hosp_id, pilot_gest_age])
    # out_lstm_labels = (output_probs >= 0.5).astype(np.int)
    # low_eng_idxes = np.where(out_lstm_labels == 1)[0]
    # low_eng_user_ids = np.array(pilot_user_ids)[low_eng_idxes]
    pilot_static_features = np.array(pilot_static_xs, dtype=np.float)
    pilot_static_features = pilot_static_features[:, : -8]

    cluster_transition_probabilities, cls, scaler, train_beneficiaries = get_individual_transition_clusters(all_beneficiaries, transitions, features_dataset, CONFIG['clusters'])
    cols = [
        "P(L, I, L)", "P(L, I, H)", "P(H, I, L)", "P(H, I, H)", "P(L, N, L)", "P(L, N, H)", "P(H, N, L)", "P(H, N, H)", 
    ]
    # ipdb.set_trace()
    for i in range(4, 8):
        if i % 2 == 0:
            curr_col = cols[i]
            cluster_transition_probabilities[curr_col] = cluster_transition_probabilities[curr_col].fillna(cluster_transition_probabilities[curr_col].mean())
        else:
            cluster_transition_probabilities[cols[i]] = 1 - cluster_transition_probabilities[cols[i - 1]]
    for i in range(4):
        if i % 2 == 0:
            curr_col = cols[i]
            est_col = cols[i + 4]
            diff_col = cluster_transition_probabilities[curr_col] - cluster_transition_probabilities[est_col]
            diff_col = diff_col.fillna(diff_col.mean())
            cluster_transition_probabilities[curr_col] = diff_col + cluster_transition_probabilities[est_col]
            cluster_transition_probabilities.loc[cluster_transition_probabilities[curr_col] >= 1, curr_col] = 1
        else:
            cluster_transition_probabilities[cols[i]] = 1 - cluster_transition_probabilities[cols[i - 1]]

    ipdb.set_trace()

    train_b_ids = train_beneficiaries['user_id'].to_list()
    cluster_transition_probabilities['mean_squared_error'] = 0.0
    cluster_counts = {}
    missing_counts = {}
    mse_vals = {}
    total_mse = 0.0
    total_count = 0
    ground_truth = []
    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    for user_id in train_b_ids:
        flag = False
        user_cluster = train_beneficiaries[train_beneficiaries['user_id'] == user_id]['cluster'].item()
        user_probs, _ = get_transition_probabilities([user_id], transitions)
        user_prob_list = [user_probs['P(L, N, L)'], user_probs['P(H, N, L)']]
        ground_truth.append(user_prob_list)
        cluster_row = cluster_transition_probabilities[cluster_transition_probabilities['cluster'] == user_cluster]
        cluster_prob_list = [cluster_row['P(L, N, L)'].item(), cluster_row['P(H, N, L)'].item()]
        try:
            mse_val = rmse(np.array(cluster_prob_list), np.array(user_prob_list))
            if mse_val < float('inf'):
                total_mse += mse_val
                total_count += 1
        except:
            flag = True
            mse_val = 0
        # ipdb.set_trace()
        if user_cluster not in cluster_counts:
            cluster_counts[user_cluster] = 1
            mse_vals[user_cluster] = mse_val
        else:
            cluster_counts[user_cluster] += 1
            mse_vals[user_cluster] += mse_val
        
        if flag:
            if user_cluster not in missing_counts:
                missing_counts[user_cluster] = 1
            else:
                missing_counts[user_cluster] += 1
        
        cluster_transition_probabilities.loc[cluster_transition_probabilities['cluster'] == user_cluster, 'mean_squared_error'] = cluster_transition_probabilities[cluster_transition_probabilities['cluster'] == user_cluster]['mean_squared_error'].item() + mse_val
    
    print(total_mse, total_count, total_mse / total_count)
    # REVIEW: Saving train cluster transition probabilities
    cluster_transition_probabilities.to_csv('outputs/new_mapping/{}_{}_transition_probabilities_{}.csv'.format(CONFIG['transitions'], CONFIG['clustering'], CONFIG['clusters']))
    # exit()

    ground_truth = np.array(ground_truth)
    with open('gt_beneficiary_probs.pkl', 'wb') as fr:
        pickle.dump(ground_truth, fr)
    fr.close()

    # ipdb.set_trace()
    # cluster_transition_probabilities['P(L] = cluster_transition_probabilities.fillna(cluster_transition_probabilities.mean())


    # REVIEW: Added code for predicting clusters using new mapping method
    pilot_benef_df = pd.DataFrame()
    pilot_benef_df['user_id']=pilot_user_ids

    if CONFIG['mapping_method']=='FO':
        pilot_cluster_predictions = cls.predict(scaler.transform(pilot_static_features))
    elif CONFIG['mapping_method'] == 'WO':
        test_warmup_transitions = test_transitions[pd.to_datetime(test_transitions['start_date'])<f(CONFIG["test_warmup_end_date"])]
        test_warmup_transition_probabilities, test_warmup_sup = get_all_transition_probabilities(pilot_benef_df, test_warmup_transitions)
        warmup_feats = np.concatenate([test_warmup_sup[CONFIG['warmup_feat_cols']].values,
                                       test_warmup_transition_probabilities[CONFIG['warmup_feat_cols']].values], axis=1)
        pilot_cluster_predictions = cls.predict(scaler.transform(warmup_feats))
    elif CONFIG['mapping_method'] == 'FW':
        test_warmup_transitions = test_transitions[pd.to_datetime(test_transitions['start_date'])<f(CONFIG["test_warmup_end_date"])]
        test_warmup_transition_probabilities, test_warmup_sup = get_all_transition_probabilities(pilot_benef_df, test_warmup_transitions)
        all_feats = np.concatenate([pilot_static_features, test_warmup_sup[CONFIG['warmup_feat_cols']].values,
                                       test_warmup_transition_probabilities[CONFIG['warmup_feat_cols']].values], axis=1)
        pilot_cluster_predictions = cls.predict(scaler.transform(all_feats))
    else:
        raise NotImplementedError
    
    # REVIEW: Save pilot predictions and GT
    pilot_benef_df['cluster'] = pilot_cluster_predictions
    pilot_benef_df = pd.merge(pilot_benef_df, cluster_transition_probabilities, on='cluster', how='left')
    pilot_benef_df.to_csv('outputs/new_mapping/pilot_predicted_prob.csv')
    test_post_warmup_transitions = test_transitions[pd.to_datetime(test_transitions['start_date'])>=f(CONFIG["test_warmup_end_date"])]
    test_post_warmup_transition_probabilities, test_post_warmup_sup = get_all_transition_probabilities(pilot_benef_df, test_post_warmup_transitions)
    test_post_warmup_transition_probabilities.to_csv('outputs/new_mapping/pilot_gt_prob.csv')

    exit()

    q_values, m_values = plan(cluster_transition_probabilities, CONFIG)

    print('-'*60)
    print('M Values: {}'.format(m_values))
    print('-'*60)

    print('-'*60)
    print('Q Values: {}'.format(q_values))
    print('-'*60)

    whittle_indices = {'user_id': [], 'whittle_index': [], 'cluster': [], 'start_state': [], 'lstm_prediction': [], 'gold_e2c': [], 'gold_label': [], 'registration_date': [], 'current_E2C': []}
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
        
        curr_cluster = pilot_cluster_predictions[idx]

        whittle_indices['user_id'].append(puser_id)
        whittle_indices['whittle_index'].append(m_values[curr_cluster, curr_state])
        whittle_indices['cluster'].append(curr_cluster)
        if curr_state == 7:
            whittle_indices['start_state'].append('NE')
        elif curr_state == 6:
            whittle_indices['start_state'].append('E')
        # if out_lstm_labels[idx] == 0:
        #     whittle_indices['lstm_prediction'].append('high_engagement')
        # elif out_lstm_labels[idx] == 1:
        #     whittle_indices['lstm_prediction'].append('low_engagement')
        whittle_indices['gold_e2c'].append(pilot_gold_e2c[idx])
        if pilot_gold_labels[idx] == 0:
            whittle_indices['gold_label'].append('high_engagement')
        elif pilot_gold_labels[idx] == 1:
            whittle_indices['gold_label'].append('low_engagement')
        
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
    df.to_csv('outputs/new_mapping/checking_{}_{}_pilot_stats_{}.csv'.format(CONFIG['transitions'], CONFIG['clustering'], CONFIG['clusters']))

    # ipdb.set_trace()

    return


run_third_pilot(all_beneficiaries, transitions, call_data, CONFIG, features_dataset, 'feb16-mar15_data', beneficiary_data)

# results, avgs = run_and_repeat(beneficiary_splits, transitions, call_data, CONFIG, features_dataset)
# print("-"*60)
# print(results)
# print(avgs)
# print("-"*60)
