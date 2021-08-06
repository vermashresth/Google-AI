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
import ipdb

plt.style.use("seaborn")
np.random.seed(1)

from training.utils import load_obj, save_obj
from training.data import load_data
from training.dataset import _preprocess_call_data

from sklearn.cluster import KMeans

stats = pd.read_csv("may_data/beneficiary_stats_v5.csv")
beneficiary_data = pd.read_csv("may_data/beneficiary/AIRegistration-20200501-20200731.csv")
_, call_data = load_data("may_data")
call_data = _preprocess_call_data(call_data)

beneficiary_splits = load_obj("may_data/RMAB_one_month/beneficiary_splits_updated.pkl")
transitions = pd.read_csv("may_data/RMAB_one_month/transitions_SI_updated.csv")

CONFIG = {
    "problem": {
        "states": ["L", "H"],
        "actions": ["N", "I"],
    },
    "time_step": 30,
    "gamma": 0.99,
    "clusters": int(sys.argv[1])
}

def kmeans_missing(X, n_clusters, max_iter=10):

    missing = ~np.isfinite(X)
    mu = np.nanmean(X, 0, keepdims=1)
    X_hat = np.where(missing, mu, X)

    for i in range(max_iter):
        if i > 0:
            cls = KMeans(n_clusters, random_state=0)
        else:
            cls = KMeans(n_clusters, n_jobs=-1, random_state=0)

        labels = cls.fit_predict(X_hat)
        centroids = cls.cluster_centers_

        X_hat[missing] = centroids[labels][missing]

        if i > 0 and np.all(labels == prev_labels):
            break

        prev_labels = labels
        prev_centroids = cls.cluster_centers_

    return labels, centroids, X_hat

def get_transition_probabilities(beneficiaries, transitions, min_support=5):
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

    return transition_probabilities

def get_group_transition_probabilities(train_beneficiaries, transitions):
    features = ["education", "phone_owner", "income_bracket", "age"]
    cols = [
        "P(L, I, L)", "P(L, I, H)", "P(H, I, L)", "P(H, I, H)", "P(L, N, L)", "P(L, N, H)", "P(H, N, L)", "P(H, N, H)", 
    ]
    education_s = [[7, 1], [2, 3, 4], [5, 6]]
    phone_owner_s = ["husband", "woman", "family"]
    income_bracket_s = [['5000-10000', '0-5000', '10000-15000'], ['30000 and above', '25000-30000', '20000-25000', '15000-20000']]
    age_s = [(0, 19), (19, 24), (24, 29), (29, 36), (36, 100)]

    transition_probabilities = pd.DataFrame(columns=features+cols)
    for education_idx in range(len(education_s)):
        for phone_owner_idx in range(len(phone_owner_s)):
            for income_bracket_idx in range(len(income_bracket_s)):
                for age_idx in range(len(age_s)):
                    beneficiaries = beneficiary_data[
                        (beneficiary_data['user_id'].isin(train_beneficiaries['user_id']))&
                        (beneficiary_data['age']>=age_s[age_idx][0])&
                        (beneficiary_data['age']<age_s[age_idx][1])&
                        (beneficiary_data['education'].isin(education_s[education_idx]))&
                        (beneficiary_data['phone_owner']==phone_owner_s[phone_owner_idx])&
                        (beneficiary_data['income_bracket'].isin(income_bracket_s[income_bracket_idx]))
                    ]["user_id"]
                    probs = get_transition_probabilities(beneficiaries, transitions)

                    probs["education"] = education_idx
                    probs["income_bracket"] = income_bracket_idx
                    probs["phone_owner"] = phone_owner_idx
                    probs["age"] = age_idx
                    probs['count'] = beneficiaries.shape[0]

                    transition_probabilities = transition_probabilities.append(probs, ignore_index=True)

    return transition_probabilities

def get_pooled_estimate(group_transition_probabilities, train_beneficiaries, transitions):
    education_s = [[7, 1], [2, 3, 4], [5, 6]]
    phone_owner_s = ["husband", "woman", "family"]
    income_bracket_s = [['5000-10000', '0-5000', '10000-15000'], ['30000 and above', '25000-30000', '20000-25000', '15000-20000']]
    age_s = [(0, 19), (19, 24), (24, 29), (29, 36), (36, 100)]

    beneficiaries = []
    for i in group_transition_probabilities.index:
        education_idx = int(group_transition_probabilities.loc[i, "education"].item())
        income_bracket_idx = int(group_transition_probabilities.loc[i, "income_bracket"].item())
        phone_owner_idx = int(group_transition_probabilities.loc[i, "phone_owner"].item())
        age_idx = int(group_transition_probabilities.loc[i, "age"].item())

        group_beneficiaries = beneficiary_data[
            (beneficiary_data['user_id'].isin(train_beneficiaries['user_id']))&
            (beneficiary_data['age']>=age_s[age_idx][0])&
            (beneficiary_data['age']<age_s[age_idx][1])&
            (beneficiary_data['education'].isin(education_s[education_idx]))&
            (beneficiary_data['phone_owner']==phone_owner_s[phone_owner_idx])&
            (beneficiary_data['income_bracket'].isin(income_bracket_s[income_bracket_idx]))
        ]["user_id"]
        beneficiaries.append(group_beneficiaries)

    beneficiaries = pd.concat(beneficiaries, axis=0)
    probs = get_transition_probabilities(beneficiaries, transitions)

    return probs

def get_reward(state, action, m):
    if state == "L":
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
        for i in range(t_probs.shape[0]):
            for j in range(t_probs.shape[1]):
                for k in range(t_probs.shape[2]):
                    s = CONFIG['problem']['states'][i]
                    s_prime = CONFIG['problem']['states'][j]
                    a = CONFIG['problem']['actions'][k]
                    t_probs[i, j, k] = transition_probabilities.loc[transition_probabilities['cluster']==cluster, "P(" + s + ", " + a + ", " + s_prime + ")"]

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

def count_overlaps(test_beneficiaries):
    call_beneficiaries = test_beneficiaries[test_beneficiaries['Group']=="Google-AI-Calls"]
    call_succ_beneficiaries = test_beneficiaries[
        (test_beneficiaries['Group']=="Google-AI-Calls")&
        (test_beneficiaries['Intervention Status']=="Successful")
    ]
    control_beneficiaries = test_beneficiaries[test_beneficiaries['Group']=="Google-AI-Control"]

    good_call_response = test_beneficiaries[
        (test_beneficiaries['user_id'].isin(call_beneficiaries['user_id']))&
        (test_beneficiaries['Post-intervention Day: E2C Ratio']>=0.5)
    ]['user_id']

    good_succ_call_response = test_beneficiaries[
        (test_beneficiaries['user_id'].isin(call_succ_beneficiaries['user_id']))&
        (test_beneficiaries['Post-intervention Day: E2C Ratio']>=0.5)
    ]['user_id']

    good_control_response = test_beneficiaries[
        (test_beneficiaries['user_id'].isin(control_beneficiaries['user_id']))&
        (test_beneficiaries['Post-intervention Day: E2C Ratio']>=0.5)
    ]['user_id']

    print([call_beneficiaries.shape[0], call_succ_beneficiaries.shape[0], control_beneficiaries.shape[0]])

    notes = np.array([
        [good_call_response.shape[0], 
        good_succ_call_response.shape[0],
        good_control_response.shape[0]]
    ])

    results = []
    for k in [100, 200]:
        top_k_call_beneficiaries = call_beneficiaries.iloc[:k, :]
        top_k_good_call_beneficiaries = top_k_call_beneficiaries[top_k_call_beneficiaries['user_id'].isin(good_call_response)]

        top_k_succ_call_beneficiaries = call_succ_beneficiaries.iloc[:k, :]
        top_k_good_succ_call_beneficiaries = top_k_succ_call_beneficiaries[top_k_succ_call_beneficiaries['user_id'].isin(good_succ_call_response)]

        top_k_control_beneficiaries = control_beneficiaries.iloc[:k, :]
        top_k_good_control_beneficiaries = top_k_control_beneficiaries[top_k_control_beneficiaries['user_id'].isin(good_control_response)]

        results.append([
            top_k_good_call_beneficiaries.shape[0]/k, 
            top_k_good_succ_call_beneficiaries.shape[0]/k,
            top_k_good_control_beneficiaries.shape[0]/k
        ])

    return np.array(results), notes

def run_experiment(train_beneficiaries, train_transitions, test_beneficiaries, call_data, CONFIG):

    group_transition_probabilities = get_group_transition_probabilities(train_beneficiaries, train_transitions)
    cols = [
        "P(L, I, L)", "P(L, I, H)", "P(H, I, L)", "P(H, I, H)", "P(L, N, L)", "P(L, N, H)", "P(H, N, L)", "P(H, N, H)", 
    ]
    labels, centroids, X_hat = kmeans_missing(group_transition_probabilities[cols], CONFIG['clusters'], max_iter=100)
    group_transition_probabilities['cluster'] = labels
    cluster_transition_probabilities = pd.DataFrame(columns=["cluster"] + cols)

    for i in range(centroids.shape[0]):
        t_probs = get_pooled_estimate(
            group_transition_probabilities[group_transition_probabilities['cluster']==i],
            train_beneficiaries, 
            train_transitions
        )
        t_probs["cluster"] = i
        # for j in range(len(cols)):
        #   t_probs[cols[j]] = centroids[i, j]

        cluster_transition_probabilities = cluster_transition_probabilities.append(t_probs, ignore_index=True)

    cluster_transition_probabilities = cluster_transition_probabilities.fillna(cluster_transition_probabilities.mean())

    q_values, m_values = plan(cluster_transition_probabilities, CONFIG)

    print('-'*60)
    print('M Values: {}'.format(m_values))
    print('-'*60)

    education_s = [[7, 1], [2, 3, 4], [5, 6]]
    phone_owner_s = ["husband", "woman", "family"]
    income_bracket_s = [['5000-10000', '0-5000', '10000-15000'], ['30000 and above', '25000-30000', '20000-25000', '15000-20000']]
    age_s = [(0, 19), (19, 24), (24, 29), (29, 36), (36, 100)]

    for i in tqdm(test_beneficiaries.index):
        user_id = test_beneficiaries.loc[i, "user_id"]
        beneficiary_features = beneficiary_data.loc[beneficiary_data['user_id']==user_id, :]
        call_features = stats.loc[stats['user_id']==user_id, :]

        for j in range(len(education_s)):
            if beneficiary_features['education'].item() in education_s[j]:
                education = j
                break

        for j in range(len(phone_owner_s)):
            if beneficiary_features['phone_owner'].item() == phone_owner_s[j]:
                phone_owner = j
                break

        for j in range(len(income_bracket_s)):
            if beneficiary_features['income_bracket'].item() in income_bracket_s[j]:
                income_bracket = j
                break

        for j in range(len(age_s)):
            if (beneficiary_features['age'].item() >= age_s[j][0]) and (beneficiary_features['age'].item() < age_s[j][1]):
                age = j
                break

        cluster = group_transition_probabilities.loc[
            (group_transition_probabilities['education']==education)&
            (group_transition_probabilities['phone_owner']==phone_owner)&
            (group_transition_probabilities['income_bracket']==income_bracket)&
            (group_transition_probabilities['age']==age),
            "cluster"
        ].item()
        state = int(test_beneficiaries.loc[i, "state"])

        # test_beneficiaries.loc[i, "Whittle Index"] = q_values[cluster, state, 1] - q_values[cluster, state, 0]
        test_beneficiaries.loc[i, "Whittle Index"] = m_values[cluster, state]

    test_beneficiaries = test_beneficiaries.sort_values(by="Whittle Index", ascending=False)
    results, notes = count_overlaps(test_beneficiaries)

    return results, notes

def run_and_repeat(beneficiaries, transitions, call_data, CONFIG):
    all_results, all_notes = [], []

    for train_beneficiaries, test_beneficiaries in beneficiaries:
        train_transitions = transitions[transitions['user_id'].isin(train_beneficiaries['user_id'])]
        results, notes = run_experiment(train_beneficiaries, train_transitions, test_beneficiaries, call_data, CONFIG)

        print(results)
        print(notes)
        
        all_results.append(results)
        all_notes.append(notes)

    return np.mean(np.stack(all_results, axis=0), axis=0), np.mean(np.stack(all_notes, axis=0), axis=0)

results, avgs = run_and_repeat(beneficiary_splits, transitions, call_data, CONFIG)
print("-"*60)
print(results)
print(avgs)
print("-"*60)