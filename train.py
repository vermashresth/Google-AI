import argparse
from cgi import test
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import tqdm 
from collections import defaultdict
from itertools import combinations

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

from clustering import kmeans_missing, get_transition_probabilities, prepare_clustering_X, get_imputed_cluster_TP_estimate
from feature_transform import prepare_feat_df



print('Loading Transitions')
# Step 1: Load Transitions
pilot_trans_df = pd.read_csv('data/pilot_transitions_5months.csv')
# Step 1.1: Define Warmup period transitions
warmup_pilot_trans_df = pilot_trans_df.groupby('user_id').head(15)
#Step 1.2: Define After Warmup period Transitions 
after_warmup_pilot_trans_df = pilot_trans_df.groupby('user_id').tail(6)



# Step 2: Load Features
print('Loading Beneficiriary Features')
with open('data/policy_dump.pkl', 'rb') as fr:
  pilot_user_ids, pilot_static_features, cls, cluster_transition_probabilities, m_values, q_values = pickle.load(fr)
fr.close()



# Step 3: Calculate All Individual Transition Probabilities
user_probs, sup = get_transition_probabilities([pilot_user_ids[0]], pilot_trans_df)

all_probs = pd.DataFrame(columns = ['user_id'] + list(user_probs.keys()))
for user_id in tqdm.tqdm(pilot_user_ids):
        user_probs, sup = get_transition_probabilities([user_id], pilot_trans_df)
        user_probs['user_id'] = user_id
        all_probs = all_probs.append(user_probs, ignore_index=True)

# Step 3.1 : Calculate Warmup Individual Transition Probabilities
warmup_all_probs = pd.DataFrame(columns = ['user_id'] + list(user_probs.keys()))
warmup_sup = pd.DataFrame(columns = ['user_id'] + list(user_probs.keys()))
for user_id in tqdm.tqdm(pilot_user_ids):
        user_probs, sup = get_transition_probabilities([user_id], warmup_pilot_trans_df)
        user_probs['user_id'] = user_id
        warmup_all_probs = warmup_all_probs.append(user_probs, ignore_index=True)
        warmup_sup = warmup_sup.append(sup, ignore_index=True)

# Step 3.2 : Calculate After warmup Individual Transition Probabilities
after_warmup_all_probs = pd.DataFrame(columns = ['user_id'] + list(user_probs.keys()))
for user_id in tqdm.tqdm(pilot_user_ids):
        user_probs, sup = get_transition_probabilities([user_id], after_warmup_pilot_trans_df)
        user_probs['user_id'] = user_id
        after_warmup_all_probs = after_warmup_all_probs.append(user_probs, ignore_index=True)



# Step 4: Prepare feature dataframe - normalize
pilot_feat_df, feat_cols = prepare_feat_df(pilot_static_features, pilot_user_ids)



# Step 5: Clustering
print('Clustering')

CLUSTERING_INPUT = 'PASSIVE_PROB'
N_CLUSTERS = 40
SPLIT_FRAC = 0.8

X, cluster_feat_cols = prepare_clustering_X(CLUSTERING_INPUT, feat_cols, pilot_feat_df, warmup_all_probs, after_warmup_all_probs, all_probs)
n = X.shape[0]
shuffled = pd.Series(range(n)).sample(frac=1).values
train_idx, test_idx = shuffled[:int(n*SPLIT_FRAC)],\
                               shuffled[int(n*SPLIT_FRAC):]
print('Train data points: ', len(train_idx))
print('Test data points: ', len(test_idx))

X_test = X.iloc[test_idx]
Y_test = after_warmup_all_probs.iloc[test_idx]
X_train = X.iloc[train_idx]

out =  kmeans_missing(X_train[cluster_feat_cols], N_CLUSTERS) 
labels, centroids, X_hat, kmeans_new, _, _ = out
X_train = pd.DataFrame(X_hat, columns=cluster_feat_cols)
X_train['fitted_cluster'] = labels
X_train['user_id'] = pilot_user_ids[train_idx]
cluster_transition_probabilities = get_imputed_cluster_TP_estimate(X_train, after_warmup_pilot_trans_df, N_CLUSTERS)


# Step 6: Fit Mapping Fn, Evaluate
mapping_X_train = pilot_feat_df.iloc[train_idx]
mapping_X_test = pilot_feat_df.iloc[test_idx]

dt_clf = RandomForestClassifier(n_estimators=100, criterion="entropy", max_depth=5, n_jobs=-1, random_state=124)
dt_clf.fit(mapping_X_train, X_train['fitted_cluster'])

predicted_labels = dt_clf.predict(mapping_X_test)
predicted_passive_prob = cluster_transition_probabilities.set_index('cluster').loc[predicted_labels][['P(L, N, L)', 'P(H, N, L)']].values   
true_passive_prob = Y_test[['P(L, N, L)', 'P(H, N, L)']].values
error = np.fabs(predicted_passive_prob - true_passive_prob )
print('Feature Only Mapping, Mean Error', np.mean(error.flatten()[~np.isnan(error.flatten())]))
# plt.hist(error.flatten()[~np.isnan(error.flatten())], bins=10)
# plt.show()
new_feat_df = pd.concat([pilot_feat_df.reset_index()[feat_cols],
                   warmup_sup[['P(L, N, L)', 'P(H, N, L)', 'P(L, I, L)', 'P(H, I, L)']]], axis=1).fillna(-1)

mapping_X_train = new_feat_df.iloc[train_idx]
mapping_X_test = new_feat_df.iloc[test_idx]

dt_clf = RandomForestClassifier(n_estimators=100, criterion="entropy", max_depth=5, n_jobs=-1, random_state=124)
dt_clf.fit(mapping_X_train, X_train['fitted_cluster'])

predicted_labels = dt_clf.predict(mapping_X_test)
predicted_passive_prob = cluster_transition_probabilities.set_index('cluster').loc[predicted_labels][['P(L, N, L)', 'P(H, N, L)']].values
true_passive_prob = Y_test[['P(L, N, L)', 'P(H, N, L)']].values
error = np.fabs(predicted_passive_prob - true_passive_prob )
print('Feature + Warmup Mapping, Mean Error',  np.mean(error.flatten()[~np.isnan(error.flatten())]))
# plt.hist(error.flatten()[~np.isnan(error.flatten())], bins=10)
# plt.show()


new_feat_df = warmup_sup[['P(L, N, L)', 'P(H, N, L)', 'P(L, I, L)', 'P(H, I, L)']]

mapping_X_train = new_feat_df.iloc[train_idx]
mapping_X_test = new_feat_df.iloc[test_idx]

dt_clf = RandomForestClassifier(n_estimators=100, criterion="entropy", max_depth=5, n_jobs=-1, random_state=124)
dt_clf.fit(mapping_X_train, X_train['fitted_cluster'])

predicted_labels = dt_clf.predict(mapping_X_test)
predicted_passive_prob = cluster_transition_probabilities.set_index('cluster').loc[predicted_labels][['P(L, N, L)', 'P(H, N, L)']].values
true_passive_prob = Y_test[['P(L, N, L)', 'P(H, N, L)']].values
error = np.fabs(predicted_passive_prob - true_passive_prob )
print('Warmup Only Mapping, Mean Error',  np.mean(error.flatten()[~np.isnan(error.flatten())]))
# plt.hist(error.flatten()[~np.isnan(error.flatten())], bins=10)
# plt.show()





