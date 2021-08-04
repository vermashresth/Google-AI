import os
import pandas as pd
import numpy as np
import pickle
import ipdb
# import random

from training.data import load_data
from training.utils import load_obj, save_obj
from training.dataset import preprocess_and_make_dataset

with open("policy_dump.pkl", 'rb') as fr:
    pilot_user_ids,pilot_static_features,cls,cluster_transition_probabilities,m_values,q_values = pickle.load(fr)
fr.close()	


prev_wi_soft = pd.read_csv("whittle_indices_kmeans_soft.csv")
prev_wi_hard = pd.read_csv("outputs/individual_clustering/weekly_kmeans_pilot_stats_40.csv")

pilot_beneficiary_data, pilot_call_data = load_data("feb16-mar15_data")

ngo_hosp_ids = set(pilot_beneficiary_data[pilot_beneficiary_data['ChannelType']==1]['ngo_hosp_id'].tolist())

def pick_beneficiaries(prev_wi_soft, prev_wi_hard, pilot_beneficiary_data):

	prev_wi_soft = prev_wi_soft.sort_values("whittle_index_NE", ascending=True)

	b_picked = {'user_id': [], 'prev_val': []}

	for idx, row in prev_wi_soft.iterrows():

		if len(b_picked['user_id']) >= 1000:
			break

		user_id = row['user_id']
		# is state NE
		if user_id in prev_wi_hard['user_id'].values:
			if prev_wi_hard[prev_wi_hard['user_id'] == user_id ]['start_state'].item() == 'NE':
				index = pilot_beneficiary_data.index[pilot_beneficiary_data['user_id'] == user_id].tolist()[0]
				#is ChannelType 0
				# if pilot_beneficiary_data[pilot_beneficiary_data['user_id'] == user_id]['ChannelType'].item() == 0:
				# 	pilot_beneficiary_data.loc[index, ['ChannelType']] = [1]
					# b_picked.append(user_id)
				# is income bracket not 1
				# income_bracket = pilot_beneficiary_data[pilot_beneficiary_data['user_id'] == user_id]['income_bracket'].item()
				# if income_bracket != 1:
				# 	b_picked['user_id'].append(user_id)
				# 	b_picked['prev_val'].append(income_bracket)
				# 	pilot_beneficiary_data.loc[index, ['income_bracket']] = [1]
				# language not 3
				# language = pilot_beneficiary_data[pilot_beneficiary_data['user_id'] == user_id]['language'].item()
				# if language != 3:
				# 	b_picked['user_id'].append(user_id)
				# 	b_picked['prev_val'].append(language)
				# 	pilot_beneficiary_data.loc[index, ['language']] = [3]
				# education not 3
				# education = pilot_beneficiary_data[pilot_beneficiary_data['user_id'] == user_id]['education'].item()
				# if education != 3:
				# 	b_picked['user_id'].append(user_id)
				# 	b_picked['prev_val'].append(education)
				# 	pilot_beneficiary_data.loc[index, ['education']] = [3]
				# age not in 20 to 30
				age = pilot_beneficiary_data[pilot_beneficiary_data['user_id'] == user_id]['age'].item()
				if not (age >= 20 and age < 25):
					b_picked['user_id'].append(user_id)
					b_picked['prev_val'].append(age)
					pilot_beneficiary_data.loc[index, ['age']] = 22

	return b_picked

b_picked = pick_beneficiaries(prev_wi_soft, prev_wi_hard, pilot_beneficiary_data)
inf_dataset = preprocess_and_make_dataset(pilot_beneficiary_data, pilot_call_data)

pilot_user_ids, pilot_dynamic_xs, pilot_gest_age, pilot_static_xs, pilot_hosp_id, pilot_labels = inf_dataset

enroll_gest_age_mean = np.mean(inf_dataset[3][:, 0])
days_to_first_call_mean = np.mean(inf_dataset[3][:, 7])

# static features preprocessing
pilot_static_xs = pilot_static_xs.astype(np.float32)
pilot_static_xs[:, 0] = (pilot_static_xs[:, 0] - enroll_gest_age_mean)
pilot_static_xs[:, 7] = (pilot_static_xs[:, 7] - days_to_first_call_mean)

pilot_static_features = np.array(pilot_static_xs, dtype=np.float)
pilot_static_features = pilot_static_features[:, : -8]

pilot_cluster_predictions = cls.predict(pilot_static_features)
pilot_cluster_soft_predictions = cls.predict_proba(pilot_static_features)

whittle_indices = {'user_id': [],'prev_feature_value': [] , 'whittle_index_curr': [], 'cluster_curr': [], 'whittle_index_prev': [], 'cluster_prev': [], 'soft_wi_curr': [], 'soft_wi_prev':[]}
for idx, puser_id in enumerate(pilot_user_ids):

	if puser_id not in b_picked['user_id']:
		continue

	soft_wi = 0
	curr_probs = pilot_cluster_soft_predictions[idx]
	for i in range(40):
		soft_wi += curr_probs[i] * m_values[i, 7]

	curr_cluster = pilot_cluster_predictions[idx]

	whittle_indices['user_id'].append(puser_id)
	whittle_indices['prev_feature_value'].append(b_picked['prev_val'][ b_picked['user_id'].index(puser_id) ])
	whittle_indices['whittle_index_curr'].append(m_values[curr_cluster, 7]) # 7 => curr_state = 'NE'
	whittle_indices['cluster_curr'].append(curr_cluster)
	whittle_indices['whittle_index_prev'].append(prev_wi_hard[prev_wi_hard['user_id']==puser_id]['whittle_index'].item())
	whittle_indices['cluster_prev'].append(prev_wi_hard[prev_wi_hard['user_id']==puser_id]['cluster'].item())
	whittle_indices['soft_wi_curr'].append(soft_wi)
	whittle_indices['soft_wi_prev'].append(prev_wi_soft[prev_wi_soft['user_id']==puser_id]['whittle_index_NE'].item())

df = pd.DataFrame(whittle_indices)
df = df.sort_values('whittle_index_curr', ascending=False)
df.to_csv("outputs/counterfactual_testing/age_1.csv")
# ipdb.set_trace()