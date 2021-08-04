import pandas as pd
import ipdb
import os
import pickle
from pandas.api.types import CategoricalDtype

from training.data import load_data
from training.utils import load_obj, save_obj
from training.dataset import _preprocess_beneficiary_data

ngo_hosp_dict = load_obj(os.path.join("training", "res", "ngo_hosp_dict.pkl"))
gest_dict = load_obj(os.path.join("training", "res", "gest_dict.pkl"))

def preprocess_b_data(data, call_data):
	# replace invalid values of each numeric column
    valid_ranges = {
        "enroll_gest_age": (1, 60, 14),
        "age": (15, 50, 25),
        "g": (1, 5, -1),
        "p": (0, 5, -1),
        "s": (0, 1, -1),
        "l": (0, 5, -1),
        "a": (0, 2, -1),
    }
    for col in valid_ranges.keys():
        min_val, max_val, replace_val = valid_ranges[col]
        data = data.round({col: 0})
        data.loc[~((min_val <= data[col]) & (data[col] <= max_val)), col] = replace_val

    # replace invalid values of each categorical column
    valid_categories = {
        "call_slots": ([1, 2, 3, 4, 5, 6], 3),
        "enroll_delivery_status": ([0, 1], 0),
        "language": ([2, 3, 4, 5], 2),
        "ChannelType": ([0, 1, 2], 0),
        "ngo_hosp_id": (ngo_hosp_dict.keys(), 4),
        "education": ([1, 2, 3, 4, 5, 6, 7], 3),
        "phone_owner": ([0, 1, 2], 2),
        "income_bracket": ([-1, 0, 1, 2, 3, 4, 5, 6], -1),
    }
    for col in valid_categories.keys():
        valid_vals, replace_val = valid_categories[col]
        data.loc[~data[col].isin(valid_vals), col] = replace_val
        data[col] = data[col].astype(CategoricalDtype(categories=valid_vals))

    # binning age into five groups
    bins = [14, 20, 25, 30, 35, 51]
    data.loc[:, "age"] = pd.cut(data["age"], bins=bins, labels=False)
    data["age"] = data["age"].astype(CategoricalDtype(categories=[0, 1, 2, 3, 4]))

    data = data.dropna()

    df = pd.DataFrame()

    for idx, row, in data.iterrows():
        if row['enroll_delivery_status'] == 1:
            row['enroll_gest_age'] += 40
        df = df.append(row, ignore_index=True)

    return df


whittle_indices = pd.read_csv("outputs/individual_clustering/weekly_kmeans_pilot_stats_40.csv")

b_data, c_data = load_data("feb16-mar15_data")
beneficiary_data = preprocess_b_data(b_data, c_data)

bins = [0, 20, 40, 60, 80, 100]
beneficiary_data.loc[:, "enroll_gest_age"] = pd.cut(beneficiary_data["enroll_gest_age"], bins=bins, labels=False)
beneficiary_data["enroll_gest_age"] = beneficiary_data["enroll_gest_age"].astype(CategoricalDtype(categories=[0, 1, 2, 3, 4]))

with open("policy_dump.pkl", 'rb') as fr:
    pilot_user_ids,pilot_static_features,cls,cluster_transition_probabilities,m_values,q_values = pickle.load(fr)
fr.close()


def get_cluster_stats(whittle_indices):
# ipdb.set_trace()

	columns = [
	        "cluster",
			"whittle_index_NE",
			"whittle_index_E",
	        "count",
	        "enroll_gest_age_0",
	        "enroll_gest_age_1",
	        "enroll_gest_age_2",
	        "enroll_gest_age_3",
	        "enroll_gest_age_4",
	        "enroll_delivery_status_0",
	        "enroll_delivery_status_1",
	        "g_-1",
	        "g_1",
	        "g_2",
	        "g_3",
	        "g_4",
	        "g_5",
	        "p_-1",
	        "p_0",
	        "p_1",
	        "p_2",
	        "p_3",
	        "p_4",
	        "p_5",
	        "s_-1",
	        "s_0",
	        "s_1",
	        "l_-1",
	        "l_0",
	        "l_1",
	        "l_2",
	        "l_3",
	        "l_4",
	        "l_5",
	        "a_-1",
	        "a_0",
	        "a_1",
	        "a_2",
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
	    ]

	cluster_stats = pd.DataFrame(columns=columns)

	for i in range(40):
		cluster_b_user_ids = whittle_indices[whittle_indices['cluster'] == i ]['user_id']
		cluster_user_features = beneficiary_data[beneficiary_data['user_id'].isin(cluster_b_user_ids)]
		# ipdb.set_trace()

		stats = dict()
		cluster_user_features = cluster_user_features.drop(columns=['ngo_hosp_id', 'user_id', 'entry_date', 'registration_date'])
		for column in cluster_user_features.columns:
			counts = cluster_user_features[column].value_counts(normalize=True)
			for key, value in counts.items():
				stats[f'{column}_{int(key)}'] = "{:.2f}".format(value*100)

		stats['count'] = len(cluster_b_user_ids)
		stats['cluster'] = i
		stats['whittle_index_NE'] = m_values[i, 7]
		stats['whittle_index_E'] = m_values[i, 6]

		for column in columns:
			stats[column] = stats.get(column, 0)

		cluster_stats = cluster_stats.append(stats, ignore_index=True)

	cluster_stats = cluster_stats.sort_values('whittle_index_NE', ascending=False)
	# ipdb.set_trace()
	cluster_stats.to_csv("kmeans_cluster_feature_dist.csv")
	return cluster_stats


get_cluster_stats(whittle_indices)