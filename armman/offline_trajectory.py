import random
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from dfl.config import policy_names, policy_map, dim_dict
from dfl.ope import opeSimulator

def get_offline_traj(policy, T):
    # load dataframes from offline data
    interv_df = pd.read_csv('../offline_data/interventions.csv')
    analysis_df = pd.read_csv('../offline_data/state-cluster-whittle-E_C.csv')
    with open('../offline_data/policy_dump.pkl', 'rb') as fr:
        pilot_user_ids, pilot_static_features, cls, cluster_transition_probabilities, m_values, q_values = pickle.load(fr)
    fr.close()
    print('loaded data')
    # if policy option is rr or random, we will fetch real world round robin trajectories
    policy_equiv_dict = {'rr':'round_robin', 'random':'round_robin'}

    # filter only entries for given policy
    pol_analysis_df = analysis_df[analysis_df['arm']==policy_equiv_dict[policy]]
    # note all the state columns
    state_cols = [f'week{i}_state' for i in range(T+1)]
    # state 0 : sleeping engaging, 1: sleeping non-engaging, 6: engaging, 7: sleeping non engaging
    # to convert this into 0: non-engaging, 1: engaging
    state_df = (pol_analysis_df[state_cols]%2==0).astype(int)
    state_matrix = state_df.values
    assert state_matrix.shape[1] == T+1

    # Reward is same as states, but exists only for T-1 steps
    reward_matrix = np.copy(state_matrix[:, :-1])

    # filter intervention df for given policy
    pol_interv_df = interv_df[interv_df['exp_group']==policy_equiv_dict[policy]]
    # note columns for every week
    week_cols = [f'week{i+1}' for i in range(T)]
    # Convert weeks from long to wide, and mark 1 for every week when intervened
    pol_interv_df = (pol_interv_df.pivot(index='user_id',
                    columns='intervene_week',
                    values='intervene_week')[week_cols].isna()!=1).astype(int).reset_index()
    # merge pol interventions df with pol analysis df, to get interventions actions 
    # in the same order as state matrix
    actions_df = pd.merge(pol_analysis_df[['user_id']],
            pol_interv_df[['user_id']+week_cols],
            how='left').fillna(0)
    action_matrix = actions_df[week_cols].values
    assert action_matrix.shape[0] == state_matrix.shape[0]
    assert action_matrix.shape[1] == T


    n_benefs = len(action_matrix)
    offline_traj = np.zeros((1, len(policy_names), T, len(dim_dict), n_benefs))

    offline_traj[0, # trial index
                        policy_map[policy], # policy index
                        :, # time index
                        dim_dict['state'], # tuple dimension
                        : # benef index
                    ] = np.copy(state_matrix.T[:-1, :])
    offline_traj[0, # trial index
                        policy_map[policy], # policy index
                        :, # time index
                        dim_dict['action'], # tuple dimension
                        : # benef index
                    ] = np.copy(action_matrix.T)
    offline_traj[0, # trial index
                        policy_map[policy], # policy index
                        :, # time index
                        dim_dict['next_state'], # tuple dimension
                        : # benef index
                    ] = np.copy(state_matrix.T[1:, :])
    offline_traj[0, # trial index
                        policy_map[policy], # policy index
                        :, # time index
                        dim_dict['reward'], # tuple dimension
                        : # benef index
                    ] = np.copy(reward_matrix.T)
    feature_df = pd.DataFrame(pilot_static_features)
    feature_df['user_id'] = pilot_user_ids
    feature_df = feature_df.set_index('user_id')
    pol_feature_df = feature_df.loc[pol_analysis_df['user_id']]

    # pol_bool = pd.Series(pilot_user_ids).isin(pol_analysis_df.user_id)
    pol_features = pol_feature_df.values

    state_matrix = state_matrix[:, :-1].T.reshape(1, 1, T, n_benefs).repeat(len(policy_map), axis = 1)
    action_matrix = action_matrix.T.reshape(1, 1, T, n_benefs).repeat(len(policy_map), axis = 1)
    reward_matrix = reward_matrix.T.reshape(1, 1, T, n_benefs).repeat(len(policy_map), axis = 1)

    return pol_features, offline_traj, state_matrix, action_matrix, reward_matrix

def get_offline_dataset(policy, T, seed=0):
    features, offline_traj, state_record, action_record, reward_record = get_offline_traj(policy, T)
    print('got offline traj')
    OPE_sim_n_trials = 100
    all_n_benefs = 7668
    n_instance = 12
    n_benefs = int(all_n_benefs/n_instance)
    n_benefs = 638
    L = T
    n_states = 2
    gamma = 0.99
    env = 'general'
    H = T
    raw_T_data = None
    raw_R_data = np.array([[0, 1]*all_n_benefs]).reshape(all_n_benefs, 2)
    R_data_ope_sim = np.array([[0, 1]*n_benefs]).reshape(n_benefs, 2)
    # ope_simulator = opeSimulator(offline_traj, n_benefs, L, n_states, OPE_sim_n_trials, gamma, beh_policy_name='random', T_data=raw_T_data, R_data=R_data_ope_sim, env=env, H=H, use_informed_prior=True)
    simulated_rewards = None

    # Setting random seed
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    # Boolean array representing 0 or 1 if a beneficiary was every intervened in T timesteps
    ever_intervened = (action_record[0, 0, :, :].sum(axis=0)>0).astype(bool)
    benef_idx_interv = np.arange(all_n_benefs)[ever_intervened]
    benef_idx_not_interv = np.arange(all_n_benefs)[~ever_intervened]

    np.random.shuffle(benef_idx_interv)
    np.random.shuffle(benef_idx_not_interv)

    dataset = []
    # break trajectories by beneficiries into n_instance
    for idx in range(n_instance):
        n_benefs_interv = int(len(benef_idx_interv)/n_instance)
        n_benefs_not_interv = int(len(benef_idx_not_interv)/n_instance)
        interv_idx = list(benef_idx_interv[idx*n_benefs_interv:(idx+1)*n_benefs_interv])
        not_interv_idx = list(benef_idx_not_interv[idx*n_benefs_not_interv:(idx+1)*n_benefs_not_interv])
        all_idx = interv_idx+not_interv_idx
        ope_simulator_subset = opeSimulator(offline_traj[:, :, :, :, all_idx], n_benefs, L, n_states, OPE_sim_n_trials, gamma, beh_policy_name='random', T_data=raw_T_data, R_data=R_data_ope_sim, env=env, H=H, use_informed_prior=True)

        instance = (features[all_idx], raw_T_data,
                    raw_R_data[all_idx],
                    offline_traj[:, :, :, :, all_idx], ope_simulator_subset, simulated_rewards,
                    state_record[:, :, :, all_idx],
                    action_record[:, :, :, all_idx],
                    reward_record[:, :, :, all_idx])
        dataset.append(instance)
    return dataset

def get_engagements(traj):
    engagement_matrix = traj[0, # trial index
                        :, # policy index
                        :, # time index
                        dim_dict['reward'], # tuple dimension
                        : # benef index
                    ]
    T = engagement_matrix.shape[2]

    for week in range(T-1):
        print(f'week {week+1}')

        for pol in policy_names.keys():
            r_n_0 = engagement_matrix[pol, 0, :].sum()
            cumm_rew = engagement_matrix[pol, :week+1, :].sum()
            inst_rew = engagement_matrix[pol, week, :].sum()

            cumm_engage_drop = cumm_rew - (week+1)*r_n_0
            inst_engage_drop = inst_rew - r_n_0


            print(policy_names[pol])
            print(f'\t r_0: {r_n_0}, inst_rew: {inst_rew}, cumm_engage_drop: {cumm_engage_drop}' )
