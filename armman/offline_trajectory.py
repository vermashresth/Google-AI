import numpy as np
import pandas as pd
import pickle
from dfl.config import policy_names, policy_map, dim_dict

def get_offline_traj(policy, T):
    # load dataframes from offline data
    interv_df = pd.read_csv('offline_data/interventions.csv')
    analysis_df = pd.read_csv('offline_data/state-cluster-whittle-E_C.csv')

    # if policy option is rr or random, we will fetch real world round robin trajectories
    policy_equiv_dict = {'rr':'round_robin', 'random':'round_robin'}

    # filter only entries for given policy
    pol_analysis_df = analysis_df[analysis_df['arm']==policy_equiv_dict[policy]]
    # note all the state columns
    state_cols = [f'week{i}_state' for i in range(T)]
    # state 0 : sleeping engaging, 1: sleeping non-engaging, 6: engaging, 7: sleeping non engaging
    # to convert this into 0: non-engaging, 1: engaging
    state_df = np.logical_not(pol_analysis_df[state_cols]%2==0).astype(int)
    state_matrix = state_df.values
    assert state_matrix.shape[1] == T

    # Reward is same as states, but exists only for T-1 steps
    reward_matrix = state_matrix[:, :T-1]

    # filter intervention df for given policy
    pol_interv_df = interv_df[interv_df['exp_group']==policy_equiv_dict[policy]]
    # note columns for every week
    week_cols = [f'week{i+1}' for i in range(T-1)]
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
    assert action_matrix.shape[1] == T-1


    n_benefs = len(action_matrix)
    offline_traj = np.zeros((1, len(policy_names), T-1, len(dim_dict), n_benefs))

    offline_traj[0, # trial index
                        policy_map[policy], # policy index
                        :, # time index
                        dim_dict['state'], # tuple dimension
                        : # benef index
                    ] = np.copy(state_matrix.T[:1, :])
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
                    ] = np.copy(state_matrix.T[:1, :])
    
    return offline_traj, state_matrix, action_matrix, reward_matrix

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
