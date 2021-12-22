import numpy as np
from dfl.config import policy_names, policy_map, dim_dict

def get_offline_traj(n_benefs, T, K, n_trials):
    interv_df = pd.read_csv('outputs/interventions.csv')
    analysis_df = pd.read_csv('outputs/state-cluster-whittle-E_C.csv')



    rr_df = analysis_df[analysis_df['arm']=='round_robin']
    state_cols = [f'week{i}_state' for i in range(T)]
    state_df = (rr_df[state_cols]%2==0).astype(int)
    state_matrix = state_df.values



    rr_interv_df = interv_df[interv_df['exp_group']=='round_robin']

    week_cols = [f'week{i+1}' for i in range(T-1)]

    rr_interv_df = (rr_interv_df.pivot(index='user_id',
                       columns='intervene_week',
                       values='intervene_week')[week_cols].isna()!=1).astype(int).reset_index()

    actions_df = pd.merge(rr_df[['user_id']],
             rr_interv_df[['user_id']+week_cols],
             how='left').fillna(0)
    action_matrix = actions_df[week_cols].values
    action_matrix.shape


    offline_traj = np.zeros((n_trials, len(policy_names), T-1, len(dim_dict), n_benefs))
    offline_traj.shape

    offline_traj[0, # trial index
                        policy_map['rr'], # policy index
                        :, # time index
                        dim_dict['state'], # tuple dimension
                        : # benef index
                    ] = np.copy(state_matrix.T[:1, :])
    offline_traj[0, # trial index
                        policy_map['rr'], # policy index
                        :, # time index
                        dim_dict['action'], # tuple dimension
                        : # benef index
                    ] = np.copy(action_matrix.T)
    offline_traj[0, # trial index
                        policy_map['rr'], # policy index
                        :, # time index
                        dim_dict['next_state'], # tuple dimension
                        : # benef index
                    ] = np.copy(state_matrix.T[1:, :])
    offline_traj[0, # trial index
                        policy_map['rr'], # policy index
                        :, # time index
                        dim_dict['reward'], # tuple dimension
                        : # benef index
                    ] = np.copy(state_matrix.T[:1, :])
    return offline_traj

def get_engagements():
    engagement_matrix = traj[0, # trial index
                        :, # policy index
                        :, # time index
                        dim_dict['reward'], # tuple dimension
                        : # benef index
                    ]

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
