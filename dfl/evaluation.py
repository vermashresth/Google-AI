import numpy as np
import pandas as pd
import tqdm

from dfl.ope import opeIS, opeISNaive
from dfl.trajectory import getSimulatedTrajectories
from dfl.trajectory import getEmpProbBenefLookup, getEmpProbClusterLookup, augmentTraj
from dfl.config import policy_map

def evaluate_by_sample(aug_traj, n_samples, w, mask, n_benefs, T, K, traj_sample_size, gamma,
                    target_policy_name, beh_policy_name):
    out = []
    print(f'Evaluation on {traj_sample_size} trajectories')
    for _ in range(n_samples):
        traj_idx_sample = np.random.choice(len(aug_traj), traj_sample_size, replace=False)

        aug_traj_subset = aug_traj[traj_idx_sample]

        ope_whittle = opeIS(aug_traj_subset, w, mask, n_benefs, T, K, traj_sample_size, gamma,
                    target_policy_name, beh_policy_name)
        out.append(ope_whittle)
    print('\tFinal OPE: ', np.mean(out))
    print('\tstd-dev', np.std(out))
    return out

def run_all_synthetic(T_data, w, cluster_ids):
    
    n_benefs = 30
    T = 20
    K = 8
    n_trials = 1
    gamma = 1
    
    target_policy_name = 'whittle'
    beh_policy_name = 'random'
    policy_id = policy_map[beh_policy_name]
    trial_id = 0
    
    n_sim_epochs = 5
    mask_seed = 0
    
    
    aug_traj_schedule = [10, 100, 1000]
    n_aug_traj = max(aug_traj_schedule)
    
    result_df = pd.DataFrame(columns=['seed', 'sim_random', 'sim_whittle', 'naive', 'decomposed']+\
                                      [f'stitched-{i}' for i in aug_traj_schedule]+\
                                      [f'similarity-{i}' for i in aug_traj_schedule])
    
    result_dict = {}

    for sim_seed in tqdm.tqdm(range(n_sim_epochs), desc='Seeded Run'):
        print('Seed ', sim_seed)
        traj, sim_whittle, simulated_rewards, mask, \
                        state_record, action_record = getSimulatedTrajectories(
                                                        n_benefs, T, K, n_trials, gamma,
                                                        sim_seed, mask_seed, T_data, w
                                                        )
        
        

        
        emp_prob_by_benef, tr_df_benef = getEmpProbBenefLookup(traj, policy_id, trial_id, n_benefs, False)

        # masked_cluster_ids = np.array(cluster_ids)[mask]
        # emp_prob_by_cluster, tr_df_cluster = getEmpProbClusterLookup(traj, policy_id, trial_id, masked_cluster_ids, False)


        benef_level_aug_traj = augmentTraj(traj, tr_df_benef, policy_id, trial_id,
                                          emp_prob_by_benef, False, n_aug_traj,
                                          T, n_benefs, cluster_ids=None)
        # cluster_level_aug_traj = augmentTraj(traj, policy_id, trial_id,
        #                           emp_prob_by_cluster, True, n_aug_traj,
        #                           T, n_benefs, masked_cluster_ids)
        opeIS_naive = opeISNaive(traj, w, mask, n_benefs, T, K, n_trials, gamma,
                    target_policy_name, beh_policy_name)

        opeIS_decomposed = opeIS(traj, w, mask, n_benefs, T, K, n_trials, gamma,
                    target_policy_name, beh_policy_name)

        for traj_sample_size in tqdm.tqdm(aug_traj_schedule):
            traj_idx_sample = np.random.choice(len(benef_level_aug_traj),
                                               traj_sample_size, replace=False)

            aug_traj_subset = benef_level_aug_traj[traj_idx_sample]
            
            opeIS_stitched_i = opeIS(aug_traj_subset, w, mask, n_benefs, T, K,
                                     traj_sample_size, gamma, target_policy_name, beh_policy_name)
            
            # aug_traj_subset = cluster_level_aug_traj[traj_idx_sample]
            # opeIS_similarity_i = opeIS(aug_traj_subset, w, mask, n_benefs, T, K,
            #                          traj_sample_size, gamma, target_policy_name, beh_policy_name)

            result_dict[f'stitched-{traj_sample_size}'] = opeIS_stitched_i
            # result_dict[f'similarity-{traj_sample_size}'] = opeIS_similarity_i
            
        result_dict['seed'] = sim_seed
        result_dict['sim_random'] = simulated_rewards[trial_id][policy_map['random']]
        result_dict['sim_whittle'] = simulated_rewards[trial_id][policy_map['whittle']]
        result_dict['naive'] = opeIS_naive
        result_dict['decomposed'] = opeIS_decomposed
        
        result_df = result_df.append(result_dict, ignore_index=True)
        result_df.to_csv('outputs/dfl/out.csv')
    return result_df
        
        
        
