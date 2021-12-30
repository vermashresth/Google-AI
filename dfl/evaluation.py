import numpy as np
import pandas as pd
import tqdm

from dfl.ope import opeIS, opeISNaive
from dfl.trajectory import getSimulatedTrajectories
from dfl.trajectory import getEmpProbBenefLookup, getEmpProbClusterLookup, augmentTraj
from dfl.utils import aux_dict_to_transition_matrix
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
    
    n_sim_epochs = 3
    mask_seed = 0
    
    
    aug_traj_schedule = [10]
    n_aug_traj = max(aug_traj_schedule)
    
    p_pruned_schedules = [0.3, 0.6]
    # p_pruned_schedules = []
    OPE_sim_n_trials = 100

    result_df = pd.DataFrame()
    
    result_dict = {}

    for sim_seed in tqdm.tqdm(range(n_sim_epochs), desc='Seeded Run'):
        print('Seed ', sim_seed)
        traj, sim_whittle, simulated_rewards, mask, \
            state_record, action_record, reward_record = getSimulatedTrajectories(
                                                            n_benefs, T, K, n_trials, gamma,
                                                            sim_seed, mask_seed, T_data, w
                                                            )
        
        
        opeIS_naive = opeISNaive(traj, w, mask, n_benefs, T, K, n_trials, gamma,
                    target_policy_name, beh_policy_name)

        opeIS_decomposed = opeIS(traj, w, mask, n_benefs, T, K, n_trials, gamma,
                    target_policy_name, beh_policy_name)
        
        emp_prob_by_benef, tr_df_benef, _ = getEmpProbBenefLookup(traj, policy_id, trial_id, n_benefs, False)
        emp_prob_by_benef_ssa, tr_df_benef_ssa, aux_dict_ssa = getEmpProbBenefLookup(traj, policy_id, trial_id, n_benefs, True)


        # masked_cluster_ids = np.array(cluster_ids)[mask]
        # emp_prob_by_cluster, tr_df_cluster = getEmpProbClusterLookup(traj, policy_id, trial_id, masked_cluster_ids, False)


        benef_level_aug_traj = augmentTraj(traj, tr_df_benef, policy_id, trial_id,
                                          emp_prob_by_benef, False, n_aug_traj,
                                          T, n_benefs, cluster_ids=None)
        benef_level_aug_traj_ssa = augmentTraj(traj, tr_df_benef_ssa, policy_id, trial_id,
                                          emp_prob_by_benef_ssa, False, n_aug_traj,
                                          T, n_benefs, cluster_ids=None)
        benef_level_aug_traj_pruned_list = []
        benef_level_aug_traj_ssa_pruned_list = []

        for p in p_pruned_schedules:
            benef_level_aug_traj_pruned = augmentTraj(traj, tr_df_benef, policy_id, trial_id,
                                            emp_prob_by_benef, False, n_aug_traj,
                                            T, n_benefs, cluster_ids=None, do_prune=p)
            benef_level_aug_traj_pruned_list.append(benef_level_aug_traj_pruned)
            
            benef_level_aug_traj_ssa_pruned = augmentTraj(traj, tr_df_benef_ssa, policy_id, trial_id,
                                            emp_prob_by_benef_ssa, False, n_aug_traj,
                                            T, n_benefs, cluster_ids=None, do_prune=p)
            benef_level_aug_traj_ssa_pruned_list.append(benef_level_aug_traj_ssa_pruned)

        # cluster_level_aug_traj = augmentTraj(traj, policy_id, trial_id,
        #                           emp_prob_by_cluster, True, n_aug_traj,
        #                           T, n_benefs, masked_cluster_ids)

        result_dict['seed'] = sim_seed
        result_dict['TRUE_sim_random'] = simulated_rewards[trial_id][policy_map['random']]
        result_dict['TRUE_sim_whittle'] = simulated_rewards[trial_id][policy_map['whittle']]
        result_dict['OPE_IS_naive'] = opeIS_naive
        result_dict['OPE_IS_decomposed'] = opeIS_decomposed

        for traj_sample_size in tqdm.tqdm(aug_traj_schedule):
            traj_idx_sample = np.random.choice(len(benef_level_aug_traj),
                                               traj_sample_size, replace=False)

            aug_traj_subset = benef_level_aug_traj[traj_idx_sample]
            aug_traj_subset_ssa = benef_level_aug_traj_ssa[traj_idx_sample]
            
            opeIS_stitched_i = opeIS(aug_traj_subset, w, mask, n_benefs, T, K,
                                     traj_sample_size, gamma, target_policy_name, beh_policy_name)
            opeIS_stitched_i_ssa = opeIS(aug_traj_subset_ssa, w, mask, n_benefs, T, K,
                                        traj_sample_size, gamma, target_policy_name, beh_policy_name)

            

            result_dict[f'OPE_IS_stitched-{traj_sample_size}'] = opeIS_stitched_i
            result_dict[f'OPE_IS_stitched-ssa-{traj_sample_size}'] = opeIS_stitched_i_ssa

            for p_idx, p in enumerate(p_pruned_schedules):
                aug_traj_subset_pruned = benef_level_aug_traj_pruned_list[p_idx][traj_idx_sample]
                aug_traj_subset_ssa_pruned = benef_level_aug_traj_ssa_pruned_list[p_idx][traj_idx_sample]

                opeIS_stitched_i_pruned = opeIS(aug_traj_subset_pruned, w, mask, n_benefs, T, K,
                                     traj_sample_size, gamma, target_policy_name, beh_policy_name)
                opeIS_stitched_i_ssa_pruned = opeIS(aug_traj_subset_ssa_pruned, w, mask, n_benefs, T, K,
                                     traj_sample_size, gamma, target_policy_name, beh_policy_name)

                result_dict[f'OPE_IS_stitched-pruned-p-{p}-{traj_sample_size}'] = opeIS_stitched_i_pruned
                result_dict[f'OPE_IS_stitched-ssa-pruned-p-{p}-{traj_sample_size}'] = opeIS_stitched_i_ssa_pruned
            
            # aug_traj_subset = cluster_level_aug_traj[traj_idx_sample]
            # opeIS_similarity_i = opeIS(aug_traj_subset, w, mask, n_benefs, T, K,
            #                          traj_sample_size, gamma, target_policy_name, beh_policy_name)
            # result_dict[f'similarity-{traj_sample_size}'] = opeIS_similarity_i
        
        est_T_data = aux_dict_to_transition_matrix(aux_dict_ssa, n_benefs)
        _, OPE_sim_whittle, _, _, _, _, _ = getSimulatedTrajectories(
                                                    n_benefs, T, K, OPE_sim_n_trials, gamma,
                                                    sim_seed, mask_seed, est_T_data, w
                                                    )
        result_dict['OPE_sim_whittle'] = OPE_sim_whittle
        result_df = result_df.append(result_dict, ignore_index=True)
        result_df.to_csv('outputs/dfl/out.csv')
    return result_df
        
        
        
