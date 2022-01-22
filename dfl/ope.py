import tensorflow as tf
import numpy as np
import tqdm

from dfl.policy import getActionProb, getActionProbNaive, getProbs
from dfl.config import dim_dict, policy_map
from dfl.trajectory import getSimulatedTrajectories
from dfl.trajectory import getEmpProbBenefLookup, getEmpProbClusterLookup, augmentTraj
from dfl.trajectory import getEmpTransitionMatrix
from dfl.utils import aux_dict_to_transition_matrix

def opeIS(traj, w, n_benefs, T, K, n_trials, gamma, target_policy_name, beh_policy_name):
    compare = {'target':policy_map[target_policy_name], 'beh':policy_map[beh_policy_name]}
    gamma_series = np.array([gamma**t for t in range(T-1)]) # Kai edited: it was **t-1** instead of **t**.

    beh_probs    = np.zeros((n_trials, T, n_benefs))
    target_probs = np.zeros((n_trials, T, n_benefs))

    v = []
    # w_mask = tf.gather(w, mask) if tf.is_tensor(w) else w[mask] # Added the tensorflow version to support tensorflow indexing
    for benef in tqdm.tqdm(range(n_benefs), desc='OPE'):
        v_i = 0
        for trial in range(n_trials):
            imp_weight = 1
            v_i_tau = 0
            for ts in range(T-1):
                a_i_t = traj[trial, # trial index
                                0, # policy index
                                ts, # time index
                                dim_dict['action'], # tuple dimension
                                benef # benef index
                                ].astype(int)

                s_t = traj[trial, # trial index
                                0, # policy index
                                ts, # time index
                                dim_dict['state'], # tuple dimension
                                : # benef index
                                ].astype(int)
                pi_tar = getActionProb(s_t, a_i_t,
                                           policy=compare['target'],
                                           benef=benef, ts=ts,
                                           w=w, k=K, N=n_benefs)
                pi_beh = getActionProb(s_t, a_i_t,
                                           policy=compare['beh'],
                                           benef=benef, ts=ts,
                                           w=w, k=K, N=n_benefs)
                imp_weight*= pi_tar/pi_beh
                # if imp_weight>1:
                #     print('weight: ', imp_weight)
                v_i_t_tau = gamma_series[ts] * traj[trial, # trial index
                                                0, # policy index
                                                ts, # time index
                                                dim_dict['reward'], # tuple dimension
                                                benef # benef index
                                                ] * imp_weight
                v_i_tau += v_i_t_tau

                beh_probs[trial, ts, benef]    = pi_beh
                target_probs[trial, ts, benef] = pi_tar

            v_i += v_i_tau
        v.append(v_i/n_trials)
    ope = np.sum(v)
    # print(f'OPE: {ope}')
    return ope

#This is the parallelized implementation of the same OPE. Ideally these two should match but the parallelized version is faster.
def opeIS_parallel(state_record, action_record, reward_record, w, n_benefs, T, K, n_trials, gamma, target_policy_name, beh_policy_name):
    compare = {'target':policy_map[target_policy_name], 'beh':policy_map[beh_policy_name]}
    gamma_series = np.array([gamma**t for t in range(T-1)])

    ntr, _, L, N = state_record.shape

    v = []
    # w_mask = tf.gather(w, mask) if tf.is_tensor(w) else w[mask] # Added the tensorflow version to support tensorflow indexing

    # state_record_beh = np.concatenate([np.tile(np.arange(N), (ntr, L, 1)).reshape(ntr, L, N, 1), state_record[:,compare['beh'],:,:].reshape(ntr, L, N, 1)], axis=-1).astype(int)
    action_record_beh = action_record[:,0,:,:]

    # Get the corresponding Whittle indices
    # whittle_indices = tf.gather_nd(w, state_record_beh)

    # Batch topk to get probabilities
    beh_probs_raw    = tf.reshape(getProbs(state_record[:,0,:,:].reshape(-1, N), policy=compare['beh'], ts=None, w=w, k=K),    (ntr, L, N))
    target_probs_raw = tf.reshape(getProbs(state_record[:,0,:,:].reshape(-1, N), policy=compare['target'], ts=None, w=w, k=K), (ntr, L, N))

    # Use action to select the corresponding probabilities
    beh_probs    = beh_probs_raw * action_record_beh + (1 - beh_probs_raw) * (1 - action_record_beh)
    target_probs = target_probs_raw * action_record_beh + (1 - target_probs_raw) * (1 - action_record_beh)

    # Importance sampling weights
    IS_weights = target_probs / beh_probs # [ntr, L, N]

    # OPE
    total_probs = np.ones((ntr, N))
    ope = 0
    ess = 0
    for t in range(T-1):
        rewards = reward_record[:, 0, t, :] 
        total_probs = total_probs * IS_weights[:,t,:] # shape: [n_trials, n_benefs]
        IS_sum = tf.reduce_sum(total_probs) / n_benefs # shape: [1]
        IS_square_sum = tf.reduce_sum(total_probs**2) / n_benefs # shape: [1]
        ope += rewards * total_probs * gamma_series[t] / IS_sum
        ess += IS_sum ** 2 / IS_square_sum # shape: [1, n_benefs]

    ope = tf.reduce_sum(ope)
    return ope, ess

def opeISNaive(traj, w, n_benefs, T, K, n_trials, gamma, target_policy_name, beh_policy_name):
    compare = {'target':policy_map[target_policy_name], 'beh':policy_map[beh_policy_name]}
    gamma_series = np.array([gamma**(t-1) for t in range(T-1)])

    v = []
    for trial in range(n_trials):
        imp_weight = 1
        v_tau = 0
        for ts in range(T-1):
            a_t = traj[trial, # trial index
                            0, # policy index
                            ts, # time index
                            dim_dict['action'], # tuple dimension
                            : # benef index
                            ].astype(int)
            # a_t_encoded = encode_vector(a_t, N_ACTIONS)

            s_t = traj[trial, # trial index
                            0, # policy index
                            ts, # time index
                            dim_dict['state'], # tuple dimension
                            : # benef index
                            ].astype(int)
            # s_t_encoded = encode_vector(s_t, N_STATES)

            pi_tar = getActionProbNaive(s_t, a_t, policy=compare['target'],
                                        w=w, k=K, N=n_benefs)
            pi_beh = getActionProbNaive(s_t, a_t, policy=compare['beh'],
                                        w=w, k=K, N=n_benefs)

            imp_weight*= pi_tar/pi_beh
            # if imp_weight>1:
            #     print('weight: ', imp_weight)
            v_t_tau = gamma_series[ts] * traj[trial, # trial index
                                            0, # policy index
                                            ts, # time index
                                            dim_dict['reward'], # tuple dimension
                                            : # benef index
                                            ].sum() * imp_weight
            v_tau += v_t_tau
  
    v.append(v_tau)
    ope = np.mean(v)
    print(f'OPE Naive: {ope}')
    return ope

# Simulation-based OPE (differentiable and parallelizable)
class opeSimulator(object):
    def __init__(self, beh_traj, n_benefs, T, m, OPE_sim_n_trials, gamma, beh_policy_name, T_data, R_data, env='general', H=None, use_informed_prior=False):
        self.n_benefs = n_benefs
        self.T = T
        self.m = m
        self.H = H
        self.OPE_sim_n_trials = OPE_sim_n_trials
        self.gamma = gamma

        policy_id = policy_map[beh_policy_name]
        self.emp_T_data, self.emp_R_data = getEmpTransitionMatrix(traj=beh_traj, policy_id=policy_id, n_benefs=n_benefs, m=m, env=env, H=H, use_informed_prior=use_informed_prior)
        if env == 'general':
            # self.emp_T_data = T_data # Directly using the real T_data
            self.emp_R_data = R_data # Reward list is explicitly given in the MDP version

    def __call__(self, w, K):
        self.K = K
        compute = tf.custom_gradient(lambda x: self._compute(x))
        return compute(w)

    def _compute(self, w_raw):
        w = tf.stop_gradient(w_raw)
        # Fast soft Whittle simulation
        traj, simulated_rewards, state_record, action_record, reward_record = getSimulatedTrajectories(
                                                    n_benefs=self.n_benefs, T=self.T, K=self.K, n_trials=self.OPE_sim_n_trials, gamma=self.gamma,
                                                    T_data=self.emp_T_data, R_data=self.emp_R_data,
                                                    w=w.numpy(), policies=[3], fast=True
                                                    )
        
        average_reward = tf.reduce_mean(tf.convert_to_tensor(simulated_rewards, dtype=tf.float32))

        def gradient_function(dsoln):
            gamma_list = np.repeat(np.reshape(self.gamma ** np.arange(self.T), (1,1,self.T,1)), repeats=self.OPE_sim_n_trials, axis=0)
            discounted_reward_record = reward_record * gamma_list
            cumulative_rewards = tf.math.cumsum(tf.convert_to_tensor(discounted_reward_record, dtype=tf.float32), axis=2, reverse=True)
            with tf.GradientTape() as tmp_tape:
                tmp_tape.watch(w)
                probs_raw = tf.reshape(getProbs(state_record[:,0,:,:].reshape(-1, self.n_benefs), policy=3, ts=None, w=w, k=self.K), (self.OPE_sim_n_trials, self.T, self.n_benefs))
                selected_probs = probs_raw * action_record[:,0,:,:] + (1 - probs_raw) * (1 - action_record[:,0,:,:]) # [ntr, 1, self.T, n_benefs]

                # tf.reshape(getProbs(state_record[:,0,:,:].reshape(-1, self.n_benefs), policy=3, ts=None, w=w, k=self.K), (-1, 1, self.T, self.n_benefs))
                selected_logprobs = tf.math.log(selected_probs)
                
                total_reward = tf.reduce_mean(tf.reduce_sum(cumulative_rewards[:,0,:,:] * selected_logprobs, axis=(1,2)))

            dtotal_dw = tmp_tape.gradient(total_reward, w)

            return dtotal_dw * dsoln

        return tf.stop_gradient(average_reward), gradient_function

