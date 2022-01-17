import numpy as np
import pandas as pd
import argparse
import tqdm

from dfl.config import policy_names, dim_dict, S_VALS, A_VALS
from dfl.policy import getActions, getSoftActions
from dfl.utils import getBenefsByCluster
from dfl.environments import generalEnv

# from armman.simulator import takeActions

from collections import defaultdict

def simulateTrajectories(args, env, k, w, gamma, start_state=None, policies=[0]):

    ##### Unpack arguments
    L=args['simulation_length']
    N=args['num_beneficiaries']
    ntr=args['num_trials']

    n_policies = len(policies)
    state_record=np.zeros((ntr, n_policies, L, N))  # Store the full state
                                                    # trajectory
    action_record=np.zeros((ntr, n_policies, L, N))  # Store the full action
                                                    # trajectory
    reward_record=np.zeros((ntr, n_policies, L, N))  # Store the full reward
                                                    # trajectory

    simulated_rewards=np.zeros((ntr, n_policies)) # Store aggregate rewards


    traj = np.zeros((ntr, n_policies, L, len(dim_dict), N))
    ##### Iterate over number of independent trials to average over
    for tr in range(ntr):

        ## For current trial, evaluate all policies
        for pol_idx, pol in enumerate(policies):

            # Initialize State
            if start_state is None:
                states=env.getStartState()
            else:
                # Explicitly given start_state, for debugging purposes
                assert start_state.shape == (N,)
                states = np.copy(start_state)
            
            ## Iterate over timesteps. Note that if simulation length is L, 
            ## there are L-1 action decisions to take.
            for timestep in range(L-1):

                ## Note Current State
                state_record[tr, pol_idx, timestep, :] = np.copy(states)
                traj[tr, pol_idx, timestep, dim_dict['state'], :] = np.copy(states)

                ## Get Actions
                actions = getActions(states=states, policy=pol, ts=timestep, w=w, k=k)
                action_record[tr, pol_idx, timestep, :] = np.copy(actions)
                traj[tr, pol_idx, timestep, dim_dict['action'], :] = np.copy(actions)

                ## Get rewards
                rewards = env.getRewards(states, actions)
                reward_record[tr, pol_idx, timestep, :] = np.copy(rewards)
                
                ## Transition to next state and get rewards
                next_states = env.takeActions(states, actions)
                traj[tr, pol_idx, timestep, dim_dict['next_state'], :] = np.copy(next_states)

                # rewards = np.copy(states)
                traj[tr, pol_idx, timestep, dim_dict['reward'], :] = np.copy(rewards)
                
                states = next_states

            # Note Rewards
            gamma_list = gamma ** np.arange(L)
            discounted_reward_record = reward_record * gamma_list.reshape(1,1,L,1)
            simulated_rewards[tr, pol_idx] = np.sum(discounted_reward_record[tr,pol_idx])

        
        ##### Print results
        # for pol in policies: 
        #     print("Expected reward of policy %s is %s"%(policy_names[pol], \
        #                         np.mean(simulated_rewards[:,policies.index(pol)])))
  
    return simulated_rewards, state_record, action_record, reward_record, traj

def fastSimulateTrajectories(args, env, k, w, gamma, start_state=None, policies=[3]):
    # Parallel implementation of simulation

    ##### Unpack arguments
    L=args['simulation_length']
    N=args['num_beneficiaries']
    ntr=args['num_trials']

    n_policies = len(policies)
    state_record=np.zeros((ntr, n_policies, L, N))  # Store the full state
                                                    # trajectory
    action_record=np.zeros((ntr, n_policies, L, N))  # Store the full action
                                                    # trajectory
    reward_record=np.zeros((ntr, n_policies, L, N))  # Store the full reward
                                                    # trajectory

    simulated_rewards=np.zeros((ntr, n_policies)) # Store aggregate rewards

    traj = np.zeros((ntr, n_policies, L, len(dim_dict), N))
    ##### Iterate over number of independent trials to average over
    ## For current trial, evaluate all policies
    for pol_idx, pol in enumerate(policies):

        assert(pol == 3) # This only works for soft whittle

        # Initialize State
        if start_state is None:
            states = np.concatenate([env.getStartState().reshape(1,N) for _ in range(ntr)], axis=0)
        else:
            # Explicitly given start_state, for debugging purposes
            assert start_state.shape == (ntr, N,)
            states = np.copy(start_state)
        
        ## Iterate over timesteps. Note that if simulation length is L, 
        ## there are L-1 action decisions to take.
        for timestep in range(L-1):

            ## Note Current State
            state_record[:, pol_idx, timestep, :] = np.copy(states)
            traj[:, pol_idx, timestep, dim_dict['state'], :] = np.copy(states)

            ## Get Actions
            actions = getSoftActions(states=states, policy=pol, ts=timestep, w=w, k=k)
            action_record[:, pol_idx, timestep, :] = np.copy(actions)
            traj[:, pol_idx, timestep, dim_dict['action'], :] = np.copy(actions)

            # Non-parallelizable part
            next_states_list = []
            for tr, tmp_states, tmp_actions in zip(range(ntr), states, actions):
                ## Get rewards
                tmp_rewards = env.getRewards(tmp_states, tmp_actions)
                reward_record[tr, pol_idx, timestep, :] = np.copy(tmp_rewards)
            
                ## Transition to next state and get rewards
                tmp_next_states = env.takeActions(tmp_states, tmp_actions)
                traj[tr, pol_idx, timestep, dim_dict['next_state'], :] = np.copy(tmp_next_states)

                # rewards = np.copy(states)
                traj[tr, pol_idx, timestep, dim_dict['reward'], :] = np.copy(tmp_rewards)
                
                next_states_list.append(tmp_next_states.reshape(1,-1))

            next_states = np.concatenate(next_states_list, axis=0)
            states = next_states

        # Note Rewards
        gamma_list = np.repeat(np.reshape(gamma ** np.arange(L), (1,1,L,1)), repeats=ntr, axis=0)
        discounted_reward_record = reward_record * gamma_list
        simulated_rewards[:, pol_idx] = np.sum(discounted_reward_record[:,pol_idx], axis=(1,2))
    
    return simulated_rewards, state_record, action_record, reward_record, traj

def getSimulatedTrajectories(n_benefs = 10, T = 5, K = 3, n_trials = 10, gamma = 1,
                             T_data=None, R_data=None, w=None, start_state=None, H=10, debug=False, replace=False,
                             policies=[0], fast=False):

    # Set args params
    args = {}
    args['num_beneficiaries']=n_benefs
    args['num_resources']=K
    args['simulation_length']=T
    args['num_trials']=n_trials

    # If transitions matrix is for larger number of benefs than `n_benefs`, generate a mask
    # Updated: I prefer to sample the mask outside of the simulation.
    # So the size of the transition matric needs to match `n_benefs`.
    assert(n_benefs == len(T_data) == len(R_data))

    # Define Simulation environment
    # Generate environment
    env = generalEnv(N=n_benefs,
                    T_data=T_data, R_data=R_data)
    
    assert(T_data.shape[2] == 2) # 2 actions
    assert(T_data.shape[1] == T_data.shape[3] == env.n_states) # n_states

    # Run simulation
    if fast: # This only supports policy_id = 3
        simulated_rewards, state_record, action_record, reward_record, traj = fastSimulateTrajectories(args=args, env=env, k=K, w=w, gamma=gamma, start_state=start_state, policies=policies)
    else:
        simulated_rewards, state_record, action_record, reward_record, traj = simulateTrajectories(args=args, env=env, k=K, w=w, gamma=gamma, start_state=start_state, policies=policies)

    if debug:
        print('trajectory shape: ', np.array(traj).shape) 
    # whittle_rew = simulated_rewards[:, 2].mean()
    return traj, simulated_rewards, state_record, action_record, reward_record



    
def getBenefsFrequency(traj, benef_ids, policy_id, trial_id):
    benef_ci_traj = traj[trial_id, # trial index
            policy_id, # policy index
            :, # time index
            :, # tuple dimension
            benef_ids # benef index
        ]
    s_traj_c =  benef_ci_traj[:, :-1, dim_dict['state']]
    a_traj_c =  benef_ci_traj[:, :-1, dim_dict['action']]
    s_prime_traj_c =  benef_ci_traj[:, :-1, dim_dict['next_state']]
    a_prime_traj_c = benef_ci_traj[:, 1:, dim_dict['action']]

    transitions_df = pd.DataFrame(columns = ['s', 's_prime', 'a', 'a_prime'])

    for s_traj, a_traj, s_prime_traj, a_prime_traj in \
                        zip(s_traj_c, a_traj_c, s_prime_traj_c, a_prime_traj_c):
        transitions_df = transitions_df.append(pd.DataFrame({'s':s_traj,
                                's_prime': s_prime_traj,
                                'a': a_traj,
                                'a_prime': a_prime_traj}), ignore_index=True)

    return transitions_df

def getBenefsFullFrequency(traj, benef_ids, policy_id):
    benef_ci_traj = traj[:, # trial index
            policy_id, # policy index
            :, # time index
            :, # tuple dimension
            benef_ids # benef index
        ]
    benef_ci_traj = benef_ci_traj.reshape(-1, benef_ci_traj.shape[2], benef_ci_traj.shape[3])
    s_traj_c =  benef_ci_traj[:, :-1, dim_dict['state']]
    a_traj_c =  benef_ci_traj[:, :-1, dim_dict['action']]
    r_traj_c =  benef_ci_traj[:, :-1, dim_dict['reward']]
    s_prime_traj_c =  benef_ci_traj[:, :-1, dim_dict['next_state']]
    a_prime_traj_c = benef_ci_traj[:, 1:, dim_dict['action']]

    transitions_df = pd.DataFrame(columns = ['s', 's_prime', 'r', 'a', 'a_prime'])

    for s_traj, a_traj, r_traj, s_prime_traj, a_prime_traj in \
                        zip(s_traj_c, a_traj_c, r_traj_c, s_prime_traj_c, a_prime_traj_c):
        transitions_df = transitions_df.append(pd.DataFrame({'s':s_traj,
                                's_prime': s_prime_traj,
                                'r': r_traj,
                                'a': a_traj,
                                'a_prime': a_prime_traj}), ignore_index=True)

    return transitions_df
    
def getBenefsEmpProbs(traj, benef_ids, policy_id, trial_id, min_sup = 1, decomposed=False):
    transition_df = getBenefsFrequency(traj, benef_ids, policy_id, trial_id)
    
    if decomposed:
        emp_prob, aux_dict = getEmpProbsDecomposed(transitions_df, min_sup)
    else:
        emp_prob, aux_dict = getEmpProbs(transitions_df, min_sup)
                        
    return transitions_df, emp_prob, aux_dict


def getEmpProbs(transitions_df, min_sup = 1):
    emp_prob = {}
    has_missing_values = False
    # Prune Logic
    transitions_df = transitions_df.copy()
    # print(transitions_df)
    while True:
        last_row = transitions_df.iloc[-1]
        last_s_prime_a_prime = last_row[['s_prime', 'a_prime']]
        # print(last_s_prime_a_prime.values)
        # print(transitions_df[['s', 'a']].values)
        match_bool = transitions_df[['s', 'a']].values==last_s_prime_a_prime.values
        # print(transitions_df)
        if 2 in np.sum(match_bool, axis=1): # There is some s, a matching last s_prime, a_prime
            break
        else:
            # print(match_bool)
            # print(transitions_df[['s', 'a']].values)
            # print(last_s_prime_a_prime.values)
            # print(transitions_df[['s', 'a', 's_prime', 'a_prime']])
            transitions_df = transitions_df.iloc[:-1]
            print('Pruned Transitions to length ', transitions_df.shape)

    for s in S_VALS:
        for a in A_VALS:
            s_a = transitions_df[(transitions_df['s']==s) &
                                    (transitions_df['a']==a)
                                ]
            s_a_count = s_a.shape[0]
            key = (s, a)
            emp_prob[key] = {}
            for s_prime in S_VALS:
                for a_prime in A_VALS:
                    s_a_s_prime_a_prime = s_a[(s_a['s_prime']==s_prime) &
                                                    (s_a['a_prime']==a_prime)
                                                ]
                    s_a_s_prime_a_prime_count = s_a_s_prime_a_prime.shape[0]
                    options_key = (s_prime, a_prime)
                    if s_a_count >= min_sup:
                        emp_prob[key][options_key] = s_a_s_prime_a_prime_count/s_a_count
                    else:
                        emp_prob[key][options_key] = None
                        has_missing_values = True
                        
    return emp_prob, {'s_a_s_prime_dict': {}, 'has_missing_values': has_missing_values}

def getEmpProbsDecomposed(transitions_df, min_sup = 1):
    emp_prob = {}
    has_missing_values = False

    transitions_df = transitions_df.copy()
    
    while True:
        last_row = transitions_df.iloc[-1]
        last_s_prime_a_prime = last_row[['s_prime', 'a_prime']]
        # print(last_s_prime_a_prime.values)
        # print(transitions_df[['s', 'a']].values)
        match_bool = transitions_df[['s', 'a']].values==last_s_prime_a_prime.values
        # print(transitions_df)
        if 2 in np.sum(match_bool, axis=1): # There is some s, a matching last s_prime, a_prime
            break
        else:
            # print(match_bool)
            # print(transitions_df[['s', 'a']].values)
            # print(last_s_prime_a_prime.values)
            # print(transitions_df[['s', 'a', 's_prime', 'a_prime']])
            transitions_df = transitions_df.iloc[:-1]
            print('Pruned Transitions to length ', transitions_df.shape)

    for s in S_VALS:
        emp_prob[(s)] = {}
        for a in A_VALS:
            s_ = transitions_df[(transitions_df['s']==s)
                                ]
            s_count = s_.shape[0]
            s_a = transitions_df[(transitions_df['s']==s) &
                                    (transitions_df['a']==a)
                                ]
            s_a_count = s_a.shape[0]
            key = (s, a)
            emp_prob[key] = {}
            for s_prime in S_VALS:
                s_a_s_prime = s_a[(s_a['s_prime']==s_prime)
                                            ]
                s_a_s_prime_count = s_a_s_prime.shape[0]
                options_key = (s_prime)
                if s_a_count >= min_sup:
                    emp_prob[key][options_key] = s_a_s_prime_count/s_a_count
                else:
                    emp_prob[key][options_key] = None
                    has_missing_values = True
            if s_count >= min_sup:
                emp_prob[(s)][(a)] = s_a_count/s_count
            else:
                emp_prob[(s)][(a)] = None
    new_emp_prob = {}
    for s in S_VALS:
        for a in A_VALS:
            new_emp_prob[(s, a)] = {}
            for s_prime in S_VALS:
                for a_prime in A_VALS:
                    if  emp_prob[(s, a)][s_prime] is None or emp_prob[(s_prime)][a_prime] is None:
                        new_emp_prob[(s, a)][(s_prime, a_prime)] = 0
                    else:
                        new_emp_prob[(s, a)][(s_prime, a_prime)] = emp_prob[(s, a)][s_prime] *\
                                                           emp_prob[(s_prime)][a_prime]

                        
    return new_emp_prob, {'s_a_s_prime_dict': emp_prob , 'has_missing_values': has_missing_values}

def getEmpTransitionMatrix(traj, policy_id, n_benefs, m, env='general', H=None, use_informed_prior=False):
    # This function does not prune the transitions of (s', a') that do not appear in (s, a)
    # The output is the empirical T_data that can be directly used for simulation.
    # This is only used for simulation but not for trajectory stitching.

    n_actions = 2

    # Construct default transition probs
    if env == 'general':
        n_states = m
        emp_prob = np.zeros((n_states, n_actions, n_states))
        default_prob = np.ones((n_states, n_actions, n_states)) / n_states
        if use_informed_prior:
            # Do not just use uniform prior. Use informed prior using population level transitions
            all_transitions_df = getBenefsFullFrequency(traj, list(range(n_benefs)), policy_id)
            for s in range(n_states):
                for a in range(n_actions):
                    s_a = all_transitions_df[(all_transitions_df['s']==s) &
                                            (all_transitions_df['a']==a)
                                        ]
                    s_a_count = s_a.shape[0]
                    for s_prime in range(n_states):
                        s_a_s_prime = s_a[(s_a['s_prime']==s_prime)
                                                    ]
                        s_a_s_prime_count = s_a_s_prime.shape[0]
                        if s_a_count >= 1:
                            default_prob[s,a,s_prime] = s_a_s_prime_count / s_a_count
        
    elif env == 'POMDP':
        n_states = m * H
        assert(H is not None)
        emp_prob = np.zeros((n_states, n_actions, n_states))
        default_passive_prob = np.zeros((m, H, 1, m, H))
        default_active_prob  = np.zeros((m, H, 1, m, H))
        for h in range(H):
            if h < H-1:
                default_passive_prob[np.arange(m), h, 0, np.arange(m), h+1] = 1
            else:
                default_passive_prob[np.arange(m), h, 0, np.arange(m), h] = 1

        default_active_prob[:, :, 0, :, 0] = 1.0 / m # reset to one of the initial states
        default_passive_prob = default_passive_prob.reshape(m*H, 1, m*H)
        default_active_prob  = default_active_prob.reshape(m*H, 1, m*H)
        default_prob = np.concatenate([default_passive_prob, default_active_prob], axis=1)
    else:
        raise NotImplementedError

    # Filling in empirical transition probs
    transition_prob_list = []
    emp_R_data = np.zeros((n_benefs, n_states))

    for benef_id in range(n_benefs):
        transitions_df = getBenefsFullFrequency(traj, [benef_id], policy_id)
        transition_prob = default_prob.copy()

        for s in range(n_states):
            if len(transitions_df[transitions_df['s'] == s]) > 0:
                emp_R_data[benef_id,s] = np.mean(transitions_df[transitions_df['s'] == s]['r'])

            for a in range(n_actions):
                s_a = transitions_df[(transitions_df['s']==s) &
                                        (transitions_df['a']==a)
                                    ]
                s_a_count = s_a.shape[0]
                for s_prime in range(n_states):
                    s_a_s_prime = s_a[(s_a['s_prime']==s_prime)
                                                ]
                    s_a_s_prime_count = s_a_s_prime.shape[0]
                    if s_a_count >= 1:
                        transition_prob[s,a,s_prime] = s_a_s_prime_count / s_a_count
        transition_prob_list.append(transition_prob.reshape(1, n_states, 2, n_states))

    emp_T_data = np.concatenate(transition_prob_list, axis=0)

    return emp_T_data, emp_R_data


def getEmpProbClusterLookup(traj, policy_id, trial_id, cluster_ids, decomposed):
    emp_prob_by_cluster = {}
    transitions_list = []
    for cluster_id in np.unique(cluster_ids):
        benefs = getBenefsByCluster(cluster_id, cluster_ids)
        transitions_df, emp_prob, has_missing_values = getBenefsEmpProbs(traj, benefs, policy_id, trial_id, min_sup = 1, decomposed=decomposed)
        emp_prob_by_cluster[cluster_id] = emp_prob
        transitions_list.append(transitions_df)
    return emp_prob_by_cluster, transitions_list

def getEmpProbBenefLookup(traj, policy_id, trial_id, n_benefs, decomposed):
    emp_prob_by_benef = {}
    aux_dict_by_benef = {}
    transitions_list = []
    for benef in tqdm.tqdm(range(n_benefs), desc='Emp Prob'):
        transitions_df, emp_prob, aux_dict = getBenefsEmpProbs(traj, [benef], policy_id, trial_id, min_sup = 1, decomposed=decomposed)
        emp_prob_by_benef[benef] = emp_prob
        aux_dict_by_benef[benef] = aux_dict
        transitions_list.append(transitions_df)
    return emp_prob_by_benef, transitions_list, aux_dict_by_benef

def imputeEmpProb(options):
    for k in options:
        if options[k]==None:
            options[k] = 0.25
    return options

def to_prune(true_count_df, aug_count_dict, percent=0.6):
    for key in aug_count_dict:
        if key in true_count_df.index:
            true = true_count_df.loc[key]
        else:
            continue
        aug = aug_count_dict[key]
        # print(np.fabs(true-aug), np.round(true*percent), true)
        if np.fabs(true-aug) > max(1, np.round(true*percent)):
            return True
    return False

def augmentTraj(traj, true_traj_dfs, policy_id, trial_id, emp_prob_lookup, lookup_by_cluster, n_aug_traj, T, n_benefs, cluster_ids, do_prune=False):
    aug_traj = np.zeros((n_aug_traj, 1, T-1, len(dim_dict), n_benefs))
    for aug_traj_i in tqdm.tqdm(range(n_aug_traj), desc='Augment Trajectory'):
        for benef in range(n_benefs):
            retry = True
            true_traj_df = true_traj_dfs[benef][['s', 'a', 's_prime', 'a_prime']]
            true_count_df = pd.Series([str(tuple(i)) for i in true_traj_df.values.astype(int)]).value_counts()
            # print('true traj df ', true_traj_df)
            # print('true count df ', true_count_df)
            while retry:
                count_dict = defaultdict(lambda: 0)
                s, a = traj[trial_id, # trial index
                    policy_id, # policy index
                    0, # time index
                    [dim_dict['state'], dim_dict['action']], # tuple dimension
                    benef # benef index
                ]
                
                if lookup_by_cluster:
                    benef_cluster = cluster_ids[benef]
                    emp_prob = emp_prob_lookup[benef_cluster]
                else:
                    emp_prob = emp_prob_lookup[benef]

                for ts in range(T-1):
                    options = emp_prob[(s, a)]
                    # print(options)
                    # options = imputeEmpProb(options)
                    choice = np.random.choice(np.arange(len(list(options.keys()))),
                                                p=list(options.values()))
                    s_prime, a_prime = list(options.keys())[choice]
                    aug_traj[aug_traj_i, 0, ts, dim_dict['state'], benef] = s
                    aug_traj[aug_traj_i, 0, ts, dim_dict['action'], benef] = a
                    aug_traj[aug_traj_i, 0, ts, dim_dict['next_state'], benef] = s_prime
                    aug_traj[aug_traj_i, 0, ts, dim_dict['reward'], benef] = s
                    count_dict[str(tuple([int(s), int(a), int(s_prime), int(a_prime)]))]+=1
                    s, a = s_prime, a_prime
                # print('count aug ', count_dict)
                retry = to_prune(true_count_df, count_dict, do_prune) and do_prune
                # print(benef, retry)
                    
    print('Generated Augmented Traj of shape: ', aug_traj.shape)
    return aug_traj

def augmentTrajDecomposed(traj, policy_id, trial_id, emp_prob_lookup, lookup_by_cluster, n_aug_traj, T, n_benefs, cluster_ids):
    aug_traj = np.zeros((n_aug_traj, 1, T-1, len(dim_dict), n_benefs))

    for aug_traj_i in tqdm.tqdm(range(n_aug_traj), desc='Augment Trajectory'):
        for benef in range(n_benefs):
            s, a = traj[trial_id, # trial index
                policy_id, # policy index
                0, # time index
                [dim_dict['state'], dim_dict['action']], # tuple dimension
                benef # benef index
            ]
            if lookup_by_cluster:
                benef_cluster = cluster_ids[benef]
                emp_prob = emp_prob_lookup[benef_cluster]
            else:
                emp_prob = emp_prob_lookup[benef]

            for ts in range(T-1):
                ## Transition from (s,a) to s_prime
                options = emp_prob[(s, a)]
                # options = imputeEmpProb(options)
                choice = np.random.choice(np.arange(len(list(options.keys()))),
                                            p=list(options.values()))
                s_prime = list(options.keys())[choice]
                
                # Transition from (s_prime) to a_prime
                options = emp_prob[(s_prime)]
                options = imputeEmpProb(options)
                choice = np.random.choice(np.arange(len(list(options.keys()))),
                                            p=list(options.values()))
                a_prime = list(options.keys())[choice]

                aug_traj[aug_traj_i, 0, ts, dim_dict['state'], benef] = s
                aug_traj[aug_traj_i, 0, ts, dim_dict['action'], benef] = a
                aug_traj[aug_traj_i, 0, ts, dim_dict['next_state'], benef] = s_prime
                aug_traj[aug_traj_i, 0, ts, dim_dict['reward'], benef] = s
                s, a = s_prime, a_prime
    print('Generated Augmented Traj of shape: ', aug_traj.shape)
    return aug_traj

