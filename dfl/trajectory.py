import numpy as np
import pandas as pd
import argparse
import tqdm

from dfl.config import policy_names, dim_dict, S_VALS, A_VALS
from dfl.policy import getActions
from dfl.utils import getBenefsByCluster
from armman.simulator import takeActions

from collections import defaultdict

def simulateTrajectories(args, T=None, w=None, start_state=None):
  E_START_STATE_PROB=0.5
  ##### Unpack arguments
  L=args.simulation_length
  N=args.num_beneficiaries
  k=args.num_resources
  ntr=args.num_trials
  
  policies=[0, 1, 2]

  state_record=np.zeros((ntr, len(policies),L,N))  # Store the full state
                                                   # trajectory
  action_record=np.zeros((ntr, len(policies),L,N))  # Store the full state
                                                   # trajectory

  simulated_rewards=np.zeros((ntr, len(policies))) # Store aggregate rewards
  history_dict = {i_pol: [] for i_pol in policies}
  
  traj = []
  ##### Iterate over number of independent trials to average over
  for tr in range(ntr):
    tr_traj = []
    ## Initialize for each trial
    np.random.seed(seed=tr+args.seed_base)
     # Random state \
    #initialization                      
    
    ## For current trial, evaluate all policies
    for pol_idx, pol in enumerate(policies):
      pol_traj = []
      if start_state is None:
          states=np.random.binomial(1, E_START_STATE_PROB, size=N)
      else:
        assert start_state.shape == (N,)
        states = np.copy(start_state)
      ## Iterate over timesteps. Note that if simulation length is L, 
      ## there are L-1 action decisions to take.
      for timestep in range(L-1):
        tupl = []
        ## Compute actions
        state_record[tr, pol_idx, timestep, :] = np.copy(states)
        tupl.append(np.copy(states))
        
        actions=getActions(states=states, policy=pol, ts=timestep, w=w, k=k)
        action_record[tr, pol_idx, timestep, :] = np.copy(actions)
        tupl.append(actions)

        states = np.copy(takeActions(states, T, actions))
        tupl.append(states)
        tupl.append(np.copy(state_record[tr, pol_idx, timestep, :]))
        pol_traj.append(tupl)

      simulated_rewards[tr, pol_idx]=np.sum(np.sum(state_record[tr,pol_idx], \
                                                    axis=1))

      tr_traj.append(pol_traj)
    traj.append(tr_traj)
  ##### Print results
  for pol in policies: 
    print("Expected reward of policy %s is %s"%(policy_names[pol], \
                            np.mean(simulated_rewards[:,policies.index(pol)])))
  
  return simulated_rewards, state_record, action_record, np.array(traj)


def getSimulatedTrajectories(n_benefs = 10, T = 5, K = 3, n_trials = 10, gamma = 1, seed = 10, mask_seed=10, T_data=None, w=None, start_state=None, debug=False, replace=False):
    parser = argparse.ArgumentParser(description='Inputs to engagement simulator module')
    parser.add_argument('-N', '--num_beneficiaries', default=7000, type=int, help='Number of Beneficiaries')
    parser.add_argument('-k', '--num_resources', default=1400, type=int, help='Number of calls available per day')
    parser.add_argument('-L', '--simulation_length', default=40, type=int, help='Number of timesteps of simulation')
    parser.add_argument('-ntr', '--num_trials', default=10, type=int, help='Number of independent trials')
    parser.add_argument('-s', '--seed_base', default=10, type=int, help='Seedbase for numpy. This is starting seedbase. Simulation will consider the seeds= {seed_base, ... seed_base+ntr-1}')
    parser.add_argument('-p', '--policy', default=-1, type=int, help='policy to run. default is all policies')
    parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
    args = parser.parse_args()
    args.num_beneficiaries=n_benefs
    args.num_resources=K
    args.simulation_length=T
    args.num_trials=n_trials
    args.seed_base = seed

    np.random.seed(mask_seed)
    mask = np.random.choice(np.arange(T_data.shape[0]), n_benefs, replace=replace)
    
    np.random.seed(args.seed_base)
    simulated_rewards, state_record, action_record, traj = simulateTrajectories(args, T=T_data[mask], w=w[mask], start_state=start_state)

    if debug:
        print(mask[:10])
        print('trajectory shape: ', np.array(traj).shape) 
    whittle_rew = simulated_rewards[:, 2].mean()
    return traj, whittle_rew, simulated_rewards, mask, state_record, action_record

    
def getBenefsEmpProbs(traj, benef_ids, policy_id, trial_id, min_sup = 1, decomposed=False):
    benef_ci_traj = traj[trial_id, # trial index
            policy_id, # policy index
            :, # time index
            :, # tuple dimension
            benef_ids # benef index
        ]
    s_traj_c =  benef_ci_traj[:, :-1, dim_dict['state']]
    a_traj_c =  benef_ci_traj[:, :-1, dim_dict['action']]
    s_prime_traj_c =  benef_ci_traj[:, :-1, dim_dict['new_state']]
    a_prime_traj_c = benef_ci_traj[:, 1:, dim_dict['action']]

    transitions_df = pd.DataFrame(columns = ['s', 's_prime', 'a', 'a_prime'])

    for s_traj, a_traj, s_prime_traj, a_prime_traj in \
                        zip(s_traj_c, a_traj_c, s_prime_traj_c, a_prime_traj_c):
        transitions_df = transitions_df.append(pd.DataFrame({'s':s_traj,
                                's_prime': s_prime_traj,
                                'a': a_traj,
                                'a_prime': a_prime_traj}), ignore_index=True)
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
                    aug_traj[aug_traj_i, 0, ts, dim_dict['new_state'], benef] = s_prime
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
                aug_traj[aug_traj_i, 0, ts, dim_dict['new_state'], benef] = s_prime
                aug_traj[aug_traj_i, 0, ts, dim_dict['reward'], benef] = s
                s, a = s_prime, a_prime
    print('Generated Augmented Traj of shape: ', aug_traj.shape)
    return aug_traj

