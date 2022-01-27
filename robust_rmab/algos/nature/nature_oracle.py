# Authors: Jackson A. Killian, 4 July, 2021
# 
# Adapted from repository by: OpenAI
#    - https://spinningup.openai.com/en/latest/

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch
from torch.optim import Adam, SGD
import time
import pdb

import matplotlib as mpl
import matplotlib.pyplot as plt
import os, sys
# mpl.use('tkagg')

from robust_rmab.algos.whittle import mathprog_methods
from robust_rmab.utils.logx import EpochLogger
#from robust_rmab.environments.bandit_env import RandomBanditEnv, Eng1BanditEnv, RandomBanditResetEnv, CirculantDynamicsEnv
from robust_rmab.environments.bandit_env_robust import ToyRobustEnv, ARMMANRobustEnv, CounterExampleRobustEnv, SISRobustEnv

from robust_rmab.algos.whittle.whittle_policy import WhittlePolicy
from robust_rmab.baselines.nature_baselines_armman import CustomPolicy


class NatureOracle:

    def __init__(self, data, env_fn, N, S, A, B, seed, REWARD_BOUND, nature_kwargs=dict(),
        home_dir="", exp_name="", sampled_nature_parameter_ranges=None,
        pop_size=0, one_hot_encode=True, non_ohe_obs_dim=None, state_norm=1,
        nature_state_norm=1):

        self.data = data
        self.home_dir = home_dir
        self.exp_name = exp_name
        self.REWARD_BOUND = REWARD_BOUND
        self.N = N
        self.S = S
        self.A = A
        self.B = B
        self.seed=seed
        self.sampled_nature_parameter_ranges = sampled_nature_parameter_ranges
        self.nature_state_norm = nature_state_norm

        self.pop_size = pop_size
        self.one_hot_encode = one_hot_encode
        self.non_ohe_obs_dim = non_ohe_obs_dim
        self.state_norm = state_norm

        self.env_fn = env_fn
        self.env = self.env_fn()

        self.nature_kwargs=nature_kwargs

        self.strat_ind = -1

        # this won't work if we go back to MPI, but doing it now to simplify seeding
        self.env.seed(seed)
        self.env.sampled_parameter_ranges = self.sampled_nature_parameter_ranges

        # create a mapping of cluster -> list of individuals in cluster
        arms_in_cluster = {}
        for cluster in range(self.env.n_clusters):
            arms_in_cluster[cluster] = []
        for arm in range(len(self.env.cluster_mapping)):
            cluster = self.env.cluster_mapping[arm]
            arms_in_cluster[cluster].append(arm)
        self.arms_in_cluster = arms_in_cluster

    def get_agent_counts(self, agent_pol, nature_pol, env, steps_per_epoch, n_iterations=100):
        """ given pure strategy for agent and nature,
        estimate the probability that each (cluster, state) is pulled """

        return self.get_agent_counts_mixed_strategy([agent_pol], [1.0], [nature_pol], [1.0], env, steps_per_epochs, n_iterations=n_iterations)

    def get_agent_counts_mixed_strategy(self, nature_strats, nature_eq, agent_strats, agent_eq, env, steps_per_epoch, n_iterations=100):
        """ given MSNE for agent and nature, estimate the probability that an individual in each (cluster, state) is pulled

        returns matrix of dim [n_cluster, n_state]
        input into the Whittle index QP solver """

        INTERVENE = 1
        counter = np.zeros((self.env.n_clusters, self.S))  # n_clusters, n_states

        # iterate through environment to track the actions of our agent policy
        for epoch in range(n_iterations):
            # sample from nature mixed strategy
            nature_pol = np.random.choice(nature_strats, p=nature_eq)
            agent_pol  = np.random.choice(agent_strats, p=agent_eq)

            o = env.reset()
            for t in range(steps_per_epoch): # horizon
                o = o.reshape(-1)
                torch_o = torch.as_tensor(o, dtype=torch.float32)

                # a_agent = agent_pol.act_test(torch_o)
                a_agent = agent_pol.act_test_cluster_to_indiv(env.cluster_mapping, env.current_arms_state, env.B)
                
                # We obtain nature's action
                a_nature = nature_pol.get_nature_action(o)

                # Bound nature's actions within allowed range
                a_nature_env = nature_pol.bound_nature_actions(a_nature, state=o, reshape=True)

                next_o, r, d, a_agent_arms = env.step(a_agent, a_nature_env, agent_pol
                                                      , debug=(epoch==0 and t==0))
                for cluster in range(self.env.n_clusters):
                    # count number of actions that are intervene
                    for indiv in self.arms_in_cluster[cluster]:
                        s = env.current_arms_state[indiv]
                        if a_agent[indiv] == INTERVENE:
                            counter[cluster, s] += 1
                o = next_o
       
        # convert to probabilities
        counter /= (steps_per_epoch * n_iterations)
        # normalize by number of individuals
        for cluster in range(self.env.n_clusters):
            counter[cluster, :] /= len(self.arms_in_cluster[cluster])

        print('agent count', counter)

        return counter


    def best_response(self, agent_strats, agent_eq, prev_nature_strats, prev_nature_eq):
        self.strat_ind += 1
        
        # temporarily just return a dummy strategy for Nature Oracle (before we implement the QP-based approach)
        #return CustomPolicy(self.sampled_nature_parameter_ranges[:,:,:,1], self.strat_ind)


        from robust_rmab.utils.run_utils import setup_logger_kwargs

        exp_name = '%s_n%is%ia%ib%.2fr%.2f'%(self.exp_name, self.N, self.S, self.A, self.B, self.REWARD_BOUND)
        data_dir = os.path.join(self.home_dir, 'data')
        logger_kwargs = setup_logger_kwargs(self.exp_name, self.seed, data_dir=data_dir)

        return self.best_response_per_cpu(agent_strats, agent_eq, prev_nature_strats, prev_nature_eq, seed=self.seed, logger_kwargs=logger_kwargs, **self.nature_kwargs)

    def best_response_per_cpu(self, agent_strats, agent_eq, prev_nature_strats, prev_nature_eq,
        seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99,
        logger_kwargs=dict(), save_freq=10,
        gurobi_time_limit=10):
        
        # Set up logger and save configuration
        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals())


        # Instantiate environment
        env = self.env
        o = env.reset()
        o = o.reshape(-1)

        obs_dim = env.observation_dimension
        action_dim_nature = env.action_dim_nature

        agent_counts = self.get_agent_counts_mixed_strategy(prev_nature_strats, prev_nature_eq, agent_strats, agent_eq, env, steps_per_epoch, n_iterations=100)

        # identify the top B (cluster, state) pairs in terms of avg pull per cluster size
        budget = -np.floor(env.B).astype('int')
        top_B = np.argsort(agent_counts, axis=None)[-budget:][::-1]  # indices of top B values
        top_B_coords = np.unravel_index(top_B, agent_counts.shape)
        top_B_coords = [(top_B_coords[0][i], top_B_coords[1][i]) for i in range(budget)]


        all_T = np.zeros((env.n_clusters, env.S, env.A))

        # optimize the WI for each cluster independently
        for cluster in range(env.n_clusters):
            
            param_ranges = env.sample_parameter_ranges()
            
            p01p_range = param_ranges[cluster,0,0,:] # [cluster, s, action, range]
            p11p_range = param_ranges[cluster,1,0,:]
            p01a_range = param_ranges[cluster,0,1,:]
            p11a_range = param_ranges[cluster,1,1,:]

            R = env.R[cluster]
            C = env.C

            # senses indicates whether the QP should try to minimize or maximize a WI
            # if (cluster, state=0) and (cluster, state=1) in top B of pairs, then senses == ['min', 'min'] and so on
            sense1 = 'min' if (cluster,0) in top_B_coords else 'max'
            sense2 = 'min' if (cluster,1) in top_B_coords else 'max'
            senses = [sense1, sense2]

            assert mathprog_methods.check_feasible_range(p01p_range, p11p_range, p01a_range, p11a_range)
    
            optimized_indexes_bqp, L_vals, z_vals, bina_vals, T_return = mathprog_methods.bqp_to_optimize_index_both_states(
                                                                     p01p_range, p11p_range, p01a_range, p11a_range,
                                                                     R, C, senses=senses, gamma=gamma, time_limit=gurobi_time_limit)


            # T_return are entries in the transition matrix: (state, action, next state)
            # note that we only care about the right column (since probabilities next state=0 and next state = 1 will sum to 1
            all_T[cluster, :, :] = T_return[:, :, 1]


        print('nature policy', np.round(all_T,3))
        nature_policy = CustomPolicy(all_T, -1)
        return nature_policy
