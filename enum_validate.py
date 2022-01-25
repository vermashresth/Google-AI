import numpy as np
from itertools import product, combinations
import tqdm

from robust_rmab.algos.whittle.whittle_policy import WhittlePolicy
from robust_rmab.environments.bandit_env_robust import ARMMANRobustEnv
from robust_rmab.baselines.agent_baselines import   (
                            PessimisticAgentPolicy, RandomAgentPolicy
                        )
from robust_rmab.baselines.nature_baselines_armman import   (
                            RandomNaturePolicy, PessimisticNaturePolicy, MiddleNaturePolicy, 
                            OptimisticNaturePolicy, SampledRandomNaturePolicy
                        )

info_dict = {}
info_dict['n_clusters'] = 2
info_dict['cluster_mapping'] = [0, 1]
info_dict['max_cluster_size'] = 1
info_dict['parameter_ranges'] = [
                                    [
                                        [[0.3, 0.7], [0.2, 0.8]],
                                        [[0.2, 0.6], [0.5, 0.6]],
                                    ],
                                    [
                                        [[0.1, 0.4], [0.6, 0.9]],
                                        [[0.2, 0.5], [0.1, 0.3]],
                                    ],
                                    # [
                                    #     [[0.5, 0.6], [0.2, 0.7]],
                                    #     [[0.1, 0.2], [0.3, 0.6]],
                                    # ],
                                    # [
                                    #     [[0.3, 0.7], [0.3, 0.7]],
                                    #     [[0.3, 0.7], [0.3, 0.7]],
                                    # ],
                                    # [
                                    #     [[0.3, 0.7], [0.3, 0.7]],
                                    #     [[0.3, 0.7], [0.3, 0.7]],
                                    # ]
                                ]

N, b, seed = 4, 1, 0
#n_states = N/2
env = ARMMANRobustEnv(N, b, seed, info_dict)
nature_pol = MiddleNaturePolicy(env.sampled_parameter_ranges, 0)
agent_pol = PessimisticAgentPolicy(env.n_arms, 0)

def agent_br(env, nature_pol):
    system_states = list(product(*[range(env.S)]*env.n_arms)) #2^n_arms
    state_lookup_dict = {system_states[i]:i for i in range(len(system_states))}
    top_b_combs = list(combinations(range(env.n_arms), b)) # n_arms C b
    all_strategies = list(product(*[top_b_combs]*len(system_states))) # (n_arms C b) ^ (s^n_arms)


    n_epochs = 10
    steps_per_epoch = 10
    gamma = 0.99


    rewards = np.zeros((len(all_strategies), n_epochs, steps_per_epoch))
    for strategy_idx in tqdm.tqdm(range(len(all_strategies))):
        strategy = all_strategies[strategy_idx]
        for epoch in range(n_epochs):
            # reseed env at every epoch
            env.seed(epoch)
            o = env.reset()
            o = o.reshape(-1)
            for t in range(steps_per_epoch):
                a_nature = nature_pol.get_nature_action(o)
                a_nature_env = nature_pol.bound_nature_actions(a_nature, state=o, reshape=True)
                print(a_nature_env.shape)
                arms_state = env.current_arms_state

                # list of arm indices to be pulled
                # print('arms state ', arms_state, state_lookup_dict[tuple(arms_state)])
                a_agent_indices = strategy[state_lookup_dict[tuple(arms_state)]] 
                a_agent = np.zeros(env.n_arms)
                a_agent[list(a_agent_indices)] = 1
                next_o, r, d, a_agent_arms = env.step(a_agent, a_nature_env, None)

                next_o = next_o.reshape(-1)
                
                actual_r = r.sum()

                rewards[strategy_idx, epoch, t] = actual_r*(gamma**t)

                o = next_o

    print(np.argmax(np.mean(np.sum(rewards, axis=2), axis=1)))

def nature_br(env, agent_pol):
    low = env.sampled_parameter_ranges[..., 0].flatten()
    high = env.sampled_parameter_ranges[..., 1].flatten()
    n_discrete_points = 3
    n_params = len(low)
    prob_vals = np.linspace(low, high, n_discrete_points)
    all_strategies = list(product(*[range(n_discrete_points)]*n_params)) # n_discrete_points ^ n_params

    n_epochs = 10
    steps_per_epoch = 10
    gamma = 0.99

    wh_policy = WhittlePolicy(env.N, env.S, env.B,
                                steps_per_epoch, gamma
                                 )

    rewards = np.zeros((len(all_strategies), n_epochs, steps_per_epoch))
    rewards_optimal = np.zeros((len(all_strategies), n_epochs, steps_per_epoch))
    for strategy_idx in tqdm.tqdm(range(len(all_strategies))):
        strategy = all_strategies[strategy_idx]
        
        a_nature = prob_vals[strategy, np.arange(n_params)]
        a_nature = a_nature.reshape((env.N//env.S, env.S, env.A))
        
        wh_policy.note_env(env)
        wh_policy.learn(a_nature)

        for epoch in range(n_epochs):
            # reseed env at every epoch
            env.seed(epoch)
            o = env.reset()
            o = o.reshape(-1)
            for t in range(steps_per_epoch):
                
                a_agent = agent_pol.act_test(o)
                next_o, r, d, a_agent_arms = env.step(a_agent, a_nature, None)

                next_o = next_o.reshape(-1)
                
                actual_r = r.sum()

                rewards[strategy_idx, epoch, t] = actual_r*(gamma**t)

                o = next_o
            
            o = env.reset()
            o = o.reshape(-1)
            for t in range(steps_per_epoch):
                
                a_agent = agent_pol.act_test(o)
                next_o, r, d, a_agent_arms = env.step(None, a_nature, wh_policy)

                next_o = next_o.reshape(-1)
                
                actual_r = r.sum()

                rewards_optimal[strategy_idx, epoch, t] = actual_r*(gamma**t)

                o = next_o

    regrets = np.sum(rewards_optimal - rewards, axis=2)

    print(np.argmax(np.mean(regrets, axis=1)))

nature_br(env, agent_pol)







