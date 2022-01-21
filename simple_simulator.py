import argparse
import numpy as np
import torch
import pickle
import tqdm

from robust_rmab.environments.bandit_env_robust import ARMMANRobustEnv
from robust_rmab.baselines.nature_baselines_armman import   (
                            RandomNaturePolicy, PessimisticNaturePolicy, MiddleNaturePolicy, 
                            OptimisticNaturePolicy, SampledRandomNaturePolicy
                        )
                    
def simulate_reward(agent_strats, agent_eq, nature_strats, nature_eq, env, seed=0, 
            steps_per_epoch=100, epochs=100, gamma=0.99):
        '''
        Given an agent pol and nature policy, we simulate the interaction for steps_per_epoch timesteps,
        and average the reward over `epochs` number of simulations
        '''
        print()
        print('Playing Agent Mixed Strategy')
        for pol, prob in zip(agent_strats, agent_eq):
            print(str(pol), ' with prob ', np.round(prob, 3))
        print()
        print('Playing Nature Mixed Strategy')
        for pol, prob in zip(nature_strats, nature_eq):
            print(str(pol), ' with prob ', np.round(prob, 3))
        env.seed(seed)

        o, ep_actual_ret, ep_len = env.reset(), 0, 0
        o = o.reshape(-1)

        rewards = np.zeros((epochs, steps_per_epoch))
        a_history = np.zeros((epochs, steps_per_epoch, env.n_arms))
        for epoch in tqdm.tqdm(range(epochs)):
            
            nature_eq = np.array(nature_eq)
            nature_eq[nature_eq < 0] = 0
            nature_eq = nature_eq / nature_eq.sum()
            nature_pol = np.random.choice(nature_strats, p=nature_eq)

            agent_eq = np.array(agent_eq)
            agent_eq[agent_eq < 0] = 0
            agent_eq = agent_eq / agent_eq.sum()

            agent_pol = np.random.choice(agent_strats, p=agent_eq)

            for t in range(steps_per_epoch):
                torch_o = torch.as_tensor(o, dtype=torch.float32)
                a_agent  = agent_pol.act_test(torch_o)
                a_nature = nature_pol.get_nature_action(torch_o)
                a_nature_env = nature_pol.bound_nature_actions(a_nature, state=o, reshape=True)
                
                
                next_o, r, d, a_agent_arms = env.step(a_agent, a_nature_env, agent_pol)

                a_history[epoch, t] = a_agent_arms.copy()

                next_o = next_o.reshape(-1)
                
                actual_r = r.sum()

                ep_actual_ret += actual_r
                ep_len += 1

                rewards[epoch,t] = actual_r*(gamma**t)

                # Update obs (critical!)
                o = next_o
        
            # loop again, reset start states
            o, ep_actual_ret, ep_len = env.reset(), 0, 0
            o = o.reshape(-1)
        
        rewards = rewards.sum(axis=1).mean()

        print("Avg Reward:", rewards)
        return rewards, {'a_history': a_history}

def load_wh_model_dict(filename):
    with open(filename, 'rb') as file:
        file.seek(0)
        model_dict = pickle.load(file)
    return model_dict


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Run adherence simulations with various methods.')
    parser.add_argument('-s', '--seed_base', type=int, help='Base for the random seed')
    parser.add_argument('-nat', '--nature', type=str, default='mixed', help='Nature stragy to play against. Can be mixed, mid, random')
    args = parser.parse_args()

    rl_info = {
        'model_file_path_rmab_wh': '/Users/vermashresth/Documents/Google-AI-RobustClusterRMAB/logs/model_dump/setup_test_n10_b5.0_h10_epoch1_dataarmman_seed0'
    }

    model_dict = load_wh_model_dict(rl_info['model_file_path_rmab_wh'])
    N, b, seed = 10, 5, 0
    # N, b, seed = model_dict['N'], model_dict['b'], args.seed_base
    env = ARMMANRobustEnv(N, b, seed)

    agent_strats, agent_eq = model_dict['agent_strategies'], model_dict['agent_eq']
    
    if args.nature=='mixed':
        nature_strats, nature_eq = model_dict['nature_strategies'], model_dict['nature_eq']
    if args.nature == 'mid':
        nature_strats = [MiddleNaturePolicy(env.sampled_parameter_ranges, 0)]
        nature_eq = [1]
    if args.nature == 'random':
        nature_strats = [SampledRandomNaturePolicy(env.sampled_parameter_ranges, 0)]
        nature_eq = [1]
    simulate_reward(agent_strats, agent_eq, nature_strats, nature_eq, env)
