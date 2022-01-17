# Authors: Jackson A. Killian, 4 July, 2021
# 
# Adapted from repository by: OpenAI
#    - https://spinningup.openai.com/en/latest/



import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import numpy as np
import torch
from torch.optim import Adam, SGD
import time
from robust_rmab.utils.logx import EpochLogger
from robust_rmab.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from robust_rmab.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
#from robust_rmab.environments.bandit_env import RandomBanditEnv, Eng1BanditEnv, RandomBanditResetEnv, CirculantDynamicsEnv, ARMMANEnv
from robust_rmab.environments.bandit_env_robust import ToyRobustEnv, ARMMANRobustEnv, CounterExampleRobustEnv, SISRobustEnv

from robust_rmab.algos.whittle.whittle_policy import WhittlePolicy


class AgentOracle:

    def __init__(self, data, N, S, A, B, seed, REWARD_BOUND, agent_kwargs=dict(),
        home_dir="", exp_name="", sampled_nature_parameter_ranges=None, robust_keyword="",
        pop_size=0, one_hot_encode=True, non_ohe_obs_dim=None, state_norm=None):

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
        self.robust_keyword = robust_keyword

        self.pop_size = pop_size
        self.one_hot_encode = one_hot_encode
        self.non_ohe_obs_dim = non_ohe_obs_dim
        self.state_norm = state_norm

        if data == 'random':
            self.env_fn = lambda : RandomBanditEnv(N,S,A,B,seed,REWARD_BOUND)

        elif data == 'random_reset':
            self.env_fn = lambda : RandomBanditResetEnv(N,S,A,B,seed,REWARD_BOUND)

        elif data == 'armman':
            self.env_fn = lambda : ARMMANRobustEnv(N,B,seed)

        elif data == 'circulant':
            self.env_fn = lambda : CirculantDynamicsEnv(N,B,seed)

        elif data == 'counterexample':
            self.env_fn = lambda : CounterExampleRobustEnv(N,B,seed)

        elif data == 'sis':
            self.env_fn = lambda : SISRobustEnv(N,B,pop_size,seed)

        self.agent_kwargs=agent_kwargs

        self.strat_ind = 0

        # this won't work if we go back to MPI, but doing it now to simplify seeding
        self.env = self.env_fn()
        self.env.seed(seed)
        self.env.sampled_parameter_ranges = self.sampled_nature_parameter_ranges


    # Todo - figure out parallelization with MPI -- not clear how to do this yet, so restrict to single cpu
    def best_response(self, nature_strats, nature_eq, add_to_seed):

        self.strat_ind += 1

        # mpi_fork(args.cpu, is_cannon=args.cannon)  # run parallel code with mpi

        from robust_rmab.utils.run_utils import setup_logger_kwargs

        exp_name = '%s_n%ib%.1fd%sr%sp%s'%(self.exp_name, self.N, self.B, self.data, self.robust_keyword, self.pop_size)
        data_dir = os.path.join(self.home_dir, 'data')
        logger_kwargs = setup_logger_kwargs(exp_name, self.seed, data_dir=data_dir)
        # logger_kwargs = setup_logger_kwargs(self.exp_name, self.seed+add_to_seed, data_dir=data_dir)

        return self.best_response_per_cpu(nature_strats, nature_eq, add_to_seed, seed=self.seed,  logger_kwargs=logger_kwargs, **self.agent_kwargs)

    # add_to_seed is obsolete
    def best_response_per_cpu(self, nature_strats, nature_eq, add_to_seed, actor_critic=None, ac_kwargs=dict(), seed=0, 
            steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
            vf_lr=1e-3, qf_lr=1e-3, lm_lr=5e-2, train_pi_iters=80, train_v_iters=80, train_q_iters=80,
            lam_OTHER=0.97,
            start_entropy_coeff=0.0, end_entropy_coeff=0.0,
            max_ep_len=1000,
            target_kl=0.01, logger_kwargs=dict(), save_freq=10,
            lamb_update_freq=10,
            init_lambda_trains=0,
            final_train_lambdas=0):
        ''' 
        This function outputs best response whittle index policy against a nature mixed strategy. 
        Many function arguments are residuals from previous deep RL based Agent Oracle and are no longer used
        '''
        # Set up logger and save configuration
        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals())

        # Instantiate environment
        env = self.env


        # Create Whittle Policy
        wh_policy = WhittlePolicy(env.N, env.S, env.B,
                                self.agent_kwargs['steps_per_epoch'], self.agent_kwargs['gamma']
                                 )


        # Sync params across processes
        sync_params(ac)

        # Set up experience buffer
        local_steps_per_epoch = int(steps_per_epoch / num_procs())


        buf = RMABPPO_Buffer(obs_dim, act_dim, env.N, ac.act_type, local_steps_per_epoch,
                        one_hot_encode=self.one_hot_encode, gamma=gamma, lam_OTHER=lam_OTHER)

        FINAL_TRAIN_LAMBDAS = final_train_lambdas


        # Set up function for computing RMABPPO policy loss
        def compute_loss_pi(data, entropy_coeff):
            ohs, act, adv, logp_old, lambdas, obs = data['ohs'], data['act'], data['adv'], data['logp'], data['lambdas'], data['obs']

            lamb_to_concat = np.repeat(lambdas, env.N).reshape(-1,env.N,1)
            full_obs = None
            if ac.one_hot_encode:
                full_obs = torch.cat([ohs, lamb_to_concat], axis=2)
            else:
                obs = obs/self.state_norm
                obs = obs.reshape(obs.shape[0], obs.shape[1], 1)
                full_obs = torch.cat([obs, lamb_to_concat], axis=2)

            loss_pi_list = np.zeros(env.N,dtype=object)
            pi_info_list = np.zeros(env.N,dtype=object)

            # Policy loss
            for i in range(env.N):
                pi, logp = ac.pi_list[i](full_obs[:, i], act[:, i])
                ent = pi.entropy().mean()
                ratio = torch.exp(logp - logp_old[:, i])
                clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv[:, i]
                loss_pi = -(torch.min(ratio * adv[:, i], clip_adv)).mean()
                
                # subtract entropy term since we want to encourage it 
                loss_pi -= entropy_coeff*ent


                loss_pi_list[i] = loss_pi

                # Useful extra info
                approx_kl = (logp_old[:, i] - logp).mean().item()
                
                clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
                clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
                pi_info = dict(kl=approx_kl, ent=ent.item(), cf=clipfrac)
                pi_info_list[i] = pi_info

            return loss_pi_list, pi_info_list

        # Set up function for computing value loss
        def compute_loss_v(data):
            ohs, ret, lambdas, obs = data['ohs'], data['ret'], data['lambdas'], data['obs']
            lamb_to_concat = np.repeat(lambdas, env.N).reshape(-1,env.N,1)
            full_obs = None
            if ac.one_hot_encode:
                full_obs = torch.cat([ohs, lamb_to_concat], axis=2)
            else:
                obs = obs/self.state_norm
                obs = obs.reshape(obs.shape[0], obs.shape[1], 1)
                full_obs = torch.cat([obs, lamb_to_concat], axis=2)

            loss_list = np.zeros(env.N,dtype=object)
            for i in range(env.N):
                loss_list[i] = ((ac.v_list[i](full_obs[:, i]) - ret[:, i])**2).mean()
            return loss_list

        def compute_loss_q(data):

            ohs, qs, oha, lambdas  = data['ohs'], data['qs'], data['oha'], data['lambdas']
            lamb_to_concat = np.repeat(lambdas, env.N).reshape(-1,env.N,1)
            full_obs = None
            if ac.one_hot_encode:
                full_obs = torch.cat([ohs, lamb_to_concat], axis=2)
            else:
                obs = obs/self.state_norm
                obs = obs.reshape(obs.shape[0], obs.shape[1], 1)
                full_obs = torch.cat([obs, lamb_to_concat], axis=2)

            loss_list = np.zeros(env.N,dtype=object)
            for i in range(env.N):
                x = torch.as_tensor(np.concatenate([full_obs[:, i], oha[:, i]], axis=1), dtype=torch.float32)
                loss_list[i] = ((ac.q_list[i](x) - qs[:, i])**2).mean()
            return loss_list

        
        def compute_loss_lambda(data):

            disc_cost = data['costs'][0]
            # lamb = data['lambdas'][0]
            obs = data['obs'][0]
            if not self.one_hot_encode:
                obs = obs/self.state_norm
            lamb = ac.lambda_net(torch.as_tensor(obs,dtype=torch.float32))
            # print('lamb',lamb, 'term 1', env.B/(1-gamma), 'cost',disc_cost, 'diff', env.B/(1-gamma) - disc_cost)
            # print('term 1', , 'cost',disc_cost)
            # print('term 1',env.B/(1-gamma))
            # print('cost',disc_cost)

            loss = lamb*(env.B/(1-gamma) - disc_cost)
            # print(loss)

            return loss

        # Set up optimizers for policy and value function
        pi_optimizers = np.zeros(env.N,dtype=object)
        vf_optimizers = np.zeros(env.N,dtype=object)
        qf_optimizers = np.zeros(env.N,dtype=object)

        for i in range(env.N):
            pi_optimizers[i] = Adam(ac.pi_list[i].parameters(), lr=pi_lr)
            # pi_optimizers[i] = SGD(ac.pi_list[i].parameters(), lr=pi_lr)
            vf_optimizers[i] = Adam(ac.v_list[i].parameters(), lr=vf_lr)
            # vf_optimizers[i] = SGD(ac.v_list[i].parameters(), lr=vf_lr)
            qf_optimizers[i] = Adam(ac.q_list[i].parameters(), lr=qf_lr)
            # qf_optimizers[i] = SGD(ac.q_list[i].parameters(), lr=qf_lr)
        # lambda_optimizer = Adam(ac.lambda_net.parameters(), lr=lm_lr)
        lambda_optimizer = SGD(ac.lambda_net.parameters(), lr=lm_lr)


        # Set up model saving
        logger.setup_pytorch_saver(ac)



        def update(epoch, head_entropy_coeff):
            data = buf.get()

            entropy_coeff = 0.0
            if (epochs - epoch) > FINAL_TRAIN_LAMBDAS:
                # cool entropy down as we relearn policy for each lambda
                entropy_coeff_schedule = np.linspace(head_entropy_coeff,0,lamb_update_freq)
                # don't rotate
                # entropy_coeff_schedule = entropy_coeff_schedule[1:] + entropy_coeff_schedule[:1]
                ind = epoch%lamb_update_freq
                entropy_coeff = entropy_coeff_schedule[ind]
            # print('entropy',entropy_coeff)

            # Train policy with multiple steps of gradient descent
            for _ in range(train_pi_iters):
                for i in range(env.N):
                    pi_optimizers[i].zero_grad()
                loss_pi, pi_info = compute_loss_pi(data, entropy_coeff)
                for i in range(env.N):
                    kl = mpi_avg(pi_info[i]['kl'])
                    # if kl > 1.5 * target_kl:
                    #     logger.log('Early stopping at step %d due to reaching max kl.'%i)
                    #     break
                    loss_pi[i].backward()
                    mpi_avg_grads(ac.pi_list[i])    # average grads across MPI processes
                    pi_optimizers[i].step()

            logger.store(StopIter=i)

            # Value function learning
            for i in range(train_v_iters):
                for i in range(env.N):
                    vf_optimizers[i].zero_grad()
                loss_v = compute_loss_v(data)
                for i in range(env.N):
                    loss_v[i].backward()
                    mpi_avg_grads(ac.v_list[i])    # average grads across MPI processes
                    vf_optimizers[i].step()


            # Lambda optimization
            if epoch%lamb_update_freq == 0 and epoch > 0 and (epochs - epoch) > FINAL_TRAIN_LAMBDAS:
                # for i in range(train_lam_iters):

                # Should only update this once because we only get one sample from the environment
                # unless we are running parallel instances
                lambda_optimizer.zero_grad()
                loss_lamb = compute_loss_lambda(data)
                
                loss_lamb.backward()
                last_param = list(ac.lambda_net.parameters())[-1]
                # print('last param',last_param)
                # print('grad',last_param.grad)

                mpi_avg_grads(ac.lambda_net)    # average grads across MPI processes
                lambda_optimizer.step()
                

        # Prepare for interaction with environment
        start_time = time.time()
        current_lamb = 0

        o, ep_actual_ret, ep_lamb_adjusted_ret, ep_len = env.reset(), 0, 0, 0
        o = o.reshape(-1)

        # Sample a nature policy
        # TODO: Currently we only calculate best response to one sample of nature policy from nature mixed strategy
        # TODO: Instead we want to calculate best response to expected nature from nature mixed strategy
        # TODO: this is only sampling one strategy. 
        nature_eq = np.array(nature_eq)
        nature_eq[nature_eq < 0] = 0
        nature_eq = nature_eq / nature_eq.sum()

        nature_pol = np.random.choice(nature_strats,p=nature_eq)


        # Main learning component: compute nature's Transition Param (TP) actions
        # and learn determinstic whittle indices against those TP

        print('Computing Agent\'s Best Response')
        print('Nature opponent chosen: ', nature_eq, nature_pol)

        torch_o = torch.as_tensor(o, dtype=torch.float32)
        a_nature = nature_pol.get_nature_action(torch_o)

        # Resample nature policy every time we update lambda
        if epoch%lamb_update_freq == 0 and epoch > 0:
            nature_pol = np.random.choice(nature_strats, p=nature_eq)
            
        torch_o = torch.as_tensor(o, dtype=torch.float32)
        a_nature = nature_pol.get_nature_action(torch_o)

        a_nature_env = nature_pol.bound_nature_actions(a_nature, state=o, reshape=True)
        print('nature transitions:', a_nature_env)

        # Whittle policy action
        wh_policy.note_env(env)
        wh_policy.learn(a_nature_env)

        return wh_policy


    def simulate_reward(self, agent_pol, nature_pol, seed=0, 
            steps_per_epoch=100, epochs=100, gamma=0.99):
        '''
        Given an agent pol and nature policy, we simulate the interaction for steps_per_epoch timesteps,
        and average the reward over `epochs` number of simulations
        '''
        # make a new env for computing returns 
        env = self.env_fn()
        # important to make sure these are always the same for all instatiations of the env
        env.sampled_parameter_ranges = self.sampled_nature_parameter_ranges

        env.seed(seed)

        o, ep_actual_ret, ep_len = env.reset(), 0, 0
        o = o.reshape(-1)

        # Store action history. This is needed for calculating fairness metrics
        # such as HHI
        a_history = []

        rewards = np.zeros((epochs, steps_per_epoch))
        for epoch in range(epochs):

            a_epoch_history = []
            for t in range(steps_per_epoch):
                torch_o  = torch.as_tensor(o, dtype=torch.float32)
                a_agent  = agent_pol.act_test(torch_o)
                a_nature = nature_pol.get_nature_action(torch_o)
                a_nature_env = nature_pol.bound_nature_actions(a_nature, state=o, reshape=True)
                
                
                next_o, r, d, a_agent_arms = env.step(a_agent, a_nature_env, agent_pol
                                                      , debug=(epoch==0 and t==0))

                a_epoch_history.append(a_agent_arms)

                next_o = next_o.reshape(-1)
                
                actual_r = r.sum()

                ep_actual_ret += actual_r
                ep_len += 1

                rewards[epoch,t] = actual_r*(gamma**t)

                # Update obs (critical!)
                o = next_o
            a_history.append(a_epoch_history)
        
            # loop again, reset start states
            o, ep_actual_ret, ep_lamb_adjusted_ret, ep_len = env.reset(), 0, 0, 0
            o = o.reshape(-1)
        
        rewards = rewards.sum(axis=1).mean()

        # Calculate HHI index
        hhi = self.calculate_fairness(a_history, epochs, steps_per_epoch, len(a_agent_arms))

        print("Rewards:", rewards)


        return rewards, hhi
    
    def calculate_fairness(self, a_history, epochs, steps_per_epoch, action_size):
        # Calculate HHI index to measure fairness in intervention distribution across arms
        """ fairness determined by percentage pulls of each arm per epoch """
        print('calculating Fairness')
        a_history = np.array(a_history)
        assert a_history.shape==(epochs, steps_per_epoch, action_size), a_history.shape

        pulls_per_epoch = np.sum(a_history, axis=1)
        budget = self.B * steps_per_epoch
        pull_proportion_per_epoch = pulls_per_epoch/budget
        hhi_per_epoch = np.sum(np.square(pull_proportion_per_epoch), axis=1) 
        mean_hhi = np.mean(hhi_per_epoch, axis=0)
        print(f'  HHI: {mean_hhi}')
        
        return mean_hhi
    
    def calculate_fairness_gini(self, a_history, epochs, steps_per_epoch, action_size):
        # Calculate gini index to measure fairness in intervention distribution across arms
        print('calculating Fairness')
        a_history = np.array(a_history)
        assert a_history.shape==(epochs, steps_per_epoch, action_size)
        pulls_per_epoch = np.sum(a_history, axis=1)
        # print('pulls per epoch: ', pulls_per_epoch)

        epoch_gini = []
        for i in range(self.N):
            for j in range(self.N):
                num = np.abs(pulls_per_epoch[i] - pulls_per_epoch[j])
                den = 2*self.N**2*np.mean(pulls_per_epoch, axis = 1)
                den[den==0]=1
                epoch_gini.append(num/den)
        out = np.mean(np.sum(epoch_gini, axis=1))
        return out
        

        





# python3 spinup/algos/pytorch/ppo/rmab_rl_lambda_ppo.py --hid 64 -l 2 --gamma 0.9 --cpu 1 --step 100 -N 4 -S 2 -A 2 -B 1 --REWARD_BOUND 2 --exp_name rmab_rl_bandit_n4s2a2b1_r2_lambda -s 0 --epochs 1000 --init_lambda_trains 1
# __main__ is now deprecated. It supported old agent_oracle with deep RL component
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--hid', type=int, default=64, help="Number of units in each layer of the neural networks used for the Oracles")
    parser.add_argument('-l', type=int, default=2, help="Depth of the neural networks used for Agent and Nature Oracles (i.e., layers)")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
    parser.add_argument('--seed', '-s', type=int, default=0, help="Seed")
    parser.add_argument('--cpu', type=int, default=1, help="Number of processes for mpi")
    
    parser.add_argument('--exp_name', type=str, default='experiment', help="Experiment name")
    parser.add_argument('-N', type=int, default=5, help="Number of arms")
    parser.add_argument('-S', type=int, default=4, help="Number of states in each arm (when applicable, e.g., SIS)")
    parser.add_argument('-A', type=int, default=2, help="Number of actions in each arm (not currently implemented)")
    parser.add_argument('-B', type=float, default=1.0, help="Budget per round")
    parser.add_argument('--reward_bound', type=int, default=1, help="Rescale rewards to this value (only some environments)")
    parser.add_argument('--save_string', type=str, default="")

    parser.add_argument('--agent_steps', type=int, default=10, help="Number of rollout steps between epochs")
    parser.add_argument('--agent_epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--agent_init_lambda_trains', type=int, default=0, help="Deprecated, leave at 0")
    parser.add_argument('--agent_clip_ratio', type=float, default=2.0, help="Clip ratio for PPO step")
    parser.add_argument('--agent_final_train_lambdas', type=int, default=10, help="Number of epochs at the end of training to update the policy and critic network, but not the lambda-network")
    parser.add_argument('--agent_start_entropy_coeff', type=float, default=0.0, help="Start entropy coefficient for the cooling procedure")
    parser.add_argument('--agent_end_entropy_coeff', type=float, default=0.0, help="End entropy coefficient for the cooling procedure")
    parser.add_argument('--agent_pi_lr', type=float, default=2e-3, help="Learning rate for policy network")
    parser.add_argument('--agent_vf_lr', type=float, default=2e-3, help="Learning rate for critic network")
    parser.add_argument('--agent_lm_lr', type=float, default=2e-3, help="Learning rate for lambda network")
    parser.add_argument('--agent_train_pi_iters', type=int, default=20, help="Training iterations to run per epoch")
    parser.add_argument('--agent_train_vf_iters', type=int, default=20, help="Training iterations to run per epoch")
    parser.add_argument('--agent_lamb_update_freq', type=int, default=4, help="Number of epochs that should pass before updating the lambda network (so really it is a period, not frequency)")

    parser.add_argument('--pop_size', type=int, default=0)

    parser.add_argument('--home_dir', type=str, default='.', help="Home directory for experiments")
    parser.add_argument('--cannon', type=int, default=0, help="Flag used for running experiments on batched slurm-based HPC resources. Leave at 0 for small experiments.")
    parser.add_argument('-d', '--data', default='counterexample', type=str, help='Environment selection',
                        choices=[   
                                    'random',
                                    'random_reset',
                                    'circulant', 
                                    'armman',
                                    'counterexample',
                                    'sis'
                                ])

    parser.add_argument('--robust_keyword', default='pess', type=str, help='Method for picking some T out of the uncertain environment',
                        choices=[   
                                    'pess',
                                    'mid',
                                    'opt', # i.e., optimistic
                                    'sample_random'
                                ])

    args = parser.parse_args()

    mpi_fork(args.cpu, is_cannon=args.cannon)  # run parallel code with mpi

    # from spinup.utils.run_utils import setup_logger_kwargs

    # exp_name = '%s_n%is%ia%ib%.2fr%.2f'%(args.exp_name, args.N, args.S, args.A, args.B, args.REWARD_BOUND)
    # print(exp_name)
    # data_dir = os.path.join(args.home_dir, 'data')
    # logger_kwargs = setup_logger_kwargs(exp_name, args.seed, data_dir=data_dir)

    N = args.N
    S = args.S
    A = args.A
    B = args.B
    budget = B
    reward_bound = args.reward_bound
    seed=args.seed
    data = args.data
    home_dir = args.home_dir
    exp_name=args.exp_name
    gamma = args.gamma

    torch.manual_seed(seed)
    np.random.seed(seed)

    agent_kwargs = {}
    agent_kwargs['steps_per_epoch'] = args.agent_steps
    agent_kwargs['epochs'] = args.agent_epochs
    agent_kwargs['init_lambda_trains'] = args.agent_init_lambda_trains
    agent_kwargs['clip_ratio'] = args.agent_clip_ratio
    agent_kwargs['final_train_lambdas'] = args.agent_final_train_lambdas
    agent_kwargs['start_entropy_coeff'] = args.agent_start_entropy_coeff
    agent_kwargs['end_entropy_coeff'] = args.agent_end_entropy_coeff
    agent_kwargs['pi_lr'] = args.agent_pi_lr
    agent_kwargs['vf_lr'] = args.agent_vf_lr
    agent_kwargs['lm_lr'] = args.agent_lm_lr
    agent_kwargs['train_pi_iters'] = args.agent_train_pi_iters
    agent_kwargs['train_v_iters'] = args.agent_train_vf_iters
    agent_kwargs['lamb_update_freq'] = args.agent_lamb_update_freq
    agent_kwargs['ac_kwargs'] = dict(hidden_sizes=[args.hid]*args.l)
    agent_kwargs['gamma'] = args.gamma

    env_fn = None

    one_hot_encode = True
    non_ohe_obs_dim = None
    state_norm = None

    if args.data == 'counterexample':
        from robust_rmab.baselines.nature_baselines_counterexample import   (
                    RandomNaturePolicy, PessimisticNaturePolicy, MiddleNaturePolicy, 
                    OptimisticNaturePolicy, DetermNaturePolicy, SampledRandomNaturePolicy
                )
        env_fn = lambda : CounterExampleRobustEnv(N,B,seed)

    if args.data == 'armman':
        from robust_rmab.baselines.nature_baselines_armman import   (
                            RandomNaturePolicy, PessimisticNaturePolicy, MiddleNaturePolicy, 
                            OptimisticNaturePolicy, SampledRandomNaturePolicy
                        )
        env_fn = lambda: ARMMANRobustEnv(N,B,seed)

    if args.data == 'sis':
        from robust_rmab.baselines.nature_baselines_sis import   (
                            RandomNaturePolicy, PessimisticNaturePolicy, MiddleNaturePolicy, 
                            OptimisticNaturePolicy, SampledRandomNaturePolicy
                        )
        env_fn = lambda: SISRobustEnv(N,B,args.pop_size,seed)
        
        # don't one hot encode this state space...
        one_hot_encode = False
        non_ohe_obs_dim = 1
        state_norm = args.pop_size


    env = env_fn()
    sampled_nature_parameter_ranges = env.sample_parameter_ranges()
    # important to make sure these are always the same for all instatiations of the env
    env.sampled_parameter_ranges = sampled_nature_parameter_ranges

    agent_oracle  = AgentOracle(data, N, S, A, budget, seed, reward_bound,
                             agent_kwargs=agent_kwargs, home_dir=home_dir, exp_name=exp_name,
                             robust_keyword=args.robust_keyword,
                             sampled_nature_parameter_ranges = sampled_nature_parameter_ranges,
                             pop_size=args.pop_size, one_hot_encode=one_hot_encode, state_norm=state_norm,
                             non_ohe_obs_dim=non_ohe_obs_dim)

    nature_strategy = None
    if args.robust_keyword == 'mid':
        nature_strategy = MiddleNaturePolicy(sampled_nature_parameter_ranges, 0)

    if args.robust_keyword == 'sample_random':
        nature_strategy = SampledRandomNaturePolicy(sampled_nature_parameter_ranges, 0)

        # init the random strategy
        nature_strategy.sample_param_setting(seed)



    add_to_seed = 0
    agent_oracle.best_response([nature_strategy], [1.0], add_to_seed)




