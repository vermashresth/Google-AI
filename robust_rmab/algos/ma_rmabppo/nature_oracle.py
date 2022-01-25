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
import robust_rmab.algos.ma_rmabppo.ma_rmabppo_core as core
from robust_rmab.algos.whittle.mathprog_methods import bqp_to_optimize_index
from robust_rmab.utils.logx import EpochLogger
from robust_rmab.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from robust_rmab.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from robust_rmab.environments.bandit_env import RandomBanditEnv, Eng1BanditEnv, RandomBanditResetEnv, CirculantDynamicsEnv
from robust_rmab.environments.bandit_env_robust import ToyRobustEnv, ARMMANRobustEnv, CounterExampleRobustEnv, SISRobustEnv

import matplotlib as mpl
import matplotlib.pyplot as plt
import os, sys
# mpl.use('tkagg')
from robust_rmab.algos.whittle.whittle_policy import WhittlePolicy
from robust_rmab.baselines.nature_baselines_armman import CustomPolicy


class MA_RMABPPO_Whittle_Buffer:
    """
    A buffer for storing trajectories experienced by a MA_RMABPPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim_agent, act_dim_nature, N, act_type, size, one_hot_encode=True, gamma=0.99, lam_OTHER=0.95):
        self.N = N
        self.obs_dim = obs_dim
        self.act_dim_agent = act_dim_agent
        self.act_dim_nature = act_dim_nature
        self.one_hot_encode = one_hot_encode

        self.obs_buf = np.zeros(core.combined_shape(size, N), dtype=np.float32)
        self.ohs_buf = np.zeros(core.combined_shape(size, (N, obs_dim)), dtype=np.float32)
        
        self.act_buf_agent = np.zeros((size, N), dtype=np.float32)
        self.act_buf_nature = np.zeros((size, act_dim_nature), dtype=np.float32)
        # self.oha_buf = np.zeros(core.combined_shape(size, (N, act_dim)), dtype=np.float32)

        self.adv_buf_agent = np.zeros((size,N), dtype=np.float32)
        self.rew_buf_agent = np.zeros((size,N), dtype=np.float32)
        self.cost_buf = np.zeros((size,N), dtype=np.float32)
        self.ret_buf_agent = np.zeros((size,N), dtype=np.float32)
        self.val_buf_agent = np.zeros((size,N), dtype=np.float32)
        self.q_buf_agent   = np.zeros((size,N), dtype=np.float32)
        self.logp_buf_agent = np.zeros((size,N), dtype=np.float32)
        self.cdcost_buf = np.zeros(size, dtype=np.float32)
        self.lamb_buf = np.zeros(size, dtype=np.float32)

        self.adv_buf_nature = np.zeros(size, dtype=np.float32)
        self.rew_buf_nature = np.zeros(size, dtype=np.float32)
        self.ret_buf_nature = np.zeros(size, dtype=np.float32)
        self.val_buf_nature = np.zeros(size, dtype=np.float32)
        self.q_buf_nature   = np.zeros(size, dtype=np.float32)
        self.logp_buf_nature = np.zeros(size, dtype=np.float32)

        self.gamma, self.lam_OTHER = gamma, lam_OTHER
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.act_type = act_type
        


    def store(self, obs, act_agent, act_nature, rew_nature, val_nature, q_nature, logp_nature):
                        
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        ohs = np.zeros((self.N, self.obs_dim))
        if self.one_hot_encode:
            for i in range(self.N):
                ohs[i, int(obs[i])] = 1
        self.ohs_buf[self.ptr] = ohs
        
        self.act_buf_agent[self.ptr] = act_agent
        self.act_buf_nature[self.ptr] = act_nature
        # oha = np.zeros((self.N, self.act_dim))
        # for i in range(self.N):
        #     oha[i, int(act[i])] = 1
        # self.oha_buf[self.ptr] = oha


        self.rew_buf_nature[self.ptr] = rew_nature
        self.val_buf_nature[self.ptr] = val_nature
        self.q_buf_nature[self.ptr]   = q_nature
        self.logp_buf_nature[self.ptr] = logp_nature
        # print('nature rew buf', self.rew_buf_nature, self.ptr, rew_nature)
        self.ptr += 1


    
    def finish_path(self, last_val_nature=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)

        arm_summed_costs = np.zeros(self.ptr - self.path_start_idx + 1)

        # for i in range(self.N):
        #     rews_agent = np.append(self.rew_buf_agent[path_slice, i], last_vals_agent[i])
        #     # TODO implement training that makes use of last_costs, i.e., use all samples to update lam
        #     costs = np.append(self.cost_buf[path_slice, i], 0)
        #     # print(costs)
        #     lambds = np.append(self.lamb_buf[path_slice], 0)

        #     arm_summed_costs += costs
        #     # adjust based on action costs

        #     rews_agent = rews_agent - lambds*costs

        #     vals_agent = np.append(self.val_buf_agent[path_slice, i], last_vals_agent[i])
            
        #     # the next two lines implement GAE-Lambda advantage calculation
        #     qs_agent = rews_agent[:-1] + self.gamma * vals_agent[1:]
        #     deltas_agent = qs_agent - vals_agent[:-1]
        #     self.adv_buf_agent[path_slice, i] = core.discount_cumsum(deltas_agent, self.gamma * self.lam_OTHER)
            
        #     # the next line computes rewards-to-go, to be targets for the value function
        #     self.ret_buf_agent[path_slice, i] = core.discount_cumsum(rews_agent, self.gamma)[:-1]

        #     # store the learned q functions
        #     self.q_buf_agent[path_slice, i]   = qs_agent
            
        self.path_start_idx = self.ptr


        # the next line computes costs-to-go, to be part of the loss for the lambda net
        # self.cdcost_buf[path_slice] = core.discount_cumsum(arm_summed_costs, self.gamma)[:-1]


        rews_nature = np.append(self.rew_buf_nature[path_slice], last_val_nature)
        vals_nature = np.append(self.val_buf_nature[path_slice], last_val_nature)

        qs_nature = rews_nature[:-1] + self.gamma * vals_nature[1:]
        deltas_nature = qs_nature - vals_nature[:-1]
        self.adv_buf_nature[path_slice] = core.discount_cumsum(deltas_nature, self.gamma * self.lam_OTHER)
        self.ret_buf_nature[path_slice] = core.discount_cumsum(rews_nature, self.gamma)[:-1]
        self.q_buf_nature[path_slice]   = qs_nature



    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # for i in range(self.N):
        #     # the next two lines implement the advantage normalization trick
        #     adv_mean_agent, adv_std_agent = mpi_statistics_scalar(self.adv_buf_agent[:, i])
        #     self.adv_buf_agent[:, i] = (self.adv_buf_agent[:, i] - adv_mean_agent) / adv_std_agent
        
        adv_mean_nature, adv_std_nature = mpi_statistics_scalar(self.adv_buf_nature)
        self.adv_buf_nature = (self.adv_buf_nature - adv_mean_nature) / adv_std_nature


        data = dict(obs=self.obs_buf, act_agent=self.act_buf_agent, act_nature=self.act_buf_nature, 
                ret_nature=self.ret_buf_nature, adv_nature=self.adv_buf_nature, 
                logp_nature=self.logp_buf_nature, qs_nature=self.q_buf_nature,
                ohs=self.ohs_buf)
        
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

class NatureOracle:

    def __init__(self, data, N, S, A, B, seed, REWARD_BOUND, nature_kwargs=dict(),
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

        # self.ma_actor_critic = core.RMABLambdaNatureOracle
        self.ma_actor_critic = core.RMABWhittleNatureOracle
        self.nature_kwargs=nature_kwargs

        self.strat_ind = -1

        # this won't work if we go back to MPI, but doing it now to simplify seeding
        self.env = self.env_fn()
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
        """ returns the probability that each cluster is pulled given its state
        matrix of dim [n_cluster, n_state]
        input into the Whittle index QP solver """

        INTERVENE = 1
        counter = np.zeros((self.env.n_clusters, self.S))  # n_clusters, n_states

        # iterate through environment to track the actions of our agent policy
        for epoch in range(n_iterations):
            o = env.reset()
            for t in range(steps_per_epoch): # horizon
                o = o.reshape(-1)
                torch_o = torch.as_tensor(o, dtype=torch.float32)

                # a_agent  = agent_pol.act_test(torch_o)
                a_agent  = agent_pol.act_test_cluster_to_indiv(env.cluster_mapping, env.current_arms_state, env.B)
                # a_nature = nature_pol.get_nature_action(torch_o)

                # ac's step function requires both world observation (for actor) and agent_policy's actions (for critic)
                # but we can only obtain agent_policy's actions after determining nature's action
                # so we split the ac.step into two parts, first pass some dummy agent_policy action
                a_agent_list_dummy = np.zeros(self.N)
                
                # We obtain nature's action
                a_nature, _, logp_nature, q_nature = nature_pol.step(torch_o, a_agent_list_dummy)

                # Bound nature's actions within allowed range
                a_nature_env = nature_pol.bound_nature_actions(a_nature, state=o, reshape=True)

                next_o, r, d, a_agent_arms = env.step(a_agent, a_nature_env, agent_pol
                                                      , debug=(epoch==0 and t==0))
                for cluster in range(self.env.n_clusters):
                    for s in range(self.S):
                        # count number of actions that are intervene
                        for indiv in self.arms_in_cluster[cluster]:
                            if env.current_arms_state[indiv] == s and a_agent[indiv] == INTERVENE:
                                counter[cluster, s] += 1
                o = next_o
       
        # convert to probabilities 
        counter /= (steps_per_epoch * n_iterations)
        print('counter:', counter)
        return counter



    def best_response(self, nature_strats, nature_eq, add_to_seed):
        self.strat_ind += 1
        
        # temporarily just return a dummy strategy for Nature Oracle (before we implement the QP-based approach)
        #return CustomPolicy(self.sampled_nature_parameter_ranges[:,:,:,1], self.strat_ind)


        # mpi_fork(args.cpu, is_cannon=args.cannon)  # run parallel code with mpi

        from robust_rmab.utils.run_utils import setup_logger_kwargs

        exp_name = '%s_n%is%ia%ib%.2fr%.2f'%(self.exp_name, self.N, self.S, self.A, self.B, self.REWARD_BOUND)
        data_dir = os.path.join(self.home_dir, 'data')
        logger_kwargs = setup_logger_kwargs(self.exp_name, self.seed, data_dir=data_dir)
        # logger_kwargs = setup_logger_kwargs(self.exp_name, self.seed+add_to_seed, data_dir=data_dir)

        return self.best_response_per_cpu(nature_strats, nature_eq, add_to_seed, seed=self.seed, logger_kwargs=logger_kwargs, **self.nature_kwargs)

    # add_to_seed is obsolete
    def best_response_per_cpu(self, agent_strats, agent_eq, add_to_seed, ma_actor_critic=core.RMABWhittleNatureOracle, ac_kwargs=dict(), 
        seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr_agent=3e-4, pi_lr_nature=3e-4,
        vf_lr_agent=1e-3, vf_lr_nature=1e-3, qf_lr=1e-3, lm_lr=5e-2, 
        train_pi_iters=80, train_v_iters=80, train_q_iters=80,
        lam_OTHER=0.97, max_ep_len=1000,
        start_entropy_coeff=0.0, end_entropy_coeff=0.0,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10,
        lamb_update_freq=10,
        init_lambda_trains=0,
        final_train_lambdas=0):
        
        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        setup_pytorch_for_mpi()

        # Set up logger and save configuration
        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals())


        # Instantiate environment
        env = self.env
        
        obs_dim = env.observation_dimension
        action_dim_nature = env.action_dim_nature


        # Create actor-critic module
        # This becomes the basis of nature's policy
        ac = ma_actor_critic(env.observation_space, env.action_space, env.sampled_parameter_ranges,
             env.action_dim_nature, env=env,
             N = env.N, C = env.C, B = env.B, strat_ind = self.strat_ind,
             one_hot_encode = self.one_hot_encode, non_ohe_obs_dim = self.non_ohe_obs_dim,
            state_norm=self.state_norm, nature_state_norm=self.nature_state_norm,
            **ac_kwargs)
        # create Whittle index policy. This helps in computing optimal policy against nature's actions
        wh_policy = WhittlePolicy(env.N, env.S, env.B,
                                10, 0.9
                                 )
        # Define the observation and action dimensions
        act_dim_agent = ac.act_dim_agent
        act_dim_nature = ac.act_dim_nature
        obs_dim = ac.obs_dim
        
        o = env.reset()
        o = o.reshape(-1)

        # Get count of agent actions
        agent_pol = wh_policy

        with torch.no_grad():
            a_nature_mu = ac.pi_nature.mu_net(torch.as_tensor(o, dtype=torch.float32))
            a_nature_env_mu = ac.bound_nature_actions(a_nature_mu, state=o, reshape=True)

        wh_policy.note_env(env)
        wh_policy.learn(a_nature_env_mu)

        nature_pol = ac
        agent_counter = self.get_agent_counts(agent_pol, nature_pol, env, steps_per_epoch)

        # Sync params across processes
        sync_params(ac)


        # Set up experience buffer
        local_steps_per_epoch = int(steps_per_epoch / num_procs())

        buf = MA_RMABPPO_Whittle_Buffer(obs_dim, act_dim_agent, act_dim_nature, env.N, ac.act_type, local_steps_per_epoch, 
            one_hot_encode=self.one_hot_encode, gamma=gamma, lam_OTHER=lam_OTHER)
        FINAL_TRAIN_LAMBDAS = final_train_lambdas


        # Set up function for computing nature policy loss
        def compute_loss_pi_nature(data, entropy_coeff):
            obs, act, adv, logp_old = data['obs'], data['act_nature'], data['adv_nature'], data['logp_nature']

            if not ac.one_hot_encode:
                obs = obs/self.nature_state_norm

            # Policy loss
            pi, logp = ac.pi_nature(obs, act)
            ent = pi.entropy().mean()
            ratio = torch.exp(logp - logp_old)

            clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv

            loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

            # subtract entropy term since we want to encourage it 
            loss_pi -= entropy_coeff*ent


            # Useful extra info
            approx_kl = (logp_old - logp).mean().item()
            ent = pi.entropy().mean().item()
            clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
            clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
            pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

            return loss_pi, pi_info

        # Set up function for computing MA_RMABPPO nature's value fn loss
        # Note that nature value function takes agent action as input. This is supplied through data
        def compute_loss_v_nature(data):
            obs, ret, act_agent = data['obs'], data['ret_nature'], data['act_agent']
            # oha_agent = np.zeros(ac.act_dim_agent)
            # oha_agent[int(act_agent)] = 1
            if not self.one_hot_encode:
                obs = obs/self.nature_state_norm

            x_s_a_agent = torch.as_tensor(np.concatenate([obs, act_agent],axis=1), dtype=torch.float32)
            return ((ac.v_nature(x_s_a_agent) - ret)**2).mean()

        # Set up optimizer objects
        pi_nature_optimizer = Adam(ac.pi_nature.parameters(), lr=pi_lr_nature)
        vf_nature_optimizer = Adam(ac.v_nature.parameters(), lr=vf_lr_nature)


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


            # with lamb_update_freq, update the nature's params
            # TODO: Since lambda network doesn't exist, rename this to nature_update_freq
            if epoch%lamb_update_freq == 0 and epoch > 0:

                # UPDATE the nature policy
                entropy_coeff = 0.0

                # Train policy with multiple steps of gradient descent
                for i in range(train_pi_iters):
                    pi_nature_optimizer.zero_grad()
                    loss_pi_nature, pi_info_nature = compute_loss_pi_nature(data, entropy_coeff)
                    kl = mpi_avg(pi_info_nature['kl'])
                    # if kl > 1.5 * target_kl:
                    #     logger.log('Early stopping at step %d due to reaching max kl.'%i)
                    #     break
                    loss_pi_nature.backward()
                    mpi_avg_grads(ac.pi_nature)    # average grads across MPI processes
                    pi_nature_optimizer.step()
                    # print('nature pi loss ', loss_pi_nature)


                for i in range(train_v_iters):

                    vf_nature_optimizer.zero_grad()
                    loss_v_nature = compute_loss_v_nature(data)
                    loss_v_nature.backward()
                    mpi_avg_grads(ac.v_nature)    # average grads across MPI processes
                    vf_nature_optimizer.step()




        # Prepare for interaction with environment
        start_time = time.time()
        current_lamb = 0


        o, ep_actual_ret_agent, ep_actual_ret_nature, ep_lamb_adjusted_ret_agent, ep_lamb_adjusted_ret_nature, ep_len = env.reset(), 0, 0, 0, 0, 0
        o = o.reshape(-1)

        init_o = np.copy(o)

        losses = {'pi_agent': [], 'v_agent': [], 'pi_nature': [], 'v_nature': [],
         'r_agent_lam':[], 'r_nature_lam':[],
         'r_agent':[], 'r_nature':[],
         'epoch_lams':[],
         'a_nature_0_01':[], 'a_nature_1_01':[],
         'a_agent_prob_01':[], 'step_lams_01':[],
         'a_nature_0_10':[], 'a_nature_1_10':[],
         'a_agent_prob_10':[], 'step_lams_10':[],
         'a_nature_0_11':[], 'a_nature_1_11':[],
         'a_agent_prob_11':[], 'step_lams_11':[],

         }



        # Review: this number is now small as compared to 50 previously
        NUM_TEST_POLICY_RUNS = 20
        


        # Sample an agent policy
        # sometimes get negative values that are tiny e.g., -6.54284594e-18, just set them to 0
        # REVIEW: Currently we heuristically believe that sampling agent_oracle with agent mixed strategy distribution
        # REVIEW: will give us a mixed agent strategy in expectation. Might want to change this logic in future
        # TODO: Potentially include whittle policy mixing part here since agent policies are all whittle index policies
        agent_eq = np.array(agent_eq)
        agent_eq[agent_eq < 0] = 0
        agent_eq = agent_eq / agent_eq.sum()

        
        head_entropy_coeff_schedule = np.linspace(start_entropy_coeff, end_entropy_coeff, epochs)
        
        print('Learning Nature\'s Best Reponse Strategy')
        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(epochs):

            # At every learning epoch, resample opponent agent policy
            agent_pol = np.random.choice(agent_strats, p=agent_eq)
            print('\nChosen agent pol', agent_pol, '\n')
            
            # Set env's initial state to init_o
            # env.current_full_state = init_o
            env.current_count_state = init_o
            o = init_o

            ## To compute agent optimal policy, use nature's avg params
            with torch.no_grad():
                a_nature_mu = ac.pi_nature.mu_net(torch.as_tensor(o, dtype=torch.float32))
                a_nature_env_mu = ac.bound_nature_actions(a_nature_mu, state=o, reshape=True)

            # Learn Optimal Whittle Index Agent policy aginst nature avg params
            wh_policy.note_env(env)
            wh_policy.learn(a_nature_env_mu)

            # Run simulation for `local_steps_per_epoch` timesteps
            for t in range(local_steps_per_epoch):
                torch_o = torch.as_tensor(o, dtype=torch.float32)
                
                # ac's step function requires both world observation (for actor) and agent_policy's actions (for critic)
                # but we can only obtain agent_policy's actions after determining nature's action
                # so we split the ac.step into two parts, first pass some dummy agent_policy action
                a_agent_list_dummy = np.zeros(self.N)
                
                # We obtain nature's action
                a_nature, _, logp_nature, q_nature = ac.step(torch_o, a_agent_list_dummy)

                # Bound nature's actions within allowed range
                a_nature_env = ac.bound_nature_actions(a_nature, state=o, reshape=True)
                
                # Use optimal agent policy `wh_policy` and pass to env.step to obtain next state and reward
                # env.step function internally calls wh_policy to obtain cluster level and arm level actions
                # Also note that r_agent is reward of optimal agent
                next_o, r_agent, d, _ = env.step(None, a_nature_env, wh_policy)
                current_lamb = env.current_lamb

                # We can now obtain actual cluster level agent actions
                a_agent_list = env.clustered_actions

                # Use actual cluster level agent actions to obtain critic network output
                _, v_nature, _, _ = ac.step(torch_o, a_agent_list)

                next_o = next_o.reshape(-1)
            
                s = time.time()
                test_r_list = np.zeros(NUM_TEST_POLICY_RUNS)
                
 
                # TODO: Optimize the sampling (reduce 50 to 5? or 1??)

                for trial in range(NUM_TEST_POLICY_RUNS):
                    # Set env's state to current world state
                    # env.current_full_state = o
                    env.current_count_state = o
                    
                    # Compute opponent agent's actions for current env worlds state
                    a_test = agent_pol.act_test(torch_o)
                    # Compute reward of opponent agent_pol for the current world state
                    _, r_test, _, _ = env.step(a_test, a_nature_env, agent_pol)
                    test_r_list[trial] = r_test.sum()
                endt = time.time()



                # env.current_full_state = next_o
                env.current_count_state = next_o

                # Compute mean opponent reward
                r_test_mean = test_r_list.mean()

                # Reward for optimal whittle policy
                actual_r_agent = r_agent.sum()
                # Compute regret. This is nature's reward
                actual_r_nature = actual_r_agent - r_test_mean
                
                # Compute lambda adjusted nature reward
                
                # cost_vec = np.zeros(env.N)
                # for i in range(env.N):
                #     cost_vec[i] = env.C[a_agent[i]]

                # only using this reward for debugging training
                # we will store and manipulate raw rewards during training, i.e., r_agent
                # lamb_adjusted_r_agent = actual_r_agent - current_lamb*cost_vec.sum()

                # # but store lambda adjusted for nature oracle...

                lamb_adjusted_r_nature = actual_r_nature - current_lamb*env.B
                ## TODO: Use Current lambda from wh policy, replace cost vec with budget
                # print('lambda adjusted r nature ', lamb_adjusted_r_nature, actual_r_nature, actual_r_agent, r_test_mean)
                ep_actual_ret_agent += actual_r_agent
                # ep_lamb_adjusted_ret_agent += lamb_adjusted_r_agent

                ep_actual_ret_nature += actual_r_nature
                ep_lamb_adjusted_ret_nature += lamb_adjusted_r_nature

                ep_len += 1

                # save and log

                buf.store(o, a_agent_list, a_nature, lamb_adjusted_r_nature, v_nature, q_nature, logp_nature)
                logger.store(VVals_nature=v_nature)


                
                # Update obs (critical!)
                o = next_o

                timeout = ep_len == max_ep_len
                terminal = d or timeout
                epoch_ended = t==local_steps_per_epoch-1

                # Some extra buf adjustment at episode end
                if terminal or epoch_ended:
                    FINAL_ROLLOUT_LENGTH = 50
                    if epoch_ended and not(terminal):
                        # print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                        pass
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v_nature, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32), a_agent_list = np.zeros(self.N,dtype=int))

                        # rollout costs for an imagined 50 steps...
                        
                        last_costs = np.zeros((FINAL_ROLLOUT_LENGTH, env.N))
                        
                    else:
                        v_nature = 0
                        last_costs = np.zeros((FINAL_ROLLOUT_LENGTH, env.N))
                    # buf.finish_path(v_agent, last_costs, v_nature)

                    buf.finish_path(v_nature)
                    # only save EpRet / EpLen if trajectory finished
                    # if terminal:
                    logger.store(EpActualRetAgent=ep_actual_ret_agent, EpActualRetNature=ep_actual_ret_nature,
                     EpLambAdjRetNature=ep_lamb_adjusted_ret_nature, EpLen=ep_len)

                    # losses['r_agent_lam'].append(ep_lamb_adjusted_ret_agent)
                    losses['r_nature_lam'].append(ep_lamb_adjusted_ret_nature)

                    losses['r_agent'].append(ep_actual_ret_agent)
                    losses['r_nature'].append(ep_actual_ret_nature)

                    losses['epoch_lams'].append(current_lamb)

                    o, ep_actual_ret_agent, ep_actual_ret_nature, ep_lamb_adjusted_ret_agent, ep_lamb_adjusted_ret_nature, ep_len = env.reset(), 0, 0, 0, 0, 0
                    o = o.reshape(-1)
                    init_o = np.copy(o)


            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({'env': env}, None)

            # Perform MA_RMABPPO nature oracle's update!
            head_entropy_coeff = head_entropy_coeff_schedule[epoch]
            # Call update. Will internally use data from buf
            update(epoch, head_entropy_coeff)


            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpActualRetAgent', with_min_and_max=True)
            logger.log_tabular('EpActualRetNature', with_min_and_max=True)
            # logger.log_tabular('EpLambAdjRetAgent', with_min_and_max=True)
            logger.log_tabular('EpLambAdjRetNature', with_min_and_max=True)
            # logger.log_tabular('EpLambAdjRet', average_only=True)
            # logger.log_tabular('EpLen', average_only=True)
            # logger.log_tabular('VVals_agent', with_min_and_max=True)
            logger.log_tabular('VVals_nature', with_min_and_max=True)
            # logger.log_tabular('Lamb', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

        return ac

