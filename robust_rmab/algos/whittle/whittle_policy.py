import numpy as np
import tqdm

def get_reward(state, action, m, lambda_g=0):
    """
    Helper function for plan2() function below
    """
    if state[0] == "L":
        reward = 1.0
    else:
        reward = -1.0
        
    if action == 'N':
        reward += m
    if action == 'I':
        reward += lambda_g

    return reward
    
def convertAxis(T):
    '''
    convert T matrix from format: a, s, s' (where s=0 is bad state) --> 
                                  s, s', a (where s=0 is good state) 
    This function is needed because most of my simulator code uses the former
    format while the whittleIndex function below is coded up using the latter. 
    '''
    P=np.zeros_like(T)
    for a in range(2):
        for s in range(2):
            for ss in range(2):
                P[1-s,1-ss,a]=T[a,s,ss]
    return P

def getWhittleIndex(two_state_probs, timesteps, gamma, sleeping_constraint = True, lambda_g=0 ):
    '''
    Function that generates whittle indices given transition probabilities. 
    Inputs: 
      two_state_probs: Transition probability matrix with axes,
                      action, starting_state, final_state. 
                      Here State=0 means engaging state
      sleeping_constraint: If True, considers the frequency constraints 
                          (and the modified MDP)
    Outputs:
      A list of len 2, corresponding to whittle indices of [NE, E] states.
    '''
    two_state_probs=convertAxis(two_state_probs)

    aug_states = []
    for i in range(6):
        if i % 2 == 0:
            aug_states.append('L{}'.format(i // 2))
        else:
            aug_states.append('H{}'.format(i // 2))

    if sleeping_constraint:
        local_CONFIG = {
            'problem': {
                "orig_states": ['L', 'H'],
                "states": aug_states + ['L', 'H'],
                "actions": ["N", "I"],
            },
            "time_step": timesteps,
            "gamma": gamma,
        }
    else:
        local_CONFIG = {
            'problem': {
                "orig_states": ['L', 'H'],
                "states": ['L', 'H'],
                "actions": ["N", "I"],
            },
            "time_step": timesteps,
            "gamma": gamma,
        }

    v_values = np.zeros(len(local_CONFIG['problem']['states']))
    q_values = np.zeros((len(local_CONFIG['problem']['states']), \
                         len(local_CONFIG['problem']['actions'])))
    high_m_values = 1 * np.ones(len(local_CONFIG['problem']['states']))
    low_m_values = -1 * np.ones(len(local_CONFIG['problem']['states']))

    t_probs = np.zeros((len(local_CONFIG['problem']['states']), \
                        len(local_CONFIG['problem']['states']), \
                        len(local_CONFIG['problem']['actions'])))

    if sleeping_constraint:    
        t_probs[0 : 2, 2 : 4, 0] = two_state_probs[:, :, 0]
        t_probs[2 : 4, 4 : 6, 0] = two_state_probs[:, :, 0]
        t_probs[4 : 6, 6 : 8, 0] = two_state_probs[:, :, 0]
        t_probs[6 : 8, 6 : 8, 0] = two_state_probs[:, :, 0]

        t_probs[0 : 2, 2 : 4, 1] = two_state_probs[:, :, 0]
        t_probs[2 : 4, 4 : 6, 1] = two_state_probs[:, :, 0]
        t_probs[4 : 6, 6 : 8, 1] = two_state_probs[:, :, 0]
        t_probs[6 : 8, 0 : 2, 1] = two_state_probs[:, :, 1]
    else:
        t_probs = two_state_probs

    max_q_diff = np.inf
    prev_m_values, m_values = None, None
    while max_q_diff > 1e-5:
        prev_m_values = m_values
        m_values = (low_m_values + high_m_values) / 2
        if type(prev_m_values) != type(None) and \
        abs(prev_m_values - m_values).max() < 1e-20:
            break
        max_q_diff = 0
        v_values = np.zeros((len(local_CONFIG['problem']['states'])))
        q_values = np.zeros((len(local_CONFIG['problem']['states']), \
                             len(local_CONFIG['problem']['actions'])))
        delta = np.inf
        while delta > 0.0001:
            delta = 0
            for i in range(t_probs.shape[0]):
                v = v_values[i]
                v_a = np.zeros((t_probs.shape[2],))
                for k in range(v_a.shape[0]):
                    for j in range(t_probs.shape[1]):
                        rew = get_reward(local_CONFIG['problem']['states'][i], \
                                    local_CONFIG['problem']['actions'][k], \
                                    m_values[i], lambda_g)
                        v_a[k] += t_probs[i, j, k] * \
                        (rew + local_CONFIG["gamma"] * v_values[j])

                v_values[i] = np.max(v_a)
                delta = max([delta, abs(v_values[i] - v)])

        state_idx = -1
        for state in range(q_values.shape[0]):
            for action in range(q_values.shape[1]):
                for next_state in range(q_values.shape[0]):
                    rew = get_reward(local_CONFIG['problem']['states'][state], \
                                local_CONFIG['problem']['actions'][action], \
                                m_values[state], lambda_g)
                    q_values[state, action] += \
                    t_probs[state, next_state, action] * \
                    (rew + local_CONFIG["gamma"] * v_values[next_state] )
            # print(state, q_values[cluster, state, 0], \
            #q_values[cluster, state, 1])

        for state in range(q_values.shape[0]):
            if abs(q_values[state, 1] - q_values[state, 0]) > max_q_diff:
                state_idx = state
                max_q_diff = abs(q_values[state, 1] - q_values[state, 0])

        # print(q_values)
        # print(low_m_values, high_m_values)
        if max_q_diff > 1e-5 and q_values[state_idx, 0] < q_values[state_idx, 1]:
            low_m_values[state_idx] = m_values[state_idx]
        elif max_q_diff > 1e-5 and q_values[state_idx, 0] > q_values[state_idx, 1]:
            high_m_values[state_idx] = m_values[state_idx]

        # print(low_m_values, high_m_values, state_idx)
        # ipdb.set_trace()
    
    m_values = (low_m_values + high_m_values) / 2

    #return q_values, m_values
    return [m_values[-1], m_values[-2]]

def getFastWhittleIndex(P, gamma):
    n_actions = 2
    N, n_states = P.shape[0], P.shape[1]

    R = np.array([[0, 1]]*N)
    # Just for clarity, add another variable n_wh_states which represents 
    # the states for which we want to calculate whittle indices. It is same as n_states
    n_wh_states = n_states

    tmp_P, tmp_R = P, R

    tmp_P = tmp_P.reshape(1, N, n_states, n_actions, n_states).repeat(n_wh_states, axis=0)

    # initialize upper and lower bounds for binary search
    w_ub = np.ones((n_wh_states, N))  # Whittle index upper bound
    w_lb = -np.ones((n_wh_states, N)) # Whittle index lower bound
    w = (w_ub + w_lb) / 2

    n_binary_search_iters = 10 # Using a fixed # of iterations or a tolerance rate instead
    n_value_iters = 100

    # Run binary search for finding whittle index corresponding to each wh_state
    for _ in range(n_binary_search_iters):
        w = (w_ub + w_lb) / 2

        # initialize value function
        V = np.zeros((n_wh_states, N, n_states))
        for _ in range(n_value_iters): # value iteration to update V

            # V is originally of shape N x n_states
            # repeat V to make it of same shape as tmp_P, i.e., N x n_states x n_actions x n_states
            V_mat = V.reshape((n_wh_states, N, 1, 1, n_states)).repeat(n_states, axis=2).repeat(n_actions, axis=3)
            
            # tmp_R is originally of shape N x n_states
            # repeat reward matrix to make it of same shape as tmp_P
            rewards_mat = tmp_R.reshape((1, N, n_states, 1, 1)).repeat(n_wh_states, axis=0).repeat(n_actions, axis=3).repeat(n_states, axis=4)

            # w is originally of shape n_states x N
            # repeat w to make it of shape n_states x N x n_states x n_states to be added to passive action in reward_mat
            w_mat = w.reshape((n_wh_states, N, 1, 1)).repeat(n_states, axis=2).repeat(n_states, axis=3)

            # Add subsidy to passive action
            rewards_mat[:, :, :, 0, :]  = rewards_mat[:, :, :, 0, :] + w_mat 

            # Compute discounted future rew
            out_mat = tmp_P * (rewards_mat + gamma*V_mat)
            Q = np.sum(out_mat, axis=4) # Q is aggregated over new_state
            V = np.max(Q, axis=3) # Find V as max over actions

        # Compute an indicator vector to mark if Whittle index is too large or too small
        # comparison = (value of not call > value of call) # a vector of size N to indicate if w is too large or not
        passive_q_vals = Q[..., 0]
        active_q_vals = Q[..., 1]
        # Return comparision of shape n_states x N
        comparison = passive_q_vals[np.arange(n_wh_states).reshape(-1, 1).repeat(N, axis=1),
                                    np.arange(N).reshape(1, -1).repeat(n_wh_states, axis=0),
                                    np.arange(n_wh_states).reshape(-1, 1).repeat(N, axis=1)] > \
                     active_q_vals[np.arange(n_wh_states).reshape(-1, 1).repeat(N, axis=1),
                                    np.arange(N).reshape(1, -1).repeat(n_wh_states, axis=0),
                                    np.arange(n_wh_states).reshape(-1, 1).repeat(N, axis=1)]  
            
        # TODO: Might want to update whittle indices for only those (arms, states) having abs(q_diff) more than a Q_delta  
        # Update lower and upper bounds of whittle index binary search
        w_ub = w_ub - (w_ub - w_lb) / 2 * comparison
        w_lb = w_lb + (w_ub - w_lb) / 2 * (1 - comparison)
    w = (w_ub + w_lb) / 2
    return w.T


def getTopk(a, k):
    '''
    Returns indices of top k elements in array a
    '''
    a=np.array(a)
    return a.argsort()[-k:][::-1]


class WhittlePolicy():
    def __init__(self, N, S, B, timesteps, gamma):
        self.N = N
        self.S = S
        self.B = B
        self.horizon = timesteps
        self.gamma = gamma
        self.id = np.random.uniform(0,1)
        n_clusters = self.N // self.S
        self.whittle_indices = np.zeros((n_clusters, S))
    
    def note_env(self, env):
        self.T = env.T
        self.env = env

    def learn(self, a_nature):
        for arm_i in range(a_nature.shape[0]):
            for arm_s in range(a_nature.shape[1]):
                for arm_a in range(a_nature.shape[2]):

                    param = a_nature[arm_i, arm_s, arm_a]
                    # if param < self.sampled_parameter_ranges[arm_i, arm_state, arm_a, 0]:
                    if param < self.env.sampled_parameter_ranges[arm_i, arm_s, arm_a, 0]:

                        # print("Warning! nature action below allowed param range. Was %s but should be in %s"%(param, self.sampled_parameter_ranges[arm_i, arm_state, arm_a]))
                        print("Setting to lower bound of range...")
                        # print('arm state',arm_state)
                        # param = self.sampled_parameter_ranges[arm_i, arm_state, arm_a, 0]
                        param = self.env.sampled_parameter_ranges[arm_i, arm_s, arm_a, 0]
                    # elif param > self.sampled_parameter_ranges[arm_i, arm_state, arm_a, 1]:
                    elif param > self.env.sampled_parameter_ranges[arm_i, arm_s, arm_a, 1]:
                        # print("Warning! nature action above allowed param range. Was %s but should be in %s"%(param, self.sampled_parameter_ranges[arm_i, arm_state, arm_a]))
                        print("Setting to upper bound of range...")
                        # print('arm state',arm_state)
                        # param = self.sampled_parameter_ranges[arm_i, arm_state, arm_a, 1]
                        param = self.env.sampled_parameter_ranges[arm_i, arm_s, arm_a, 1]

                    # self.T[arm_i,arm_state,arm_a,0] = param
                    # self.T[arm_i,arm_state,arm_a,1] = 1-param

                    self.T[arm_i,arm_s, arm_a,0] = param
                    self.T[arm_i,arm_s, arm_a,1] = 1-param
        
        # self.cluster_T = (self.T[::self.S] + self.T[1::self.S])/2
        self.cluster_T = np.copy(self.T)
        self.n_clusters = self.env.n_clusters
        # self.whittle_indices = np.array([getWhittleIndex(self.T[i], self.horizon, self.gamma)\
        #      for i in range(self.N)])
        print('Computing Whittle Indices!!!')
        # self.whittle_indices = np.array([getWhittleIndex(self.cluster_T[i], self.horizon, self.gamma)\
        #      for i in tqdm.tqdm(range(self.n_clusters))])
        self.whittle_indices = getFastWhittleIndex(self.cluster_T, self.gamma)

        # self.whittle_indices = np.random.uniform(0, 1*np.sum(a_nature), (self.N, self.S))

    def act_test_indiv(self, observations):
        if isinstance(observations, np.ndarray):
            observations_array = observations.astype(int)
        else:
            observations_array = observations.numpy().astype(int)
        whittle_indices = self.whittle_indices[np.arange(self.N), observations_array]
        actions = np.zeros(self.N)
        actions[getTopk(whittle_indices, int(self.B))] = 1
        actions = actions.astype(int)
        assert np.sum(actions)==self.B
        return actions
    
    def act_test_cluster_to_indiv(self, cluster_mapping, observations, actual_B):
        if isinstance(observations, np.ndarray):
            observations_array = observations.astype(int)
        else:
            observations_array = observations.numpy().astype(int)
        cluster_whittle_index = self.whittle_indices[cluster_mapping]
        self.n_arms = len(cluster_mapping)
        whittle_indices = cluster_whittle_index[np.arange(self.n_arms), observations_array]
        actions = np.zeros(self.n_arms)
        actions[getTopk(whittle_indices, int(actual_B))] = 1
        try:
            self.current_lamb = (whittle_indices[getTopk(whittle_indices, int(self.B))[-1]]+\
                whittle_indices[getTopk(whittle_indices, int(self.B))[-2]])/2
        except:
            self.current_lamb = 0
        actions = actions.astype(int)
        assert np.sum(actions)==actual_B
        return actions

    def act_test(self, observations):
        actions = np.zeros(self.N)
        return actions
    
    def __str__(self):
        # Give a unique name to whittle policy using ranking of whittle indices
        # return f'Whittle Policy {np.round(self.whittle_indices.flatten(), 1)}-{np.argsort(self.whittle_indices.flatten())}'
        top_3_rank = [str(i) for i in np.argsort(self.whittle_indices.flatten())[:3]]
        str_top_3_rank = '_'.join(top_3_rank)
        return 'Whittle Policy '+str_top_3_rank
