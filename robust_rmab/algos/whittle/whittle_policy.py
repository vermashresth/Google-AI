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
        self.whittle_indices = np.zeros(self.N)
    
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
        self.whittle_indices = np.array([getWhittleIndex(self.cluster_T[i], self.horizon, self.gamma)\
             for i in tqdm.tqdm(range(self.n_clusters))])

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
        self.current_lamb = (whittle_indices[getTopk(whittle_indices, int(self.B))[-1]]+\
            whittle_indices[getTopk(whittle_indices, int(self.B))[-2]])/2
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
