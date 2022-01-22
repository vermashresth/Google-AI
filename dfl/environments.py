import tensorflow as tf
import numpy as np
from dfl.utils import getSoftTopk
from armman.simulator import getTopk

EPS = 1e-4

class baseEnv:
    def __init__(self, N, T_data, R_data, n_states, n_actions):
        self.N = N
        self.T_data = T_data
        self.R_data = R_data
        self.n_states = n_states
        self.n_actions = n_actions
        self.check_shapes()

    def check_shapes(self):
        # Check if shape of transition probability params is correct
        assert self.T_data.shape==(len(self.T_data), self.n_states, self.n_actions, self.n_states), 'Transition Probabilities not of right shape'

    def takeActions(self, states, actions):
        # Vectorized multinomial for fast transitions
        def vec_multinomial(prob_matrix):
            if np.isnan(prob_matrix.sum()):
                raise
            s = prob_matrix.cumsum(axis=1)
            r = np.random.rand(prob_matrix.shape[0])
            k = (s < np.expand_dims(r, 1)).sum(axis=1)
            return k

        next_states = vec_multinomial(self.T_data[np.arange(self.N), states, np.array(actions).astype(int), :])
        return next_states

    def getRewards(self, states, actions):
        rewards = self.R_data[np.arange(self.N), states]
        # print('reward', np.sum(rewards))
        return rewards

    def getStartState(self):
        return np.random.multinomial(1, [1/self.n_states]*self.n_states, self.N).argmax(axis=1)

# class armmanEnv(baseEnv):
#     def __init__(self, N, T_data, R_data):
#         
#         # Review: Remember, in OPE sticthed, we use n_states from global config
#         n_states = 2
#         n_actions = 2
#         super().__init__(N, T_data, R_data, n_states, n_actions)
    
class generalEnv(baseEnv):
    def __init__(self, N, T_data, R_data):
        
        # Figure out n_states and n_actions from T_data
        n_states = T_data.shape[1]
        n_actions = T_data.shape[2]

        super().__init__(N, T_data, R_data, n_states, n_actions)
    
def POMDP2MDP(T_data, R_data, H):
    # This function converts the POMDP transition probabilities and rewards to MDP.
    N, m, _, _ = T_data.shape
    assert(R_data.shape[1] == m and R_data.shape[0] == N)

    # For loop version
    # new_T_data = np.zeros((N, m, H, 2, m, H)) # Will be reshaped to N, mH, 2, mH later
    # new_R_data = np.zeros((N, m, H))
    # for i in range(N):
    #     T0 = T_data[i][:,0,:]
    #     T1 = T_data[i][:,1,:]

    #     belief_states = T1 # row-wise [b1, b2, ..., bm]^T
    #     for h in range(H):
    #         # Passive action
    #         if h < H-1:
    #             new_T_data[i,np.arange(m),h,0,np.arange(m),h+1] = 1 # h -> h+1
    #         else:
    #             new_T_data[i,np.arange(m),h,0,np.arange(m),h] = 1 # stationary state -> stationary state

    #         # Active action
    #         new_T_data[i,:,h,1,:,0] = belief_states

    #         # Reward function
    #         new_R_data[i, :, h] = belief_states @ R_data[i]

    #         # Transition
    #         belief_states = belief_states @ T0

    # Tensorflow parallelized version
    passive_T_data = np.zeros((N, m, H, 1, m, H)) # Will be reshaped to N, mH, 1, mH later
    active_T_data_list = []
    R_data_list = []
    
    T0 = T_data[:,:,0,:] # [N, m, m] # batch transition matrices
    T1 = T_data[:,:,1,:] # [N, m, m]

    belief_states = T1 # batch of row-wise [b1, b2, ..., bm]^T
    for h in range(H):
        # Passive action
        if h < H-1:
            passive_T_data[:,np.arange(m),h,0,np.arange(m),h+1] = 1 # h -> h+1
        else:
            passive_T_data[:,np.arange(m),h,0,np.arange(m),h] = 1 # stationary state -> stationary state

        # Active action
        active_T_data_list.append(tf.reshape(belief_states, (N, m, 1, 1, m, 1)))

        # Reward function
        R_data_list.append(belief_states @ tf.reshape(R_data, (N,m,1)))

        # Transition
        belief_states = belief_states @ T0

    passive_T_data = tf.constant(passive_T_data.reshape(N, m*H, 1, m*H), dtype=tf.float32) # [N, m*H, 1, m*H]
    active_T_data_part = tf.concat(active_T_data_list, axis=2) # [N, m, H, 1, m, 1]
    active_T_data = tf.pad(active_T_data_part, paddings=[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, H-1]], mode='CONSTANT')
    active_T_data = tf.reshape(active_T_data, (N, m*H, 1, m*H))

    new_T_data = tf.concat([passive_T_data, active_T_data], axis=2)
    new_R_data = tf.reshape(tf.concat(R_data_list, axis=2), (N, m*H))

    # if not (np.all(np.sum(new_T_data, axis=-1) > np.ones((N, m*H, 2)) - EPS) and np.all(np.sum(new_T_data, axis=-1) < np.ones((N, m*H, 2)) + EPS)):
    #     print(np.sum(new_T_data, axis=-1))

    assert(np.all(np.sum(new_T_data, axis=-1) > np.ones((N, m*H, 2)) - EPS) \
            and np.all(np.sum(new_T_data, axis=-1) < np.ones((N, m*H, 2)) + EPS))

    new_T_data = new_T_data / np.sum(new_T_data, axis=-1, keepdims=True) # normalize to be exact

    return new_T_data, new_R_data


