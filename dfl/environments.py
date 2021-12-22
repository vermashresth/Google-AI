import numpy as np
from dfl.utils import getSoftTopk
from armman.simulator import getTopk

class baseEnv:
    def __init__(self, N, k, w, T_data, n_states, n_actions, seed):
        self.N = N
        self.k = k
        self.w = w
        self.T_data = T_data
        self.n_states = n_states
        self.n_actions = n_actions
        np.random.seed(seed)
        self.check_shapes()

    def check_shapes(self):
        # Check if shapes of whittle indices and 
        # transition probability params are correct
        assert self.w.shape==(len(self.w), self.n_states), 'Whittle Indices not of right shape'
        assert self.T_data.shape==(len(self.T_data), self.n_states, self.n_actions, self.n_states), 'Transition Probabilities not of right shape'

    def takeActions(self, states, actions):
        # Vectorized multinomial for fast transitions
        def vec_multinomial(prob_matrix):
            s = prob_matrix.cumsum(axis=1)
            r = np.random.rand(prob_matrix.shape[0])
            k = (s < np.expand_dims(r, 1)).sum(axis=1)
            return k

        next_states = vec_multinomial(self.T_data[np.arange(self.N), states, np.array(actions).astype(int), :])
        return next_states
    
    def getStartState(self):
        return np.random.multinomial(1, [1/self.n_states]*self.n_states, self.N).argmax(axis=1)

class armmanEnv(baseEnv):
    def __init__(self, N, k, w, T_data, seed):
        
        # Review: Remember, in OPE sticthed, we use n_states from global config
        n_states = 2
        n_actions = 2
        super().__init__(N, k, w, T_data, n_states, n_actions, seed)
    
    def getRewards(self, states):
        # Env specific reward function
        return np.copy(states)


class dummy3StatesEnv(baseEnv):
    def __init__(self, N, k, w, T_data, seed):
        
        n_states = 3
        n_actions = 2
        super().__init__(N, k, w, T_data, n_states, n_actions, seed)
    
    def getRewards(self, states):
        # Env specific reward function
        return np.copy(states)

class generalEnv(baseEnv):
    def __init__(self, N, k, w, T_data, seed):
        
        # Figure out n_states and n_actions from T_data
        n_states = T_data.shape[1]
        n_actions = T_data.shape[2]

        super().__init__(N, k, w, T_data, n_states, n_actions, seed)
    
    def getRewards(self, states):
        # Env specific reward function
        return np.copy(states)