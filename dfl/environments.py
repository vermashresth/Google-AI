import numpy as np
from dfl.utils import getSoftTopk
from armman.simulator import getTopk
from dfl.config import E_START_STATE_PROB_ARMMAN

class baseEnv:
    def __init__(self, N, T_data, seed):
        self.N = N
        self.T_data = T_data
        np.random.seed(seed)

    def takeActions(self, states, actions):
        def vec_multinomial(prob_matrix):
            s = prob_matrix.cumsum(axis=1)
            r = np.random.rand(prob_matrix.shape[0])
            k = (s < np.expand_dims(r, 1)).sum(axis=1)
            return k

        next_states = vec_multinomial(self.T_data[np.arange(self.N), states, np.array(actions).astype(int), :])
        return next_states

class armmanEnv(baseEnv):
    def __init__(self, N, k, w, T_data, seed):
        self.N = N
        self.k = k
        
        # Review: Remember, in OPE, we use n_states from global config
        self.n_states = 2
        self.n_actions = 2

        assert w.shape==(len(w), self.n_states), 'Whittle Indices not of right shape'
        self.w = w
        np.random.seed(seed)
        assert T_data.shape==(len(T_data), self.n_states, self.n_actions, self.n_states), 'Transition Probabilities not of right shape'
        super().__init__(N, T_data, seed)

    def getStartState(self):
        return np.random.binomial(1, E_START_STATE_PROB_ARMMAN, size=self.N)

    def getActions(self, states, policy, ts):
        actions=np.zeros(self.N)

        if policy == 0:
            # Select k arms at random
            actions[np.random.choice(np.arange(self.N), self.k)] = 1

        elif policy == 1:
            # Select k arms in round robin
            actions[[(ts*self.k+i)%self.N for i in range(self.k)]] = 1

        elif policy == 2:
            # select k arms by Whittle
            whittle_indices=self.w[np.arange(self.N), states]

            top_k_whittle=getTopk(whittle_indices, self.k)
            actions[top_k_whittle]=1

        elif policy == 3:
            # select k arms by Whittle using soft top k
            whittle_indices=self.w[np.arange(self.N), states]

            soft_top_k_whittle=getSoftTopk([whittle_indices], self.k)
            actions[soft_top_k_whittle]=1

        return actions
    
    def getRewards(self, states):
        return np.copy(states)


class dummy3StatesEnv(baseEnv):
    def __init__(self, N, k, w, T_data, seed):
        self.N = N
        self.k = k
        self.n_states = 3
        self.n_actions = 2
        
        self.w = w
        
        assert T_data.shape==(len(T_data), self.n_states, self.n_actions, self.n_states), 'Transition Probabilities not of right shape'

        np.random.seed(seed)
        super().__init__(N, T_data, seed)

    def getStartState(self):
        return np.random.multinomial(1, [1/self.n_states]*self.n_states, self.N ).argmax(axis=1)

    def getActions(self, states, policy, ts):
        actions=np.zeros(self.N)

        if policy == 0:
            # Select k arms at random
            actions[np.random.choice(np.arange(self.N), self.k)] = 1

        elif policy == 1:
            # Select k arms in round robin
            actions[[(ts*self.k+i)%self.N for i in range(self.k)]] = 1

        elif policy == 2:
            # select k arms by Whittle
            whittle_indices=self.w[np.arange(self.N), states]

            top_k_whittle=getTopk(whittle_indices, self.k)
            actions[top_k_whittle]=1

        elif policy == 3:
            # select k arms by Whittle using soft top k
            whittle_indices=self.w[np.arange(self.N), states]

            soft_top_k_whittle=getSoftTopk([whittle_indices], self.k)
            actions[soft_top_k_whittle]=1

        return actions
    
    def getRewards(self, states):
        return np.copy(states)