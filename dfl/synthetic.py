import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, '../')

from dfl.model import SyntheticANN
from dfl.trajectory import getSimulatedTrajectories
from dfl.utils import generateRandomTMatrix
from dfl.whittle import whittleIndex, newWhittleIndex
from dfl.environments import POMDP2MDP
from dfl.ope import opeSimulator

def generateDataset(n_benefs, n_states, n_instances, n_trials, L, K, gamma, env='general', H=10, run_whittle=False):
    # n_benefs: number of beneficiaries in a cohort
    # n_states: number of states
    # n_instances: number of cohorts in the whole dataset
    # n_trials: number of trajectories we obtained from each cohort. In the real-world dataset, n_trials=1
    # L: number of time steps
    # K: budget
    # gamma: discount factor

    # Generate T_data => ANN => features
    # Return a tensorflow dataset that includes (features, traj, T_data) as an instance.
    dataset = []

    # Randomly initializing a neural network
    model = SyntheticANN()

    # Prepare for parallelization
    state_matrix = np.zeros((n_instances, L, n_benefs))
    action_matrix = np.zeros((n_instances, L, n_benefs))

    # Generating synthetic data
    for i in range(n_instances):
        # Generate transition probabilities
        raw_T_data = generateRandomTMatrix(n_benefs, n_states=n_states) # numpy array

        # Generate rewards from uniform distribution
        R = sorted(np.random.uniform(size=n_states))
        R = (R - np.min(R)) / np.ptp(R) # normalize rewards
        raw_R_data = np.repeat(R.reshape(1,-1), n_benefs, axis=0) # using the same rewards across all arms (for simplicity)

        # Generate features using the transition probabilities
        feature = model(tf.constant(raw_T_data.reshape(-1,2*n_states*n_states), dtype=tf.float32))

        # Generate environment parameters
        if env=='general':
            T_data, R_data = raw_T_data, raw_R_data
        elif env=='POMDP':
            T_data, R_data = POMDP2MDP(tf.constant(raw_T_data, dtype=tf.float32), tf.constant(raw_R_data, dtype=tf.float32), H)
            T_data, R_data = T_data.numpy(), R_data.numpy()
            # print('raw R', raw_R_data)
            # print('new R', R_data)

        # Different choices of Whittle indices
        if run_whittle:
            # w = whittleIndex(tf.constant(T_data, dtype=tf.float32)).numpy() # Old Whittle index computation. This only works for n_states=2.
            w = newWhittleIndex(tf.constant(T_data, dtype=tf.float32), \
                tf.constant(R_data, dtype=tf.float32)).numpy() # New Whittle index computation. It should work for multiple states.
        else:
            w = np.zeros((n_benefs, T_data.shape[1])) # All zeros. This is to disable Whittle index policy to speed up simulation.
        
        assert w.shape == (n_benefs, T_data.shape[1])

        sim_seed = i  # just a randomness
        mask_seed = i # just a randomness
        traj, sim_whittle, simulated_rewards, mask, \
                state_record, action_record, reward_record = getSimulatedTrajectories(
                                                                n_benefs=n_benefs, T=L, K=K, n_trials=n_trials, gamma=gamma,
                                                                seed=sim_seed, mask_seed=mask_seed, T_data=T_data, R_data=R_data,
                                                                w=w, replace=False, select_full=True
                                                                )

        OPE_sim_n_trials = 100
        ope_simulator = opeSimulator(traj, mask_seed, n_benefs, L, K, n_states, OPE_sim_n_trials, gamma, beh_policy_name='random', env=env, H=H)

        # print('real T data:', T_data[0])
        # print('empirical T data:', ope_simulator.emp_T_data[0])

        instance = (feature, raw_T_data, raw_R_data, traj, ope_simulator, simulated_rewards, mask, state_record, action_record, reward_record)
        print('average simulated rewards (random, rr, whittle):', np.mean(simulated_rewards, axis=0))
        dataset.append(instance)

    return dataset

def loadSyntheticData(n_benefs=100, n_states=2, n_trials=10, L=10, K=10, gamma=0.99):
    T_data = tf.constant(generateRandomTMatrix(n_benefs, n_states=n_states), dtype=tf.float32)
    w = whittleIndex(T_data)
    cluster_ids = np.arange(len(T_data)) # one cluster per person

    return T_data.numpy(), w.numpy(), None, cluster_ids

if __name__ == '__main__':
    # Testing data generation
    n_benefs = 50
    n_instances = 20
    n_trials = 100
    L = 10
    K = 3
    n_states = 2
    gamma = 0.99
    env = 'POMDP'
    H = 10

    T_data = generateRandomTMatrix(n_benefs, n_states=n_states)
    dataset = generateDataset(n_benefs, n_states, n_instances, n_trials, L, K, gamma=gamma, env=env, H=H, run_whittle=False)
