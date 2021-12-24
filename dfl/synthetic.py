import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, '../')

from dfl.model import SyntheticANN
from dfl.trajectory import getSimulatedTrajectories
from dfl.utils import generateRandomTMatrix
from dfl.whittle import whittleIndex

def generateDataset(n_benefs, n_states, n_instances, n_trials, L, K, gamma):
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
        T_data = generateRandomTMatrix(n_benefs, n_states=n_states) # numpy array
        feature = model(tf.constant(T_data.reshape(-1,2*n_states*n_states), dtype=tf.float32))
        # w = np.zeros((n_benefs, n_states)) # Not running Whittle policy so this in not important
        w = whittleIndex(tf.constant(T_data, dtype=tf.float32)).numpy()
        assert w.shape == (n_benefs, n_states)

        sim_seed = i  # just a randomness
        mask_seed = i # just a randomness
        traj, sim_whittle, simulated_rewards, mask, \
                state_record, action_record = getSimulatedTrajectories(
                                                n_benefs=n_benefs, T=L, K=K, n_trials=n_trials, gamma=gamma,
                                                seed=sim_seed, mask_seed=mask_seed, T_data=T_data, w=w, replace=False
                                                )

        instance = (feature, T_data, traj, simulated_rewards, mask, state_record, action_record)
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
    n_trials = 10
    L = 10
    K = 3
    n_states = 2
    gamma = 0.99

    T_data = generateRandomTMatrix(n_benefs, n_states=n_states)
    dataset = generateDataset(n_benefs, n_states, n_instances, n_trials, L, K, gamma=gamma)