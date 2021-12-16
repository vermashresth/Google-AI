import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, "/mnt/chromeos/GoogleDrive/SharedWithMe/436/Decision Focused Learning/")

from dfl.model import SyntheticANN
from dfl.trajectory import getSimulatedTrajectories


def generateDataset(n_benefs, n_instances, n_trials, L, K, gamma):
    # n_benefs: number of beneficiaries in a cohort
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
        T_data = generateTransitionProbability(n_benefs)
        feature = model(tf.constant(T_data.reshape(-1,8))).numpy()
        w = np.zeros((n_benefs, 2)) # Not running Whittle policy so this in not important

        sim_seed = i  # just a randomness
        mask_seed = i # just a randomness
        traj, sim_whittle, simulated_rewards, mask, \
                state_record, action_record = getSimulatedTrajectories(
                                                n_benefs, L, K, n_trials, gamma,
                                                sim_seed, mask_seed, T_data, w, replace=False
                                                )

        instance = (feature, T_data, traj, simulated_rewards, mask, state_record, action_record)
        dataset.append(instance)

    return dataset


def generateTransitionProbability(n_benefs):
    # Generate T_data
    transition_probability_list = []
    for _ in range(n_benefs):
        probs = np.zeros((2,2,2))
        
        # sampling feasible probabilities
        prob_candidates = sorted(np.random.uniform(size=4))
        probs[0,0,1] = prob_candidates[0] # smallest
        probs[1,1,1] = prob_candidates[3] # largest
        if np.random.randint(2) == 0:
            probs[0,1,1] = prob_candidates[1]
            probs[1,0,1] = prob_candidates[2]
        else:
            probs[0,1,1] = prob_candidates[2]
            probs[1,0,1] = prob_candidates[1]

        probs[:,:,0] = 1 - probs[:,:,1]

        transition_probability_list.append(probs)

    return np.array(transition_probability_list)

if __name__ == '__main__':
    # Testing data generation
    n_benefs = 10
    n_instances = 20
    n_trials = 10
    L = 10
    K = 3

    T_data = generateTransitionProbability(n_benefs)
    dataset = generateDataset(n_benefs, n_instances, n_trials, L, K)
