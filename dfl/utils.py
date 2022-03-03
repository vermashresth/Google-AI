import numpy as np
import math

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import tqdm
import tensorflow as tf
import sys
sys.path.insert(0, "../")

from itertools import combinations

from dfl.config import N_ACTIONS, N_STATES, dim_dict, policy_map

def nck(n, k):
    return math.factorial(n)/(math.factorial(k)*math.factorial(n-k))

def print_r(text, p):
    dice = np.random.random()
    if dice<p:
        print(text)

def getBenefsByCluster(cluster_id, cluster_ids):
    benefs = np.arange(len(cluster_ids))[np.array(cluster_ids)==cluster_id]
    return benefs

def aux_dict_to_transition_matrix(aux_dict_ssa, n_benefs):
    T_data = np.zeros((n_benefs, N_STATES, N_ACTIONS, N_STATES))
    for benef in aux_dict_ssa.keys():
        sub_dict = aux_dict_ssa[benef]['s_a_s_prime_dict']
        keys = [k for k in sub_dict.keys() if isinstance(k, tuple)]
        for (s, a) in keys:
            for s_prime in sub_dict[(s, a)]:
                prob = sub_dict[(s, a)][s_prime]
                T_data[benef, s, a, s_prime] = prob
    T_data[np.isnan(T_data)] = 1/N_STATES
    return T_data

def getRandomProbabilityDistribution(m):
    random_points = np.random.uniform(size=m-1)
    random_points = np.append(random_points, [0,1])
    sorted_points = sorted(random_points)
    diffs = np.diff(sorted_points)
    assert np.sum(diffs) == 1 and len(diffs) == m
    return diffs

def generateRandomTMatrix(n_benefs, n_states, R_data):

    """
    This function is to replace the function in armman/simulator.py to support multiple states

    Generates a n_benefs x n_states x 2 x n_states T matrix indexed as: \
    T[beneficiary_number][current_state][action][next_state]
    action=0 denotes passive action, a=1 is active action
    """
        
    T = np.zeros((n_benefs, n_states, 2, n_states))
    for i in range(n_benefs):
        for j in range(n_states):
            while True:
                passive_transition = getRandomProbabilityDistribution(n_states)
                active_transition  = getRandomProbabilityDistribution(n_states)
                if active_transition @ R_data > passive_transition @ R_data + 0.2: # Ensure that calling is significantly better
                    T[i,j,0,:] = passive_transition
                    T[i,j,1,:] = active_transition
                    break
    return T  


def takeActions(states, T, actions):
    '''
    This function is to replace the function takeAction in ARMMAN/simulator.py
    This function supports multiple states (more than 2)

    Inputs:
    states: A vector of size N with integer values {0,1,...,m-1} respresenting the states\
          of beneficiaries
    T: Transition matrix of size Nx2xmxm (num_beneficiaries x starting_state x \
          action x ending_state)
    actions: A vector of size N with integer values {0,1,...,m-1} representing action \
          chosen for each beneficiary

    Outputs:
    next_states:  A vector of size N with integer values {0,1,...,m-1} respresenting the\
          *next* states of beneficiaries, determined using coin tosses
    '''

    N, m = T.shape[0], T.shape[2] # T: [N, m, 2, m]
    next_states = np.zeros(N)
    for i in range(N):
        next_states[i] = np.random.choice(a=m, size=1, p=T[i,states[i],actions[i],:])
    return next_states.astype('int64')

def twoStageNLLLoss(traj, prediction, policy):
    n_tr, T, n_benefs = traj.shape[0], traj.shape[2], traj.shape[4]
    
    s = traj[:, # trial index
                                policy_map[policy], # policy index
                                :T-1, # time index
                                dim_dict['state'], # tuple dimension
                                : # benef index
                                ].astype(int).flatten()
    a = traj[:, # trial index
                                policy_map[policy], # policy index
                                :T-1, # time index
                                dim_dict['action'], # tuple dimension
                                : # benef index
                                ].astype(int).flatten()
    s_prime = traj[:, # trial index
                                policy_map[policy], # policy index
                                :T-1, # time index
                                dim_dict['next_state'], # tuple dimension
                                : # benef index
                                ].astype(int).flatten()
    
    benef_idx = np.arange(n_benefs).reshape(1, -1).repeat(n_tr*(T-1), axis=0).flatten()
    indices = list(zip(benef_idx, s, a, s_prime))
    trans_probs = tf.gather_nd(prediction, indices)
    return -tf.reduce_sum(tf.math.log(trans_probs)) / n_tr

"""
DIFFERENTIALBLE TOP-K LAYER
"""
class DiffTopK(object):
    """
    A differentiable Top-k layer
    Based on the paper: https://proceedings.neurips.cc/paper/2020/file/ec24a54d62ce57ba93a531b460fa8d18-Paper.pdf
    """

    def __init__(self, k, epsilon=0.1, max_iter=100):
        """Construct a Top-k Layer
        Args:v 
          k: 
          max_iter: 
        """
        self.k=k
        self.epsilon=epsilon
        self.anchors=tf.constant([0,1], shape=(1,2), dtype=tf.float32)
        self.max_iter=max_iter


    def __call__(self, scores):
        """ Computes the Top-k and gradient function
        Args:
          scores: a sequence of 'n' tf.Tensors corresponding to the 'n' \
                  whittle indices.
        Returns: a vector of size ()

        """
        bs, n = scores.shape # Batch version
        self.bs, self.n = bs, n
        scores = tf.reshape(scores, [bs, n, 1]) 

        # Preprocessing scores vector
        # find the -inf value and replace it with the minimum value except -inf
        # scores_ = tf.identity(scores).numpy() # copy
        # max_scores = tf.reduce_max(scores)
        # scores_[scores_==float('-inf')] = float('inf')
        # min_scores = tf.reduce_min(scores) 
        # filled_value = min_scores-(max_scores-min_scores)
        # mask = scores==float('-inf')
        # print(mask)

        ## Replacement for masked_fill() below:
        # scores_ = tf.identity(scores)
        # scores_[mask.numpy()] = filled_value
        # scores = tf.convert_to_tensor(scores_)

        C = (scores-self.anchors)**2 #TODO: Fix tf syntax
        C = C / (tf.reduce_max(C))

        mu = tf.ones((1,n,1)) /n
        nu = tf.constant([self.k/n, (n-self.k)/n], shape=(1,1,2), dtype=tf.float32)

        compute = tf.custom_gradient(lambda x,y,z: self._compute(x,y,z))
        return compute(C, mu, nu)

    def _compute(self, C_raw, mu, nu):
        bs, n = self.bs, self.n
        m = 2 # This is the dimensionality of the optimal transport. For top-k, m=2 to denote 2 categories of mass (top-k or not). For sorted top-k, m should be set larger.
        C = tf.stop_gradient(C_raw)

        f = tf.zeros([bs, n, 1])
        g = tf.zeros([bs, 1, m])

        # m = 2 in this simplified version
        # mu: [n, 1]
        # nu: [1, m]
        # f:  [n, 1] this is the dual variable 
        # g:  [1, m] this is the dual variable
        # Gamma: [n, m]

        # Computing Gamma

        def min_epsilon_row(Z, epsilon):
            return -epsilon*tf.reduce_logsumexp(Z / epsilon, axis=-1, keepdims=True)
            
        def min_epsilon_col(Z, epsilon):
            return -epsilon*tf.reduce_logsumexp(Z / epsilon, axis=-2, keepdims=True)
    
        for i in range(self.max_iter):
            f = min_epsilon_row(-C+g, self.epsilon) + self.epsilon*tf.math.log(mu)
            g = min_epsilon_col(-C+f, self.epsilon) + self.epsilon*tf.math.log(nu)
        f = min_epsilon_row(-C+g, self.epsilon) + self.epsilon*tf.math.log(mu)
 
        Gamma = tf.math.exp((-C+f+g) / self.epsilon)
        solution = Gamma[:,:,:-1] # [bs, n, m-1] i.e. [bs, n, 1]

        def gradient_function(dsoln):
            nu_ = nu[:,:,:-1] # [1, 1, m-1], or the first entry of nu
            inv_mu = 1./tf.reshape(mu, (1, n, 1))  # [1, n, 1]

            # TODO
            Kappa = tf.linalg.diag(tf.reshape(nu_, (1,m-1))) - (tf.transpose(solution, [0,2,1]) * tf.reshape(inv_mu, (1,1,n))) @ solution # [bs, m-1, m-1] This should be a list of scalars 
            padding_value = 1e-10
            inv_Kappa = tf.linalg.inv(Kappa + padding_value) # [bs, m-1, m-1] list of scalars
            mu_Gamma_Kappa = (inv_mu * solution) @ inv_Kappa # [bs, n, m-1]
            H1 = tf.linalg.diag(tf.reshape(inv_mu, (1,n))) + mu_Gamma_Kappa @ (tf.transpose(solution, [0,2,1]) * tf.reshape(inv_mu, (1,1,n))) # [bs, n, n]
            H2 = -mu_Gamma_Kappa # [bs, n, m-1]
            H3 = tf.transpose(H2, [0,2,1]) # [bs, m-1, n]
            H4 = inv_Kappa # [bs, m-1, m-1]

            H2_pad = tf.pad(H2, paddings=[[0,0], [0,0], [0,1]], mode="CONSTANT")
            H4_pad = tf.pad(H4, paddings=[[0,0], [0,0], [0,1]], mode="CONSTANT")

            grad_f_C = tf.expand_dims(H1, axis=-1) * tf.expand_dims(Gamma, axis=-3) \
                      + tf.expand_dims(H2_pad, axis=-2) * tf.expand_dims(Gamma, axis=-3) # [bs,n,n,m]
            grad_g_C = tf.expand_dims(H3, axis=-1) * tf.expand_dims(Gamma, axis=-3) \
                      + tf.expand_dims(H4_pad, axis=-2) * tf.expand_dims(Gamma, axis=-3) # [bs,m,n,m]

            grad_g_C_pad = tf.pad(grad_g_C, paddings=[[0,0], [0,1], [0,0], [0,0]], mode="CONSTANT")
            grad_C1 = dsoln * Gamma # [bs,n,m]
            grad_C2 = tf.reduce_sum(tf.reshape(grad_C1, (bs,n,m,1,1)) * tf.expand_dims(grad_f_C, axis=-3), axis=[1,2])
            grad_C3 = tf.reduce_sum(tf.reshape(grad_C1, (bs,n,m,1,1)) * tf.expand_dims(grad_g_C_pad, axis=-4), axis=[1,2])

            grad_C = (-grad_C1 + grad_C2 + grad_C3) / self.epsilon

            return grad_C, None, None

        return tf.stop_gradient(Gamma), gradient_function

def getSoftTopk(a, k, epsilon=0.1):
    bs, n = a.shape
    diffTopK = DiffTopK(k=k, epsilon=epsilon)
    gamma = diffTopK(-a).numpy()
    probs = gamma[:,:,0] / np.sum(gamma[:,:,0], axis=1, keepdims=True)

    selection = []
    for i in range(bs):
        selection.append(np.random.choice(a=n, p=probs[i], size=k, replace=False))
    return selection

