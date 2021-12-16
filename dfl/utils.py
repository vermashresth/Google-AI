import numpy as np
import math

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import tqdm
import tensorflow as tf

from itertools import combinations

def nck(n, k):
    return math.factorial(n)/(math.factorial(k)*math.factorial(n-k))

def print_r(text, p):
    dice = np.random.random()
    if dice<p:
        print(text)

def getBenefsByCluster(cluster_id, cluster_ids):
    benefs = np.arange(len(cluster_ids))[np.array(cluster_ids)==cluster_id]
    return benefs


"""
DIFFERENTIALBLE TOP-K LAYER
"""
class DiffTopK(object):
    """
    A differentiable Top-k layer
    Based on the paper: https://proceedings.neurips.cc/paper/2020/file/ec24a54d62ce57ba93a531b460fa8d18-Paper.pdf
    """

    def __init__(self, k, epsilon=0.1, max_iter=200):
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
        bs, n, = scores.shape # Batch version
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

    def _compute(self, C, mu, nu):
        bs, n = self.bs, self.n
        m = 2 # This is the dimensionality of the optimal transport. For top-k, m=2 to denote 2 categories of mass (top-k or not). For sorted top-k, m should be set larger.

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

def getSoftTopk(a, k):
    bs, n = a.shape
    diffTopK = DiffTopK(k=k)
    gamma = diffTopK(-a)
    probs = gamma[:,:,0].numpy() * n

    selection = []
    for i in range(bs):
        selection.append(np.random.choice(a=n, p=probs[i], size=k, replace=False))
    return selection

