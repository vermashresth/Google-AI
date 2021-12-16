import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import tqdm 
import tensorflow as tf

from itertools import combinations

"""
DIFFERENTIALBLE WHITTLE INDEX LAYER
"""
def whittleIndex(P, gamma=0.99):
    '''
    Inputs:
        P: Transition matrix of dimensions N X 2 X 2 X 2 where axes are:
          batchsize(N), action, start_state, end_state

        gamma: Discount factor

    Returns:
        index: NX2 Tensor of Whittle index for states (0,1)

    '''
    N=int(tf.shape(P)[0])

    ### Matrix equations for state 0
    row1_s0=tf.concat([tf.ones([N,1]), tf.reshape(gamma*P[:, 0,0,0], shape=[N,1]) -tf.ones([N,1]),  tf.reshape(gamma*P[:, 0,0,1], shape=[N,1])], 1)
    row2_s0=tf.concat([tf.zeros([N,1]), tf.reshape(gamma*P[:, 1,0,0], shape=[N,1]) -tf.ones([N,1]),  tf.reshape(gamma*P[:, 1,0,1], shape=[N,1])], 1)
    row3a_s0=tf.concat([tf.ones([N,1]), tf.reshape(gamma*P[:, 0,1,0], shape=[N,1]) ,  tf.reshape(gamma*P[:, 0,1,1], shape=[N,1])-tf.ones([N,1])], 1)
    row3b_s0=tf.concat([tf.zeros([N,1]), tf.reshape(gamma*P[:, 1,1,0], shape=[N,1]) ,  tf.reshape(gamma*P[:, 1,1,1], shape=[N,1])-tf.ones([N,1])], 1)

    A1_s0= tf.concat([tf.reshape(row1_s0, shape=[N,1,3]),
                  tf.reshape(row2_s0, shape=[N, 1,3]),
                  tf.reshape(row3a_s0, shape=[N, 1,3])],1)

    A2_s0= tf.concat([tf.reshape(row1_s0, shape=[N,1,3]),
                  tf.reshape(row2_s0, shape=[N, 1,3]),
                  tf.reshape(row3b_s0, shape=[N, 1,3])],1)
    b_s0=tf.constant(np.array([0,0,-1]).reshape(3,1), dtype=tf.float32)


    ### Matrix equations for state 1
    row1_s1=tf.concat([tf.ones([N,1]), tf.reshape(gamma*P[:, 0,1,0], shape=[N,1]) ,  tf.reshape(gamma*P[:, 0,1,1], shape=[N,1])-tf.ones([N,1])], 1)
    row2_s1=tf.concat([tf.zeros([N,1]), tf.reshape(gamma*P[:, 1,1,0], shape=[N,1]) ,  tf.reshape(gamma*P[:, 1,1,1], shape=[N,1])-tf.ones([N,1])], 1)
    row3a_s1=tf.concat([tf.ones([N,1]), tf.reshape(gamma*P[:, 0,0,0], shape=[N,1]) -tf.ones([N,1]) ,  tf.reshape(gamma*P[:, 0,0,1], shape=[N,1])], 1)
    row3b_s1=tf.concat([tf.zeros([N,1]), tf.reshape(gamma*P[:, 1,0,0], shape=[N,1]) -tf.ones([N,1]) ,  tf.reshape(gamma*P[:, 1,0,1], shape=[N,1])], 1)

    A1_s1= tf.concat([tf.reshape(row1_s1, shape=[N,1,3]),
                  tf.reshape(row2_s1, shape=[N, 1,3]),
                  tf.reshape(row3a_s1, shape=[N, 1,3])],1)

    A2_s1= tf.concat([tf.reshape(row1_s1, shape=[N,1,3]),
                  tf.reshape(row2_s1, shape=[N, 1,3]),
                  tf.reshape(row3b_s1, shape=[N, 1,3])],1)

    b_s1=tf.constant(np.array([-1, -1, 0]).reshape(3,1), dtype=tf.float32)


    ### Compute candidates
    cnd1_s0 = tf.reshape((tf.linalg.inv(A1_s0) @ b_s0), shape=[-1, 3])
    cnd2_s0 = tf.reshape((tf.linalg.inv(A2_s0) @ b_s0), shape=[-1,3])

    cnd1_s1 = tf.reshape((tf.linalg.inv(A1_s1) @ b_s1), shape=[-1, 3])
    cnd2_s1 = tf.reshape((tf.linalg.inv(A2_s1) @ b_s1), shape=[-1, 3])

    ## Create a mask to select the correct candidate:
    c1s0= cnd1_s0.numpy()
    c2s0= cnd2_s0.numpy()
    c1s1= cnd1_s1.numpy()
    c2s1= cnd2_s1.numpy()
    Pnp=P.numpy()

    ## Following line implements condition checking when candidate1 is correct
    ## It results in an array of size N, with value 1 if candidate1 is correct else 0.
    cand1_s0_mask= tf.constant(1.0*(c1s0[:, 0] + 1.0 + gamma*(Pnp[:,0,1,0]*c1s0[:,1] + Pnp[:,0,1,1]*c1s0[:,2]) >= 1.0+ gamma* (Pnp[:,1,1,0]*c1s0[:,1] + Pnp[:,1,1,1]*c1s0[:,2])), dtype=tf.float32)
    cand1_s1_mask= tf.constant(1.0*(c1s1[:, 0] + gamma*(Pnp[:,0,0,0]*c1s0[:,1] + Pnp[:,0,0,1]*c1s0[:,2]) >=  gamma* (Pnp[:,1,0,0]*c1s0[:,1] + Pnp[:,1,0,1]*c1s0[:,2])), dtype=tf.float32)

    cand2_s0_mask= (1.0- cand1_s0_mask)
    cand2_s1_mask= (1.0- cand1_s1_mask)

    return tf.concat([tf.reshape(cnd1_s0[:, 0]*cand1_s0_mask + cnd2_s0[:, 0]*cand2_s0_mask, shape=[N,1]), tf.reshape(cnd1_s1[:, 0]*cand1_s1_mask + cnd2_s1[:, 0]*cand2_s1_mask, shape=[N,1])], 1)

