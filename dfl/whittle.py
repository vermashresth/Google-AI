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
def get_reward(R, state, action, m):
    rewards = np.copy(R[:, state])
    if not action: # Passive action
        rewards += m # Add subsidy
    return rewards
    
def newWhittleIndex(P, R, gamma=0.99):
    '''
    Inputs:
        P: Transition matrix of dimensions N X n_states X 2 X n_states where axes are:
          batchsize(N), start_state, action, end_state

        R: Rewards of the corresponding states (0,1,...,n_states-1)
           R is a matrix of size N X n_states
           The rewards of each batch is not neccesary identical.

        gamma: Discount factor

        Tensorflow should keep track of the gradient of matrix P and R.

    Returns:
        index: N x n_states Tensor of Whittle index for states (0,1,...,n_states-1)
    '''

    # Part 1: disable gradient tracking and use value iteration and binary search to find Whittle index
    #         This part should support parallelization
    #         This part should always disable tracking gradient of P and R by using "tf.stop_gradient(P)"
    n_actions = 2
    N, n_states = P.shape[0], P.shape[1]
    tmp_P, tmp_R = tf.stop_gradient(P), tf.stop_gradient(R)

    # initialize upper and lower bounds
    w_ub = np.ones((N, n_states))  # Whittle index upper bound
    w_lb = np.zeros((N, n_states)) # Whittle index lower bound
    w = (w_ub + w_lb) / 2

    n_binary_search_iters = 20 # Using a fixed # of iterations or a tolerance rate instead
    n_value_iters = 100

    for _ in range(n_binary_search_iters):
        w = (w_ub + w_lb) / 2
        # initialize value function
        V = np.zeros((N, n_states))
        action_max_Q = np.zeros((N, n_states)) # vector to store which of action results in max Q
        for _ in range(n_value_iters): # value iteration to update V
            # Reset Q values in every value iteration
            Q = np.zeros((N, n_states, n_actions)) 
            # Iterate over state, action, new_state
            for state in range(n_states):
                for action in range(n_actions):
                    for new_state in range(n_states):
                        # Compute reward given the subsidy
                        rew = get_reward(tmp_R, state, action, w[:, state]) # vector of size N
                        # Update Q value for the current state and action
                        Q[:, state, action] += tmp_P[:, state, action, new_state] * \
                            (rew + gamma*V[:, new_state])
                # Update Value by taking max of Q over actions
                V[:, state] = np.max(Q[:, state, :], axis=1)
                # Note the action which resulted in max Q value
                action_max_Q[:, state] = np.argmax(Q[:, state, :], axis=1)

        # Compute an indicator vector to mark if Whittle index is too large or too small
        # comparison = (value of not call > value of call) # a vector of size N to indicate if w is too large or not
        comparison = (Q[:, :, 0] > Q[:, :, 1]).astype(int)
        
        # TODO: Might want to update whittle indices for only those (arms, states) having abs(q_diff) more than a Q_delta  
        # Update lower and upper bounds of whittle index binary search
        w_ub = w_ub - (w_ub - w_lb) / 2 * comparison
        w_lb = w_lb + (w_ub - w_lb) / 2 * (1 - comparison)

    # outcome: w = (w_ub + w_lb) / 2
    w = (w_ub + w_lb) / 2


    # action_max_Q stores the information mentioned below
    # Part 2: figure out which set of argmax in the Bellman equation holds.
    #         V[state] = argmax 0 + w + sum_{next_state} P_passive[state, next_state] V[next_state]
    #                           R[state] + sum_{next_state} P_active[state, next_state] V[next_state]
    # In total, there are n_states-1 argmax, each with 2 actions. 
    # You can maintain a vector of size (N, n_states, ) to mark which one in the argmax holds.
    # This vector will later be used to formulate the corresponding linear equation that Whittle should satisfy.


    # Part 3: reformulating Whittle index computation as a solution to a linear equation.
    #         Since now we know which argmax holds in Bellman equation, we can express Whittle index as a solution to a linear equation.
    #         We will keep tracking gradient in this part and recompute the Whittle index again to allow Tensorflow to backpropagate.

    # Express w, V as a solutions to linear equations.
    # w = tf.linalg.solve(compute_the_chosen_matrix(P, R, gamma, indicator_function), rhs)

    return w

def whittleIndex(P, gamma=0.99):
    '''
    Inputs:
        P: Transition matrix of dimensions N X 2 X 2 X 2 where axes are:
          [Old]     batchsize(N), action, start_state, end_state
          [Updated] batchsize(N), start_state, action, end_state

        gamma: Discount factor

    Returns:
        index: NX2 Tensor of Whittle index for states (0,1)

    '''
    N=int(tf.shape(P)[0])

    ### Matrix equations for state 0
    row1_s0=tf.concat([tf.ones([N,1]), tf.reshape(gamma*P[:, 0,0,0], shape=[N,1]) -tf.ones([N,1]),  tf.reshape(gamma*P[:, 0,0,1], shape=[N,1])], 1)
    row2_s0=tf.concat([tf.zeros([N,1]), tf.reshape(gamma*P[:, 0,1,0], shape=[N,1]) -tf.ones([N,1]),  tf.reshape(gamma*P[:, 0,1,1], shape=[N,1])], 1)
    row3a_s0=tf.concat([tf.ones([N,1]), tf.reshape(gamma*P[:, 1,0,0], shape=[N,1]) ,  tf.reshape(gamma*P[:, 1,0,1], shape=[N,1])-tf.ones([N,1])], 1)
    row3b_s0=tf.concat([tf.zeros([N,1]), tf.reshape(gamma*P[:, 1,1,0], shape=[N,1]) ,  tf.reshape(gamma*P[:, 1,1,1], shape=[N,1])-tf.ones([N,1])], 1)

    A1_s0= tf.concat([tf.reshape(row1_s0, shape=[N,1,3]),
                  tf.reshape(row2_s0, shape=[N, 1,3]),
                  tf.reshape(row3a_s0, shape=[N, 1,3])],1)

    A2_s0= tf.concat([tf.reshape(row1_s0, shape=[N,1,3]),
                  tf.reshape(row2_s0, shape=[N,1,3]),
                  tf.reshape(row3b_s0, shape=[N,1,3])],1)
    b_s0=tf.constant(np.array([0,0,-1]).reshape(3,1), dtype=tf.float32)


    ### Matrix equations for state 1
    row1_s1=tf.concat([tf.ones([N,1]), tf.reshape(gamma*P[:, 1,0,0], shape=[N,1]) ,  tf.reshape(gamma*P[:, 1,0,1], shape=[N,1])-tf.ones([N,1])], 1)
    row2_s1=tf.concat([tf.zeros([N,1]), tf.reshape(gamma*P[:, 1,1,0], shape=[N,1]) ,  tf.reshape(gamma*P[:, 1,1,1], shape=[N,1])-tf.ones([N,1])], 1)
    row3a_s1=tf.concat([tf.ones([N,1]), tf.reshape(gamma*P[:, 0,0,0], shape=[N,1]) -tf.ones([N,1]) ,  tf.reshape(gamma*P[:, 0,0,1], shape=[N,1])], 1)
    row3b_s1=tf.concat([tf.zeros([N,1]), tf.reshape(gamma*P[:, 0,1,0], shape=[N,1]) -tf.ones([N,1]) ,  tf.reshape(gamma*P[:, 0,1,1], shape=[N,1])], 1)

    A1_s1= tf.concat([tf.reshape(row1_s1, shape=[N,1,3]),
                  tf.reshape(row2_s1, shape=[N,1,3]),
                  tf.reshape(row3a_s1, shape=[N,1,3])],1)

    A2_s1= tf.concat([tf.reshape(row1_s1, shape=[N,1,3]),
                  tf.reshape(row2_s1, shape=[N,1,3]),
                  tf.reshape(row3b_s1, shape=[N,1,3])],1)

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
    cand1_s0_mask= tf.constant(1.0*(c1s0[:, 0] + 1.0 + gamma*(Pnp[:,1,0,0]*c1s0[:,1] + Pnp[:,1,0,1]*c1s0[:,2]) >= 1.0+ gamma* (Pnp[:,1,1,0]*c1s0[:,1] + Pnp[:,1,1,1]*c1s0[:,2])), dtype=tf.float32)
    cand1_s1_mask= tf.constant(1.0*(c1s1[:, 0] + gamma*(Pnp[:,0,0,0]*c1s0[:,1] + Pnp[:,0,0,1]*c1s0[:,2]) >=  gamma* (Pnp[:,1,0,0]*c1s0[:,1] + Pnp[:,1,0,1]*c1s0[:,2])), dtype=tf.float32)

    cand2_s0_mask= (1.0- cand1_s0_mask)
    cand2_s1_mask= (1.0- cand1_s1_mask)

    return tf.concat([tf.reshape(cnd1_s0[:, 0]*cand1_s0_mask + cnd2_s0[:, 0]*cand2_s0_mask, shape=[N,1]), tf.reshape(cnd1_s1[:, 0]*cand1_s1_mask + cnd2_s1[:, 0]*cand2_s1_mask, shape=[N,1])], 1)

