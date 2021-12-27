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
    w_lb = -np.ones((N, n_states)) # Whittle index lower bound
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
    
    # Loop over each state
    w_list = []
    for state in range(n_states):
        # Define batch of indicator matrix `A`
        indicator_mat = np.zeros((N, n_states+1, 2*n_states))

        # Since indicator is dependent on both argmax action and state,
        # obtain batch indices and states correponding to every element in
        # action_max_Q array
        batch_indices, s_primes = np.meshgrid(np.arange(action_max_Q.shape[0]),
                                             np.arange(action_max_Q.shape[1]),
                                             indexing='ij')
        # Define a function which returns marked index given scaler batch idx, state, argmax action
        def indicate_index_fn(batch_idx, s, a):
            if a==0: # corresponds to first equation
                return (batch_idx, s, 2*s)
            else:
                return (batch_idx, s, 2*s+1)
            # returns three different vectors containing indices for the three dimensions

        # Vectrize the indicate function
        vec_indicate_index_fn = np.vectorize(indicate_index_fn)

        # We will get three vectors as output correponding to the indices of three dimensions
        # of indicator_mat which have to be marked as 1
        output_index_seq = vec_indicate_index_fn(batch_indices.flatten(),
                                    s_primes.flatten(),
                                    action_max_Q.flatten())
        indicator_mat[output_index_seq] = 1

        # For m+1 th entry, take inverse of entry for s = `state`
        # Review: This assumes binary action 0 or 1
        last_a_argmax = np.logical_not(action_max_Q[:, state]).astype(int) # Note action not selected earlier for s = state
        output_last_index_seq = vec_indicate_index_fn(np.arange(N), # all batch indices
                                                      np.array([state]*N), # same s=state for all batches
                                                      last_a_argmax) # inverted argmax actions of s=state for all batches
        
        indicator_mat[output_last_index_seq[0], # indices for batch dimension
                      np.array([n_states]*N), # same index (last row) for all batches
                      output_last_index_seq[2] # indices for chosen argmax
                      ] = 1

        ## Build the rhs matrix for solving linear equations
        rhs = tf.repeat(-1*R, 2, axis=1) # obtain vector [-r(0), -r(0), -r(1), -r(1), ...]
        rhs = tf.expand_dims(rhs, axis=2) # outputs rhs of shape N x 2*n_states x 1
        rhs = tf.matmul(tf.constant(indicator_mat, dtype=tf.float32),
                        rhs) # filter the rhs using indicator matrix

        ## Build lhs matrix for solving linear equations

        # obtain P_matrix containing transition probabilities * gamma
        lhs_P_mat = gamma*tf.reshape(P, (N, n_actions*n_states, n_states))

        # Generate matrix to subtract from tranmsition probabilities depending on state
        # Generate a matrix having 1 for [s, s] and 0 otherwise 
        lhs_sub_matrix = np.eye(n_states)
        # Generate a matrix having 1 for [2s, s] and [2s+1, s] and 0 otherwise
        lhs_sub_matrix = np.repeat(lhs_sub_matrix, 2, axis=0)
        # Generate a batch of matrices having 1 for [2s, s] and [2s+1, s] and 0 otherwise
        lhs_sub_matrix = np.broadcast_to(lhs_sub_matrix, (N, 2*n_states, n_states))
        
        # subtract the sub_matrix
        lhs_P_mat = lhs_P_mat - tf.constant(lhs_sub_matrix, dtype=tf.float32)

        # generate the first column of lhs matrix, it should be column vector [1, 0, 1, 0, ...] 
        # broadcasted to N batches
        lhs_one_zero_matrix = np.broadcast_to([1, 0], (N, n_states, 2)).reshape(N, 2*n_states, 1)

        # concatenate one zero column vector to p_matrix to obtain lhs matrix
        lhs_mat = tf.concat([tf.constant(lhs_one_zero_matrix, dtype=tf.float32), lhs_P_mat],
                      axis=2)
        # Filter lhs mat using indicator matrix
        lhs = tf.matmul(tf.constant(indicator_mat, dtype=tf.float32),
                        lhs_mat)

        # Solve batch of linear equations
        solution = tf.linalg.solve(lhs, rhs) # matrix of shape N, m+1
        w_state = solution[:, :1, 0] # vector of shape N, 1
        w_list.append(w_state)

    # Part 3: reformulating Whittle index computation as a solution to a linear equation.
    #         Since now we know which argmax holds in Bellman equation, we can express Whittle index as a solution to a linear equation.
    #         We will keep tracking gradient in this part and recompute the Whittle index again to allow Tensorflow to backpropagate.

    # Express w, V as a solutions to linear equations.
    # w = tf.linalg.solve(compute_the_chosen_matrix(P, R, gamma, indicator_function), rhs)

    return tf.concat(w_list, axis=1)

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

