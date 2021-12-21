import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import tqdm 

from itertools import combinations


"""
BLACK BOX: This cell is a black box from Lovish that reads/parses the\
beneficiary data and returns arrays corresponding to beneficiary IDs,\ 
cluster assignments, transition probs, whittle indices
"""

def loadBeneficiaryData():
  """
  This function cd's to the folder where the relevant data pickle files are \
  stored. Reads and returns:

  rmab_grou_probs:
    Numpy matrix storing transition probabilites of size Nx2x2x2 with axes \
    num_beneficiariesX action X start_state x end_state
    N: num of beneficiaries
    action: 0= passive, 1=active
    start/end state: 0= Not engaging 1=Engaging

  rmab_group_whittle_indices:
    Numpy array of size Nx2. i-th entry corresponds to the whittle indices\
    for states [NE, E] for beneficiary i
  
  engagement_matrix: 
    Dictionary with keys: ['rmab', 'round_robin', 'control']. 
    engagement_matrix[key] is a NxL numpy matrix storing binary engagement data
    N=num of beneficiaries
    L=num of timesteps  
  """
#   from google.colab import drive
#   drive.mount('/content/drive')
  '''
  This line reads files present in my drive. For it to work on your system,
  please make sure to have the necessary files present in your cwd.
  '''
  ## Loading Whittle Policy related files
  aug_states = []
  for i in range(6):
      if i % 2 == 0:
          aug_states.append('E{}'.format(i // 2))
      else:
          aug_states.append('NE{}'.format(i // 2))

  CONFIG = {
      "problem": {
          "orig_states": ['E', 'NE'],
          "states": aug_states + ['E', 'NE'],
          "actions": ["A", "I"],
      },
      "time_step": 7,
      "gamma": 0.99
  }

#   %cd drive/My\ Drive/ARMMAN/code/ 
  with open('policy_dump.pkl', 'rb') as fr:
    pilot_user_ids, pilot_static_features, cls, cluster_transition_probabilities, m_values, q_values = pickle.load(fr)
  fr.close()

  #engagement_matrix_file='full_matrix.pkl'
#   engagement_matrix_file='full_matrix_week11_end.pkl'
#   with open(engagement_matrix_file, 'rb') as fr:
#     engagement_matrix = pickle.load(fr)
#   fr.close()
  engagement_matrix = None

  cluster_assignments = cls.predict(pilot_static_features)

  rmab_group_results = pd.read_csv('outputs/pilot_outputs/rmab_pilot.csv')
  rmab_user_ids = rmab_group_results['user_id'].to_list()

  rmab_group_probs, rmab_group_whittle_indices = [], []
  cluster_ids = []

  for idx, user_id in enumerate(rmab_user_ids):
    locate_idx = np.where(pilot_user_ids == user_id)[0][0]
    curr_cluster = cluster_assignments[locate_idx]
    cluster_ids.append(curr_cluster)
    whittle_indices = m_values[curr_cluster]

    start_state = rmab_group_results[rmab_group_results['user_id'] == user_id]['start_state'].item()

    t_probs = np.zeros((len(CONFIG['problem']['states']), len(CONFIG['problem']['states']), len(CONFIG['problem']['actions'])))
    two_state_probs = np.zeros((2, 2, 2))
    for i in range(two_state_probs.shape[0]):
      for j in range(two_state_probs.shape[1]):
        for k in range(two_state_probs.shape[2]):
          s = CONFIG['problem']['orig_states'][i]
          s_prime = CONFIG['problem']['orig_states'][j]
          a = CONFIG['problem']['actions'][k]
          #two_state_probs[i, j, k] = cluster_transition_probabilities.loc[cluster_transition_probabilities['cluster']==curr_cluster, "P(" + s + ", " + a + ", " + s_prime + ")"]
          two_state_probs[k, int(1-i), int(1-j)] = cluster_transition_probabilities.loc[cluster_transition_probabilities['cluster']==curr_cluster, "P(" + s + ", " + a + ", " + s_prime + ")"]
          # The indices are adjusted to return transition matrix in the format T[action, start_state, end_state] where state=0 is bad state and state=1 is good state
          
    t_probs[0 : 2, 2 : 4, 0] = two_state_probs[0, :, :]
    t_probs[2 : 4, 4 : 6, 0] = two_state_probs[0, :, :]
    t_probs[4 : 6, 6 : 8, 0] = two_state_probs[0, :, :]
    t_probs[6 : 8, 6 : 8, 0] = two_state_probs[0, :, :]

    t_probs[0 : 2, 2 : 4, 1] = two_state_probs[0, :, :]
    t_probs[2 : 4, 4 : 6, 1] = two_state_probs[0, :, :]
    t_probs[4 : 6, 6 : 8, 1] = two_state_probs[0, :, :]
    t_probs[6 : 8, 0 : 2, 1] = two_state_probs[1, :, :]

    rmab_group_probs.append(two_state_probs)
    user_whittle_idx = rmab_group_results[rmab_group_results['user_id'] == user_id]['whittle_index'].item()
    #rmab_group_whittle_indices.append(user_whittle_idx)
    rmab_group_whittle_indices.append([whittle_indices[-1],whittle_indices[-2]]) ### whittle_indeces[-1] is for NE and whittle_indices[-2] is for E

  rmab_group_probs = np.array(rmab_group_probs)
  rmab_group_whittle_indices = np.array(rmab_group_whittle_indices)
  return rmab_group_probs, rmab_group_whittle_indices, engagement_matrix, cluster_ids



def get_reward(state, action, m, lambda_g=0):
    """
    Helper function for plan2() function below
    """
    if state[0] == "L":
        reward = 1.0
    else:
        reward = -1.0
        
    if action == 'N':
        reward += m
    if action == 'I':
        reward += lambda_g

    return reward

def convertAxis(T):
    '''
    convert T matrix from format: a, s, s' (where s=0 is bad state) --> 
                                  s, s', a (where s=0 is good state) 
    This function is needed because most of my simulator code uses the former
    format while the whittleIndex function below is coded up using the latter. 
    '''
    P=np.zeros_like(T)
    for a in range(2):
        for s in range(2):
            for ss in range(2):
                P[1-s,1-ss,a]=T[a,s,ss]
    return P

def getWhittleIndex(two_state_probs, sleeping_constraint = True, lambda_g=0 ):
    '''
    Function that generates whittle indices given transition probabilities. 
    Inputs: 
      two_state_probs: Transition probability matrix with axes,
                      action, starting_state, final_state. 
                      Here State=0 means engaging state
      sleeping_constraint: If True, considers the frequency constraints 
                          (and the modified MDP)
    Outputs:
      A list of len 2, corresponding to whittle indices of [NE, E] states.
    '''
    two_state_probs=convertAxis(two_state_probs)

    aug_states = []
    for i in range(6):
        if i % 2 == 0:
            aug_states.append('L{}'.format(i // 2))
        else:
            aug_states.append('H{}'.format(i // 2))

    if sleeping_constraint:
        local_CONFIG = {
            'problem': {
                "orig_states": ['L', 'H'],
                "states": aug_states + ['L', 'H'],
                "actions": ["N", "I"],
            },
            "time_step": 7,
            "gamma": 0.99,
        }
    else:
        local_CONFIG = {
            'problem': {
                "orig_states": ['L', 'H'],
                "states": ['L', 'H'],
                "actions": ["N", "I"],
            },
            "time_step": 7,
            "gamma": 0.99,
        }

    v_values = np.zeros(len(local_CONFIG['problem']['states']))
    q_values = np.zeros((len(local_CONFIG['problem']['states']), \
                         len(local_CONFIG['problem']['actions'])))
    high_m_values = 1 * np.ones(len(local_CONFIG['problem']['states']))
    low_m_values = -1 * np.ones(len(local_CONFIG['problem']['states']))

    t_probs = np.zeros((len(local_CONFIG['problem']['states']), \
                        len(local_CONFIG['problem']['states']), \
                        len(local_CONFIG['problem']['actions'])))

    if sleeping_constraint:    
        t_probs[0 : 2, 2 : 4, 0] = two_state_probs[:, :, 0]
        t_probs[2 : 4, 4 : 6, 0] = two_state_probs[:, :, 0]
        t_probs[4 : 6, 6 : 8, 0] = two_state_probs[:, :, 0]
        t_probs[6 : 8, 6 : 8, 0] = two_state_probs[:, :, 0]

        t_probs[0 : 2, 2 : 4, 1] = two_state_probs[:, :, 0]
        t_probs[2 : 4, 4 : 6, 1] = two_state_probs[:, :, 0]
        t_probs[4 : 6, 6 : 8, 1] = two_state_probs[:, :, 0]
        t_probs[6 : 8, 0 : 2, 1] = two_state_probs[:, :, 1]
    else:
        t_probs = two_state_probs

    max_q_diff = np.inf
    prev_m_values, m_values = None, None
    while max_q_diff > 1e-5:
        prev_m_values = m_values
        m_values = (low_m_values + high_m_values) / 2
        if type(prev_m_values) != type(None) and \
        abs(prev_m_values - m_values).max() < 1e-20:
            break
        max_q_diff = 0
        v_values = np.zeros((len(local_CONFIG['problem']['states'])))
        q_values = np.zeros((len(local_CONFIG['problem']['states']), \
                             len(local_CONFIG['problem']['actions'])))
        delta = np.inf
        while delta > 0.0001:
            delta = 0
            for i in range(t_probs.shape[0]):
                v = v_values[i]
                v_a = np.zeros((t_probs.shape[2],))
                for k in range(v_a.shape[0]):
                    for j in range(t_probs.shape[1]):
                        rew = get_reward(local_CONFIG['problem']['states'][i], \
                                    local_CONFIG['problem']['actions'][k], \
                                    m_values[i], lambda_g)
                        v_a[k] += t_probs[i, j, k] * \
                        (rew + local_CONFIG["gamma"] * v_values[j])

                v_values[i] = np.max(v_a)
                delta = max([delta, abs(v_values[i] - v)])

        state_idx = -1
        for state in range(q_values.shape[0]):
            for action in range(q_values.shape[1]):
                for next_state in range(q_values.shape[0]):
                    rew = get_reward(local_CONFIG['problem']['states'][state], \
                                local_CONFIG['problem']['actions'][action], \
                                m_values[state], lambda_g)
                    q_values[state, action] += \
                    t_probs[state, next_state, action] * \
                    (rew + local_CONFIG["gamma"] * v_values[next_state] )
            # print(state, q_values[cluster, state, 0], \
            #q_values[cluster, state, 1])

        for state in range(q_values.shape[0]):
            if abs(q_values[state, 1] - q_values[state, 0]) > max_q_diff:
                state_idx = state
                max_q_diff = abs(q_values[state, 1] - q_values[state, 0])

        # print(q_values)
        # print(low_m_values, high_m_values)
        if max_q_diff > 1e-5 and q_values[state_idx, 0] < q_values[state_idx, 1]:
            low_m_values[state_idx] = m_values[state_idx]
        elif max_q_diff > 1e-5 and q_values[state_idx, 0] > q_values[state_idx, 1]:
            high_m_values[state_idx] = m_values[state_idx]

        # print(low_m_values, high_m_values, state_idx)
        # ipdb.set_trace()
    
    m_values = (low_m_values + high_m_values) / 2

    #return q_values, m_values
    return [m_values[-1], m_values[-2]]


"""This cell contains all utils-like helper functions"""

def getTopk(a, k):
    '''
    Returns indices of top k elements in array a
    '''
    a=np.array(a)
    return a.argsort()[-k:][::-1]

def verify_T_matrix(T):
    '''
    Checks whether it satisfies the natural constraints assumed in the problem \
    (active is better than passive etc.)
    '''
    valid = True
    valid &= T[0, 0, 1] <= T[0, 1, 1] #non-oscillate condition
    valid &= T[1, 0, 1] <= T[1, 1, 1] #must be true for active as well
    valid &= T[0, 1, 1] <= T[1, 1, 1] #action has positive "maintenance" value
    valid &= T[0, 0, 1] <= T[1, 0, 1] #action has non-negative "influence" value
    return valid

def generateRandomTmatrix(N):

    """
    Generates a Nx2x2x2 T matrix indexed as: \
    T[beneficiary_number][action][current_state][next_state]
    action=0 denotes passive action, a=1 is active action
    """
        
    T=np.zeros((N,2,2,2))
    for i in range(N):
        p_pass_01, p_pass_11, p_act_01, p_act_11=np.random.uniform(size=4)
        T[i,0]=np.array([[1-p_pass_01, p_pass_01],[1-p_pass_11, p_pass_11]])
        T[i,1]=np.array([[1-p_act_01, p_act_01],[1-p_act_11, p_act_11]])
    return T  

#

def takeActions(states, T, actions):
    '''
    Simulates random transitions by flipping coins for each beneficiaries and \
    updating their states given previous states and actions taken

    Inputs: 
    states: A vector of size N with binary values {0,1} respresenting the states\
          of beneficiaries
    T: Transition matrix of size Nx2x2x2 (num_beneficiaries x action x \
          starting_state x ending_state)
    actions: A vector of size N with binary values {0,1} representing action \
          chosen for each beneficiary

    Outputs:
    next_states:  A vector of size N with binary values {0,1} respresenting the\
          *next* states of beneficiaries, determined using coin tosses
    '''

    next_states=np.random.binomial(1, T[np.arange(T.shape[0]),actions.astype(int)\
                                      , states,1])
    return next_states

