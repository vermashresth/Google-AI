import numpy as np 
from gurobipy import *


def check_feasible_range(p01p_range, p11p_range, p01a_range, p11a_range, no_equal=False):



    # Create a new model
    m = Model("Quick program for checking feasibility")
    m.setParam( 'OutputFlag', False )

    # need one 
    p01p = m.addVar(vtype=GRB.CONTINUOUS, lb=p01p_range[0], ub=p01p_range[1], name='p01p')
    p00p = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name='p00p')
    m.addConstr(p00p == 1 - p01p)

    p11p = m.addVar(vtype=GRB.CONTINUOUS, lb=p11p_range[0], ub=p11p_range[1], name='p11p')
    p10p = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name='p10p')
    m.addConstr(p10p == 1 - p11p)

    p01a = m.addVar(vtype=GRB.CONTINUOUS, lb=p01a_range[0], ub=p01a_range[1], name='p01a')
    p00a = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name='p00a')
    m.addConstr(p00a == 1 - p01a)

    p11a = m.addVar(vtype=GRB.CONTINUOUS, lb=p11a_range[0], ub=p11a_range[1], name='p11a')
    p10a = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name='p10a')
    m.addConstr(p10a == 1 - p11a)

    T = np.zeros((2,2,2),dtype=object)

    T[0,0,0] = p00p
    T[0,0,1] = p01p
    T[1,0,0] = p10p
    T[1,0,1] = p11p

    T[0,1,0] = p00a
    T[0,1,1] = p01a
    T[1,1,0] = p10a
    T[1,1,1] = p11a

    # "Natural" constraints

    # better to be in the good state already
    if no_equal:
        eps=1e-2
        m.addConstr(p11a >= p01a+eps)
        m.addConstr(p11p >= p01p+eps)

        # better to act than not
        m.addConstr(p11a >= p11p+eps)
        m.addConstr(p01a >= p01p+eps)
    else:
        m.addConstr(p11a >= p01a)
        m.addConstr(p11p >= p01p)

        # better to act than not
        m.addConstr(p11a >= p11p)
        m.addConstr(p01a >= p01p)

    # Optimize model
    m.optimize()
    # print("model status",m.status)

    feasible = int(m.status) != 3

    return feasible

def get_armman_param_ranges(N, seed, size_type=-1):
    np.random.seed(seed)
    SMALL_RANGE_SIZE = 0.2
    MEDIUM_RANGE_SIZE = 0.5
    LARGE_RANGE_SIZE = 0.7
    MAX_TRIES = 1000000
    prob_epsilon = 0.05
    lower_prob = 0 + prob_epsilon
    upper_prob = 1 - prob_epsilon

    # this N_GRID is just for sampling feasible ranges so want some decent descritization
    N_GRID = 10
    choices = np.linspace(lower_prob, upper_prob, N_GRID)

    arm_transition_ranges = np.zeros((N,2,2,2))

    # 1: get a random set of ranges for each arm that is feasible wrt to 'natural' constraints
    n_tries = 0
    for arm in range(N):
        feasible = False
        while not feasible:
            p01p_range = np.sort(np.random.choice(choices,2,replace=True))
            p11p_range = np.sort(np.random.choice(choices,2,replace=True))
            p01a_range = np.sort(np.random.choice(choices,2,replace=True))
            p11a_range = np.sort(np.random.choice(choices,2,replace=True))
            all_ranges = [p01p_range, p11p_range, p01a_range, p11a_range]
            all_range_sizes = np.fabs(np.array(all_ranges)[:, 0] - np.array(all_ranges)[:, 1])
            feasible = check_feasible_range(*all_ranges, no_equal=True)
            
            if size_type == 'small':
                feasible = feasible and max(all_range_sizes)<SMALL_RANGE_SIZE
            elif size_type == 'medium':
                feasible = feasible and max(all_range_sizes)<MEDIUM_RANGE_SIZE and min(all_range_sizes)>SMALL_RANGE_SIZE
            elif size_type == 'large':
                feasible = feasible and max(all_range_sizes)>LARGE_RANGE_SIZE and min(all_range_sizes)>SMALL_RANGE_SIZE
            
            feasible = feasible and np.mean(p01p_range)+0.2<np.mean(p01a_range)
            feasible = feasible and np.mean(p11p_range)+0.2<np.mean(p11a_range)

            n_tries+=1
            if n_tries > MAX_TRIES:
                raise
        arm_transition_ranges[arm,0,0] = p01p_range
        arm_transition_ranges[arm,1,0] = p11p_range
        arm_transition_ranges[arm,0,1] = p01a_range
        arm_transition_ranges[arm,1,1] = p11a_range


    return arm_transition_ranges

