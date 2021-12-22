
# policy_names={0:'random',1:'rr', 2:'whittle'}
policy_names={0:'random',1:'rr', 2:'whittle', 3:'soft-whittle'}
policy_map = {policy_names[key]:key for key in policy_names}
dim_dict = {'state':0, 'action':1, 'next_state':2, 'reward':3}

S_VALS = [0, 1]
A_VALS = [0, 1]

N_STATES = len(S_VALS)
N_ACTIONS = len(A_VALS)

