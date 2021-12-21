
# policy_names={0:'random',1:'rr', 2:'whittle'}
policy_names={0:'random',1:'rr', 2:'whittle', 3:'soft-whittle'}
policy_map = {policy_names[key]:key for key in policy_names}
dim_dict = {'state':0, 'action':1, 'new_state':2, 'reward':3}

s_vals = [0, 1]
a_vals = [0, 1]

N_STATES = len(s_vals)
N_ACTIONS = len(a_vals)

