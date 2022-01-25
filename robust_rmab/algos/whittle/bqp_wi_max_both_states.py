import numpy as np
import pandas as pd
import mathprog_methods
import pickle
import time
import params

par = params.Params()

np.random.seed(2)
pd.set_option("display.max_rows", None, "display.max_columns", None)



N = 2
S = 2
A = 2
gamma = par.gamma
C = np.array([0, 1])
R = np.array([0, 1])
T = np.zeros((N,S,A,S))

N_GRID = par.N_GRID
prob_epsilon = par.prob_epsilon
lower_prob = 0 + prob_epsilon
upper_prob = 1 - prob_epsilon

wi_grid_df = None
with open('indexes/blp_indexes_df_NGRID%s.pickle'%N_GRID, 'rb') as handle:
    wi_grid_df = pickle.load(handle)


# for ensuring indexes are same between the grid and the bqp
diff_tolerance = 1e-2

# for avoiding annoying issues with filtering the pandas df
choice_eps = 1e-3
choices = np.linspace(lower_prob, upper_prob, N_GRID)

no_differences = True

N_TRIALS = 10
start = time.time()

time_limit = 5

diffs = np.zeros((N_TRIALS,S))

times = np.zeros(N_TRIALS)
range_gaps = np.zeros(N_TRIALS)
for trial in range(N_TRIALS):
    s1 = time.time()
    # TODO: sample random senses for both states, get the optimized objective+solution from the grid, print all, compare with bqp

    senses = np.random.choice(['min','max'],2,replace=True)

    senses_signs = [ -1 if i=='max' else 1 for i in senses]

    feasible = False
    while not feasible:
        p01p_range = np.sort(np.random.choice(choices,2,replace=True))
        p11p_range = np.sort(np.random.choice(choices,2,replace=True))
        p01a_range = np.sort(np.random.choice(choices,2,replace=True))
        p11a_range = np.sort(np.random.choice(choices,2,replace=True))
        all_ranges = [p01p_range, p11p_range, p01a_range, p11a_range]

        feasible = mathprog_methods.check_feasible_range(*all_ranges)

    

    range_gaps[trial] = sum([r[1] - r[0] for r in all_ranges])


    # get ranges from the pre-computed df
    sub_df = wi_grid_df[(wi_grid_df['p01p'] >= p01p_range[0]-choice_eps) & (wi_grid_df['p01p'] <= p01p_range[1]+choice_eps)]
    sub_df = sub_df[(sub_df['p11p'] >= p11p_range[0]-choice_eps) & (sub_df['p11p'] <= p11p_range[1]+choice_eps)]
    sub_df = sub_df[(sub_df['p01a'] >= p01a_range[0]-choice_eps) & (sub_df['p01a'] <= p01a_range[1]+choice_eps)]
    sub_df = sub_df[(sub_df['p11a'] >= p11a_range[0]-choice_eps) & (sub_df['p11a'] <= p11a_range[1]+choice_eps)]
    

    # Now impose the 'Natural' constraints
    sub_df = sub_df[sub_df['p11a'] >= sub_df['p01a']]
    sub_df = sub_df[sub_df['p11p'] >= sub_df['p01p']]
    sub_df = sub_df[sub_df['p11a'] >= sub_df['p11p']]
    sub_df = sub_df[sub_df['p01a'] >= sub_df['p01p']]


    sub_df['objective'] = sub_df['index0']*senses_signs[0] + sub_df['index1']*senses_signs[1]

    solution_epsilon = 1e-3
    solution_value = sub_df['objective'].min()

    solution_row = sub_df[(sub_df['objective'] <= solution_value + solution_epsilon) & (sub_df['objective'] >= solution_value - solution_epsilon)]
    # print(wi_grid_df)
    print(sub_df)
    print('solution row')
    print(solution_row)
    print('solution value')
    print(solution_value)

    optimized_indexes_grid = solution_row[['index0','index1']].values[0]
    
    print('senses')
    print(senses)
    print('ranges')
    print('p01p_range',p01p_range)
    print('p11p_range',p11p_range)
    print('p01a_range',p01a_range)
    print('p11a_range',p11a_range)


    print('jointly optimizing the wi\'s')
    # bqp_to_optimize_index_both_states(p01p_range, p11p_range, p01a_range, p11a_range, 
    #                               R, C, start_state, sense=['min','min'], gamma=0.95, lambda_lim=None):
    optimized_indexes_bqp, L_vals, z_vals, bina_vals, T_return  = mathprog_methods.bqp_to_optimize_index_both_states(p01p_range, p11p_range, p01a_range, p11a_range,
                                                                        R, C, senses=senses, gamma=gamma, time_limit=time_limit)

    print(optimized_indexes_bqp)

    print("transition settings")
    print('p01p',T_return[0,0,1])
    print('p11p',T_return[1,0,1])
    print('p01a',T_return[0,1,1])
    print('p11a',T_return[1,1,1])
    print()


    # check for just one arm
    Q_ind0 = np.zeros((T.shape[0], T.shape[1]))
    for s in range(T.shape[0]):
        for a in range(T.shape[1]):
            Q_ind0[s,a] = R[s] - optimized_indexes_bqp[0]*C[a] + gamma*L_vals[0].dot(T_return[s,a])


    Q_ind1 = np.zeros((T.shape[0], T.shape[1]))
    for s in range(T.shape[0]):
        for a in range(T.shape[1]):
            Q_ind1[s,a] = R[s] - optimized_indexes_bqp[1]*C[a] + gamma*L_vals[1].dot(T_return[s,a])


    print('ind0: BQP L vals')
    print(L_vals[0])
    print('ind0: BQP Q vals')
    print(Q_ind0)
    print()

    print('ind1: BQP L vals')
    print(L_vals[1])
    print('ind1: BQP Q vals')
    print(Q_ind1)
    print()

    print('ind0 (grid)', optimized_indexes_grid[0])
    print('ind0 (bqp)', optimized_indexes_bqp[0])
    print()
    print('ind1 (grid)', optimized_indexes_grid[1])
    print('ind1 (bqp)', optimized_indexes_bqp[1])
    print()
    print()

    no_differences = no_differences & (abs(optimized_indexes_grid[0] - optimized_indexes_bqp[0]) < diff_tolerance)
    no_differences = no_differences & (abs(optimized_indexes_grid[1] - optimized_indexes_bqp[1]) < diff_tolerance)

    s2 = time.time()

    times[trial] = s2-s1

    diffs[trial] = optimized_indexes_grid - optimized_indexes_bqp



end = time.time()
bqp_time = end - start

print('total time taken',bqp_time)
print("no differences:",no_differences)
print()
print('distribution of solve times')
print(times)
print('average:',times.mean())
print('std:',times.std())
print()
print('distribution of size of ranges')
print(range_gaps)

for trial in range(N_TRIALS):
    print(times[trial], '\t', diffs[trial].round(3))

# colnames = ['p01p','p11p','p01a','p11a','index0','index1']
# blp_df = pd.DataFrame(wi_grid_blp_compute_list, columns=colnames)

# with open('blp_indexes_df_NGRID%s.pickle'%N_GRID, 'wb') as handle:
#     pickle.dump(blp_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print(blp_df)




