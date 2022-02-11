import numpy as np
import pandas as pd
import gym
import torch
from scipy.special import comb
import time
import pickle
from robust_rmab.algos.whittle import mathprog_methods


N = 80
B = 100
data = 'large_bootstrapped'

if data == 'new_large':
    in_file = 'new_armman_params.pickle' # n_clusters = 40, n_arms = 7668. use N = 80, B<7668
    assert N == 80
elif data == 'old_large':
    in_file = 'armman_params.pickle' # n_clusters = 40, n_arms = 7668. use N = 80, B<7668
    assert N == 80
elif data == 'large_bootstrapped':
    in_file = 'bootstrapped_armman_params.pickle' # n_clusters = 40, n_arms = 7668. use N = 80, B<7668
    assert N == 80
# elif data == 'small':
#     in_file = 'armman_params_small.pickle' # n_clusters = 26, n_arms = 100. use N = 52, B<100
#     assert N == 52
elif data == 'very_small':
    in_file = 'armman_params_very_small.pickle' # n_clusters = 5, n_arms = 15, use N = 10, B<15
    assert N == 10
elif data =='toy':
    in_file = 'armman_params_toy.pickle' # n_clusters = 2, n_arms = 2, use N = 4, B<2
    assert N == 6
else:
    in_file = data

with open(in_file, 'rb') as handle:
    print(f'Loading Pickled Params from {in_file}...')
    info_dict = pickle.load(handle)


assert B < len(info_dict['cluster_mapping'])

S = 2
A = 2
n_clusters = info_dict['n_clusters']
param_ranges = info_dict['parameter_ranges']


#########################

# scale up number of individuals
scale_up_factor = 20
new_cluster_mapping = np.repeat(info_dict['cluster_mapping'], scale_up_factor)

new_info_dict = {
  'n_clusters': n_clusters,
  'cluster_mapping': new_cluster_mapping,
  'parameter_ranges': info_dict['parameter_ranges'],
  'max_cluster_size': info_dict['max_cluster_size'] * scale_up_factor,
  }

f = open(f'bootstrapped_armman_params_scaleup{scale_up_factor}.pickle', 'wb')
pickle.dump(new_info_dict, f)

sys.exit(0)

########################

# use clusters with medium equality in cluster sizes
orig_n_indiv = len(info_dict['cluster_mapping'])
eq_cluster_size = orig_n_indiv // n_clusters
new_cluster_mapping = []

# count custer sizes
cluster_size = {}
for cluster in range(n_clusters):
    cluster_size[cluster] = 0
for indiv, cluster in enumerate(info_dict['cluster_mapping']):
    cluster_size[cluster] += 1

for cluster in range(n_clusters):
    new_cluster_size = int((cluster_size[cluster] + eq_cluster_size) / 2)
    new_cluster_mapping.append([cluster] * new_cluster_size)

new_cluster_mapping = [x for sublist in new_cluster_mapping for x in sublist]
new_cluster_mapping = np.array(new_cluster_mapping)

new_info_dict = {
  'n_clusters': n_clusters,
  'cluster_mapping': new_cluster_mapping,
  'parameter_ranges': info_dict['parameter_ranges'],
  'max_cluster_size': np.max([x for x in cluster_size.values()]),
  }

f = open('bootstrapped_armman_params_small_variance_cluster_size.pickle', 'wb')
pickle.dump(new_info_dict, f)
import pdb, sys
pdb.set_trace()
sys.exit(0)

#########################

# enforce all clusters to be of the same size
orig_n_indiv = len(info_dict['cluster_mapping'])
cluster_size = orig_n_indiv // n_clusters
new_cluster_mapping = np.arange(n_clusters)
new_cluster_mapping = np.repeat(new_cluster_mapping, cluster_size)

new_info_dict = {
  'n_clusters': n_clusters,
  'cluster_mapping': new_cluster_mapping,
  'parameter_ranges': info_dict['parameter_ranges'],
  'max_cluster_size': cluster_size,
  }

f = open('bootstrapped_armman_params_equal_cluster_size.pickle', 'wb')
pickle.dump(new_info_dict, f)
import pdb, sys
pdb.set_trace()
sys.exit(0)
info_dict['cluster_mapping'] = np.arange(n_clusters)



feasible_param_ranges = np.zeros((self.n_clusters, S, A, 2))
num_feasible, num_infeasible = 0, 0

# ensure parameter ranges fulfill feasibility constraints
for cluster in range(self.n_clusters):
    p01p_range = param_ranges[cluster,0,0,:] # [cluster, s, action, range]
    p11p_range = param_ranges[cluster,1,0,:]
    p01a_range = param_ranges[cluster,0,1,:]
    p11a_range = param_ranges[cluster,1,1,:]

    # ensure start at good state is beneficial
    if not p01p_range[1] <= p11p_range[1]:
        p01p_range[:] = p11p_range

    if not p01a_range[1] <= p11a_range[1]:
        p01a_range[:] = p11a_range
    
    # ensure active action is beneficial
    if not p01a_range[1] >= p01p_range[1]:
        p01p_range[:] = p01a_range
    
    if not p11a_range[1] >= p11p_range[1]:
        p11p_range[:] = p11a_range

    feasibility = mathprog_methods.check_feasible_range(p01p_range, p11p_range, p01a_range, p11a_range)
    # print(f'cluster {cluster}, feasible {feasibility}')
    if feasibility:
        num_feasible += 1
    else:
        num_infeasible += 1
        print('infeasible! --------------------')
        print('p01p', p01p_range)
        print('p11p', p11p_range)
        print('p01a', p01a_range)
        print('p11a', p11a_range)

    # # better to be in the good state already
    # p01a_range[1] = np.min([p01a_range[1], p11a_range[1]]) # p11a >= p01a
    # p01p_range[1] = np.min([p01p_range[1], p11p_range[1]]) # p11p >= p01p

    # # better to act than not act
    # p01p_range[1] = np.min([p01p_range[1], p01a_range[1]])
    # p11p_range[1] = np.min([p11p_range[1], p11a_range[1]])

    # # p11a upper bound should be equal or higher than all ubs
    # p01p_range[1] = np.min([p01p_range[1], p11a_range[1]])
    # # p11p_range[1] = np.min([p11p_range[1], p11a_range[1]])
    # # p01a_range[1] = np.min([p01a_range[1], p11a_range[1]])

    # # lower bound no more than upper bound
    # p01a_range[0] = np.min(p01a_range)
    # p01p_range[0] = np.min(p01p_range)
    # p11p_range[0] = np.min(p11p_range)
    # p11a_range[0] = np.min(p11a_range)

    # # intervene lower bound should be more than passive lower bound
    # p01a_range[0] = np.max([p01a_range[0], p01p_range[0]])
    # p11a_range[0] = np.max([p11a_range[0], p11p_range[0]])

    # # start in good start lower bound should be more than start in bad state lb
    # p11p_range[0] = np.max([p11p_range[0], p01p_range[0]])
    # p11a_range[0] = np.max([p11a_range[0], p11a_range[0]])

    # if not feasibility:
    #     print('ranges after -----')
    #     print('p01p', p01p_range)
    #     print('p11p', p11p_range)
    #     print('p01a', p01a_range)
    #     print('p11a', p11a_range)

    # print('--------------------------')
    # print('p01p', p01p_range)
    # print('p11p', p11p_range)
    # print('p01a', p01a_range)
    # print('p11a', p11a_range)
    feasible_param_ranges[cluster, 0, 0, :] = p01p_range  # [cluster, s, action, range]
    feasible_param_ranges[cluster, 1, 0, :] = p11p_range
    feasible_param_ranges[cluster, 0, 1, :] = p01a_range
    feasible_param_ranges[cluster, 1, 1, :] = p11a_range

# # compute interval stats
# interval_widths = []
# widths_p01p = []
# widths_p11p = []
# widths_p01a = []
# widths_p11a = []
# means_p01p = []
# means_p11p = []
# means_p01a = []
# means_p11a = []
# for cluster in range(self.n_clusters):
#     p01p_range = param_ranges[cluster,0,0,:] # [cluster, s, action, range]
#     p11p_range = param_ranges[cluster,1,0,:]
#     p01a_range = param_ranges[cluster,0,1,:]
#     p11a_range = param_ranges[cluster,1,1,:]
#     interval_widths.append(p01p_range[1] - p01p_range[0])
#     interval_widths.append(p11p_range[1] - p11p_range[0])
#     interval_widths.append(p01a_range[1] - p01a_range[0])
#     interval_widths.append(p11a_range[1] - p11a_range[0])
#     widths_p01p.append(p01p_range[1] - p01p_range[0])
#     widths_p11p.append(p11p_range[1] - p11p_range[0])
#     widths_p01a.append(p01a_range[1] - p01a_range[0])
#     widths_p11a.append(p11a_range[1] - p11a_range[0])
#     means_p01p.append((p01p_range[1] + p01p_range[0])/2)
#     means_p11p.append((p11p_range[1] + p11p_range[0])/2)
#     means_p01a.append((p01a_range[1] + p01a_range[0])/2)
#     means_p11a.append((p11a_range[1] + p11a_range[0])/2)


# avg_interval_width = np.mean(interval_widths)
# std_interval_width = np.std(interval_widths)
# print(f'avg interval width {avg_interval_width:.3f} std {std_interval_width:.3f}')
# print(f'p01p width avg {np.mean(widths_p01p):.3f} std {np.std(widths_p01p):.3f}')
# print(f'p11p width avg {np.mean(widths_p11p):.3f} std {np.std(widths_p11p):.3f}')
# print(f'p01a width avg {np.mean(widths_p01a):.3f} std {np.std(widths_p01a):.3f}')
# print(f'p11a width avg {np.mean(widths_p11a):.3f} std {np.std(widths_p11a):.3f}')
# print('')
# print(f'p01p mean avg {np.mean(means_p01p):.3f} std {np.std(means_p01p):.3f}')
# print(f'p11p mean avg {np.mean(means_p11p):.3f} std {np.std(means_p11p):.3f}')
# print(f'p01a mean avg {np.mean(means_p01a):.3f} std {np.std(means_p01a):.3f}')
# print(f'p11a mean avg {np.mean(means_p11a):.3f} std {np.std(means_p11a):.3f}')


# # print(f'{num_feasible} feasible, {num_infeasible} infeasible')
# import sys
# sys.exit(0)

self.PARAMETER_RANGES = feasible_param_ranges

# Set these parameters from global loaded pickled file
# TODO: make this more clean
self.cluster_mapping, self.max_cluster_size = info_dict['cluster_mapping'], info_dict['max_cluster_size']


assert self.n_clusters*S == N, f'n_clusters = {self.n_clusters}, S={S}, N={N}. self.n_clusters*S should be same as N'

# Here N is not number of arms but size of problem. It is equal to number of clusters times number of states
self.N = N
self.n_arms = len(self.cluster_mapping)

# Arm observation space is state at arm level
self.arm_observation_space = np.arange(S)
# This observation space is at cluster-state level. Since it contains counts of beneficiries
# its maximum value is size of largest cluster
self.observation_space = np.arange(self.max_cluster_size+1)
self.action_space = np.arange(A)
self.observation_dimension = 1
self.action_dimension = 1

## Nature outputs transition probabilities of shape n_clusters x n_states x n_actions
# N is already n_clusters * n_states, so we can write it as below
self.action_dim_nature = N*A

self.S = S
self.A = A
self.B = B

assert self.B < self.n_arms, f'self.B = {self.B}, self.n_arms = {self.n_arms}. self.B should be less than self.n_arms'
self.init_seed = seed

self.random_stream = np.random.RandomState()

# Obtain placeholder Transition Probablity matrix, Reward function, Cost function
self.T, self.R, self.C = self.get_experiment(self.n_clusters)

self.sampled_parameter_ranges = self.sample_parameter_ranges() 


self.seed(seed=seed)


