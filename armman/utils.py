## Helper function which allows getting probablity from two_state_prob matrix using readable format
## P(NE, A, E) -> two_state_prob[0, 0, 1]
import numpy as np

def prob_string_to_prob(prob_string, two_state_prob):
    key_mapping = {'NE':0, 'E':1, 'A':0, 'I':1}
    s, a, new_s = prob_string[2:-1].replace(' ', '').split(',')
    return two_state_prob[key_mapping[a], key_mapping[s], key_mapping[new_s]]

## Helper function to create two_state_prob matrix using given transitions
def gen_trans(e_i_e=0.9, ne_i_e=0.8, ne_a_e=0.3, e_a_e=0.1):
    no_interv = [[1-ne_a_e, ne_a_e], [1-e_a_e, e_a_e]]
    interv = [[1-ne_i_e, ne_i_e], [1-e_i_e, e_i_e]]
    return np.array([no_interv, interv])

def percentile_rank(v, x):
    v = np.sort(v)
    return (v<x).sum() / len(v)

def sum_up_to(number):
    return sum(range(1, number + 1))