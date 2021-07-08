import pandas as pd
import numpy as np

transition_probabilities = pd.read_csv("groundtruth_analysis/transition_probabilities_week_9_comp.csv")
test_transition_probabilities = transition_probabilities[transition_probabilities['TEST/TRAIN'] == 'Test - pilot data']
train_transition_probabilities = transition_probabilities[transition_probabilities['TEST/TRAIN'] == 'Train']

def compute_kl_divergence(p_probs, q_probs):
	print(p_probs, q_probs)
	try: 
		kl_div = 0.0
		if q_probs[0] == 0 or p_probs[0] == 0 or q_probs[1] == 0 or p_probs[1] == 0:
			return '-'
		kl_div += p_probs[0] * np.log(p_probs[0] / q_probs[0])
		kl_div += p_probs[1] * np.log(p_probs[1] / q_probs[1])
	except Exception:
		kl_div = '-'
	return kl_div


cols = [
        'KL(E, I, s)', 'KL(NE, I, s)', 'KL(E, A, s)', 'KL(NE, A, s)', 
    ]


kl_divergence = pd.DataFrame(columns=['cluster'] + cols)

for i in range(1):
	test_row = test_transition_probabilities[test_transition_probabilities['cluster'] == i]
	train_row = train_transition_probabilities[train_transition_probabilities['cluster'] == i]
	kl_divs = dict()
	kl_divs['cluster'] = i
	p_probs = [ test_row['P(E, I, E)'].item(), test_row['P(E, I, NE)'].item() ]
	q_probs = [ train_row['P(E, I, E)'].item(), train_row['P(E, I, NE)'].item() ]
	kl_divs['KL(E, I, s)'] = compute_kl_divergence(p_probs, q_probs)
	p_probs = [ test_row['P(NE, I, E)'].item(), test_row['P(NE, I, NE)'].item() ]
	q_probs = [ train_row['P(NE, I, E)'].item(), train_row['P(NE, I, NE)'].item() ]
	kl_divs['KL(NE, I, s)'] = compute_kl_divergence(p_probs, q_probs)
	p_probs = [ test_row['P(E, A, E)'].item(), test_row['P(E, A, NE)'].item() ]
	q_probs = [ train_row['P(E, A, E)'].item(), train_row['P(E, A, NE)'].item() ]
	kl_divs['KL(E, A, s)'] = compute_kl_divergence(p_probs, q_probs)
	p_probs = [ test_row['P(NE, A, E)'].item(), test_row['P(NE, A, NE)'].item() ]
	q_probs = [ train_row['P(NE, A, E)'].item(), train_row['P(NE, A, NE)'].item() ]
	kl_divs['KL(NE, A, s)'] = compute_kl_divergence(p_probs, q_probs)
	kl_divergence = kl_divergence.append(kl_divs, ignore_index=True)

kl_divergence.to_csv("groundtruth_analysis/kl_divergence.csv")