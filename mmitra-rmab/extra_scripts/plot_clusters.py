import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import ipdb

with open('gt_beneficiary_probs.pkl', 'rb') as fr:
    gt = pickle.load(fr)
fr.close()

df20 = pd.read_csv('outputs/individual_clustering/weekly_kmeans_transition_probabilities_20.csv')
df40 = pd.read_csv('outputs/individual_clustering/weekly_kmeans_transition_probabilities_40.csv')

vals20_x = np.array(df20['P(L, N, L)'].to_list())
vals20_y = np.array(df20['P(H, N, L)'].to_list())

vals40_x = np.array(df40['P(L, N, L)'].to_list())
vals40_y = np.array(df40['P(H, N, L)'].to_list())

plt.scatter(gt[:, 0], gt[:, 1], s=10, label='GT')
plt.scatter(vals20_x, vals20_y, marker='^', s=50, label='PPF-20', c='lawngreen')
plt.scatter(vals40_x, vals40_y, marker='s', s=40, label='PPF-40', c='red')
plt.xlabel(r'$P_{E, E}^p$',fontsize=20)
plt.ylabel(r'$P_{NE, E}^p$',fontsize=20)
plt.legend(fontsize=14)
# plt.title(r'Comparison of FO with GT: $P_{E, E}^p$ and $P_{NE, E}^p$')
plt.savefig('PPF.png', dpi=500, bbox_inches='tight')
# plt.show()