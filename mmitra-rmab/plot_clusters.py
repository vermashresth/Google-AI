import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import ipdb


# Enter pickle file containing ground truth transition probablity array
# TODO: Ground Truth currently must be containing only passive action probs. 
# We would need active actions ones as well
gt_beneficiries_transition_prob_file = 'gt_beneficiary_probs.pkl'

# Enter paths to csv files containing clustered transition probablities, 
# Will generate two plots corresponding to every file - (intervention probs, non-intervention probs)
clustered_transition_prob_df_csvs = ['outputs/individual_clustering/weekly_kmeans_transition_probabilities_20.csv',
                                    'outputs/individual_clustering/weekly_kmeans_transition_probabilities_20.csv']

# Enter label to show for every output csv file. This will show up in plots
exp_labels = ['PPF-20-Clusters',
              'PPF-40-Clusters']


n_exps =  len(clustered_transition_prob_df_csvs)

with open(gt_beneficiries_transition_prob_file, 'rb') as fr:
    gt = pickle.load(fr)
fr.close()

plt.figure(figsize=(18,8*n_exps))
for idx, csv_file in enumerate(clustered_transition_prob_df_csvs):
    df20 = pd.read_csv(csv_file)
    
    plt.subplot(n_exps,2,1+idx*2)

    vals20_x = np.array(df20['P(L, N, L)'].to_list())
    vals20_y = np.array(df20['P(H, N, L)'].to_list())

    plt.scatter(gt[:, 0], gt[:, 1], s=10, label='GT')
    plt.scatter(vals20_x, vals20_y, marker='^', s=50, label=exp_labels[idx], c='red')
    plt.xlabel(r'$P_{E, E}^p$',fontsize=20)
    plt.ylabel(r'$P_{NE, E}^p$',fontsize=20)
    plt.legend(fontsize=14)
    plt.title(r'under NO Intervention')
    
    plt.subplot(n_exps,2,2+idx*2)

    vals20_x = np.array(df20['P(L, I, L)'].to_list())
    vals20_y = np.array(df20['P(H, I, L)'].to_list())

    plt.scatter(gt[:, 0], gt[:, 1], s=10, label='GT')
    plt.scatter(vals20_x, vals20_y, marker='^', s=50, label=exp_labels[idx], c='red')
    plt.xlabel(r'$P_{E, E}^a$',fontsize=20)
    plt.ylabel(r'$P_{NE, E}^a$',fontsize=20)
    plt.legend(fontsize=14)
    plt.title(r'under Intervention')
    
plt.savefig('clustered_prob_viz.png', dpi=500, bbox_inches='tight')
plt.show()
