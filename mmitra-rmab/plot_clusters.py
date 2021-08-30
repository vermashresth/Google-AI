import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import ipdb


# Enter pickle file containing ground truth transition probablity array
# TODO: Ground Truth currently must be containing only passive action probs. 
# We would need active actions ones as well
gt_beneficiries_transition_prob_file = 'outputs/cluster_outputs/gt_beneficiary_probs.pkl'

# Enter paths to csv files containing clustered transition probablities, 
# Will generate two plots corresponding to every file - (intervention probs, non-intervention probs)
clustered_transition_prob_df_csvs  = [
"outputs/cluster_outputs/weekly_kmeans_transition_probabilities_10.csv",
"outputs/cluster_outputs/weekly_kmeans_transition_probabilities_20.csv",
"outputs/cluster_outputs/weekly_kmeans_transition_probabilities_30.csv",
"outputs/cluster_outputs/weekly_kmeans_transition_probabilities_40.csv",
"outputs/cluster_outputs/weekly_kmeans_transition_probabilities_50.csv",
"outputs/cluster_outputs/weekly_kmeans_transition_probabilities_75.csv",
"outputs/cluster_outputs/weekly_kmeans_transition_probabilities_100.csv",
"outputs/cluster_outputs/weekly_kmeans_transition_probabilities_200.csv",
"outputs/cluster_outputs/weekly_kmeans_transition_probabilities_300.csv",
"outputs/cluster_outputs/weekly_kmeans_transition_probabilities_400.csv",
"outputs/cluster_outputs/weekly_kmeans_transition_probabilities_500.csv"]

# Enter label to show for every output csv file. This will show up in plots
exp_labels = ['PPF-10-Clusters',
              'PPF-20-Clusters',
             'PPF-30-Clusters',
             'PPF-40-Clusters',
             'PPF-50-Clusters',
             'PPF-75-Clusters',
             'PPF-100-Clusters',
             'PPF-200-Clusters',
             'PPF-300-Clusters',
             'PPF-400-Clusters',
             'PPF-500-Clusters']


n_exps =  len(clustered_transition_prob_df_csvs)

with open(gt_beneficiries_transition_prob_file, 'rb') as fr:
    gt = pickle.load(fr)
fr.close()

for idx, csv_file in enumerate(clustered_transition_prob_df_csvs):
    df = pd.read_csv(csv_file)
    plt.figure(figsize=(9,8))

    vals_x = np.array(df['P(L, N, L)'].to_list())
    vals_y = np.array(df['P(H, N, L)'].to_list())

    plt.scatter(gt[:, 0], gt[:, 1], s=20, label='GT', alpha=0.1)
    plt.scatter(vals_x, vals_y, marker='^', s=50, label=exp_labels[idx], c='red')
    plt.xlabel(r'$P_{E, E}^p$',fontsize=20)
    plt.ylabel(r'$P_{NE, E}^p$',fontsize=20)
    plt.legend(fontsize=14)
    plt.title(r'under NO Intervention')
    
    plt.savefig(f'outputs/cluster_outputs/plots/clustered_prob_viz_{exp_labels[idx]}.png', dpi=500, bbox_inches='tight')
    plt.show()
    
