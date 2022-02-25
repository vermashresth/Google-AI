import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set CONFIG
CONFIG = {
    'clusters': 20,
    'mapping_method': 'FO'
        }
print('Clusters: ', CONFIG["clusters"])
print('Mapping Method: ', CONFIG["mapping_method"])

pred = pd.read_csv(f'outputs/clusters_{CONFIG["clusters"]}_mapping_{CONFIG["mapping_method"]}_predicted_prob.csv')
gt = pd.read_csv(f'outputs/clusters_{CONFIG["clusters"]}_mapping_{CONFIG["mapping_method"]}_gt_prob.csv')

non_nan = ~(gt[['P(L, N, L)', 'P(H, N, L)']].isna().sum(axis=1).astype(bool))

diff = pred[['P(L, N, L)', 'P(H, N, L)']][non_nan] - gt[['P(L, N, L)', 'P(H, N, L)']][non_nan]
diff.abs()

## Distribution of Erros
plt.hist(diff.abs().values.flatten())
plt.show()

print('MAE: ', diff.abs().values.flatten().mean())

