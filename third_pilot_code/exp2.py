import os
import sys
import numpy as np
import seaborn as sns
from numpy.lib.arraysetops import unique
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
import ipdb

import pickle

plt.style.use("seaborn")
np.random.seed(1)

from training.utils import load_obj, save_obj
from training.data import load_data
from training.dataset import _preprocess_call_data, preprocess_and_make_dataset

from sklearn.ensemble import RandomForestClassifier


def plot_whittle_indices():
    whittle_indices = pd.read_csv("whittle_indices_kmeans_soft.csv")
    # whittle_indices = whittle_indices[[ 'whittle_index_E', 'whittle_index_NE' ]]
    # data = sns.load_dataset(whittle_indices)
    # sns.pointplot(x = "NE Whittle Index",
    #     y = "E Whittle Index",
    #     data = data
    #     )

    all_whittle_indices = list(whittle_indices['whittle_index_NE'].values)
    all_whittle_indices += list(whittle_indices['whittle_index_E'].values)
    # x_cord = list(whittle_indices['whittle_index_NE'].values)
    # y_cord = list(whittle_indices['whittle_index_E'].values)
    plot_indices = dict()
    for i in all_whittle_indices:
        if  i < 0:
            i = i - 0.05
        plot_indices[i//0.05] = plot_indices.get(i//0.05, 0) + 1
    # for i in range()
    # for i in range(20):
    #     plot_indices[i] = plot_indices.get(i, 0)

    df = pd.DataFrame(columns=['Whittle Index', 'frequency'])
    
    # Change labels of x-axis plot to probability ranges
    for i in sorted( plot_indices.keys() ) :
        if int(i < 0):
            df = df.append({'Whittle Index': "[{:.2f}, {:.2f})".format(0.05*int(i+1), 0.05*int(i+2)),
            'frequency': plot_indices[i]}, ignore_index=True)
        else:
            df = df.append({'Whittle Index': "[{:.2f}, {:.2f})".format(0.05*int(i), 0.05*int(i+1)),
            'frequency': plot_indices[i]}, ignore_index=True)

    plt.figure(figsize=(8, 8))

    plots = sns.barplot(x="Whittle Index", y="frequency", data=df)

    # Annotate the bars of the plot
    for bar in plots.patches:
        if bar.get_height() != 0:
        # ipdb.set_trace()
            plots.annotate(format(bar.get_height(), '.2f'), 
                           (bar.get_x() + bar.get_width() / 2, 
                            bar.get_height()), ha='center', va='center',
                           size=7, xytext=(0, 8),
                           textcoords='offset points')
    plt.tick_params(labelsize=5)      
    plt.xlabel("Whittle Index", size=8)
    plt.ylabel("Frequency of Beneficiaries", size=8)
    plt.show()


plot_whittle_indices()