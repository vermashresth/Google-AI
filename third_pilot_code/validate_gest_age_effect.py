import os
import sys
import numpy as np
from numpy.lib.arraysetops import unique
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from copy import deepcopy

from datetime import datetime
from pprint import pprint
from tqdm import tqdm
import ipdb
import pickle

plt.style.use("seaborn")
np.random.seed(1)

from training.utils import load_obj, save_obj
from training.data import load_data
from training.dataset import _preprocess_call_data, preprocess_and_make_dataset

from sklearn.cluster import KMeans, OPTICS, SpectralClustering, DBSCAN, AffinityPropagation, AgglomerativeClustering, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn_som.som import SOM
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from training.modelling.metrics import F1, Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model
from training.modelling.dataloader import get_train_val_test

pilot_beneficiary_data, pilot_call_data = load_data("feb16-mar15_data")
inf_dataset = preprocess_and_make_dataset(pilot_beneficiary_data, pilot_call_data)
pilot_call_data = _preprocess_call_data(pilot_call_data)
pilot_user_ids, pilot_dynamic_xs, pilot_gest_age, pilot_static_xs, pilot_hosp_id, pilot_labels = inf_dataset


with open('policy_dump.pkl', 'rb') as fr:
	pilot_user_ids,pilot_static_features,cls,cluster_transition_probabilities,m_values,q_values = pickle.load(fr)
fr.close()

ipdb.set_trace()