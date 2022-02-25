import tensorflow as tf
import os
import sys
import numpy as np
import itertools
import pickle
import matplotlib.pyplot as plt
import argparse
import tqdm
import imageio
sys.path.insert(0, "../")

from sklearn.decomposition import PCA

from dfl.model import ANN
from dfl.synthetic import generateDataset
from dfl.whittle import whittleIndex, newWhittleIndex
from dfl.utils import getSoftTopk, twoStageNLLLoss
from dfl.ope import opeIS, opeIS_parallel
from dfl.environments import POMDP2MDP
from dfl.trajectory import getEmpTransitionMatrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARMMAN decision-focused learning')
    parser.add_argument('--method', default='TS', type=str, help='TS (two-stage learning) or DF (decision-focused learning).')
    parser.add_argument('--env', default='general', type=str, help='general (MDP) or POMDP.')
    parser.add_argument('--data', default='synthetic', type=str, help='synthetic or pilot')
    parser.add_argument('--epochs', default=50, type=int, help='num epochs')
    parser.add_argument('--ope', default='sim', type=str, help='importance sampling (IS) or simulation-based (sim).')

    args = parser.parse_args()
    L = 10
    K = 20
    n_states = 2
    gamma = 0.99
    target_policy_name = 'soft-whittle'
    beh_policy_name    = 'random'
    TS_WEIGHT=0.1

    # Environment setup
    env = args.env
    H = 10
    training_mode = 'two-stage' if args.method == 'TS' else 'decision-focused'

    # Loading pretrained model
    folder_path = 'pretrained/{}'.format(args.data)
    TS_model_path = '{}/TS.pickle'.format(folder_path)
    DF_model_path = '{}/DF.pickle'.format(folder_path)
    f_TS = open(TS_model_path, 'rb')
    f_DF = open(DF_model_path, 'rb')
    train_dataset, val_dataset, test_dataset, TS_model_list = pickle.load(f_TS)
    _, _, _, DF_model_list = pickle.load(f_DF)

    dataset_list = [('train', train_dataset), ('val', val_dataset), ('test', test_dataset)]

    for mode, dataset in dataset_list:
        feature_list = []
        TS_prediction_list = []
        DF_prediction_list = []
        label_list = []
        for feature, label, raw_R_data, traj, ope_simulator, _, state_record, action_record, reward_record in tqdm.tqdm(dataset):
            feature       = tf.constant(feature, dtype=tf.float32)
            raw_R_data    = tf.constant(raw_R_data, dtype=tf.float32)
            TS_prediction = [] # Recording the predictions of all epochs
            DF_prediction = [] # Recording the predictions of all epochs

            for epoch in range(len(TS_model_list)):
                TS_model = ANN(n_states=2)
                TS_model.build((None, feature.shape[1]))
                TS_model.set_weights(TS_model_list[epoch])
                TS_prediction_epoch = TS_model(feature) # prediction of this epoch

                DF_model = ANN(n_states=2)
                DF_model.build((None, feature.shape[1]))
                DF_model.set_weights(DF_model_list[epoch])
                DF_prediction_epoch = DF_model(feature) # prediction of this epoch

                TS_prediction.append(TS_prediction_epoch.numpy())
                DF_prediction.append(DF_prediction_epoch.numpy())

            # Computing empirical transitions
            n_benefs, n_states, H = len(feature), raw_R_data.shape[1], traj.shape[2]
            emp_T_data, emp_R_data = getEmpTransitionMatrix(traj=traj, policy_id=0, n_benefs=n_benefs, m=n_states, env='general', H=H, use_informed_prior=False)
            emp_label = emp_T_data

            feature_list.append(feature.numpy())
            label_list.append(emp_label)
            TS_prediction_list.append(TS_prediction)
            DF_prediction_list.append(DF_prediction)

        # visualization
        features    = np.array(feature_list)
        labels      = np.array(label_list)
        TS_predictions = np.array(TS_prediction_list) # Shape: num instances x num epochs x num beneficiaries x 2 x 2 x 2
        DF_predictions = np.array(DF_prediction_list) # Shape: num instances x num epochs x num beneficiaries x 2 x 2 x 2

        TS_predictions = np.swapaxes(TS_predictions, 0, 1) # Shape: num epochs x num instances x num beneficiaries x 2 x 2 x 2
        DF_predictions = np.swapaxes(DF_predictions, 0, 1) # Shape: num epochs x num instances x num beneficiaries x 2 x 2 x 2

        features    = features.reshape((features.shape[0] * features.shape[1], features.shape[2]))
        labels      = labels.reshape((labels.shape[0] * labels.shape[1], -1))
        TS_predictions = TS_predictions.reshape((TS_predictions.shape[0], TS_predictions.shape[1] * TS_predictions.shape[2], -1))
        DF_predictions = DF_predictions.reshape((DF_predictions.shape[0], DF_predictions.shape[1] * DF_predictions.shape[2], -1))

        # Dimension reduction
        pca = PCA(n_components=1)
        # pca = PCA(n_components=2)
        pca.fit(features)

        processed_features = pca.transform(features)

        # Transition probability names
        name_list = ['NE - N - NE', 'NE - N - E', 'NE - I - NE', 'NE - I - E', 'E - N - NE', 'E - N - E', 'E - I - NE', 'E - I - E']
        for i in range(8): # There are in total 2 x 2 x 2 transition probabilities
            image_path = '{}/{}/var{}/'.format(folder_path, mode, i)
            if not os.path.exists(image_path):
                os.mkdir(image_path)

            # Transition probability names
            name_list = ['NE - N - NE', 'NE - N - E', 'NE - I - NE', 'NE - I - E', 'E - N - NE', 'E - N - E', 'E - I - NE', 'E - I - E']

            images = []
            for epoch in range(len(TS_model_list)):
                fig = plt.figure()
                ax = fig.add_subplot()
                ax.scatter(processed_features[:,0], labels[:,i],         c='r', s=1, marker='^', label='label')
                ax.scatter(processed_features[:,0], TS_predictions[epoch,:,i], c='b', s=1, marker='o', label='TS prediction')
                ax.scatter(processed_features[:,0], DF_predictions[epoch,:,i], c='g', s=1, marker='s', label='DF prediction')
                ax.set_title(name_list[i] + ' Epoch {}'.format(epoch))

                # ax = fig.add_subplot(projection='3d')
                # ax.scatter(processed_features[:,0], processed_features[:,1], labels[:,i], c='r', marker='o', label='label')
                # ax.scatter(processed_features[:,0], processed_features[:,1], predictions[:,i], c='b', marker='o', label='prediction')

                ax.legend()

                filename = image_path + 'epoch{}.png'.format(epoch)
                plt.savefig(image_path + 'epoch{}.png'.format(epoch))
                plt.close()

                images.append(imageio.imread(filename))

            gif_path = '{}/{}/var{}.gif'.format(folder_path, mode, i)
            imageio.mimsave(gif_path, images)
