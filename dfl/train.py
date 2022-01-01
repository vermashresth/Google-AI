import tensorflow as tf
import numpy as np
import argparse
import tqdm
import time
import sys
import pickle
sys.path.insert(0, "../")

from dfl.model import ANN
from dfl.synthetic import generateDataset
from dfl.whittle import whittleIndex, newWhittleIndex
from dfl.utils import getSoftTopk
from dfl.ope import opeIS, opeIS_parallel
from dfl.environments import POMDP2MDP

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARMMAN decision-focused learning')
    parser.add_argument('--method', default='TS', type=str, help='TS (two-stage learning) or DF (decision-focused learning).')
    parser.add_argument('--env', default='general', type=str, help='general (MDP) or POMDP.')
    parser.add_argument('--sv', default='.', type=str, help='save string name')
    parser.add_argument('--epochs', default=10, type=int, help='num epochs')
    parser.add_argument('--ope', default='IS', type=str, help='importance sampling (IS) or simulation-based (sim).')

    args = parser.parse_args()
    print('argparser arguments', args)

    n_benefs = 50
    n_instances = 10
    n_trials = 10
    L = 10
    K = 10
    n_states = 3
    gamma = 0.99
    target_policy_name = 'soft-whittle'
    beh_policy_name    = 'random'

    # Environment setup
    env = args.env
    H = 10

    # Evaluation setup
    ope_mode = args.ope

    # dataset generation
    full_dataset  = generateDataset(n_benefs, n_states, n_instances, n_trials, L, K, gamma, env=env, H=H)
    train_dataset = full_dataset[:int(n_instances*0.7)]
    val_dataset   = full_dataset[int(n_instances*0.7):int(n_instances*0.8)]
    test_dataset  = full_dataset[int(n_instances*0.8):]

    dataset_list = [('train', train_dataset), ('val', val_dataset), ('test', test_dataset)]

    # model initialization
    model = ANN(n_states=n_states)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss_fn = tf.keras.losses.mean_squared_error

    # training
    training_mode = 'two-stage' if args.method == 'TS' else 'decision-focused'
    total_epoch = args.epochs
    overall_loss={}
    overall_ope={}
    for epoch in range(total_epoch+1):
        for mode, dataset in dataset_list:
            if epoch==0:
                overall_loss[mode]=[]
                overall_ope[mode]=[]
            loss_list = []
            ope_list = []
            if mode == 'train':
                dataset = tqdm.tqdm(dataset)

            for (feature, label, raw_R_data, traj, ope_simulator, simulated_rewards, mask, state_record, action_record, reward_record) in dataset:
                feature, label = tf.constant(feature, dtype=tf.float32), tf.constant(label, dtype=tf.float32)
                raw_R_data = tf.constant(raw_R_data, dtype=tf.float32)

                with tf.GradientTape() as tape:
                    prediction = model(feature) # Transition probabilities
                    if epoch==total_epoch:
                        prediction=label
                    
                    loss = tf.reduce_sum((label - prediction)**2) # Two-stage loss
                    

                    # Setup MDP or POMDP environment
                    if env=='general':
                        T_data, R_data = prediction, raw_R_data
                    elif env=='POMDP':
                        T_data, R_data = POMDP2MDP(prediction, raw_R_data, H)
                    
                    # Batch Whittle index computation
                    # w = whittleIndex(prediction)
                    w = newWhittleIndex(T_data, R_data)
                    
                    if ope_mode == 'IS': # importance-sampling based OPE
                        ope = opeIS_parallel(state_record, action_record, reward_record, w, mask, n_benefs, L, K, n_trials, gamma,
                                target_policy_name, beh_policy_name)
                    elif ope_mode == 'sim': # simulation-based OPE
                        ope = ope_simulator(w)
                    else:
                        raise NotImplementedError

                    performance = -ope

                # backpropagation
                if mode == 'train' and epoch<total_epoch:
                    if training_mode == 'two-stage':
                        grad = tape.gradient(loss, model.trainable_variables)
                    elif training_mode == 'decision-focused':
                        grad = tape.gradient(performance, model.trainable_variables)
                    else:
                        raise NotImplementedError
                    optimizer.apply_gradients(zip(grad, model.trainable_variables))
                del tape

                loss_list.append(loss)
                ope_list.append(ope)

            print(f'Epoch {epoch}, {mode} mode, average loss {np.mean(loss_list)}, average ope {np.mean(ope_list)}')
            
            overall_loss[mode].append(np.mean(loss_list))
            overall_ope[mode].append(np.mean(ope_list))

    
    if not(args.sv == '.'):
        ### Output to be saved, else do nothing. 
        with open(args.sv, 'wb') as filename:
            pickle.dump([overall_loss, overall_ope], filename)

