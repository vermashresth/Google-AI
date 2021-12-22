import tensorflow as tf
import numpy as np
import argparse
import tqdm
import sys
sys.path.insert(0, "../")

from dfl.model import ANN
from dfl.synthetic import generateDataset
from dfl.whittle import whittleIndex
from dfl.utils import getSoftTopk
from dfl.ope import opeIS, opeIS_parallel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARMMAN decision-focused learning')
    parser.add_argument('--method', default='TS', type=str, help='TS (two-stage learning) or DF (decision-focused learning).')

    args = parser.parse_args()
    print('argparser arguments', args)

    n_benefs = 50
    n_instances = 10
    n_trials = 10
    L = 10
    K = 10
    m = 3
    gamma = 0.99
    target_policy_name = 'soft-whittle'
    beh_policy_name    = 'random'

    # dataset generation
    full_dataset  = generateDataset(n_benefs, m, n_instances, n_trials, L, K, gamma)
    train_dataset = full_dataset[:int(n_instances*0.7)]
    val_dataset   = full_dataset[int(n_instances*0.7):int(n_instances*0.8)]
    test_dataset  = full_dataset[int(n_instances*0.8):]

    dataset_list = [('train', train_dataset), ('val', val_dataset), ('test', test_dataset)]

    # model initialization
    model = ANN(m=m)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss_fn = tf.keras.losses.mean_squared_error

    # training
    training_mode = 'two-stage' if args.method == 'TS' else 'decision-focused'
    total_epoch = 10
    for epoch in range(total_epoch):
        for mode, dataset in dataset_list:
            loss_list = []
            ope_list = []
            for (feature, label, traj, simulated_rewards, mask, state_record, action_record) in tqdm.tqdm(dataset):
                feature, label = tf.constant(feature, dtype=tf.float32), tf.constant(label, dtype=tf.float32)

                with tf.GradientTape() as tape:
                    prediction = model(feature) # Transition probabilities
                    loss = tf.reduce_sum((label - prediction)**2) # Two-stage loss

                    # Batch Whittle index computation
                    w = whittleIndex(prediction)
                    
                    # ========== Non-parallel version of OPE implementation ===========
                    # This is fine in the inference part but can be slow in the training part
                    # Especially when soft top k is involved.
                    # opeIS_decomposed = opeIS(traj, w.numpy(), mask, n_benefs, L, K, n_trials, gamma,
                    #         target_policy_name, beh_policy_name)
                    # print('opeIS (original)', opeIS_decomposed)

                    # ============ Parallel version of OPE implementation =============
                    opeIS_decomposed_parallel = opeIS_parallel(state_record, action_record, w, mask, n_benefs, L, K, n_trials, gamma,
                            target_policy_name, beh_policy_name)
                    # print('opeIS (parallel)', opeIS_decomposed_parallel)

                    performance = -opeIS_decomposed_parallel

                # backpropagation
                if mode == 'train':
                    if training_mode == 'two-stage':
                        grad = tape.gradient(loss, model.trainable_variables)
                    elif training_mode == 'decision-focused':
                        grad = tape.gradient(performance, model.trainable_variables)
                    else:
                        raise NotImplementedError
                    optimizer.apply_gradients(zip(grad, model.trainable_variables))
                del tape

                loss_list.append(loss)
                ope_list.append(opeIS_decomposed_parallel)

            print(f'Epoch {epoch}, {mode} mode, average loss {np.mean(loss_list)}, average ope {np.mean(ope_list)}')

