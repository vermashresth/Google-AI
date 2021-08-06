import tensorflow as tf
import numpy as np
import sys
import os

from ..utils import load_obj, save_obj


def get_train_val_test(config):
    train_d, val_d, test_d = get_datasets(config["problem"])

    train = get_generator(train_d, config)
    train_num_batches = get_num_batches(train_d, config)

    val = get_generator(val_d, config)
    val_num_batches = get_num_batches(val_d, config)

    test = get_generator(test_d, config)
    test_num_batches = get_num_batches(test_d, config)

    return (
        (train, train_num_batches, train_d),
        (val, val_num_batches, val_d),
        (test, test_num_batches, test_d),
    )


def get_generator(dataset, config):
    generator = tf.data.Dataset.from_generator(
        lambda: data_generator(dataset, config),
        output_types=(
            {
                "static": tf.float32,
                "dynamic": tf.float32,
                "ngo_hosp_id": tf.float32,
                "gest_age": tf.float32,
            },
            tf.int8,
        ),
        output_shapes=(
            {
                "static": tf.TensorShape(
                    (None, config["model"]["static_features_dim"])
                ),
                "dynamic": tf.TensorShape(
                    (None, None, config["model"]["dynamic_features_dim"],)
                ),
                "ngo_hosp_id": tf.TensorShape(
                    (None, config["model"]["ngo_hosp_id_dim"],)
                ),
                "gest_age": tf.TensorShape(
                    (None, None, config["model"]["gest_ages_dim"])
                ),
            },
            tf.TensorShape((None,)),
        ),
    ).prefetch(tf.data.experimental.AUTOTUNE)

    return generator


def data_generator(dataset, config):
    num_examples = dataset[0].shape[0]
    num_batches = int(np.ceil(num_examples / config["train"]["batch_size"]))

    enroll_gest_age_mean = np.mean(dataset[3][:, 0])
    enroll_gest_age_std = np.std(dataset[3][:, 0])

    days_to_first_call_mean = np.mean(dataset[3][:, 7])
    days_to_first_call_std = np.std(dataset[3][:, 7])

    while True:
        perm = np.random.permutation(num_examples)
        shuffled_dataset = tuple([item[perm] for item in list(dataset)])

        for batch_no in range(num_batches):
            start_idx = batch_no * config["train"]["batch_size"]
            stop_idx = (batch_no + 1) * config["train"]["batch_size"]
            batch_dataset = tuple([item[start_idx:min(stop_idx, num_examples)] for item in list(shuffled_dataset)])

            (user_ids, dynamic_xs, gest_ages, static_xs, ngo_hosp_ids, labels) = batch_dataset

            # dynamic features preprocessing
            dynamic_xs = dynamic_xs.astype(np.float32)
            dynamic_xs[:, :, 2] = dynamic_xs[:, :, 2] / 60
            dynamic_xs[:, :, 3] = dynamic_xs[:, :, 3] / 60
            dynamic_xs[:, :, 4] = dynamic_xs[:, :, 4] / 12

            # static features preprocessing
            static_xs = static_xs.astype(np.float32)
            static_xs[:, 0] = (static_xs[:, 0] - enroll_gest_age_mean) 
            static_xs[:, 7] = (static_xs[:, 7] - days_to_first_call_mean)
            
            yield {
                "static": static_xs,
                "dynamic": dynamic_xs,
                "gest_age": gest_ages,
                "ngo_hosp_id": ngo_hosp_ids,
            }, labels


def get_num_batches(dataset, config):
    num_examples = dataset[0].shape[0]
    num_batches = int(np.ceil(num_examples / config["train"]["batch_size"]))

    return num_batches


def get_datasets(config):
    dataset_file_name = os.path.join("data", "saves", config["label"], str(config["risk_threshold"]), str(config["min_denominator"]))
    if os.path.exists(dataset_file_name):
        train = load_obj(os.path.join(dataset_file_name, "train_1month.pkl"))
        val = load_obj(os.path.join(dataset_file_name, "val_1month.pkl"))
        test = load_obj(os.path.join(dataset_file_name, "test_1month.pkl"))
    else:
        numerator_idx, denominator_idx = None, None
        if config["label"] == "E2C":
            numerator_idx, denominator_idx = 3, 2
        elif config["label"] == "E2SA":
            numerator_idx, denominator_idx = 3, 1
        elif config["label"] == "C2SA":
            numerator_idx, denominator_idx = 2, 1
        elif config["label"] == "SA2A":
            numerator_idx, denominator_idx = 1, 0

        dataset = load_obj("data/saves/dataset_1month.pkl")
        user_ids, dynamic_xs, gest_ages, static_xs, ngo_hosp_ids, labels = dataset
        valid_idx = np.where(labels[:, denominator_idx] >= config["min_denominator"])

        dataset = tuple([item[valid_idx] for item in dataset])
        user_ids, dynamic_xs, gest_ages, static_xs, ngo_hosp_ids, labels = dataset
        ratios = labels[:, numerator_idx]/labels[:, denominator_idx]
        labels = (ratios < config["risk_threshold"])*1
        labels = labels.astype(np.int8)

        dataset = (user_ids, dynamic_xs, gest_ages, static_xs, ngo_hosp_ids, labels)

        mask1 = np.random.rand(dataset[0].shape[0]) < 0.70
        train_idx = dataset[0][mask1]
        rest_idx = dataset[0][~mask1]

        mask2 = np.random.rand(rest_idx.shape[0]) < 0.5
        val_idx = rest_idx[mask2]
        test_idx = rest_idx[~mask2]

        train = tuple([item[mask1] for item in dataset])
        rest = tuple([item[~mask1] for item in dataset])

        val= tuple([item[mask2] for item in rest])
        test = tuple([item[~mask2] for item in rest])

        save_obj(train, os.path.join(dataset_file_name, "train.pkl"))
        save_obj(val, os.path.join(dataset_file_name, "val.pkl"))
        save_obj(test, os.path.join(dataset_file_name, "test.pkl"))

    print("Training Size:", train[0].shape[0], "; High Risk", train[5][np.where(train[5]==1)].shape[0], "; Low Risk", train[5][np.where(train[5]==0)].shape[0])
    print("Validation Size:", val[0].shape[0], "; High Risk", val[5][np.where(val[5]==1)].shape[0], "; Low Risk", val[5][np.where(val[5]==0)].shape[0])
    print("Testing Size:", test[0].shape[0], "; High Risk", test[5][np.where(test[5]==1)].shape[0], "; Low Risk", test[5][np.where(test[5]==0)].shape[0])

    return train, val, test