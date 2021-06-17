import os

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Dense,
    Conv1D,
    GlobalAveragePooling1D,
    BatchNormalization,
    Embedding,
    Masking,
    RNN,
    LSTMCell,
    concatenate,
)
from tensorflow.keras.regularizers import L2
from ..utils import load_obj

gest_dict = load_obj(os.path.join("training", "res", "gest_dict.pkl"))
ngo_hosp_dict = load_obj(os.path.join("training", "res", "ngo_hosp_dict.pkl"))


def get_conv_model(CONFIG):

    static_input = Input(
        shape=(CONFIG["static_features_dim"],), dtype=tf.float32, name="static"
    )

    ngo_hosp_id_input = Input(
        shape=(CONFIG["ngo_hosp_id_dim"],), dtype=tf.float32, name="ngo_hosp_id"
    )

    dynamic_input = Input(
        shape=(None, CONFIG["dynamic_features_dim"]), dtype=tf.float32, name="dynamic",
    )

    gest_ages_input = Input(
        shape=(None, CONFIG["gest_ages_dim"]), dtype=tf.float32, name="gest_age",
    )

    ngo_hosp_id_embedding = Embedding(
        input_dim=len(ngo_hosp_dict.keys()) + 1,
        output_dim=CONFIG["ngo_hosp_id_embedding_dim"],
    )(ngo_hosp_id_input)[:, 0]

    gest_age_embedding = Embedding(
        input_dim=len(gest_dict.keys()) + 1,
        output_dim=CONFIG["gest_age_embedding_dim"],
    )(gest_ages_input)[:, :, 0]

    static_x = concatenate([static_input, ngo_hosp_id_embedding])

    for units, activation, batch_norm in zip(
        CONFIG["static_units"],
        CONFIG["static_activation"],
        CONFIG["static_batch_norm"],
    ):
        static_x = Dense(units=units, activation=activation)(static_x)

        if batch_norm:
            static_x = BatchNormalization()(static_x)

    dynamic_x = concatenate([dynamic_input, gest_age_embedding])

    for filters, kernel_size, activation in zip(
        CONFIG["dynamic_filters"],
        CONFIG["dynamic_kernel_sizes"],
        CONFIG["dynamic_activation"],
    ):
        dynamic_x = Conv1D(filters, kernel_size, activation=activation)(dynamic_x)

    dynamic_x = GlobalAveragePooling1D()(dynamic_x)

    concat_x = concatenate([static_x, dynamic_x],)

    for units, activation, batch_norm in zip(
        CONFIG["concat_units"],
        CONFIG["concat_activation"],
        CONFIG["concat_batch_norm"],
    ):
        concat_x = Dense(units=units, activation=activation)(concat_x)

        if batch_norm:
            concat_x = BatchNormalization()(concat_x)

    model = Model(
        inputs=[static_input, dynamic_input, ngo_hosp_id_input, gest_ages_input],
        outputs=[concat_x],
    )

    return model


def get_rnn_model(CONFIG):

    static_input = Input(
        shape=(CONFIG["static_features_dim"],), dtype=tf.float32, name="static"
    )

    ngo_hosp_id_input = Input(
        shape=(CONFIG["ngo_hosp_id_dim"],), dtype=tf.float32, name="ngo_hosp_id"
    )

    dynamic_input = Input(
        shape=(None, CONFIG["dynamic_features_dim"]), dtype=tf.float32, name="dynamic",
    )

    gest_ages_input = Input(
        shape=(None, CONFIG["gest_ages_dim"]), dtype=tf.float32, name="gest_age",
    )

    ngo_hosp_id_embedding = Embedding(
        input_dim=len(ngo_hosp_dict.keys()) + 1,
        output_dim=CONFIG["ngo_hosp_id_embedding_dim"],
    )(ngo_hosp_id_input)[:, 0]

    gest_age_embedding = Embedding(
        input_dim=len(gest_dict.keys()) + 1,
        output_dim=CONFIG["gest_age_embedding_dim"],
    )(gest_ages_input)[:, :, 0]

    static_x = concatenate([static_input, ngo_hosp_id_embedding])

    for units, activation, batch_norm in zip(
        CONFIG["static_units"],
        CONFIG["static_activation"],
        CONFIG["static_batch_norm"],
    ):
        static_x = Dense(
          units=units, 
          activation=activation,
          kernel_regularizer=L2(l2=CONFIG["lambda"])
        )(static_x)

        if batch_norm:
            static_x = BatchNormalization()(static_x)

    dynamic_x = concatenate([dynamic_input, gest_age_embedding])

    dynamic_x = Masking(mask_value=0.0)(dynamic_x)

    dynamic_x = RNN(
      cell=LSTMCell(
        CONFIG["lstm_units"],
        kernel_regularizer=L2(l2=CONFIG["lambda"]))
    )(dynamic_x)

    concat_x = concatenate([static_x, dynamic_x],)

    for units, activation, batch_norm in zip(
        CONFIG["concat_units"],
        CONFIG["concat_activation"],
        CONFIG["concat_batch_norm"],
    ):
        concat_x = Dense(
          units=units, 
          activation=activation,
          kernel_regularizer=L2(l2=CONFIG["lambda"])
        )(concat_x)

        if batch_norm:
            concat_x = BatchNormalization()(concat_x)

    model = Model(
        inputs=[static_input, dynamic_input, ngo_hosp_id_input, gest_ages_input],
        outputs=[concat_x],
    )

    return model


if __name__ == "__main__":
    MODEL_CONFIG = {
        "static_features_dim": 50,
        "dynamic_features_dim": 4,
        "ngo_hosp_id_dim": 1,
        "gest_ages_dim": 1,
        "ngo_hosp_id_embedding_dim": 4,
        "gest_age_embedding_dim": 4,
        "static_units": [8],
        "static_activation": ["relu"],
        "static_batch_norm": [True],
        "dynamic_filters": [8, 8],
        "dynamic_kernel_sizes": [3, 3],
        "dynamic_activation": ["relu", "relu"],
        "concat_units": [8, 1],
        "concat_activation": ["relu", "sigmoid"],
        "concat_batch_norm": [True, False],
    }
    conv_model = get_conv_model(MODEL_CONFIG)

    MODEL_CONFIG = {
        "static_features_dim": 50,
        "dynamic_features_dim": 4,
        "ngo_hosp_id_dim": 1,
        "gest_ages_dim": 1,
        "ngo_hosp_id_embedding_dim": 4,
        "gest_age_embedding_dim": 4,
        "static_units": [8],
        "static_activation": ["relu"],
        "static_batch_norm": [True],
        "lstm_units": 10,
        "concat_units": [8, 1],
        "concat_activation": ["relu", "sigmoid"],
        "concat_batch_norm": [True, False],
    }
    rnn_model = get_rnn_model(MODEL_CONFIG)
