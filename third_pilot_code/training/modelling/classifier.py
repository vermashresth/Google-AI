import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import load_model

from .models import get_conv_model, get_rnn_model
from .metrics import BinaryAccuracy, Precision, Recall, F1

class Classifier:
    def __init__(self, CONFIG):

        self.CONFIG = CONFIG

        if CONFIG["train"]["load"]:
            self.model = load_model(CONFIG["train"]["model_path"])
        else:
            if CONFIG["model"]["architecture"] == "cnn":
                print("Using CNN Architecture")
                self.model = get_conv_model(CONFIG["model"])
            else:
                print("Using RNN Architecture")
                self.model = get_rnn_model(CONFIG["model"])

            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(CONFIG["train"]["learning_rate"]),
                loss=[BinaryCrossentropy(from_logits=2)],
                metrics=[
                    BinaryAccuracy(name="binary_accuracy-" + str(CONFIG["problem"]["risk_threshold"])),
                    Precision(name="precision-" + str(CONFIG["problem"]["risk_threshold"]) + "-0", class_id=0),
                    Precision(name="precision-" + str(CONFIG["problem"]["risk_threshold"]) + "-1", class_id=1),
                    Recall(name="recall-" + str(CONFIG["problem"]["risk_threshold"]) + "-0", class_id=0),
                    Recall(name="recall-" + str(CONFIG["problem"]["risk_threshold"]) + "-1", class_id=1),
                    F1(name="F1-" + str(CONFIG["problem"]["risk_threshold"]) + "-0", type="0"),
                    F1(name="F1-" + str(CONFIG["problem"]["risk_threshold"]) + "-1", type="1"),
                ],
            )

    def train(self, train_dataset, val_dataset, num_batches, num_val_batches):

        log_dir = os.path.join(self.CONFIG["train"]["logs_path"],)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            profile_batch=0,
            update_freq=self.CONFIG["train"]["log_update_freq"],
        )

        history = self.model.fit(
            train_dataset,
            epochs=self.CONFIG["train"]["epochs"],
            class_weight=self.CONFIG["train"]["class_weights"],
            validation_data=val_dataset,
            validation_steps=num_val_batches,
            verbose=1,
            steps_per_epoch=num_batches,
            callbacks=[tensorboard_callback],
            workers=8
        )

        self.model.save(self.CONFIG["train"]["model_path"])

        with open(
            os.path.join(self.CONFIG["train"]["model_path"], "config.txt"), "w"
        ) as f:
            print(self.CONFIG, file=f)

    def test(self, test_dataset, num_batches=None):
        if num_batches:
            result = self.model.evaluate(test_dataset, steps=num_batches)
        else:
            result = self.model.evaluate(test_dataset)
        
        return dict(zip(self.model.metrics_names, result))

    def predict(self, test_dataset, num_batches=None):
        if num_batches:
            return self.model.predict(test_dataset, steps=num_batches)
        else:
            return self.model.predict(test_dataset)