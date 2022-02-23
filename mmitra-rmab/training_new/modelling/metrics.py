import tensorflow as tf
import sys


class Precision(tf.keras.metrics.Metric):
    def __init__(self, name="precision", class_id=1, threshold=None, **kwargs):
        super(Precision, self).__init__(name=name, **kwargs)
        self.class_id = class_id
        self.threshold = threshold
        self.precision = self.add_weight(name="p", initializer="zeros")
        self.true_positives = self.add_weight(
            name="true_positives", initializer="zeros"
        )
        self.false_positives = self.add_weight(
            name="false_positives", initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        if self.threshold:
            y_true = tf.math.greater_equal(y_true, self.threshold)
            y_pred = tf.math.greater_equal(y_pred, self.threshold)
        else:
            y_true = tf.cast(y_true, tf.bool)
            y_pred = tf.cast(tf.math.round(y_pred), tf.bool)

        if self.class_id == 0:
            y_true = tf.logical_not(y_true)
            y_pred = tf.logical_not(y_pred)

        tp = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        true_positives = tf.reduce_sum(tf.cast(tp, tf.float32))
        self.true_positives.assign_add(true_positives)

        fp = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        false_positives = tf.reduce_sum(tf.cast(fp, tf.float32))
        self.false_positives.assign_add(false_positives)

        self.precision.assign(
            tf.divide(self.true_positives, self.true_positives + self.false_positives)
        )

    def result(self):
        return self.precision

    def reset_states(self):
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)


class CategoricalPrecision(tf.keras.metrics.Metric):
    def __init__(self, name="categorical_precision", class_id=None, **kwargs):
        super(CategoricalPrecision, self).__init__(name=name, **kwargs)
        self.class_id = class_id
        self.precision = self.add_weight(name="cp", initializer="zeros")
        self.true_positives = self.add_weight(
            name="true_positives", initializer="zeros"
        )
        self.false_positives = self.add_weight(
            name="false_positives", initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.argmax(y_pred, axis=-1)

        tp = tf.logical_and(tf.equal(y_true, self.class_id), tf.equal(y_pred, self.class_id))
        true_positives = tf.reduce_sum(tf.cast(tp, tf.float32))
        self.true_positives.assign_add(true_positives)

        fp = tf.logical_and(tf.not_equal(y_true, self.class_id), tf.equal(y_pred, self.class_id))
        false_positives = tf.reduce_sum(tf.cast(fp, tf.float32))
        self.false_positives.assign_add(false_positives)

        self.precision.assign(
            tf.divide(self.true_positives, self.true_positives + self.false_positives)
        )

    def result(self):
        return self.precision

    def reset_states(self):
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)


class Recall(tf.keras.metrics.Metric):
    def __init__(self, name="recall_0", class_id=1, threshold=None, **kwargs):
        super(Recall, self).__init__(name=name, **kwargs)
        self.class_id = class_id
        self.threshold = threshold
        self.recall = self.add_weight(name="r", initializer="zeros")
        self.true_positives = self.add_weight(
            name="true_positives", initializer="zeros"
        )
        self.false_negatives = self.add_weight(
            name="false_negatives", initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        if self.threshold:
            y_true = tf.math.greater_equal(y_true, self.threshold)
            y_pred = tf.math.greater_equal(y_pred, self.threshold)
        else:
            y_true = tf.cast(y_true, tf.bool)
            y_pred = tf.cast(tf.math.round(y_pred), tf.bool)

        if self.class_id == 0:
            y_true = tf.logical_not(y_true)
            y_pred = tf.logical_not(y_pred)

        tp = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        true_positives = tf.reduce_sum(tf.cast(tp, tf.float32))
        self.true_positives.assign_add(true_positives)

        fn = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
        false_negatives = tf.reduce_sum(tf.cast(fn, tf.float32))
        self.false_negatives.assign_add(false_negatives)

        self.recall.assign(
            tf.divide(self.true_positives, self.true_positives + self.false_negatives)
        )

    def result(self):
        return self.recall

    def reset_states(self):
        self.true_positives.assign(0.0)
        self.false_negatives.assign(0.0)


class CategoricalRecall(tf.keras.metrics.Metric):
    def __init__(self, name="categorical_recall", class_id=None, **kwargs):
        super(CategoricalRecall, self).__init__(name=name, **kwargs)
        self.class_id = class_id
        self.recall = self.add_weight(name="cr", initializer="zeros")
        self.true_positives = self.add_weight(
            name="true_positives", initializer="zeros"
        )
        self.false_negatives = self.add_weight(
            name="false_negatives", initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.argmax(y_pred, axis=-1)

        tp = tf.logical_and(tf.equal(y_true, self.class_id), tf.equal(y_pred, self.class_id))
        true_positives = tf.reduce_sum(tf.cast(tp, tf.float32))
        self.true_positives.assign_add(true_positives)

        fn = tf.logical_and(tf.equal(y_true, self.class_id), tf.not_equal(y_pred, self.class_id))
        false_negatives = tf.reduce_sum(tf.cast(fn, tf.float32))
        self.false_negatives.assign_add(false_negatives)

        self.recall.assign(
            tf.divide(self.true_positives, self.true_positives + self.false_negatives)
        )

    def result(self):
        return self.recall

    def reset_states(self):
        self.true_positives.assign(0.0)
        self.false_negatives.assign(0.0)


class F1(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", type="micro", threshold=None, **kwargs):
        super(F1, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.f1 = self.add_weight(name="f1", initializer="zeros")
        self.true_positives = self.add_weight(
            name="true_positives", initializer="zeros"
        )
        self.true_negatives = self.add_weight(
            name="true_negatives", initializer="zeros"
        )
        self.false_positives = self.add_weight(
            name="false_positives", initializer="zeros"
        )
        self.false_negatives = self.add_weight(
            name="false_negatives", initializer="zeros"
        )

        if type not in ["0", "1", "macro", "weighted"]:
            self.type = "weighted"
        else:
            self.type = type

    def update_state(self, y_true, y_pred, sample_weight=None):

        if self.threshold:
            y_true = tf.math.greater_equal(y_true, self.threshold)
            y_pred = tf.math.greater_equal(y_pred, self.threshold)
        else:
            y_true = tf.cast(y_true, tf.bool)
            y_pred = tf.cast(tf.math.round(y_pred), tf.bool)

        tp = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        true_positives = tf.reduce_sum(tf.cast(tp, tf.float32))
        self.true_positives.assign_add(true_positives)

        tn = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, False))
        true_negatives = tf.reduce_sum(tf.cast(tn, tf.float32))
        self.true_negatives.assign_add(true_negatives)

        fp = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        false_positives = tf.reduce_sum(tf.cast(fp, tf.float32))
        self.false_positives.assign_add(false_positives)

        fn = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
        false_negatives = tf.reduce_sum(tf.cast(fn, tf.float32))
        self.false_negatives.assign_add(false_negatives)

        total = (
            self.true_positives
            + self.true_negatives
            + self.false_positives
            + self.false_negatives
        )

        w_1 = (self.true_positives + self.false_negatives) / total
        precision_1 = self.true_positives / (self.true_positives + self.false_positives)
        recall_1 = self.true_positives / (self.true_positives + self.false_negatives)

        w_0 = (self.true_negatives + self.false_positives) / total
        precision_0 = self.true_negatives / (self.true_negatives + self.false_negatives)
        recall_0 = self.true_negatives / (self.true_negatives + self.false_positives)

        f1_1 = (2 * precision_1 * recall_1) / (precision_1 + recall_1)
        f1_0 = (2 * precision_0 * recall_0) / (precision_0 + recall_0)

        macro_f1 = (f1_0 + f1_1) / 2
        weighted_f1 = w_0 * f1_0 + w_1 * f1_1

        if self.type == "0":
            self.f1.assign(f1_0)
        elif self.type == "1":
            self.f1.assign(f1_1)
        elif self.type == "macro":
            self.f1.assign(macro_f1)
        elif self.type == "weighted":
            self.f1.assign(weighted_f1)

    def result(self):
        return self.f1

    def reset_states(self):
        self.true_positives.assign(0.0)
        self.true_negatives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)


class BinaryAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="accuracy", threshold=None, **kwargs):
        super(BinaryAccuracy, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.accuracy = self.add_weight(name="accuracy", initializer="zeros")
        self.true_positives = self.add_weight(
            name="true_positives", initializer="zeros"
        )
        self.true_negatives = self.add_weight(
            name="true_negatives", initializer="zeros"
        )
        self.false_positives = self.add_weight(
            name="false_positives", initializer="zeros"
        )
        self.false_negatives = self.add_weight(
            name="false_negatives", initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):

        if self.threshold:
            y_true = tf.math.greater_equal(y_true, self.threshold)
            y_pred = tf.math.greater_equal(y_pred, self.threshold)
        else:
            y_true = tf.cast(y_true, tf.bool)
            y_pred = tf.cast(tf.math.round(y_pred), tf.bool)

        tp = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        true_positives = tf.reduce_sum(tf.cast(tp, tf.float32))
        self.true_positives.assign_add(true_positives)

        tn = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, False))
        true_negatives = tf.reduce_sum(tf.cast(tn, tf.float32))
        self.true_negatives.assign_add(true_negatives)

        fp = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        false_positives = tf.reduce_sum(tf.cast(fp, tf.float32))
        self.false_positives.assign_add(false_positives)

        fn = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
        false_negatives = tf.reduce_sum(tf.cast(fn, tf.float32))
        self.false_negatives.assign_add(false_negatives)

        total = (
            self.true_positives
            + self.true_negatives
            + self.false_positives
            + self.false_negatives
        )

        self.accuracy.assign((self.true_positives + self.true_negatives) / total)

    def result(self):
        return self.accuracy

    def reset_states(self):
        self.true_positives.assign(0.0)
        self.true_negatives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)
