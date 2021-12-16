import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model

class SyntheticANN(Model):
    def __init__(self):
        super(SyntheticANN, self).__init__()
        self.d1 = Dense(64, activation='relu', trainable=False)
        self.d2 = Dense(64, activation='relu', trainable=False)
        self.d3 = Dense(16, trainable=False) # generating feature of size 16

    def call(self, x):
        x1 = self.d1(x)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        # x4 = self.d4(x3)
        return x3


class ANN(Model):
    def __init__(self):
        super(ANN, self).__init__()
        self.d1 = Dense(64, activation='relu', trainable=True, input_shape=(16,))
        self.d2 = Dense(4, activation='sigmoid', trainable=True)

    def call(self, x):
        x1 = self.d1(x)
        x2 = self.d2(x1)

        output_part = tf.reshape(x2, (-1,2,2,1))
        output = tf.concat([1.-output_part, output_part], -1)

        return output


