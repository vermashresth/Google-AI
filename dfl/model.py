import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Softmax, Dropout
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
    def __init__(self, n_states):
        '''
        init
            m: state size
        input:
            x: feature of size 16
        output
            transition prob of size 2*m
        '''
        super(ANN, self).__init__()
        self.n_states = n_states
        self.d1 = Dense(64, activation='relu', trainable=True)
        # self.d2 = Dense(64, activation='relu', trainable=True)
        self.dropout = Dropout(.2)
        self.d2 = Dense(n_states*2*n_states, activation=None, trainable=True)
        self.softmax = Softmax()

    def call(self, x):
        x1 = self.d1(x)
        x1 = self.dropout(x1)
        x2 = self.d2(x1)
        x2 = self.dropout(x2)
        # x3 = self.d3(x2)

        output_raw = tf.reshape(x2, (-1, self.n_states, 2, self.n_states))
        output = self.softmax(output_raw)

        # print(output[0])

        return output


