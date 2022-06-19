import numpy as np
import tensorflow as tf

class DQN(tf.keras.Model):

    def __init__(self, observation_space:tuple = (8,), action_space:int = 4):
        super().__init__()

        self.input_layer = tf.keras.layers.Input(shape=observation_space)
        self.linear1 = tf.keras.layers.Dense(32, activation="relu")
        self.linear2 = tf.keras.layers.Dense(64, activation="relu")
        self.linear3 = tf.keras.layers.Dense(64, activation="relu")
        self.linear4 = tf.keras.layers.Dense(action_space)

        self.layers.append(self.input_layer)
        self.layers.append(self.linear1)
        self.layers.append(self.linear2)
        self.layers.append(self.linear3)
        self.layers.append(self.linear4)

    def call(self, x):

        for layer in self.layers:
            x = layer(x)

        return x

