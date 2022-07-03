import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class Actor(tf.keras.Model):

    def __init__(self, observation_space:tuple = (96,96,1), action_space:int = 3):
        super().__init__()

        self.action_space = action_space

        self.input_layer = tf.keras.layers.Input(shape=observation_space)
        self.conv_layer1 = tf.keras.layers.Conv2D(32, kernel_size=2, activation="tanh")
        self.conv_layer2 = tf.keras.layers.Conv2D(64, kernel_size=2, activation="tanh")
        self.flatten = tf.keras.layers.Flatten()
        self.linear1 = tf.keras.layers.Dense(action_space, activation=None)

        self.layers.append(self.input_layer)
        self.layers.append(self.conv_layer1)
        self.layers.append(self.conv_layer2)
        self.layers.append(self.flatten)
        self.layers.append(self.linear1)

    def call(self, x):

        for layer in self.layers:
            x = layer(x)

        return x

    def get_state_action_log_probability(self, states, actions):
        logits = self.call(states)

        log_probabilities_all = tf.nn.log_softmax(logits)
        log_probabilities = tf.reduce_sum(actions * log_probabilities_all)

        return log_probabilities

    def get_state_log_probabilities(self, states):
        logits = self.call(states)

        log_probabilities_all = tf.nn.log_softmax(logits)

        actions_sampled = np.random.sample(shape=(states.shape[0], self.action_space)) # TODO Corner case if a single state comes as input_layer

        log_probabilities = tf.reduce_sum(actions_sampled * log_probabilities_all)

        return log_probabilities

    def get_log_probabilities(self, logits, actions):
        log_probabilities_all = tf.nn.log_softmax(logits)
        log_probabilities = tf.reduce_sum(log_probabilities_all * actions)

        return log_probabilities
