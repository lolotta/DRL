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
        self.mu = tf.keras.layers.Dense(action_space, activation="tanh", kernel_initializer=tf.keras.initializers.Zeros())
        self.sigma = tf.keras.layers.Dense(action_space, activation="softplus", kernel_initializer=tf.keras.initializers.Zeros())


    def call(self, x):

        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.flatten(x)
        mu = self.mu(x)
        sigma = self.sigma(x)

        return mu, sigma
