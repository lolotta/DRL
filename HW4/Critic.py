import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class Critic(tf.keras.Model):

    def __init__(self, observation_space:tuple = (96,96,1), action_space:int = 3):
        super().__init__()

        self.action_space = action_space

        self.input_layer = tf.keras.layers.Input(shape=observation_space)
        self.conv_layer1 = tf.keras.layers.Conv2D(32, kernel_size=2, activation="tanh")
        self.conv_layer2 = tf.keras.layers.Conv2D(64, kernel_size=2, activation="tanh")
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(1, activation="linear")


    def call(self, x):

        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return x
