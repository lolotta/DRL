import random

import numpy as np
import tensorflow as tf

class Critic(tf.keras.Model):

    def __init__(self, observation_space:tuple = (96,96,1)):
        super().__init__()

        self.input_layer = tf.keras.layers.Input(shape=observation_space)
        self.conv_layer1 = tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu")
        self.conv_layer2 = tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu")
        self.max_pool1 = tf.keras.layers.MaxPooling2D(pool_size=3)
        self.conv_layer3 = tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu")
        self.conv_layer4 = tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu")
        self.max_pool2 = tf.keras.layers.MaxPooling2D(pool_size=3)

        # self.flatten = tf.keras.layers.Flatten()
        self.globalPooling = tf.keras.layers.GlobalAvgPool2D()

        self.value = tf.keras.layers.Dense(1, activation="linear")


    def call(self, x):

        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.max_pool1(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.max_pool2(x)
        x = self.globalPooling(x)
        x = self.value(x)

        return x