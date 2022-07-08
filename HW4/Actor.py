import random

import numpy as np
import tensorflow as tf

class Actor(tf.keras.Model):

    def __init__(self, observation_space:tuple = (96,96,1), action_space:int = 3):
        super().__init__()

        self.action_space = action_space

        self.input_layer = tf.keras.layers.Input(shape=observation_space)
        self.conv_layer1 = tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu")
        self.conv_layer2 = tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu")
        self.max_pool1 = tf.keras.layers.MaxPooling2D(pool_size=3)
        self.conv_layer3 = tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu")
        self.conv_layer4 = tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu")
        self.max_pool2 = tf.keras.layers.MaxPooling2D(pool_size=3)

        # self.flatten = tf.keras.layers.Flatten()
        self.globalPooling = tf.keras.layers.GlobalAvgPool2D()

        # Noch eine Dense Layer jeweils
        self.mu1 = tf.keras.layers.Dense(1, activation="tanh")
        self.mu2 = tf.keras.layers.Dense(1, activation="sigmoid")
        self.mu3 = tf.keras.layers.Dense(1, activation="sigmoid")
        self.sigma1 = tf.keras.layers.Dense(1, activation="softplus")
        self.sigma2 = tf.keras.layers.Dense(1, activation="softplus")
        self.sigma3 = tf.keras.layers.Dense(1, activation="softplus")


    def call(self, x):

        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.max_pool1(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.max_pool2(x)
        x = self.globalPooling(x)

        mu1 = self.mu1(x)
        mu2 = self.mu2(x)
        mu3 = self.mu3(x)

        sigma1 = self.sigma1(x) # linear activation function --> Parametrisiert taking Exp
        sigma2 = self.sigma2(x) # linear activation function --> Parametrisiert taking Exp
        sigma3 = self.sigma3(x) # linear activation function --> Parametrisiert taking Exp

        # sigma1 = tf.exp(sigma1)
        # sigma2 = tf.exp(sigma2)
        # sigma3 = tf.exp(sigma3)

        sigma1 = tf.clip_by_value(sigma1, 0.01, 1.0)
        sigma2 = tf.clip_by_value(sigma2, 0.01, 1.0)
        sigma3 = tf.clip_by_value(sigma3, 0.01, 1.0)

        return tf.concat([mu1, mu2, mu3], 1), tf.concat([sigma1, sigma2, sigma3], 1)
