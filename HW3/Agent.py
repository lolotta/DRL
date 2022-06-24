import copy

import gym
import numpy as np

from HW3.DQN import DQN
import random
import tensorflow as tf

class Agent:
    dqn: tf.keras.Model
    target_dqn: tf.keras.Model
    env: gym.Env
    epsilon: float
    gamma: float
    optimizer: tf.keras.optimizers.Optimizer
    loss_function = tf.keras.losses.MeanSquaredError()

    def __init__(self, env: gym.Env, epsilon: float = 0.15, gamma = 0.99, learning_rate = 0.001):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.dqn = DQN()
        self.target_dqn = DQN()

        self.optimizer = tf.optimizers.Adam(lr=self.learning_rate)


    def take_action(self, state):
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            state = np.array([state, ])
            q_values = self.dqn(state)
            action = np.argmax(q_values)

        return action

    def learn(self, experiences):

        states, actions, rewards, next_states, dones = experiences

        with tf.GradientTape() as tape:

            Q_prime_target_values = self.target_dqn(next_states, training=True)
            Q_prime_targets = np.max(Q_prime_target_values, axis=1)

            Q_targets = rewards + (self.gamma * Q_prime_targets * (1-dones))

            Q_values = self.dqn(states, training=True)
            # actions = np.expand_dims(actions, 1)
            Q_values = [Q_value[actions[index]] for index, Q_value in enumerate(Q_values)]

            loss = self.loss_function(Q_values, Q_targets)
            gradients = tape.gradient(loss, self.dqn.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.dqn.trainable_variables))


    def soft_update_network(self, local_network, target_network, update_ratio: float):
        for target_param, local_param in zip(target_network.get_weights(), local_network.get_weights()):
            target_param.data.copy_(update_ratio * local_param.data + (1.0 - update_ratio) * target_param.data)

    def delayed_update(self):
        self.target_dqn.set_weights(self.dqn.get_weights())