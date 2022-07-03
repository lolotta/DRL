import copy

import cv2
import gym
import numpy as np

from HW4.Actor import Actor
import random
import tensorflow as tf

class Agent:
    actor: tf.keras.Model
    critic: tf.keras.Model
    env: gym.Env
    epsilon: float
    gamma: float
    optimizer: tf.keras.optimizers.Optimizer

    def __init__(self, env: gym.Env, epsilon: float = 0.15, gamma: float = 0.99, learning_rate: float = 0.001):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.actor = Actor()
        # self.critic = Critic()

        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)

    def sample_trajectory(self, step_size=None):
        step = 0
        done = False
        state = self.env.reset()
        state = self.process_state_image(state)
        state = np.array([state])


        state_buffer = []
        action_buffer = []
        reward_buffer = []
        log_probabilities_buffer = []

        while not done:
            logits, action = self.take_action(state)
            action = np.array(logits[0])
            log_probabilities = self.actor.get_log_probabilities(logits, action)

            next_state, reward, done, info = self.env.step(action)
            next_state = self.process_state_image(next_state)
            next_state = np.array([next_state])

            state_buffer.append(state)
            action_buffer.append(action)
            reward_buffer.append(reward)
            log_probabilities_buffer.append(log_probabilities)

            step += 1
            state = copy.copy(next_state)

            if step == step_size:
                break

        return (state_buffer, action_buffer, reward_buffer, log_probabilities_buffer)


    def take_action(self, states):
        logits = self.actor(states)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        return logits, action

    def learn(self, trajectories: tuple):
        states, actions, rewards, log_probabilities = trajectories
        # np.squeeze(np.vstack())

        reward_to_go = [np.sum(rewards[i:] * (self.gamma**np.array(range(i, len(rewards))))) for i in range(len(rewards))]

        with tf.GradientTape() as tape:
            states = np.array(states)
            logits, actions = self.take_action(states)
            actions = np.array(logits[0])
            log_probabilities = self.actor.get_log_probabilities(logits, actions)
            loss = log_probabilities * reward_to_go

        gradients = tape.gradient(loss, self.actor.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))

    def process_state_image(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = state.astype(float)
        state /= 255.0
        state = np.expand_dims(state, axis=-1)
        return state