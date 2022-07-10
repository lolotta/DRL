import copy

import cv2
import gym
import numpy as np

from HW4.Actor import Actor
from HW4.Critic import Critic
import random
import tensorflow as tf
import tensorflow_probability as tfp

class Agent:
    actor: tf.keras.Model
    critic: tf.keras.Model
    env: gym.Env
    epsilon: float
    gamma: float
    lmbda: float
    optimizer_actor: tf.keras.optimizers.Optimizer
    optimizer_critic: tf.keras.optimizers.Optimizer

    def __init__(self, env: gym.Env, gamma: float = 0.99, learning_rate: float = 0.0005, lmbda: float = 0.95):
        self.env = env
        self.gamma = gamma
        self.lmbda = lmbda
        self.learning_rate = learning_rate

        self.actor = Actor()
        self.critic = Critic()

        self.optimizer_actor = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.optimizer_critic = tf.optimizers.Adam(learning_rate=self.learning_rate)

    def sample_trajectory(self, step_size=None):
        step = 0
        done = False
        state = self.env.reset()
        state = self.process_state_image(state)
        state = np.array([state])

        state_buffer = []
        action_buffer = []
        reward_buffer = []
        next_state_buffer = []
        log_probabilities_buffer = []

        while not done:
            log_probability, action = self.take_action(state)
            action = np.squeeze(action.numpy())

            next_state, reward, done, info = self.env.step(action)
            next_state = self.process_state_image(next_state)
            next_state = np.array([next_state])

            state_buffer.append(state)
            action_buffer.append(action)
            reward_buffer.append(reward)
            next_state_buffer.append(next_state)
            log_probabilities_buffer.append(log_probability)

            step += 1
            state = copy.copy(next_state)

            if step == step_size:
                break

        return (state_buffer, action_buffer, reward_buffer, next_state_buffer, log_probabilities_buffer)


    def take_action(self, states, training=False, action=None):
        # Kein Numpy verwenden
        # Evaluate log_probabilities
        # Batch normalization as image normalization
        mu, sigma = self.actor(states, training=training)

        if action is None:
            epsilon = tf.random.normal([1], mean=0, stddev=1)
            action = mu + epsilon * sigma

        # Obtain pdf of Gaussian distribution
        dist = tfp.distributions.Normal(loc=mu, scale=sigma)
        log_probability = dist.log_prob(action)
        # pdf_value = tf.exp(-0.5 * ((action - mu) / (sigma))**2) * 1 / (sigma * tf.sqrt(2 * np.pi))

        # Compute log probability
        # log_probability = tf.math.log(pdf_value + 1e-5)
        log_probability = tf.reduce_sum(log_probability, 1, keepdims=True)

        return log_probability, action

    def learn(self, trajectories: tuple):
        gaes = []
        td_targets = []
        for trajectory in trajectories:
            states, actions, rewards, next_states, log_probabilities = trajectory
            states = np.vstack(states)
            next_states = np.vstack(next_states)

            values = self.critic(states)
            next_values = self.critic(next_states)

            values = tf.squeeze(values)
            next_values = tf.squeeze(next_values)

            gae, td_target = self.gae_target(rewards, values, next_values)

            gaes.append(gae)
            td_targets.append(td_target)

        self.train_actor(trajectories, gaes)
        self.train_critic(trajectories, td_targets)

    def train_actor(self, trajectories, gaes):
        accumulated_gradients = [tf.zeros_like(trainable_variables) for trainable_variables in self.actor.trainable_variables]
        for trajectory, gae in zip(trajectories, gaes):
            states, actions, rewards, next_states, log_probabilities = trajectory
            states = np.vstack(states)
            next_states = np.vstack(next_states)

            with tf.GradientTape() as tape:
                log_probabilities, actions = self.take_action(states, action=actions, training=True)
                loss = tf.reduce_sum(- log_probabilities * gae)

            gradients = tape.gradient(loss, self.actor.trainable_variables)
            accumulated_gradients = [(acum_grad + grad) for acum_grad, grad in zip(accumulated_gradients, gradients)]
        accumulated_gradients = [this_grad / len(trajectories) for this_grad in accumulated_gradients]
        self.optimizer_actor.apply_gradients(zip(accumulated_gradients, self.actor.trainable_variables))

    def process_state_image(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = state.astype(float)
        state /= 255.0
        state = np.expand_dims(state, axis=-1)
        return state

    def train_critic(self, trajectories, td_targets):
        accumulated_gradients = [tf.zeros_like(trainable_variables) for trainable_variables in self.actor.trainable_variables]
        for trajectory, td_target in zip(trajectories, td_targets):
            states, actions, rewards, next_states, log_probabilities = trajectory
            states = np.vstack(states)

            with tf.GradientTape() as tape:
                value_prediction = self.critic(states, training=True)

                mse = tf.keras.losses.MeanSquaredError()
                loss = mse(value_prediction, td_target)

            gradients = tape.gradient(loss, self.critic.trainable_variables)
            accumulated_gradients = [(acum_grad + grad) for acum_grad, grad in zip(accumulated_gradients, gradients)]
        accumulated_gradients = [this_grad / len(trajectories) for this_grad in accumulated_gradients]
        self.optimizer_critic.apply_gradients(zip(accumulated_gradients, self.critic.trainable_variables))

    def gae_target(self, rewards, values, next_value):
        n_step_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        forward_value = 0

        forward_value = next_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + self.gamma * forward_value - values[k]
            gae_cumulative = self.gamma * self.lmbda * gae_cumulative + delta
            gae[k] = tf.reduce_sum(gae_cumulative)
            forward_value = values[k]
            n_step_targets[k] = gae[k] + values[k]

        return gae, n_step_targets