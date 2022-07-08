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
        # self.critic = Critic()

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
            log_probabilities_buffer.append(log_probability)

            step += 1
            state = copy.copy(next_state)

            if step == step_size:
                break

        return (state_buffer, action_buffer, reward_buffer, log_probabilities_buffer)


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
        self.train_actor(trajectories)
        # self.train_critic(trajectories)

    def train_actor(self, trajectories):
        accumulated_gradients = [tf.zeros_like(trainable_variables) for trainable_variables in self.actor.trainable_variables]
        for trajectory in trajectories:
            states, actions, rewards, log_probabilities = trajectory
            states = np.vstack(states)
            # states = np.array(states)
            # actions = np.array(actions)
            # rewards = np.array(rewards)
            # log_probabilities = np.array(log_probabilities)

            # np.squeeze(np.vstack())

            reward_to_go = [np.sum(rewards[i:] * (self.gamma ** np.array(range(i, len(rewards))))) for i in range(len(rewards))]

            with tf.GradientTape() as tape:
                log_probabilities, actions = self.take_action(states, action=actions, training=True)
                loss = tf.reduce_sum(- log_probabilities * reward_to_go)

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

    def train_critic(self, trajectories):
        accumulated_gradients = [tf.zeros_like(trainable_variables) for trainable_variables in self.actor.trainable_variables]
        for trajectory in trajectories:
            states, actions, rewards, log_probabilities = trajectory

            with tf.GradientTape() as tape:
                value_prediction = self.critic(states, training=True)
                assert value_prediction.shape == tdc_targets.shape
                mse = tf.keras.losses.MeanSquaredError()
                mse(tf.stop_gradient(td_targets), value_prediction)
                loss = mse(tf.stop_gradient(td_targets), value_prediction)

            gradients = tape.gradient(loss, self.critic.trainable_variables)
            accumulated_gradients = [(acum_grad + grad) for acum_grad, grad in zip(accumulated_gradients, gradients)]
        accumulated_gradients = [this_grad / len(trajectories) for this_grad in accumulated_gradients]
        self.optimizer_critic.apply_gradients(zip(accumulated_gradients, self.critic.trainable_variables))

    def gae_target(self, rewards, v_values, next_v_value, done):
        n_step_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + self.gamma * forward_val - v_values[k]
            gae_cumulative = self.gamma * self.lmbda * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]
        return gae, n_step_targets