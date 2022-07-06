import copy
import random
import time
import gym
import numpy as np
import cv2

from HW4.Agent import Agent
from HW4.Actor import Actor
from HW4.ExperienceReplayBuffer import ReplayBuffer

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('WARNING')
tf.autograph.set_verbosity(1)


def process_state_image(state):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = state.astype(float)
    state /= 255.0
    state = np.expand_dims(state, axis=-1)
    return state

episodes = 10000
epsilon = 0.1
experiences_threshold = 256
update_interval = 100
steps = 0

env = gym.make('CarRacing-v1')
# replay_buffer = ReplayBuffer(10000)
agent: Agent = Agent(env, epsilon)
def test():
    done = False
    state = env.reset()
    state = process_state_image(state)
    episode_rewards = []
    steps = 0
    while True:

        log_probability, action = agent.take_action(np.array([state]))
        action = np.array(action[0])

        print(action)

        next_state, reward, done, info = env.step(action)
        next_state = process_state_image(next_state)

        episode_rewards.append(reward)
        steps += 1

        if done:
            state = env.reset()
            state = process_state_image(state)
            done = False
            break

        state = copy.copy(next_state)

    print(f"Reward:{sum(episode_rewards)} Steps: {steps}")


def train():
    trajectories = []
    for i in range(5):
        trajectory = agent.sample_trajectory()
        trajectories.append(trajectory)

    agent.learn(trajectories)


for episode in range(episodes):
    train()
    test()

# Close the env
env.close()



