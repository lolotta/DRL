import copy
import random
import time
import gym
import numpy as np

from HW3.Agent import Agent
from HW3.DQN import DQN
from HW3.ExperienceReplayBuffer import ReplayBuffer

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('WARNING')
tf.autograph.set_verbosity(1)

episodes = 10000
epsilon = 0.1
experiences_threshold = 256
update_interval = 100
steps = 0

env = gym.make('LunarLander-v2')
replay_buffer = ReplayBuffer(10000)
agent = Agent(env, epsilon)

done = False
state = env.reset()

for episode in range(episodes):
    episode_rewards = []
    while True:
        action = agent.take_action(state)

        next_state, reward, done, info = env.step(action)

        replay_buffer.add(state, action, reward, next_state, done)



        if len(replay_buffer.memory) > experiences_threshold:
            experiences = replay_buffer.sample(128)

            agent.learn(experiences)

            if steps % 100 == 0:
                agent.delayed_update()

        state = copy.copy(next_state)
        episode_rewards.append(reward)
        steps += 1

        if done:
            state = env.reset()
            done = False
            break


    print(f"Reward:{sum(episode_rewards)} Steps: {steps}")

# Close the env
env.close()



