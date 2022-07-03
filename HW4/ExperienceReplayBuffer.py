from typing import NamedTuple

import numpy as np
from collections import deque, namedtuple
import random

class ReplayBuffer:

    experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    memory_capacity: int
    memory: deque

    def __init__(self, memory_capacity=100000):
        self.memory_capacity = memory_capacity
        self.memory = deque(maxlen=memory_capacity)


    def add(self, state, action, reward, next_state, done):
        experience = self.experience(state, action, reward, next_state, done)

        self.memory.append(experience)


    def sample(self, sample_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        experiences = random.sample(self.memory, sample_size)

        states = np.squeeze(np.vstack([experience.state for experience in experiences if experience is not None]))
        actions = np.squeeze(np.vstack([experience.action for experience in experiences if experience is not None]))
        rewards = np.squeeze(np.vstack([experience.reward for experience in experiences if experience is not None]))
        next_states = np.squeeze(np.vstack([experience.next_state for experience in experiences if experience is not None]))
        dones = np.squeeze(np.vstack([experience.done for experience in experiences if experience is not None]))

        return (states, actions, rewards, next_states, dones)



