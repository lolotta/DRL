import numpy as np
from typing import Tuple
import random

from GridWorld import GridWorld
class Agent:
    state_space: Tuple[int, int]
    action_space: int
    Q_table: np.ndarray
    epsilon: float
    n_step: int
    learning_rate: float
    gamma: float
    def __init__(self, state_space: int, action_space: int, epsilon: float = 0.1, n_step: int = 1, learning_rate: float = 0.1, gamma: float = 0.9) -> None:
        self.state_space = state_space
        self.action_space = action_space
        self.epsilon = epsilon
        self.n_step = n_step
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.create_Q_table()

    def create_Q_table(self):
        self.Q_table = np.zeros((self.state_space[0], self.state_space[1], self.action_space))#


    def take_action(self, state: np.ndarray):
        if random.random() < self.epsilon:
            #print(self.action_space)
            action = random.randint(0,self.action_space - 1) 
        else:
            #print(self.Q_table.shape)
            #print(state, np.argmax(self.Q_table[state]))
            action = np.argmax(self.Q_table[state[0], state[1]])
        
        return action

    def run(self, environment: GridWorld, episodes: int = 100):
        for episode in range(episodes):
            states= []
            actions = []
            rewards = []

            start_state = environment.reset()
            action = self.take_action(start_state)
            
            states.append(start_state)
            actions.append(action)

            terminal_step = float("Inf")
            tau = 0
            step = 0
            while not tau == (terminal_step - 1):
                if step < terminal_step:
                    state, reward, terminal = environment.step(action)
                    states.append(state)
                    rewards.append(reward)
                    
                    if terminal:
                        terminal_step = step + 1

                    else:
                        action = self.take_action(state)
                        actions.append(action)

                tau = step - self.n_step + 1

                if tau >= 0:
                    G_return = 0
                    for i in range(tau + 1, min(tau + self.n_step, terminal_step)):
                        G_return += self.gamma ** ( i - tau - 1) * rewards[i]

                    if tau + self.n_step < terminal_step:
                        G_return += self.gamma ** self.n_step * self.Q_table[states[tau + self.n_step][0], states[tau + self.n_step][1], actions[tau + self.n_step]] #TODO G????

                    self.Q_table[states[tau][0], states[tau][1], actions[tau]] += self.learning_rate * (G_return - self.Q_table[states[tau][0], states[tau][1], actions[tau]])

                step+=1

            


agent = Agent((5,5), 4)

agent.run(environment=GridWorld())
print(agent.Q_table)