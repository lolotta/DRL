from GridWorld import GridWorld
from Agent import Agent

environment = GridWorld()

print("1-Step SARSA")
agent = Agent((5,5), 4, n_step=1)
agent.run(environment=environment, episodes=300)
print("Q-Values: \n", agent.Q_table)

print("5-Step SARSA")
agent = Agent((5,5), 4, n_step=5)
agent.run(environment=environment, episodes=300)
print("Q-Values: \n", agent.Q_table)

print("10-Step SARSA")
agent = Agent((5,5), 4, n_step=10)
agent.run(environment=environment, episodes=300)
print("Q-Values: \n", agent.Q_table)