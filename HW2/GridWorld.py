import random
import numpy as np

class Tile:
    blocked: bool = False
    reward: float = 0.0
    terminal: bool = False
    starting: bool = False
    agent_presence: bool = False
    windiness: float = 0.0

    def __str__(self):
        property = "N"
        if self.blocked:
            property = "B"
        elif self.terminal:
            property = "T"
        elif self.starting:
            property = "S"

        agent = ""
        if self.agent_presence:
            agent = "X "

        windy = ""
        if self.windiness > 0.0:
            windy = " W"
        return f"| {agent}{property} {self.reward}{windy}|"

    def __repr__(self):
        return self.__str__()

class GridWorld:

    def __init__(self, size=(5,5)):
        self.grid_size: tuple[int, int] = size
        self.grid: np.ndarray = np.array([[Tile() for j in range(size[0])] for i in range(size[1])])
        self.hard_code_grid()

        self.agent_position: np.ndarray = np.array([0, 0])

        self.action_mapping: dict = {0: np.array([-1, 0]), 1: np.array([1, 0]), 2: np.array([0, -1]), 3: np.array([0, 1])}


    def reset(self):
        self.__init__()

    def step(self, action: int):

        self.move(action)

        state = self.get_state()
        reward = self.get_reward()
        terminal = self.get_terminal()

        return state, reward, terminal

    def visualize(self):
        print(self)

    def hard_code_grid(self):
        # Set Starting Point
        self.grid[0, 0].starting = True

        # Set Terminal Point
        self.grid[4, 4].terminal = True
        self.grid[4, 4].reward = 1.0

        # Set Blocked Tiles
        self.grid[2, 2].blocked = True
        self.grid[2, 3].blocked = True
        self.grid[3, 2].blocked = True

        # Set Negative Reward Tiles
        self.grid[0, 2].reward = -0.5
        self.grid[1, 1].reward = -0.5
        self.grid[3, 3].reward = -0.5

        # Set Undeterministic Tiles / Windy Tiles
        self.grid[3,4].windiness = 0.5
        self.grid[2,4].windiness = 0.2
        self.grid[1,3].windiness = 0.2
        self.grid[4,1].windiness = 0.2

    def __repr__(self):
        return ''.join(f'{str(list)} \n' for list in self.grid)

    def move(self, action):
        if random.random() < self.grid[tuple(self.agent_position)].windiness:
            action = random.randint(0,3)

        new_position = self.agent_position + self.action_mapping[action]

        if self.check_position(new_position):
            self.grid[tuple(self.agent_position)].agent_presence = False
            self.agent_position = new_position
            self.grid[tuple(self.agent_position)].agent_presence = True


    def check_position(self, new_position):
        if not self.check_borders(new_position):
            return False
        elif not self.check_blocked(new_position):
            return False

        return True

    def check_borders(self, new_position):
        if new_position[0] < 0 or new_position[0] >= self.grid_size[0]:
            return False

        if new_position[1] < 0 or new_position[1] >= self.grid_size[1]:
            return False

        return True

    def check_blocked(self, new_position):
        if self.grid[tuple(new_position)].blocked:
            return False
        return True

    def get_state(self):
        return self.agent_position

    def get_reward(self):
        return self.grid[tuple(self.agent_position)].reward

    def get_terminal(self):
        return self.grid[tuple(self.agent_position)].terminal


gridWorld = GridWorld()
print(gridWorld)
print(gridWorld.agent_position)
print(gridWorld.step(3))
print(gridWorld.step(1))
print(gridWorld.step(3))
print(gridWorld.step(3))
print(gridWorld.step(3))
print(gridWorld.step(1))
print(gridWorld.step(1))

print(gridWorld)
print(gridWorld.agent_position)


