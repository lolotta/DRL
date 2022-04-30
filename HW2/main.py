from GridWorld import GridWorld

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

gridWorld.visualize()
gridWorld.reset()
print(gridWorld)

gridWorld.visualize()
