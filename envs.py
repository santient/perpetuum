import numpy
import torch

from env import Env


class Inertia(Env):
    def __init__(self, map_height, map_width, food_density, hazard_density, device="cpu"):
        self.map_height = map_height
        self.map_width = map_width
        self.food_density = food_density
        self.hazard_density = hazard_density
        self.device = torch.device(device)
        self.map = None
        self.position = None
        self.velocity = None
        self.score = 0
        self.reset()
        super().__init__()

    def step(self, actions):
        for t, a in actions:
            if a[0]:
                self.velocity[0] += 0.1
            if a[1]:
                self.velocity[0] -= 0.1
            if a[2]:
                self.velocity[1] += 0.1
            if a[3]:
                self.velocity[1] -= 0.1
        self.position += self.velocity
        if self.position[0] < 0:
            self.position[0] = 0
            self.velocity[0] = 0
        elif self.position[0] >= self.map_height:
            self.position[0] = self.map_height - 1
            self.velocity[0] = 0
        if self.position[1] < 0:
            self.position[1] = 0
            self.velocity[1] = 0
        elif self.position[1] >= self.map_width:
            self.position[1] = self.map_width - 1
            self.velocity[1] = 0
        pos = self.position.astype(int)
        feat = self.map[tuple(pos)]
        if feat == "+":
            reward = 1
        elif feat == "x":
            reward = -1
        else:
            reward = 0
        self.map[tuple(pos)] = " "
        observation = torch.zeros(12, dtype=torch.float32, device=self.device)
        for i, x in enumerate(range(pos[0] + 1, self.map_height + 1)):
            i += 1
            if x == self.map_height:
                observation[2] = 1 / i
            elif self.map[x, pos[1]] == "+":
                observation[0] = 1 / i
                break
            elif self.map[x, pos[1]] == "x":
                observation[1] = 1 / i
                break
        for i, x in enumerate(reversed(range(-1, pos[0]))):
            i += 1
            if x == -1:
                observation[5] = 1 / i
            elif self.map[x, pos[1]] == "+":
                observation[3] = 1 / i
                break
            elif self.map[x, pos[1]] == "x":
                observation[4] = 1 / i
                break
        for i, y in enumerate(range(pos[1] + 1, self.map_width + 1)):
            i += 1
            if y == self.map_width:
                observation[8] = 1 / i
            elif self.map[pos[0], y] == "+":
                observation[6] = 1 / i
                break
            elif self.map[pos[0], y] == "x":
                observation[7] = 1 / i
                break
        for i, y in enumerate(reversed(range(-1, pos[1]))):
            i += 1
            if y == -1:
                observation[11] = 1 / i
            elif self.map[pos[0], y] == "+":
                observation[9] = 1 / i
                break
            elif self.map[pos[0], y] == "x":
                observation[10] = 1 / i
                break
        self.score += reward
        return observation, reward

    def reset(self):
        self.map = numpy.random.choice(
            ["+", "x", " "],
            size=(self.map_height, self.map_width),
            p=[self.food_density, self.hazard_density, 1 - self.food_density - self.hazard_density])
        self.position = numpy.zeros(2)
        self.velocity = numpy.zeros(2)
        self.score = 0

    def visualize(self):
        print(self.map)
        while True:
            print(self.position.astype(int), self.velocity, self.score, end="\r", flush=True)
