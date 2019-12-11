import numpy
import torch

from env.env import Env


class Inertia(Env):
    def __init__(self, map_height, map_width, food_density, hazard_density, speed=1e-2, episode_steps=10000, device=None):
        super().__init__()
        self.map_height = map_height
        self.map_width = map_width
        self.food_density = food_density
        self.hazard_density = hazard_density
        self.speed = speed
        self.episode_steps = episode_steps
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        self.map = None
        # self.food_count = None
        self.position = None
        self.velocity = None
        self.score = 0
        self.reset()

    def __random_teleport(self):
        self.velocity[0] = 0
        self.velocity[1] = 0
        self.position[0] = torch.randint(self.map_height, ()).item()
        self.position[1] = torch.randint(self.map_width, ()).item()

    # def __count_food(self):
    #     return Counter(self.map.flatten())["+"]

    def step(self, action):
        if action is not None:
            if action[0]:
                self.velocity[0] += self.speed
            if action[1]:
                self.velocity[0] -= self.speed
            if action[2]:
                self.velocity[1] += self.speed
            if action[3]:
                self.velocity[1] -= self.speed
        self.position += self.velocity
        if self.position[0] < 0 \
        or self.position[0] >= self.map_height \
        or self.position[1] < 0 \
        or self.position[1] >= self.map_width:
            self.__random_teleport()
        # if self.position[0] < 0:
        #     self.position[0] = 0
        #     self.velocity[0] = 0
        # elif self.position[0] >= self.map_height:
        #     self.position[0] = self.map_height - 1
        #     self.velocity[0] = 0
        # if self.position[1] < 0:
        #     self.position[1] = 0
        #     self.velocity[1] = 0
        # elif self.position[1] >= self.map_width:
        #     self.position[1] = self.map_width - 1
        #     self.velocity[1] = 0
        pos = self.position.astype(int)
        feat = self.map[tuple(pos)]
        if feat == "+":
            # self.food_count -= 1
            reward = 1.0
        elif feat == "x":
            reward = -1.0
        else:
            reward = None
        self.map[tuple(pos)] = " "
        observation = torch.zeros(13, dtype=torch.float32, device=self.device)
        for x in range(pos[0] + 1, self.map_height + 1):
            dist = abs(x - self.position[0]) + 1
            if x == self.map_height:
                observation[2] = 1 / dist
            elif self.map[x, pos[1]] == "+":
                observation[0] = 1 / dist
                break
            elif self.map[x, pos[1]] == "x":
                observation[1] = 1 / dist
                break
        for x in reversed(range(-1, pos[0])):
            dist = abs(x - self.position[0])
            if x == -1:
                observation[5] = 1 / dist
            elif self.map[x, pos[1]] == "+":
                observation[3] = 1 / dist
                break
            elif self.map[x, pos[1]] == "x":
                observation[4] = 1 / dist
                break
        for y in range(pos[1] + 1, self.map_width + 1):
            dist = abs(y - self.position[1]) + 1
            if y == self.map_width:
                observation[8] = 1 / dist
            elif self.map[pos[0], y] == "+":
                observation[6] = 1 / dist
                break
            elif self.map[pos[0], y] == "x":
                observation[7] = 1 / dist
                break
        for y in reversed(range(-1, pos[1])):
            dist = abs(y - self.position[1])
            if y == -1:
                observation[11] = 1 / dist
            elif self.map[pos[0], y] == "+":
                observation[9] = 1 / dist
                break
            elif self.map[pos[0], y] == "x":
                observation[10] = 1 / dist
                break
        observation[12] = 1
        if reward is not None:
            # print(reward)
            self.score += reward
        if self.timestep == self.episode_steps:
            print("Episode score:", self.score)
            self.reset()
        # print(action, observation, reward)
        return observation, reward

    def reset(self):
        self.map = numpy.random.choice(
            ["+", "x", " "],
            size=(self.map_height, self.map_width),
            p=[self.food_density, self.hazard_density, 1 - self.food_density - self.hazard_density])
        # self.food_count = self.__count_food()
        self.position = numpy.zeros(2)
        self.velocity = numpy.zeros(2)
        self.score = 0
        self.timestep = 0
        self.__random_teleport()
        if self.model is not None:
            self.model.prune(0.01)

    def visualize(self):
        print(self.map)
        while True:
            print(self.timestep)
            # print(self.position, self.velocity, self.score, end="\r", flush=True)
            # print(end="\r", flush=True)
