import random

import torch

from perpetuum.env import Env


class Debug(Env):
    def __init__(self, steps=1000, device=None):
        super().__init__()
        self.steps = steps
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        self.reset()

    def step(self, action):
        reward = 0.0
        if action[0]:
            reward += 2 * float(self.state) - 1
        if action[1]:
            reward -= 2 * float(not self.state) - 1
        reward *= 0.1
        self.score += reward
        self.state = bool(random.getrandbits(1))
        observation = self.get_observation()
        terminal = self.timestep == self.steps
        self.timestep += 1
        return observation, reward, terminal

    def get_observation(self):
        observation = torch.zeros(2, device=self.device)
        observation[int(self.state)] = 1.0
        return observation

    def reset(self):
        self.score = 0
        self.state = False
        self.timestep = 0
        return None
