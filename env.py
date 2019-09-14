from abc import ABC, abstractmethod
from threading import Thread


class Env(ABC):
    def __init__(self):
        self.model = None
        self.running = False
        self.thread = None
        self.timestep = 0
        super().__init__()

    @abstractmethod
    def step(self, actions):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def visualize(self):
        pass

    def __run(self):
        action = None
        while self.running:
            observation, reward = self.step(action)
            action = self.model.step(observation)
            if reward is not None:
                self.model.reward(reward)
            self.timestep += 1

    def start(self, model, visualize=False):
        if not self.running:
            self.running = True
            self.model = model
            self.thread = Thread(target=self.__run)
            self.thread.start()
            if visualize:
                self.visualize()

    def stop(self):
        if self.running:
            self.running = False
            self.thread.join()
            self.thread = None
            self.model = None
