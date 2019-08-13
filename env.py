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
        while self.running:
            actions = []
            while True:
                time, action = self.model.output()
                if action is not None:
                    actions.append((time, action))
                else:
                    break
            observation, reward = self.step(actions)
            self.model.input(observation)
            if reward is not None:
                self.model.reward(reward)
            self.timestep += 1

    def start(self, model, visualize=False):
        self.running = True
        self.model = model
        self.thread = Thread(target=self.__run)
        self.thread.start()
        self.model.start()
        if visualize:
            self.visualize()

    def stop(self):
        self.running = False
        self.model.stop()
        self.thread.join()
        self.thread = None
        self.model = None
