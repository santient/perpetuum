from queue import Queue
from threading import Thread

import torch

from sna import SNA


class AsynchronousSNA(SNA):
    def __init__(self, input_neurons, hidden_neurons, output_neurons, input_limit=0, output_limit=0, device="cpu"):
        super().__init__(input_neurons, hidden_neurons, output_neurons, device)
        self.inputs = Queue(self.input_limit)
        self.outputs = Queue(self.output_limit)
        self.running = False
        self.thread = None
        self.timestep = 0

    def reset(self):
        if not self.running:
            super().reset()
            self.inputs.queue.clear()
            self.outputs.queue.clear()
            self.timestep = 0
    
    def input(self, inputs):
        if self.inputs.full():
            self.inputs.get()
        self.inputs.put(inputs)

    def output(self):
        out = None
        if not self.outputs.empty():
            out = self.outputs.get()
        return self.timestep, out

    def __run(self):
        while self.running:
            inputs = None
            if not self.inputs.empty():
                inputs = self.inputs.get()
            outputs = self.step(inputs)
            if outputs is not None:
                if self.outputs.full():
                    self.outputs.get()
                self.outputs.put(outputs)
            self.timestep += 1

    def start(self):
        if not self.running:
            self.running = True
            self.thread = Thread(target=self.__run)
            self.thread.start()

    def stop(self):
        if self.running:
            self.running = False
            self.thread.join()
            self.thread = None
