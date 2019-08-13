from queue import Queue
from threading import Thread

import torch


class SNA():
    def __init__(self, input_neurons, hidden_neurons, output_neurons, limit=0, device="cpu"):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.total_neurons = input_neurons + hidden_neurons + output_neurons
        self.limit = limit
        self.device = torch.device(device)
        self.potentials = torch.zeros(self.total_neurons, device=self.device)
        self.weights = torch.zeros(self.hidden_neurons + self.output_neurons, self.input_neurons + self.hidden_neurons, device=self.device)
        self.mask = torch.zeros_like(self.weights, dtype=torch.bool)
        self.mask[-self.output_neurons:, :self.input_neurons] = False
        self.mask[:self.hidden_neurons, -self.hidden_neurons:] = torch.eye(self.hidden_neurons, self.hidden_neurons, dtype=torch.bool, device=self.device)
        self.inputs = Queue()
        self.outputs = Queue(limit)
        self.running = False
        self.thread = None
        self.timestep = 0

    def initialize(self, density):
        self.weights.uniform_(-1, 1)
        self.weights[torch.empty_like(self.weights).uniform_(0, 1) > density] = 0
        self.weights[self.mask] = 0

    def step(self, inputs=None):
        if inputs is not None:
            self.potentials[:self.input_neurons] += inputs
        fire = self.potentials >= 1
        self.potentials[self.input_neurons:] += self.weights.mv(fire[:-self.output_neurons].float())
        self.potentials[fire] = 0
        self.potentials.clamp_(min=0)
        outputs = None
        if fire[-self.output_neurons:].any():
            outputs = fire[-self.output_neurons:]
        return outputs

    def reset(self):
        if not self.running:
            self.potentials[:] = 0
            self.inputs.queue.clear()
            self.outputs.queue.clear()
            self.timestep = 0
    
    def input(self, inputs):
        self.inputs.put(inputs)

    def output(self):
        out = None
        if not self.outputs.empty():
            out = self.outputs.get()
        return self.timestep, out

    def run(self):
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
        self.running = True
        self.thread = Thread(target=self.run)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        self.thread = None
