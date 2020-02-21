import torch


class SNA():
    def __init__(self, input_neurons, hidden_neurons, output_neurons, potential_decay=0.99, device=None):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.total_neurons = input_neurons + hidden_neurons + output_neurons
        self.potential_decay = potential_decay
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        self.potentials = torch.zeros(self.total_neurons, device=self.device)
        self.weights = torch.zeros(self.hidden_neurons + self.output_neurons, self.input_neurons + self.hidden_neurons, device=self.device)
        self.mask = torch.ones_like(self.weights, dtype=torch.uint8)
        self.mask[-self.output_neurons:, :self.input_neurons] = 0
        self.mask[:self.hidden_neurons, -self.hidden_neurons:] = 1 - torch.eye(self.hidden_neurons, self.hidden_neurons, dtype=torch.uint8, device=self.device)
        self.timestep = 0

    def initialize(self, density):
        self.weights.uniform_(-1, 1)
        mask = (torch.empty_like(self.weights).uniform_(0, 1) <= density).type(torch.uint8)
        self.weights *= mask * self.mask

    def prune(self, threshold):
        mask = (self.weights.abs() >= threshold).type(torch.uint8)
        self.weights *= mask

    def step(self, inputs=None):
        if inputs is not None:
            self.potentials[:self.input_neurons] += inputs
        fire = (self.potentials >= 1).type(torch.uint8)
        delta = self.weights @ fire[:-self.output_neurons].type_as(self.weights)
        self.potentials *= self.potential_decay
        self.potentials[self.input_neurons:] += delta
        self.potentials *= 1 - fire
        mask = (self.potentials >= 0).type(torch.uint8)
        self.potentials *= mask
        outputs = fire[-self.output_neurons:]
        self.timestep += 1
        return outputs

    def reset(self):
        self.potentials[:] = 0
        self.timestep = 0
