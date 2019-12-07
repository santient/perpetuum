import torch


class SNA:
    def __init__(self, input_neurons, hidden_neurons, output_neurons, device="cpu"):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.total_neurons = input_neurons + hidden_neurons + output_neurons
        self.device = torch.device(device)
        self.potentials = torch.zeros(self.total_neurons, device=self.device)
        self.weights = torch.zeros(self.hidden_neurons + self.output_neurons, self.input_neurons + self.hidden_neurons, device=self.device)
        self.mask = torch.zeros_like(self.weights, dtype=torch.bool)
        self.mask[-self.output_neurons:, :self.input_neurons] = False
        self.mask[:self.hidden_neurons, -self.hidden_neurons:] = torch.eye(self.hidden_neurons, self.hidden_neurons, dtype=torch.bool, device=self.device)

    def initialize(self, density):
        self.weights.uniform_(-1, 1)
        self.weights[torch.empty_like(self.weights).uniform_(0, 1) > density] = 0
        self.weights[self.mask] = 0

    def step(self, inputs=None):
        if inputs is not None:
            self.potentials[:self.input_neurons] += inputs
        fire = self.potentials >= 1
        delta = torch.sum(self.weights[:, fire[:-self.output_neurons]], dim=1)
        self.potentials[self.input_neurons:] += delta
        self.potentials[fire] = 0
        self.potentials[self.potentials < 0] = 0
        outputs = fire[-self.output_neurons:]
        return outputs

    def reset(self):
        self.potentials[:] = 0
