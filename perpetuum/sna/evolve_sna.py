import torch

from sna.sna import SNA


class EvolvingSNA(SNA):
    def __init__(self, input_neurons, hidden_neurons, output_neurons, potential_decay=0.99, device=None):
        super().__init__(input_neurons, hidden_neurons, output_neurons, potential_decay, device)

    def mutate(self, density, intensity):
        sign = self.weights.sign()
        delta = torch.empty_like(self.weights).uniform_(-intensity, intensity)
        delta[torch.empty_like(self.weights).uniform_(0, 1) > density] = 0
        self.weights += delta
        self.weights[self.mask | (self.weights.sign() == -sign)] = 0
        self.weights.clamp_(-1, 1)
