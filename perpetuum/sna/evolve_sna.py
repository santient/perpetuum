import torch

from perpetuum.sna import SNA


class EvolvingSNA(SNA):
    def __init__(self, input_neurons, hidden_neurons, output_neurons, potential_decay=0.99, device=None):
        super().__init__(input_neurons, hidden_neurons, output_neurons, potential_decay, device)

    def mutate(self, change_rate, change_intensity, prune_rate):
        sign = self.weights.sign()
        delta = torch.empty_like(self.weights).uniform_(-change_intensity, change_intensity)
        delta[torch.empty_like(self.weights).uniform_(0, 1) > change_rate] = 0
        self.weights += delta
        self.weights[self.mask | (self.weights.sign() == -sign) | (torch.empty_like(self.weights).uniform_(0, 1) <= prune_rate)] = 0
        self.weights.clamp_(-1, 1)
