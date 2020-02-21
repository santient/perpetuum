import torch

from perpetuum.sna import SNA


class NoisySNA(SNA):
    def __init__(self, input_neurons, hidden_neurons, output_neurons, potential_decay=0.99, device=None, noise_eps=0.1):
        super().__init__(input_neurons, hidden_neurons, output_neurons, potential_decay, device)
        self.noise_eps = noise_eps

    def step(self, inputs=None):
        if inputs is not None:
            self.potentials[:self.input_neurons] += inputs
        fire = (self.potentials >= 1).type(torch.uint8)
        delta = self.weights @ fire[:-self.output_neurons].type_as(self.weights)
        self.potentials *= self.potential_decay
        self.potentials[self.input_neurons:] += delta + self.noise_eps * torch.randn_like(self.potentials[self.input_neurons:])
        self.potentials *= 1 - fire
        mask = (self.potentials >= 0).type(torch.uint8)
        self.potentials *= mask
        outputs = fire[-self.output_neurons:]
        self.timestep += 1
        return outputs
