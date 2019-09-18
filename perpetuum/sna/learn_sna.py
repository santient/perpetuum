import torch

import SNA


class LearningSNA(SNA):
    def __init__(self, input_neurons, hidden_neurons, output_neurons, learn=True, lr=1e-3, discount=0.99, decay=0.99, history_limit=101, device="cpu"):
        super().__init__(input_neurons, hidden_neurons, output_neurons, device)
        self.learn = learn
        self.lr = lr
        self.discount = discount
        self.decay = decay
        self.history_limit = history_limit
        self.stdp_weights = torch.tensor([1 / x if x != 0 else 0 for x in range(-self.history_limit // 2, self.history_limit - self.history_limit // 2)], device=self.device)
        self.bias = torch.zeros_like(self.potentials)
        self.history = torch.zeros(self.history_limit, self.total_neurons, dtype=torch.bool, device=self.device)

    def reset(self):
        super().reset()
        self.bias[:] = 0
        self.history[:, :] = 0

    def stdp(self):
        if self.learn:
            sign = self.weights / self.weights.sign()
            spike = self.history[self.history_limit / 2]
            delta = self.lr * self.stdp_weights * torch.sum(self.history.float(), dim=0)
            delta[spike] = 0
            self.weights[spike[self.input_neurons:]] += delta
            self.weights[self.mask or self.weights.sign() != sign] = 0

    def step(self, inputs=None):
        if inputs is not None:
            self.potentials[:self.input_neurons] += inputs
        spike = self.potentials + self.bias >= 1
        if self.learn:
            self.stdp(spike)
            self.bias *= self.decay
            self.history = self.history.roll(1)
            self.history[0] = spike
        delta = torch.sum(self.weights[:, spike[:-self.output_neurons]], dim=1)
        self.potentials[self.input_neurons:] += delta
        self.potentials[spike] = 0
        torch.nn.functional.relu(self.potentials, inplace=True)
        outputs = spike[-self.output_neurons:]
        return outputs

    def reward(self, reward):
        if self.learn:
            self.bias += reward

    def enable_learning(self):
        self.learn = True

    def disable_learning(self):
        self.learn = False
        self.bias[:] = 0
        self.history[:, :] = 0
