import torch

from sna.sna import SNA


class LearningSNA(SNA):
    def __init__(self, input_neurons, hidden_neurons, output_neurons, potential_decay=0.95, device=None, learn=True, lr=1e-2, reward_discount=0.95, bias_decay=0.95, history_limit=101):
        super().__init__(input_neurons, hidden_neurons, output_neurons, potential_decay, device)
        self.learn = learn
        self.lr = lr
        self.reward_discount = reward_discount
        self.bias_decay = bias_decay
        self.history_limit = history_limit
        self.stdp_weights = torch.tensor([1 / x if x != 0 else 0 for x in range(-(self.history_limit // 2), self.history_limit - self.history_limit // 2)], device=self.device).unsqueeze(1)
        # self.discount_weights = torch.tensor([self.reward_discount ** x for x in range(self.history_limit)], device=self.device)
        self.bias = torch.zeros_like(self.potentials)
        self.history = torch.zeros(self.history_limit, self.total_neurons, dtype=torch.bool, device=self.device)

    def reset(self):
        super().reset()
        self.bias[:] = 0
        self.history[:, :] = 0

    def stdp(self):
        if self.learn:
            sign = self.weights.sign()
            spike = self.history[self.history_limit // 2]
            delta = self.lr * torch.sum(self.stdp_weights * self.history, dim=0).unsqueeze(1)
            delta[spike] = 0
            self.weights[:, spike[:-self.output_neurons]] += delta[self.input_neurons:]
            self.weights[self.mask | (self.weights.sign() == -sign)] = 0
            self.weights.clamp_(-1, 1)

    def prune(self, threshold):
        self.weights[self.weights.abs() < threshold] = 0

    def step(self, inputs=None):
        if inputs is not None:
            self.potentials[:self.input_neurons] += inputs
        spike = self.potentials + self.bias >= 1
        if self.learn:
            self.history = self.history.roll(1, dims=0)
            self.history[0] = spike
            self.stdp()
            self.bias *= self.bias_decay
        delta = torch.sum(self.weights[:, spike[:-self.output_neurons]], dim=1)
        self.potentials *= self.potential_decay
        self.potentials[self.input_neurons:] += delta
        self.potentials[spike] = 0
        self.potentials[self.potentials < 0] = 0
        outputs = spike[-self.output_neurons:]
        return outputs

    def reward(self, r):
        if self.learn:
            # TODO parallelize
            for i, out_spike in enumerate(self.history):
                delta = torch.zeros_like(self.potentials)
                delta[-self.output_neurons:] = out_spike[-self.output_neurons:] * r
                for in_spike in self.history[i + 1:]:
                    delta[:-self.output_neurons] = self.weights.T @ delta[self.input_neurons:]
                    delta *= in_spike
                    delta[:self.input_neurons] = 0
                    delta[-self.output_neurons:] = 0
                    self.bias += delta
                    delta *= self.reward_discount
                r *= self.reward_discount

    def enable_learning(self):
        self.learn = True

    def disable_learning(self):
        self.learn = False
        self.bias[:] = 0
        self.history[:, :] = 0
