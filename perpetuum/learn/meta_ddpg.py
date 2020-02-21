from collections import deque
import random

import torch
from torch import nn
from torch import optim


class OUNoise():
    def __init__(self, action_shape, low, high, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000, device=None):
        self.action_shape = action_shape
        self.low = low
        self.high = high
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        self.reset()
        
    def reset(self):
        self.state = torch.ones(*self.action_shape, device=self.device) * self.mu
        self.t = 0
        
    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * torch.randn(*self.action_shape, device=self.device)
        self.state = x + dx
        self.t += 1
        return self.state
    
    def get_action(self, action):
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, self.t / self.decay_period)
        ou_state = self.evolve_state()
        return torch.clamp(action + ou_state, self.low, self.high)

class Memory():
    def __init__(self, max_size, device=None):
        self.buffer = deque(maxlen=max_size)
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
    
    def push(self, weights, reward):
        self.buffer.append((weights, reward))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        weights_batch, reward_batch = zip(*batch)
        return torch.stack(weights_batch).to(self.device), torch.tensor(reward_batch, device=self.device)

    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, sna):
        super(Actor, self).__init__()
        self.mask = sna.mask
        self.weights = nn.Parameter(self.mask * torch.randn_like(sna.weights))

    def forward(self):
        return self.mask * torch.tanh(self.weights)

class Critic(nn.Module):
    def __init__(self, sna, n_layers=1):
        super(Critic, self).__init__()
        self.mask = sna.mask
        m = sna.weights.shape[0]
        n = sna.weights.shape[1]
        self.n_layers = n_layers
        self.L = nn.ParameterList(nn.Parameter(torch.randn(m, m, device=sna.device)) for i in range(n_layers))
        self.R = nn.ParameterList(nn.Parameter(torch.randn(n, n, device=sna.device)) for i in range(n_layers))
        self.outL = nn.Parameter(torch.randn(m, m, device=sna.device))
        self.outR = nn.Parameter(torch.randn(n, n, device=sna.device))

    def forward(self, W):
        out = W
        for i in range(self.n_layers):
            out = torch.relu(self.L[i] @ out @ self.R[i])
        out = self.mask * (self.outL @ out @ self.outR)
        return out

class MetaDDPGAgent():
    def __init__(self, sna, prune_threshold=0.1, actor_learning_rate=1e-4, critic_learning_rate=1e-3, critic_layers=1, gamma=0.99, tau=1e-2, max_memory_size=50000, device=None):
        self.sna = sna
        self.prune_threshold = prune_threshold
        self.gamma = gamma
        self.tau = tau
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        self.actor = Actor(sna).to(self.device)
        self.actor_target = Actor(sna).to(self.device)
        self.critic = Critic(sna, critic_layers).to(self.device)
        self.critic_target = Critic(sna, critic_layers).to(self.device)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        self.memory = Memory(max_memory_size, device=self.device)        
        self.critic_criterion = nn.MSELoss().to(self.device)
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    def step(self, state, env, noise):
        weights = noise.get_action(self.actor().detach())
        self.sna.weights = weights
        self.sna.prune(self.prune_threshold)
        action = self.sna.step(state)
        next_state, reward = env.step(action)
        return weights, next_state, reward
    
    def update(self, batch_size):
        weights, rewards = self.memory.sample(batch_size)    
        Qvals = self.critic(weights)
        next_weights = torch.stack([self.actor_target()] * batch_size)
        next_Q = self.critic_target(next_weights.detach())
        Qprime = rewards.view(batch_size, 1, 1) + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)
        policy_loss = -self.critic(self.actor()).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

class MetaDDPGTrainer():
    def __init__(self, sna, env, agent_params, noise_params):
        self.sna = sna
        self.env = env
        self.agent = MetaDDPGAgent(sna, **agent_params, device=sna.device)
        self.noise = OUNoise(sna.weights.shape, -1.0, 1.0, **noise_params, device=sna.device)

    def train(self, steps, batch_size, report_iter):
        print("Training model...")
        self.sna.reset()
        state = self.env.reset()
        self.noise.reset()
        avg_reward = []
        for step in range(steps):
            weights, next_state, reward = self.agent.step(state, self.env, self.noise)
            self.agent.memory.push(weights.cpu(), reward)
            state = next_state
            if len(self.agent.memory) >= batch_size:
                self.agent.update(batch_size)
            avg_reward.append(reward)
            if len(avg_reward) == report_iter:
                avg_reward = sum(avg_reward) / len(avg_reward)
                print("Step {} average reward: {}".format(step, avg_reward))
                avg_reward = []
        print("Training complete.")

    def save(path):
        pass

    def load(path):
        pass
