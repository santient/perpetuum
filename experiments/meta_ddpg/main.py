import torch

from perpetuum.sna import NoisySNA
from perpetuum.env import Inertia
from perpetuum.learn.meta_ddpg import MetaDDPGTrainer


if __name__ == "__main__":
    device = torch.device("cuda")
    sna = NoisySNA(13, 100, 4, device=device)
    env = Inertia(100, 100, 0.1, 0.1, device=device)
    trainer = MetaDDPGTrainer(sna, env, {"critic_layers": 3}, {})
    trainer.train(100000, 32, 1000)
    print(sna.weights, sna.potentials)
