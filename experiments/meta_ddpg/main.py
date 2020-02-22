import torch

from perpetuum.sna import NoisySNA
from perpetuum.env import Debug
from perpetuum.learn.meta_ddpg import MetaDDPGTrainer


if __name__ == "__main__":
    device = torch.device("cuda")
    sna = NoisySNA(2, 4, 2, device=device)
    env = Debug(steps=1000, device=device)
    trainer = MetaDDPGTrainer(sna, env, agent_params={"critic_layers": 1})
    trainer.train(100, 32)
    # trainer.save("test.pkl")
