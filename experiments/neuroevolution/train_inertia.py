import copy

import torch

from perpetuum.sna import EvolvingSNA
from perpetuum.env import Inertia

def train(): # TODO generalize and parameterize
    torch.multiprocessing.set_start_method("spawn")
    device = torch.device("cuda:0")
    population = [EvolvingSNA(12, 100, 4, potential_decay=0.99, device=device) for i in range(100)]
    for model in population:
        model.initialize(0.1)
    env = Inertia(10, 10, 0.1, 0.1, speed=1e-2, episode_steps=1000, device=device)
    best = evolution(population, env, 10, (0.01, 0.1, 0.01), 100)
    print("Training finished, final top score:", best)

def evolution(population, env, bottleneck, mutation_params, generations):
    assert bottleneck < len(population)
    for i in range(generations):
        scores = generation(population, env)
        survivors, top_scores = zip(*list(sorted(zip(population, scores), key=lambda x: -x[1]))[:bottleneck])
        survivors = list(survivors)
        for model in survivors:
            model.reset()
        torch.save(survivors[0].weights, "experiments/neuroevolution/checkpoints/best_gen_{}.pt".format(i))
        top_scores = list(top_scores)
        print("(Generation {}) (Top Score {})".format(i, top_scores[0]))
        mutants = [copy.deepcopy(survivors[i % len(survivors)]) for i in range(len(population) - bottleneck)]
        for model in mutants:
            model.mutate(*mutation_params)
        population[:] = survivors + mutants
        env.reset()
    return top_scores[0]

def generation(population, env):
    envs = [copy.deepcopy(env) for model in population]
    # scores = [0] * len(population)
    # torch.multiprocessing.spawn(trial, (population, envs, scores), len(population))
    # return scores
    pool = torch.multiprocessing.Pool(8)
    return pool.starmap(trial, zip(population, envs)) # TODO fix memory issues
    # return [trial(model, env) for model, env in zip(population, envs)]

def trial_multi(i, population, envs, scores):
    model = population[i]
    env = envs[i]
    score = 0
    terminal = False
    observation = None
    while not terminal:
        action = model.step(observation)
        observation, reward, terminal = env.step(action)
        if reward is not None:
            score += reward
    scores[i] = score

def trial(model, env):
    score = 0
    terminal = False
    observation = None
    while not terminal:
        action = model.step(observation)
        observation, reward, terminal = env.step(action)
        if reward is not None:
            score += reward
    return score
