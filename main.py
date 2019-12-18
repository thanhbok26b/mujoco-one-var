import gym
import mujoco_py
import numpy as np
import matplotlib.pyplot as plt

def estimate_observation_range(env, num_episode):
    observations = []
    for i_episode in range(num_episode):
        observation = env.reset()
        observations.append(observation)
        while 1:
            observation, _, done, _ = env.step(env.action_space.sample())
            observations.append(observation)
            if done:
                break
    return np.mean(observations, axis=0), np.std(observations, axis=0)

class Policy(object):

    def __init__(self, env, seed, mean, std):
        n = env.observation_space.shape[0]
        p = env.action_space.shape[0]
        np.random.seed(seed)
        self.weights = np.random.rand(p, n)
        self.mean = mean
        self.std = std

    def get_action(self, observation, weight=None):
        if weight is not None:
            self.weights[0, 0] = weight
        norm_observation = (observation - self.mean) / self.std
        action = self.weights @ norm_observation
        action = np.clip(action, 0, 1)
        return action

def run_episode(env, policy, weight=None):
    observation = env.reset()
    fitness = 0
    while 1:
        action = policy.get_action(observation, weight)
        observation, reward, done, info = env.step(env.action_space.sample())
        fitness += reward
        if done:
            break
    return fitness

if __name__ == '__main__':
    env = gym.make('Hopper-v1')
    num_episode = 1000
    seed = 0
    # Estimate state Mean and Std
    mean, std = estimate_observation_range(env, num_episode)
    # Initialize Policy
    policy = Policy(env, seed, mean, std)
    x = np.linspace(0, 1, 11)
    y = []
    for weight in x:
        fitnesses = []
        for i_episode in range(num_episode):
            fitnesses.append(run_episode(env, policy, weight))
        y.append(np.mean(fitnesses))
        print(weight, y[-1])
    # Plot the analysis
    plt.plot(x, y)
    plt.show()
