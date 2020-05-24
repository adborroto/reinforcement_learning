import gym
import numpy as np
import math
import time


def test(env):
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            observation_ram = env.unwrapped._get_ram()
            print(len(observation_ram))
            time.sleep(.03)
            if done:
                break


if __name__ == "__main__":
    env = gym.make('Breakout-v0')
    test(env)
    env.close()
