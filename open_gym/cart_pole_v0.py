import gym
import numpy as np
import math
import time
Q = {}
np.random.seed = 32

def epsilon_greedy(env, current_action, eps=0.1):
    if np.random.random() < eps or current_action == None:
        return env.action_space.sample()
    return current_action


def max_action(d, env):
    max_key = env.action_space.sample()
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key

def get_state_code(state, env):

    bounds = list(zip(env.observation_space.low, env.observation_space.high))
    bounds[1] = [-0.5, 0.5]
    bounds[3] = [-math.radians(50), math.radians(50)]
    scale_factors = [3,3,8,6]
    encode_state = []
    for i in range(len(state)):
        value = state[i]
        min_bound = bounds[i][0]
        max_bound = bounds[i][1]
        value = min_bound if value< min_bound else value
        value = max_bound if value > max_bound else value
        width = (max_bound - min_bound ) / scale_factors[i]
        x = int(((value - min_bound ) / width) )
        encode_state.append(x)
    return tuple(encode_state)

def Q_state(state):
    if not state in Q:
        Q[state] = {}
    return Q[state]


def Q_state_action(state, action, initial_value = 0):
    qa = Q_state(state)
    if not action in qa:
        Q[state][action] = initial_value
    return Q[state][action]


def train(env, iter=3000, alpha=0.1, gamma=0.9):
    for i_episode in range(iter):
        s = env.reset()
        a = epsilon_greedy(env, env.action_space.sample())
        while True:
            state = get_state_code(s, env)
            a = epsilon_greedy(env, max_action((Q_state(state)),env))

            s2, reward, game_over, info = env.step(a)
            state2 = get_state_code(s2, env)
            a2 = max_action((Q_state(state2)),env)
            
            qsa = Q_state_action(state,a)
            qsa2 = Q_state_action(state2,a2)
            
            if game_over:
                Q[state][a] = -1
            else: 
                Q[state][a] = qsa + alpha * (reward + gamma * qsa2 - qsa)
            s = s2
            a = a2

            if game_over:
                break
    return (Q,env)


def test(env):
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            state  = get_state_code(observation, env)
            action = max_action(Q[state],env)
            observation, reward, done, info = env.step(action)
            if done:
                print("Game over after {} timesteps".format(t+1))
                break
            if t == 99:
                print("Win after {} timesteps".format(t+1))


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    train(env)
    test(env)
    env.close()
