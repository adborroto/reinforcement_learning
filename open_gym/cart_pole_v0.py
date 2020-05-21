import gym
import numpy as np
import math
import time
Q = {}

def epsilon_greedy(env, current_action, eps=0.1):
    if np.random.random() < eps or current_action == None:
        return env.action_space.sample()
    return current_action


def max_action(d):
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key

def get_state_code(state, env):

    bounds = list(zip(env.observation_space.low, env.observation_space.high))
    bounds[0] = [-0.2,0.2]
    bounds[1] = [-0.5, 0.5]
    bounds[3] = [-math.radians(50), math.radians(50)]
    scale_factors = [20,10,20,10]
    encode_state = []
    for i in range(len(state)):
        value = state[i]
        min_bound = bounds[i][0]
        max_bound = bounds[i][1]
        value = min_bound if value< min_bound else value
        value = max_bound if value > max_bound else value
        width = max_bound - min_bound
        x = int(((value - min_bound ) / width) * scale_factors[i])
        encode_state.append(x)
    return tuple(encode_state)

        

    return state_value_bounds

def Q_state(state):
    if not state in Q:
        Q[state] = {}
    return Q[state]


def Q_state_action(state, action, initial_value = 0):
    qa = Q_state(state)
    if not action in qa:
        Q[state][action] = initial_value
    return Q[state][action]


def train(env, iter=2000, alpha=0.1, gamma=0.9):
    for i_episode in range(iter):
        s = env.reset()
        a = epsilon_greedy(env, env.action_space.sample())
        while True:
            state = get_state_code(s, env)
            a2 = max_action(Q_state(state))
            a2 = epsilon_greedy(env, a2)
            s2, reward, game_over, info = env.step(a2)
            state2 = get_state_code(s2, env)
            qsa = Q_state_action(state,a)
            qsa2 = Q_state_action(state2,a2)
            r = -10 * reward  if game_over else reward
            Q[state][a] = qsa + alpha * (r + gamma * qsa2 - qsa)

            s = s2
            a = a2

            if game_over:
                break
    return (Q,env)


def play(env):
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            time.sleep(.300)
            print(observation)
            state  = get_state_code(observation, env)
            print(state)
            action = max_action(Q[state])
            #action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    train(env)
    play(env)
    env.close()
