import gym
import numpy as np
import math
import time
import seaborn as sns;
import matplotlib.pyplot as plt
from cart_pole_v0 import epsilon_greedy, max_action
sns.set()
Q = {}

np.random.seed = 32

def Q_state(state):
    if not state in Q:
        Q[state] = {}
    return Q[state]


def Q_state_action(state, action, initial_value = 0):
    qa = Q_state(state)
    if not action in qa:
        Q[state][action] = initial_value
    return Q[state][action]


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[(I == 144) | (I==109)] = 0 # erase background
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()
   

def ball_state(ball):
    if ball is None:
        return (float('NaN'),float('NaN')) # No ball
    y =  ball[0].mean()
    x =  ball[1].mean()
    return (x,y)

def observation_to_state(obs):
    if obs is None:
        return
    map = np.array(prepro(obs)).reshape((80,80 ))
    left_paddel = np.where(map[:,7:9] == 1)[0]
    right_paddel = np.where(map[:,69:71] == 1)[0]

    ball = np.where(map[:,10:68] != 0)
    ball_s = ball_state(ball)
    state = (int(np.nan_to_num(np.mean(left_paddel))), int(np.mean(right_paddel)),ball_s[0], ball_s[1])
    return state

def test(env):
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            
            state = observation_to_state(observation)
            if state in Q:
                action = max_action(Q[state],env)
                print(action)
            else:
                print('I dont know')
                action = env.action_space.sample()

            observation, reward, done, info = env.step(action)
            time.sleep(.1)
            if done:
                break


def train(env, iter=5, alpha=0.1, gamma=0.9):
    for i_episode in range(iter):
        if i_episode%10 ==0:
            print(i_episode)
        s = env.reset()
        a = epsilon_greedy(env, env.action_space.sample())
        while True:
            state = observation_to_state(s)
            a = epsilon_greedy(env, max_action((Q_state(state)),env))

            s2, reward, game_over, info = env.step(a)
            state2 = observation_to_state(s2)
            a2 = max_action((Q_state(state2)),env)
            
            qsa = Q_state_action(state,a)
            qsa2 = Q_state_action(state2,a2)
            
            if reward < 0:
                Q[state][a] = -1
            if reward > 0:
                Q[state][a] = 1
            else: 
                Q[state][a] = qsa + alpha * (reward + gamma * qsa2 - qsa)
            s = s2
            a = a2

            if game_over:
                break
    return (Q,env)


if __name__ == "__main__":
    env = gym.make('Pong-v0')
    train(env, iter=150)
    test(env)
    env.close()
