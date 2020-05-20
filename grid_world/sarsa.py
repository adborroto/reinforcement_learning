from grid_world import standard_grid
from value_iteration import initial_policy, print_policy, update_policy
import numpy as np


def choose_action(available_actions, current_action, eps=0.2):
    if np.random.random() < eps:
        return np.random.choice(available_actions)
    return current_action


def max_dict(d):
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val

def initial_Q(env, initial_value = 0):
    Q = {}
    for s in env.all_states():
        Q[s] = {}
        for a in env.all_actions:
            Q[s][a] = initial_value
    return Q

def sarsa(episodes = 1000, initial_state = (2, 0), alpha=0.1, gamma=0.9):
    env = standard_grid()
    Q = initial_Q(env, initial_value=0)
    s = initial_state

    for episode in range(episodes):
        s = initial_state
        a = max_dict(Q[s])[0]
        env.set_state(s)

        while not env.game_over():
            s2 = env.move(a)  # state
            r = env.get_state_reward()  # reward
            a2 = max_dict(Q[s2])[0]
            a2 = choose_action(env.all_actions, a2)  # action
            Q[s][a] = Q[s][a] + alpha * (r + gamma * Q[s2][a2] - Q[s][a])
            s = s2
            a = a2
    return (Q,env)
        

if __name__ == "__main__":
    Q,env = sarsa()
    print(Q)

    policy = {}
    V = {}
    for s in env.actions.keys():
        a, max_q = max_dict(Q[s])
        policy[s] = a
        V[s] = max_q

    print_policy(policy, title='Final policy')
