from grid_world import standard_grid
from value_iteration import initial_policy, print_policy, update_policy
import numpy as np


def choose_action(available_actions, current_action, eps=0.2):
    if np.random.random() < eps:
        return np.random.choice(available_actions)
    return current_action


def play_game(env, policy, initial_state):
    env.set_state(initial_state)
    s = initial_state
    s_r = [(s, 0)]

    while not env.game_over():
        a = choose_action(env.all_actions, policy[s])  # action
        s = env.move(a)  # state
        r = env.get_state_reward()  # reward
        s_r.append((s, r))

    return s_r


def temporal_difference(alpha=0.1, gamma=0.9):
    env = standard_grid()
    V = {}
    policy = initial_policy(env)
    states = env.all_states()
    for s in states:
        V[s] = 0

    for i in range(2000):

        s_r = play_game(env, policy, (2, 0))
        for t in range(len(s_r)-1):
            s, r = s_r[t]
            s1, r1 = s_r[t+1]
            V[s] = V[s] + alpha * (r1 + gamma * V[s1] - V[s])

    return (env, V, policy)


if __name__ == "__main__":
    env, v, policy = temporal_difference()
    print(v)

    print_policy(policy, 'Initial policy')
    update_policy(v, policy, env)
    print_policy(policy, 'Final policy')
