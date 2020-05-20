from grid_world import standard_grid
from value_iteration import initial_policy, print_policy, update_policy
from sarsa import max_dict, initial_Q, choose_action
import numpy as np

def q_learning(episodes = 2000, initial_state = (2, 0), alpha=0.1, gamma=0.9):
    env = standard_grid()
    Q = initial_Q(env, initial_value=0)
    s = initial_state

    for episode in range(episodes):
        s = initial_state
        env.set_state(s)

        while not env.game_over():
            a = choose_action(env.all_actions, max_dict(Q[s])[0])  # action
            s2 = env.move(a)  # state
            r = env.get_state_reward()  # reward
            a2 = max_dict(Q[s2])[0]
            Q[s][a] = Q[s][a] + alpha * (r + gamma * Q[s2][a2] - Q[s][a])
            s = s2
            a = a2
    return (Q,env)
         

if __name__ == "__main__":
    Q,env = q_learning()
    print(Q)

    policy = {}
    V = {}
    for s in env.actions.keys():
        a, max_q = max_dict(Q[s])
        policy[s] = a
        V[s] = max_q

    print_policy(policy, title='Final policy')
