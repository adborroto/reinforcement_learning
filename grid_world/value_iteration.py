from grid_world import standard_grid
import numpy as np
import sys



def print_policy(P, title):
    print("---------------------------")
    print(title)
    for i in range(3):
        print("---------------------------")
        for j in range(4):
            a = P.get((i,j), ' ')
            print("  %s  |" % a, end="")
        print("")

def initial_policy(grid):
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(grid.all_actions)
    return policy 

def value_iteration(grid,policy, e = 1e-4, gamma=0.9):
    
    V = {}
    states = grid.all_states()
    for s in states:
        V[s] = 0.1
    
    previous_update = float('inf')
    
    while True:
        for s in states:
            Vs = V[s]

            if not s in policy:
                continue
            best_v = float('-inf')
            for a in grid.all_actions:
                grid.set_state(s)
                new_state = grid.move(a)
                r = grid.get_state_reward()
                new_v = r + gamma  * V[new_state]
                if new_v > best_v:
                    best_v = new_v
            V[s] = best_v
            previous_update = min(previous_update, np.abs(Vs - V[s]))

        if previous_update < e:
            return V

def update_policy(V, policy, grid, gamma=0.9):
    for s in policy.keys():
        best_a = None
        best_value = float('-inf')
        for a in grid.all_actions:
            grid.set_state(s)
            new_state = grid.move(a)
            r = grid.get_state_reward()
            new_v = r + gamma  * V[new_state]
            if new_v > best_value:
                best_value = new_v
                best_a = a
        policy[s] = best_a

if __name__ == "__main__":
    env = standard_grid()
    policy = initial_policy(env)
    v = value_iteration(env,policy)

    print_policy(policy, 'Initial policy')
    update_policy(v,policy,env)
    print_policy(policy, 'Final policy')
