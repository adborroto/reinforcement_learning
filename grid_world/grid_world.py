import numpy as np


class Environment:
    def __init__(self, initial_position):
        self.i = initial_position[0]
        self.j = initial_position[1]
        self.all_actions = ('U', 'D', 'L', 'R')

    def set(self, rewards, actions):
        self.rewards = rewards
        self.actions = actions

    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        return (self.i, self.j)

    def is_terminal_state(self, s):
        return s not in self.actions

    def get_state_reward(self):
        return self.rewards.get((self.i, self.j), 0)

    def move(self, action):
        if action in self.actions[(self.i, self.j)]:
            if action == 'U':
                self.i -= 1
            elif action == 'D':
                self.i += 1
            elif action == 'R':
                self.j += 1
            elif action == 'L':
                self.j -= 1
            
        return (self.i, self.j)
        
    def game_over(self):
        return (self.i, self.j) not in self.actions
    
    def all_states(self):
        return set(self.actions.keys()) | set(self.rewards.keys())


def standard_grid():
    """[summary]
        .  .  .  1
        .  x  . -1
        s  .  .  .
        Returns:
           Default grid world environment
        """
    g = Environment((2, 0))
    rewards = {(0, 3): 1, (1, 3): -1}
    actions = {
        (0, 0): ('D', 'R'),
        (0, 1): ('L', 'R'),
        (0, 2): ('L', 'D', 'R'),
        (1, 0): ('U', 'D'),
        (1, 2): ('U', 'D', 'R'),
        (2, 0): ('U', 'R'),
        (2, 1): ('L', 'R'),
        (2, 2): ('L', 'R', 'U'),
        (2, 3): ('L', 'U'),
    }
    g.set(rewards, actions)
    return g
